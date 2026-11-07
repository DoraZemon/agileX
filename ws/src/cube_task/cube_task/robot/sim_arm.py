"""
Kinematic PiPER arm + parallel gripper for simulation and tests.

Joint state, FK and IK come from robot.kinematics.PiperKinematics, so a target
the real controller cannot reach (joint limits) also fails here. Time passes
only through the injected Clock (core.clock.SimClock in simulation): a motion
calls clock.sleep() in steps of at most 0.1 s and moves a held object along
via world.update_tcp(), so a camera rendering the world sees it move.

Timing model (config, units s / m / rad):
    duration = robot.motion.sim_overhead_s
               + linear: distance / (robot.motion.sim_lin_speed * speed / 100)
               + P2P:    max |dq| / (sim.joint_speed * speed / 100)
A gripper command takes robot.gripper.settle_s.

Faults for tests: ``inject(kind, at_call=None, count=1)``, see FAULT_KINDS.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.interfaces import ArmInterface, Clock
from ..core.transforms import inv_T, pose_error
from ..core.types import ArmHealth, GripperResult, MotionResult, Stamped
from .kinematics import flange_to_tcp_from_cfg, PiperKinematics
from .safety import interpolate_pose

LOG = logging.getLogger('cube_task.robot.sim_arm')

#: timeout      next motion takes its timeout and stops half way -> error 'timeout'
#: unreachable  next motion fails at once -> error 'unreachable'
#: comm_loss    health() not ok from now on; every command -> error 'comm'
#: grasp_miss   next closing gripper command grabs nothing
#: drop         the held object falls during the next motion; width then reads
#:              the commanded close width (about 0)
#: arm_error    next motion fails at once -> error 'arm_error'
FAULT_KINDS = ('timeout', 'unreachable', 'comm_loss', 'grasp_miss', 'drop', 'arm_error')
_MOTION_FAULTS = ('timeout', 'unreachable', 'arm_error', 'drop')
_MAX_STEP_S = 0.1
_WIDTH_EPS = 1e-4


def _cfg_float(cfg, key: str, default: float, positive: bool = True) -> float:
    v = cfg.get(key, default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError('config %s must be a number, got %r' % (key, v))
    if not math.isfinite(f) or (positive and f <= 0) or f < 0:
        raise ValueError('config %s must be a finite %s number, got %r'
                         % (key, 'positive' if positive else 'non-negative', v))
    return f


class _Fault:
    def __init__(self, kind: str, at_call: Optional[int], count: int):
        self.kind = kind
        self.at_call = at_call
        self.remaining = count


class SimArm(ArmInterface):
    """
    Simulated PiPER implementing ArmInterface.

    cfg: core.config.Config. clock: Clock (SimClock for fast simulation).
    world: object with try_grasp(T_base_tcp, open_width, close_width) ->
    Optional[float], update_tcp(T_base_tcp, width), release(T_base_tcp) and a
    ``held`` attribute (sim.world.SimWorld), or None for a bare kinematic arm.

    Like the real arm, motions need connect() and enable() first. The
    gripper starts closed (width 0), so a pick that forgets to open fails.
    """

    def __init__(self, cfg, clock: Clock, world: Any = None):
        self._cfg = cfg
        self._clock = clock
        self._world = world
        self.kin = PiperKinematics(str(cfg.get('robot.model', None)))
        self._T_ft = flange_to_tcp_from_cfg(cfg)
        self._T_tf = inv_T(self._T_ft)
        self._overhead = _cfg_float(cfg, 'robot.motion.sim_overhead_s', 0.35, positive=False)
        self._lin_speed = _cfg_float(cfg, 'robot.motion.sim_lin_speed', 0.15)
        self._joint_speed = _cfg_float(cfg, 'sim.joint_speed', 2.0)
        self._timeout_base = _cfg_float(cfg, 'robot.motion.timeout_base_s', 6.0)
        self._timeout_per_m = _cfg_float(cfg, 'robot.motion.timeout_per_m_s', 15.0,
                                         positive=False)
        self._settle = _cfg_float(cfg, 'robot.gripper.settle_s', 0.6, positive=False)
        self._close_width = _cfg_float(cfg, 'robot.gripper.close_width', 0.0, positive=False)
        if cfg.get('robot.gripper.max_width', None) is None:
            raise ValueError('config robot.gripper.max_width is required (m)')
        self._max_width = _cfg_float(cfg, 'robot.gripper.max_width', 0.0)
        self._noise = _cfg_float(cfg, 'sim.arm_joint_noise_std', 0.0, positive=False)
        self._rng = np.random.RandomState(int(cfg.get('sim.seed', 0) or 0))
        self._speed = 100
        self.set_speed_percent(cfg.get('robot.speed_percent', 100))
        q0 = cfg.get('robot.poses.home_joints', None)
        q0 = [0.0] * 6 if q0 is None else q0
        if not self.kin.within_limits(q0):
            raise ValueError('robot.poses.home_joints outside joint limits: %r' % (q0,))
        self._q = np.asarray(q0, dtype=float).copy()
        self._width = 0.0
        self._cmd_width = 0.0
        self._connected = False
        self._enabled = False
        self._estop = False
        self._comm_lost = False
        self._faults: List[_Fault] = []
        self.n_calls = 0
        self.fault_log: List[Dict[str, Any]] = []
        self.command_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ test hooks
    def inject(self, kind: str, at_call: Optional[int] = None, count: int = 1) -> None:
        """
        Arm a fault (see FAULT_KINDS).

        at_call: 1-based index of the command (move_tcp / move_joints /
        gripper, counted in n_calls) from which the fault is armed; None = the
        next applicable command. count: how many applicable commands it hits.
        """
        if kind not in FAULT_KINDS:
            raise ValueError('unknown fault %r, expected one of %s' % (kind, FAULT_KINDS))
        if count < 1:
            raise ValueError('count must be >= 1')
        if at_call is not None and at_call < 1:
            raise ValueError('at_call is 1-based')
        self._faults.append(_Fault(kind, at_call, int(count)))

    def set_joints(self, q: Sequence[float]) -> None:
        """Teleport the simulated arm (scenario setup only)."""
        if not self.kin.within_limits(q):
            raise ValueError('joints outside limits: %r' % (list(q),))
        self._q = np.asarray(q, dtype=float).copy()
        self._update_world()

    @property
    def world(self) -> Any:
        return self._world

    def _take_fault(self, kinds: Sequence[str]) -> Optional[str]:
        for f in self._faults:
            if f.kind in kinds and (f.at_call is None or self.n_calls >= f.at_call):
                f.remaining -= 1
                if f.remaining <= 0:
                    self._faults.remove(f)
                self.fault_log.append({'call': self.n_calls, 'kind': f.kind,
                                       't': self._clock.now()})
                LOG.info('sim fault %s at call %d', f.kind, self.n_calls)
                return f.kind
        return None

    def _peek_fault(self, kind: str) -> bool:
        return any(f.kind == kind and (f.at_call is None or self.n_calls >= f.at_call)
                   for f in self._faults)

    # ------------------------------------------------------------ state
    def _tcp(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        return self.kin.fk(self._q if q is None else q) @ self._T_ft

    def _update_world(self, T_tcp: Optional[np.ndarray] = None) -> None:
        if self._world is not None:
            self._world.update_tcp(self._tcp() if T_tcp is None else T_tcp, self._width)

    def _feedback_ok(self) -> bool:
        return self._connected and not self._comm_lost

    def connect(self) -> None:
        """Open the simulated link. Raises nothing (the sim is always there)."""
        self._connected = True
        self._update_world()

    def enable(self, timeout: float = 5.0) -> bool:
        """Enable the simulated motors; False without communication."""
        if not self._feedback_ok():
            return False
        self._enabled = True
        return True

    def health(self) -> ArmHealth:
        """Comm / e-stop state of the simulated arm."""
        if not self._connected:
            return ArmHealth(ok=False, reason='not connected', comm_ok=False,
                             enabled=self._enabled)
        if self._comm_lost:
            return ArmHealth(ok=False, reason='simulated communication loss', comm_ok=False,
                             enabled=self._enabled)
        if self._estop:
            return ArmHealth(ok=False, reason='emergency stop active', comm_ok=True,
                             enabled=self._enabled, ctrl_mode=1, arm_status=1)
        return ArmHealth(ok=True, comm_ok=True, enabled=self._enabled, ctrl_mode=1,
                         arm_status=0, detail={'gripper_ok': True})

    def get_joints(self) -> Optional[Stamped]:
        """Joint angles (rad), None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._q.copy(), self._clock.now())

    def get_flange_pose(self) -> Optional[Stamped]:
        """T_base_flange, None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self.kin.fk(self._q), self._clock.now())

    def get_tcp_pose(self) -> Optional[Stamped]:
        """T_base_tcp, None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._tcp(), self._clock.now())

    def gripper_width(self) -> Optional[Stamped]:
        """Measured opening (m), None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(float(self._width), self._clock.now())

    def set_speed_percent(self, percent: int) -> None:
        """Set the speed scale used by the timing model (1..100)."""
        try:
            p = int(percent)
        except (TypeError, ValueError):
            raise ValueError('speed percent must be an integer, got %r' % (percent,))
        if p != percent or not 1 <= p <= 100:
            raise ValueError('speed percent must be an integer in [1, 100], got %r' % (percent,))
        self._speed = p

    def stop(self) -> None:
        """Nothing to do: simulated motions are atomic."""

    def emergency_stop(self) -> None:
        """Latch an e-stop: health() not ok and motions fail until a new SimArm."""
        self._estop = True
        self._enabled = False

    def close(self) -> None:
        """Disconnect (motors stay enabled, as on the real arm)."""
        self._connected = False

    # ------------------------------------------------------------ motion
    def _motion_gate(self) -> Optional[MotionResult]:
        self.n_calls += 1
        if not self._connected:
            return MotionResult(ok=False, error='comm', detail='not connected')
        if self._comm_lost or self._take_fault(('comm_loss',)):
            self._comm_lost = True
            return MotionResult(ok=False, error='comm', detail='simulated communication loss')
        if self.abort_event.is_set():
            return MotionResult(ok=False, error='aborted', detail='abort requested')
        if self._estop:
            return MotionResult(ok=False, error='arm_error', detail='emergency stop active')
        if not self._enabled:
            return MotionResult(ok=False, error='arm_error',
                                detail='motors not enabled, call enable() first')
        return None

    def _instant_failure(self, error: str, detail: str) -> MotionResult:
        t0 = self._clock.now()
        self._clock.sleep(self._overhead)
        return MotionResult(ok=False, error=error, detail=detail,
                            duration=self._clock.now() - t0, final_tcp=self._tcp())

    def _result(self, ok: bool, error: str, detail: str, t0: float,
                T_goal: np.ndarray) -> MotionResult:
        T_now = self._tcp()
        dp, dr = pose_error(T_now, T_goal)
        return MotionResult(ok=ok, error=error, detail=detail,
                            duration=self._clock.now() - t0, final_tcp=T_now,
                            pos_err=dp, rot_err_deg=math.degrees(dr))

    def _execute(self, q_goal: np.ndarray, T_goal: np.ndarray, linear: bool,
                 timeout: Optional[float]) -> MotionResult:
        t0 = self._clock.now()
        q0 = self._q.copy()
        T0 = self._tcp()
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        scale = self._speed / 100.0
        if linear:
            move_t = dist / (self._lin_speed * scale)
        else:
            move_t = float(np.max(np.abs(q_goal - q0))) / (self._joint_speed * scale)
        duration = self._overhead + move_t
        limit = (self._timeout_base + self._timeout_per_m * dist) if timeout is None \
            else float(timeout)
        stop_frac, total, error = 1.0, duration, ''
        if self._take_fault(('timeout',)):
            stop_frac, total, error = 0.5, limit, 'timeout'
        elif duration > limit:
            stop_frac = 0.0 if move_t <= 0 else max(0.0, min(1.0, (limit - self._overhead)
                                                             / move_t))
            total, error = limit, 'timeout'
        holding = self._world is None or getattr(self._world, 'held', None) is not None
        drop = holding and self._peek_fault('drop') and self._take_fault(('drop',)) == 'drop'
        n = max(1, int(math.ceil(total / _MAX_STEP_S - 1e-9)))
        for k in range(1, n + 1):
            self._clock.sleep(total / n)
            t = total * k / n
            s = 1.0 if move_t <= 0 else max(0.0, min(1.0, (t - self._overhead) / move_t))
            s = min(s, stop_frac)
            T_s = None
            if linear and s > 0:
                T_s = interpolate_pose(T0, T_goal, s)
                q_s, _ep, _eo = self.kin.ik(T_s @ self._T_tf, q_seed=self._q, n_random=0)
                self._q = q_s if q_s is not None else q0 + s * (q_goal - q0)
            else:
                self._q = q0 + s * (q_goal - q0)
            if drop and s >= 0.5:
                self._drop()
                drop = False
            self._update_world(T_s)
            if self.abort_event.is_set() and k < n:
                return self._result(False, 'aborted', 'abort requested during motion', t0,
                                    T_goal)
        if drop:
            self._drop()
            self._update_world()
        if error:
            return self._result(False, error, 'motion did not finish within %.2f s' % limit,
                                t0, T_goal)
        q_end = q_goal.copy()
        if self._noise > 0:
            q_end = self.kin.clamp(q_end + self._rng.normal(0.0, self._noise, 6))
        self._q = q_end
        self._update_world()
        return self._result(True, '', '', t0, T_goal)

    def _drop(self) -> None:
        where = None
        if self._world is not None and getattr(self._world, 'held', None) is not None:
            where = self._world.release(self._tcp())
        self._width = self._cmd_width
        LOG.info('sim: held object dropped (landed in %s)', where or 'table')

    def move_tcp(self, T_base_tcp: np.ndarray, linear: bool = False,
                 timeout: Optional[float] = None) -> MotionResult:
        """Move the TCP to T_base_tcp (IK from the current joints, then timed motion)."""
        gate = self._motion_gate()
        if gate is not None:
            return gate
        T = np.asarray(T_base_tcp, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return MotionResult(ok=False, error='rejected', detail='invalid target pose')
        self.command_log.append({'cmd': 'move_tcp', 'linear': bool(linear), 'T': T.copy(),
                                 't': self._clock.now()})
        fault = self._take_fault(('unreachable', 'arm_error'))
        if fault:
            return self._instant_failure(fault, 'simulated %s' % fault)
        try:
            q_goal, ep, eo = self.kin.ik(T @ self._T_tf, q_seed=self._q)
        except ValueError as exc:
            return MotionResult(ok=False, error='rejected', detail=str(exc))
        if q_goal is None:
            return self._instant_failure('unreachable', 'no IK solution (residual %.4f m, '
                                         '%.2f deg)' % (ep, math.degrees(eo)))
        q_goal, _ep, _eo = self.kin.refine(T @ self._T_tf, q_goal)   # exact final pose
        return self._execute(q_goal, T, linear, timeout)

    def move_joints(self, q: Sequence[float], timeout: Optional[float] = None) -> MotionResult:
        """Joint-space move to q (rad)."""
        gate = self._motion_gate()
        if gate is not None:
            return gate
        qt = np.asarray(q, dtype=float).reshape(-1)
        if qt.shape[0] != 6 or not np.all(np.isfinite(qt)):
            return MotionResult(ok=False, error='rejected', detail='invalid joint target')
        self.command_log.append({'cmd': 'move_joints', 'q': qt.copy(), 't': self._clock.now()})
        if not self.kin.within_limits(qt):
            return self._instant_failure('unreachable', 'joint limits: '
                                         + ', '.join(self.kin.limit_violations(qt)))
        fault = self._take_fault(('unreachable', 'arm_error'))
        if fault:
            return self._instant_failure(fault, 'simulated %s' % fault)
        return self._execute(qt, self._tcp(qt), False, timeout)

    # ------------------------------------------------------------ gripper
    def gripper(self, width: float, force: Optional[float] = None,
                wait: bool = True) -> GripperResult:
        """
        Command the opening width (m, clipped to [0, robot.gripper.max_width]).

        Opening releases a held object (world.release); closing asks
        world.try_grasp(T_tcp, current width, target) and the measured width
        becomes the held object's width, or the target when nothing is caught.
        """
        self.n_calls += 1
        if not self._connected:
            return GripperResult(ok=False, error='comm', detail='not connected')
        if self._comm_lost or self._take_fault(('comm_loss',)):
            self._comm_lost = True
            return GripperResult(ok=False, error='comm', detail='simulated communication loss')
        if self.abort_event.is_set():
            return GripperResult(ok=False, width=self._width, error='aborted',
                                 detail='abort requested')
        try:
            w = float(width)
        except (TypeError, ValueError):
            w = float('nan')
        if not math.isfinite(w):
            return GripperResult(ok=False, width=self._width, error='fault',
                                 detail='invalid width %r' % (width,))
        target = min(max(w, 0.0), self._max_width)
        self.command_log.append({'cmd': 'gripper', 'width': target, 't': self._clock.now()})
        T_tcp = self._tcp()
        held = None if self._world is None else getattr(self._world, 'held', None)
        if target > self._width + _WIDTH_EPS:
            if held is not None:
                zone = self._world.release(T_tcp)
                LOG.debug('sim: released object into %s', zone or 'table')
            self._width = target
        elif target < self._width - _WIDTH_EPS:
            if held is None:
                held_w = None
                # Only a real close (not the pre-open to object width) can miss.
                closing = target <= self._close_width + 0.001
                miss = closing and self._take_fault(('grasp_miss',)) is not None
                if not miss and self._world is not None:
                    held_w = self._world.try_grasp(T_tcp, self._width, target)
                self._width = float(held_w) if held_w is not None else target
        self._cmd_width = target
        self._update_world(T_tcp)
        if wait:
            self._clock.sleep(self._settle)
        return GripperResult(ok=True, width=float(self._width))

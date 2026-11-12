"""
Workspace / reachability guard and the SafeArm wrapper.

Every motion of the real arm goes through SafeArm: it checks the target (and
the path) against config robot.workspace and the kinematic model before the
wrapped arm sees the command. A rejected target returns
``MotionResult(ok=False, error='rejected', detail=reason)`` and the wrapped arm
is not called at all.

Frames: all poses are 4x4 in the robot base frame; TCP = flange @ T_flange_tcp.
Units: m, rad (config keys ending in _deg are degrees).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.interfaces import ArmInterface
from ..core.transforms import inv_T, is_rotation, make_T, pose_error
from ..core.types import ArmHealth, GripperResult, MotionResult, Stamped
from .kinematics import PiperKinematics, rot_log, tilt_outward

LOG = logging.getLogger('cube_task.robot.safety')

_PATH_STEP_M = 0.02        # spacing of path samples for linear moves
_PATH_MAX_SAMPLES = 40
_JOINT_PATH_SAMPLES = 10   # samples along a joint-space (P2P / move_joints) path
#: default of robot.workspace.max_tilt_free_deg (deg).
DEFAULT_MAX_TILT_FREE_DEG = 40.0


def _need(cfg, key: str):
    """Read a safety-relevant config value; refuse to guess when it is missing."""
    v = cfg.get(key, None)
    if v is None:
        raise ValueError('config %s is required by the safety guard' % key)
    return v


def _need_range(cfg, key: str) -> Tuple[float, float]:
    v = _need(cfg, key)
    try:
        lo, hi = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError('config %s must be [min, max], got %r' % (key, v))
    if len(v) != 2 or not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError('config %s must be [min, max] with min < max, got %r' % (key, v))
    return lo, hi


def _need_float(cfg, key: str, positive: bool = False) -> float:
    v = _need(cfg, key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError('config %s must be a number, got %r' % (key, v))
    if not math.isfinite(f) or (positive and f <= 0):
        raise ValueError('config %s must be a finite%s number, got %r'
                         % (key, ' positive' if positive else '', v))
    return f


class SafetyGuard:
    """
    Static TCP workspace limits from config robot.workspace.

    Keys (all required, no defaults): x, y, z ([min, max] m, base frame),
    r_min / r_max (m, horizontal distance of the TCP from the base z axis),
    min_table_clearance (m, TCP z above the table z), max_tilt_from_down_deg
    (deg, angle between the tool z axis and base -Z). Optional:
    max_tilt_free_deg (default 40, never below max_tilt_from_down_deg), the
    largest per-call tilt limit (``max_tilt_deg``) a caller may ask for; used for
    free-space poses (pre-grasp, lift, transit, pre-place, retreat) only.
    """

    def __init__(self, cfg):
        self.x = _need_range(cfg, 'robot.workspace.x')
        self.y = _need_range(cfg, 'robot.workspace.y')
        self.z = _need_range(cfg, 'robot.workspace.z')
        self.r_min = _need_float(cfg, 'robot.workspace.r_min')
        self.r_max = _need_float(cfg, 'robot.workspace.r_max', positive=True)
        if self.r_min < 0 or self.r_min >= self.r_max:
            raise ValueError('config robot.workspace needs 0 <= r_min < r_max')
        self.min_table_clearance = _need_float(cfg, 'robot.workspace.min_table_clearance')
        tilt = _need_float(cfg, 'robot.workspace.max_tilt_from_down_deg')
        if not 0.0 <= tilt <= 180.0:
            raise ValueError('config robot.workspace.max_tilt_from_down_deg must be in [0, 180]')
        self.max_tilt = math.radians(tilt)
        free = cfg.get('robot.workspace.max_tilt_free_deg', DEFAULT_MAX_TILT_FREE_DEG)
        try:
            free = float(DEFAULT_MAX_TILT_FREE_DEG if free is None else free)
        except (TypeError, ValueError):
            raise ValueError('config robot.workspace.max_tilt_free_deg must be a number')
        if not (math.isfinite(free) and 0.0 <= free <= 180.0):
            raise ValueError('config robot.workspace.max_tilt_free_deg must be in [0, 180]')
        self.max_tilt_free = max(self.max_tilt, math.radians(free))

    def tilt_limit(self, max_tilt_deg: Optional[float] = None) -> float:
        """
        Return the tilt limit (rad) of one call.

        None: max_tilt_from_down_deg; else max_tilt_deg (deg) capped at
        max_tilt_free_deg (a NaN or negative request gives 0).
        """
        if max_tilt_deg is None:
            return self.max_tilt
        try:
            req = math.radians(float(max_tilt_deg))
        except (TypeError, ValueError):
            return 0.0
        if not req >= 0.0:
            return 0.0
        return min(req, self.max_tilt_free)

    def check_tcp(self, T_base_tcp: np.ndarray, table_z: Optional[float] = None,
                  check_tilt: bool = True,
                  max_tilt_deg: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check a TCP target. Returns (ok, reason); reason is '' when ok.

        table_z: table height (m, base z) under the target, or None to skip
        the clearance test. check_tilt=False skips the tool-down test (used
        for joint-space targets such as a folded home pose). max_tilt_deg:
        tilt limit of this call (see tilt_limit); None = max_tilt_from_down_deg.
        """
        try:
            T = np.asarray(T_base_tcp, dtype=float)
        except (TypeError, ValueError):
            return False, 'pose is not numeric'
        if T.shape != (4, 4):
            return False, 'pose must be 4x4, got shape %s' % (T.shape,)
        if not np.all(np.isfinite(T)):
            return False, 'pose contains NaN / inf'
        if not is_rotation(T[:3, :3], tol=1e-3):
            return False, 'pose rotation is not orthonormal'
        x, y, z = (float(v) for v in T[:3, 3])
        for name, v, (lo, hi) in (('x', x, self.x), ('y', y, self.y), ('z', z, self.z)):
            if not lo <= v <= hi:
                return False, 'tcp %s=%.3f outside workspace [%.3f, %.3f]' % (name, v, lo, hi)
        r = math.hypot(x, y)
        if not self.r_min <= r <= self.r_max:
            return False, 'tcp radius %.3f outside [%.3f, %.3f]' % (r, self.r_min, self.r_max)
        if table_z is not None:
            tz = float(table_z)
            if not math.isfinite(tz):
                return False, 'table_z is not finite'
            if z < tz + self.min_table_clearance:
                return False, ('tcp z=%.4f below table z %.4f + clearance %.4f'
                               % (z, tz, self.min_table_clearance))
        if check_tilt:
            tilt = self.tilt_from_down(T)
            limit = self.tilt_limit(max_tilt_deg)
            if tilt > limit + 1e-9:
                return False, ('tool tilt %.1f deg from straight down exceeds %.1f deg'
                               % (math.degrees(tilt), math.degrees(limit)))
        return True, ''

    @staticmethod
    def tilt_from_down(T_base_tcp: np.ndarray) -> float:
        """Angle (rad) between the tool z axis and base -Z."""
        c = -float(np.asarray(T_base_tcp, dtype=float)[2, 2])
        return math.acos(max(-1.0, min(1.0, c)))


def _rot_exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def interpolate_pose(T_a: np.ndarray, T_b: np.ndarray, s: float) -> np.ndarray:
    """Pose at fraction s in [0, 1] of the straight line T_a -> T_b (lerp + slerp)."""
    Ra, Rb = T_a[:3, :3], T_b[:3, :3]
    R = Ra @ _rot_exp(s * rot_log(Ra.T @ Rb))
    return make_T(R, (1.0 - s) * T_a[:3, 3] + s * T_b[:3, 3])


def _rejected(reason: str) -> MotionResult:
    LOG.warning('motion rejected: %s', reason)
    return MotionResult(ok=False, error='rejected', detail=reason)


class SafeArm(ArmInterface):
    """
    ArmInterface wrapper that refuses unsafe or unreachable motions.

    arm: the wrapped ArmInterface (SimArm or PyAgxArm). guard: SafetyGuard.
    kin: PiperKinematics for reachability and path checks (optional, but
    always given by robot.factory.make_arm). T_flange_tcp: 4x4 TCP offset,
    required when kin is given.

    Checks before any command reaches the wrapped arm:
    - abort_event set -> MotionResult(error='aborted'); health().ok False ->
      error 'comm' (communication) or 'arm_error' (controller state).
    - move_tcp: workspace box / radius / table clearance / tool tilt, IK
      within joint limits from the current joints, then the path: samples of a
      linear move must each be reachable; a P2P move is swept in joint space
      (our IK solution, an approximation of the controller's) and must keep
      the TCP and flange above the table.
    - move_joints: joint limits, final TCP above the table, joint-path sweep.
    Rejections return MotionResult(ok=False, error='rejected', detail=reason)
    without calling the wrapped arm.
    """

    def __init__(self, arm: ArmInterface, guard: SafetyGuard,
                 kin: Optional[PiperKinematics] = None,
                 T_flange_tcp: Optional[np.ndarray] = None,
                 table_check_joint_moves: bool = True):
        if kin is not None and T_flange_tcp is None:
            raise ValueError('SafeArm needs T_flange_tcp when kin is given')
        self._arm = arm
        self._guard = guard
        self._kin = kin
        if T_flange_tcp is not None:
            T = np.asarray(T_flange_tcp, dtype=float)
            if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not is_rotation(T[:3, :3]):
                raise ValueError('T_flange_tcp must be a finite rigid 4x4 transform')
            self._T_ft: Optional[np.ndarray] = T.copy()
            self._T_tf: Optional[np.ndarray] = inv_T(T)
        else:
            self._T_ft = None
            self._T_tf = None
        self._table_z: Optional[float] = None
        # False: move_joints checks only the joint limits
        # (config robot.safety.table_check_joint_moves)
        self._table_check_joint_moves = bool(table_check_joint_moves)

    # ----------------------------------------------------------- properties
    @property
    def inner(self) -> ArmInterface:
        """Return the wrapped arm. Commands sent to it bypass every safety check."""
        return self._arm

    @property
    def guard(self) -> SafetyGuard:
        return self._guard

    @property
    def kin(self) -> Optional[PiperKinematics]:
        return self._kin

    @property
    def max_tilt_free_deg(self) -> float:
        """Largest per-call tilt limit (deg) move_tcp / check_reachable accept."""
        return math.degrees(self._guard.max_tilt_free)

    @property
    def table_z(self) -> Optional[float]:
        """Table height (m, base z) used for clearance checks, or None."""
        return self._table_z

    def set_table_z(self, z: Optional[float]) -> None:
        """Set the table height (m, base z) for clearance checks; None disables them."""
        if z is None:
            self._table_z = None
            return
        zf = float(z)
        if not math.isfinite(zf):
            raise ValueError('table z must be finite, got %r' % (z,))
        self._table_z = zf

    def set_abort_event(self, event: threading.Event) -> None:
        """Use event for this wrapper and the wrapped arm (its waits abort too)."""
        ArmInterface.set_abort_event(self, event)
        self._arm.set_abort_event(event)

    # ----------------------------------------------------------- delegation
    def connect(self) -> None:
        """Connect the wrapped arm."""
        self._arm.connect()

    def enable(self, timeout: float = 5.0) -> bool:
        """Enable the wrapped arm's motors."""
        return self._arm.enable(timeout)

    def health(self) -> ArmHealth:
        """Health of the wrapped arm."""
        return self._arm.health()

    def get_tcp_pose(self) -> Optional[Stamped]:
        """Latest T_base_tcp of the wrapped arm."""
        return self._arm.get_tcp_pose()

    def get_flange_pose(self) -> Optional[Stamped]:
        """Latest T_base_flange of the wrapped arm."""
        return self._arm.get_flange_pose()

    def get_joints(self) -> Optional[Stamped]:
        """Latest joint angles (rad) of the wrapped arm."""
        return self._arm.get_joints()

    def gripper_width(self) -> Optional[Stamped]:
        """Latest measured gripper opening (m)."""
        return self._arm.gripper_width()

    def set_speed_percent(self, percent: int) -> None:
        """Set the controller speed percentage of the wrapped arm."""
        self._arm.set_speed_percent(percent)

    def stop(self) -> None:
        """Hold the current position (wrapped arm)."""
        self._arm.stop()

    def emergency_stop(self) -> None:
        """Trigger the electronic emergency stop of the wrapped arm."""
        self._arm.emergency_stop()

    def close(self) -> None:
        """Disconnect the wrapped arm without disabling the motors."""
        self._arm.close()

    def supports_teach(self) -> bool:
        return self._arm.supports_teach()

    def exit_teach_mode(self) -> bool:
        """Leave teach mode via the wrapped arm (real arm only, read its warning)."""
        fn = getattr(self._arm, 'exit_teach_mode', None)
        if fn is None:
            raise NotImplementedError('%s has no teach mode' % type(self._arm).__name__)
        return fn()

    def read_teach_pose(self):
        """Return the wrapped arm's current teach pose (on-site tools)."""
        fn = getattr(self._arm, 'read_teach_pose', None)
        if fn is None:
            raise NotImplementedError('%s has no teach mode' % type(self._arm).__name__)
        return fn()

    def gripper(self, width: float, force: Optional[float] = None,
                wait: bool = True) -> GripperResult:
        """Validate and forward a gripper command (width m, force N)."""
        if self.abort_event.is_set():
            return GripperResult(ok=False, error='aborted', detail='abort requested')
        try:
            w = float(width)
        except (TypeError, ValueError):
            w = float('nan')
        if not math.isfinite(w) or w < 0.0:
            return GripperResult(ok=False, error='fault', detail='invalid width %r' % (width,))
        if force is not None:
            try:
                f = float(force)
            except (TypeError, ValueError):
                f = float('nan')
            if not math.isfinite(f) or f < 0.0:
                return GripperResult(ok=False, error='fault',
                                     detail='invalid force %r' % (force,))
        return self._arm.gripper(w, force=force, wait=wait)

    # --------------------------------------------------------------- checks
    def _gate(self) -> Optional[MotionResult]:
        """Refuse motion while aborted or unhealthy; None means go ahead."""
        if self.abort_event.is_set():
            return MotionResult(ok=False, error='aborted', detail='abort requested')
        try:
            h = self._arm.health()
        except Exception as exc:  # a broken health() must not let motion through
            return MotionResult(ok=False, error='comm', detail='health() failed: %s' % exc)
        if not h.ok:
            err = 'comm' if not h.comm_ok else 'arm_error'
            LOG.warning('motion refused, arm not healthy: %s', h.reason)
            return MotionResult(ok=False, error=err, detail='arm not healthy: %s' % h.reason)
        return None

    def _current_joints(self) -> Optional[np.ndarray]:
        st = self._arm.get_joints()
        if st is None or st.value is None:
            return None
        q = np.asarray(st.value, dtype=float).reshape(-1)
        if q.shape[0] != 6 or not np.all(np.isfinite(q)):
            return None
        return q

    def _ik_tcp(self, T_tcp: np.ndarray, seed: Optional[Sequence[float]]):
        return self._kin.ik(T_tcp @ self._T_tf, q_seed=seed)

    def check_reachable(self, T_base_tcp: np.ndarray,
                        q_seed: Optional[Sequence[float]] = None,
                        max_tilt_deg: Optional[float] = None) -> Tuple[bool, str]:
        """
        Workspace + IK check of a TCP target without moving. Returns (ok, reason).

        q_seed: joint angles (rad) to start IK from; None uses the current
        joints when available. max_tilt_deg: tilt limit of this check
        (SafetyGuard.tilt_limit).
        """
        ok, reason = self._guard.check_tcp(T_base_tcp, self._table_z,
                                           max_tilt_deg=max_tilt_deg)
        if not ok:
            return False, reason
        if self._kin is None:
            return True, ''
        if q_seed is None:
            q_seed = self._current_joints()
        q, ep, eo = self._ik_tcp(np.asarray(T_base_tcp, dtype=float), q_seed)
        if q is None:
            return False, ('no IK solution within joint limits (best residual %.4f m, %.2f deg)'
                           % (ep, math.degrees(eo)))
        return True, ''

    def find_reachable_tilt(self, T_base_tcp: np.ndarray,
                            q_seed: Optional[Sequence[float]] = None,
                            step_deg: float = 5.0, max_tilt_deg: Optional[float] = None
                            ) -> Tuple[Optional[np.ndarray], float, str]:
        """
        Lean a TCP target outward (kinematics.tilt_outward) until it is reachable.

        Tries 0, step_deg, 2 * step_deg, ... up to the guard's
        max_tilt_from_down_deg (or max_tilt_deg, capped at max_tilt_free_deg).
        Returns (T_base_tcp to use or None, tilt rad, reason of the last
        rejection). Planners use this for pre-grasp and transit poses, which
        joint5's limit makes unreachable when vertical.
        """
        step = math.radians(step_deg)
        if not (math.isfinite(step) and step > 0):
            raise ValueError('step_deg must be > 0')
        T0 = np.asarray(T_base_tcp, dtype=float)
        reason = ''
        k = 0
        limit = self._guard.tilt_limit(max_tilt_deg)
        while k * step <= limit + 1e-9:
            tilt = k * step
            Tc = tilt_outward(T0, tilt) if k > 0 else T0
            ok, reason = self.check_reachable(Tc, q_seed=q_seed, max_tilt_deg=max_tilt_deg)
            if ok:
                return Tc, tilt, ''
            k += 1
        return None, 0.0, reason

    def _tcp_above_table(self, T_tcp: np.ndarray, T_flange: np.ndarray) -> Tuple[bool, str]:
        z_min = self._guard.z[0]
        if self._table_z is not None:
            z_min = max(z_min, self._table_z + self._guard.min_table_clearance)
        if T_tcp[2, 3] < z_min:
            return False, 'tcp z=%.4f below %.4f' % (T_tcp[2, 3], z_min)
        if T_flange[2, 3] < z_min:
            return False, 'flange z=%.4f below %.4f' % (T_flange[2, 3], z_min)
        return True, ''

    def _check_joint_path(self, q_from: np.ndarray, q_to: np.ndarray) -> Tuple[bool, str]:
        for k in range(1, _JOINT_PATH_SAMPLES + 1):
            s = k / float(_JOINT_PATH_SAMPLES)
            Tf = self._kin.fk(q_from + s * (q_to - q_from))
            ok, reason = self._tcp_above_table(Tf @ self._T_ft, Tf)
            if not ok:
                return False, 'joint-space path at %.0f%%: %s' % (100 * s, reason)
        return True, ''

    def _check_linear_path(self, T_from: np.ndarray, T_to: np.ndarray,
                           q_from: Optional[np.ndarray],
                           max_tilt_deg: Optional[float] = None) -> Tuple[bool, str]:
        dist = float(np.linalg.norm(T_to[:3, 3] - T_from[:3, 3]))
        ang = float(np.linalg.norm(rot_log(T_from[:3, :3].T @ T_to[:3, :3])))
        n = int(min(_PATH_MAX_SAMPLES, max(2, math.ceil(max(dist, 0.1 * ang) / _PATH_STEP_M))))
        seed = q_from
        for k in range(1, n):
            s = k / float(n)
            Ts = interpolate_pose(T_from, T_to, s)
            ok, reason = self._guard.check_tcp(Ts, self._table_z, max_tilt_deg=max_tilt_deg)
            if not ok:
                return False, 'linear path at %.0f%%: %s' % (100 * s, reason)
            if self._kin is not None:
                q, ep, eo = self._ik_tcp(Ts, seed)
                if q is None:
                    return False, ('linear path at %.0f%% has no IK solution (residual %.4f m)'
                                   % (100 * s, ep))
                seed = q
        return True, ''

    # --------------------------------------------------------------- motion
    def move_tcp(self, T_base_tcp: np.ndarray, linear: bool = False,
                 timeout: Optional[float] = None,
                 max_tilt_deg: Optional[float] = None) -> MotionResult:
        """
        Check, then forward a TCP move (see the class docstring for the checks).

        max_tilt_deg: tilt limit of the target (SafetyGuard.tilt_limit; None =
        max_tilt_from_down_deg). Samples of a linear path may lean as far as the
        start pose does (at most max_tilt_free_deg), so a move from a leaned
        free-space pose down to a grasp pose is accepted.
        """
        gate = self._gate()
        if gate is not None:
            return gate
        ok, reason = self._guard.check_tcp(T_base_tcp, self._table_z, max_tilt_deg=max_tilt_deg)
        if not ok:
            return _rejected(reason)
        T = np.asarray(T_base_tcp, dtype=float)
        q_now = None
        q_goal = None
        if self._kin is not None:
            q_now = self._current_joints()
            if q_now is None:
                return _rejected('current joint angles unavailable')
            q_goal, ep, eo = self._ik_tcp(T, q_now)
            if q_goal is None:
                return _rejected('target not reachable within joint limits '
                                 '(best residual %.4f m, %.2f deg)' % (ep, math.degrees(eo)))
        if linear:
            T_now = self._current_tcp(q_now)
            if T_now is None:
                return _rejected('current TCP pose unknown, cannot check the linear path')
            limit = self._guard.tilt_limit(max_tilt_deg)
            start = min(self._guard.tilt_from_down(T_now) + math.radians(0.5),
                        self._guard.max_tilt_free)
            path_deg = math.degrees(max(limit, start))
            ok, reason = self._check_linear_path(T_now, T, q_now, path_deg)
        elif self._kin is not None:
            ok, reason = self._check_joint_path(q_now, q_goal)
        if not ok:
            return _rejected(reason)
        res = self._arm.move_tcp(T, linear=linear, timeout=timeout)
        if not linear and q_goal is not None and not res.ok and res.error == 'unreachable':
            return self._p2p_by_joints(q_goal, res, timeout)
        return res

    def _p2p_by_joints(self, q_goal: np.ndarray, res: MotionResult,
                       timeout: Optional[float]) -> MotionResult:
        """
        After the controller found no solution for a P2P target, move_j to our IK solution.

        The controller solves move_p itself and refuses some leaned poses that
        q_goal (within the joint limits) reaches; the joint path is checked again
        from where the arm is now. Returns the original result when that fails.
        """
        gate = self._gate()
        if gate is not None:
            return res
        q_now = self._current_joints()
        if q_now is None:
            return res
        ok, reason = self._check_joint_path(q_now, q_goal)
        if not ok:
            LOG.warning('P2P target refused by the controller (%s); joint fallback refused: %s',
                        res.detail, reason)
            return res
        LOG.warning('P2P target refused by the controller (%s); moving to the checked joint '
                    'solution instead', res.detail)
        return self._arm.move_joints(q_goal, timeout=timeout)

    def _current_tcp(self, q_now: Optional[np.ndarray]) -> Optional[np.ndarray]:
        st = self._arm.get_tcp_pose()
        if st is not None and st.value is not None:
            T = np.asarray(st.value, dtype=float)
            if T.shape == (4, 4) and np.all(np.isfinite(T)):
                return T
        if q_now is not None and self._kin is not None:
            return self._kin.fk(q_now) @ self._T_ft
        return None

    def move_joints(self, q: Sequence[float], timeout: Optional[float] = None) -> MotionResult:
        """Check the joint limits (plus TCP height and path unless disabled), then forward."""
        gate = self._gate()
        if gate is not None:
            return gate
        try:
            qt = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return _rejected('joint target is not numeric: %r' % (q,))
        if qt.shape[0] != 6 or not np.all(np.isfinite(qt)):
            return _rejected('joint target must be 6 finite values, got %r' % (q,))
        if self._kin is not None:
            bad: List[str] = self._kin.limit_violations(qt)
            if bad:
                return _rejected('joint limits: ' + ', '.join(bad))
            if self._table_check_joint_moves:
                Tf = self._kin.fk(qt)
                ok, reason = self._tcp_above_table(Tf @ self._T_ft, Tf)
                if not ok:
                    return _rejected('joint target: ' + reason)
                q_now = self._current_joints()
                if q_now is None:
                    return _rejected('current joint angles unavailable')
                ok, reason = self._check_joint_path(q_now, qt)
                if not ok:
                    return _rejected(reason)
        return self._arm.move_joints(qt, timeout=timeout)


def tcp_error(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(position error m, rotation error deg) between two TCP poses."""
    dp, dr = pose_error(T_a, T_b)
    return dp, math.degrees(dr)

"""
PiPER arm + AGX parallel gripper simulated with MuJoCo physics.

``MujocoArm`` implements core.interfaces.ArmInterface on an engine.MujocoSim
with the same behavior contract as robot.sim_arm.SimArm and
robot.pyagxarm_arm.PyAgxArm: motion calls block while the injected clock
advances (each clock.sleep runs the physics steps), and they report
failures in MotionResult instead of raising.

Inside MuJoCo the MuJoCo model is the robot: its own kinematics (ik.MujocoIK)
define the flange (meta flange_body) and the TCP = flange @ T_flange_tcp from
config robot.tcp_offset, like a well-calibrated real controller.

Control: a controller registered with sim.add_controller writes the position
actuator targets before every physics step from the active trajectory, with
velocity and (diagonal) inertia feed-forward:

    P2P / move_joints: joint-space line, duration from sim.joint_speed (rad/s
        at 100 %) x robot.speed_percent.
    linear: Cartesian straight line (lerp + slerp) at robot.motion.sim_lin_speed
        x speed, joint waypoints precomputed by IK every ~3 mm by continuation
        from the measured joints; 'unreachable' when a waypoint has no IK.

Both use a cosine-ramped trapezoid time scaling (ramp _ACCEL_TIME_S). A move
is done when the trajectory ended, the measured TCP (MuJoCo state) is within
robot.motion.pos_tol / rot_tol_deg (move_joints: every joint within 0.01 rad)
and the joints are still for robot.motion.settle_samples polls; otherwise
'timeout' (robot.motion.timeout_base_s + timeout_per_m_s x distance) or
'stalled' (the arm stopped short, or a tracking error showed it is blocked by
a contact; the arm then holds the measured joints).

Gripper (AGX): a speed-limited finger motor moves toward the commanded width
at _GRIP_RATE m/s, at most a few mm ahead of the fingers, and the finger
actuator force is limited so each finger presses with the commanded grip
force; a close therefore stops on the object and squeezes it, and the fingers
do not slam shut when it slips out. Measured width = 2 x finger joint position.

Faults for tests: ``inject(kind, at_call=None, count=1)`` with the kinds of
robot.sim_arm.FAULT_KINDS (see inject()).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...core.interfaces import ArmInterface, Clock
from ...core.transforms import inv_T, is_rotation, pose_error, rotation_angle
from ...core.types import ArmHealth, GripperResult, MotionResult, Stamped
from ...robot.kinematics import flange_to_tcp_from_cfg, JOINT_LIMITS
from ...robot.pyagxarm_arm import GRIPPER_FORCE_RANGE
from ...robot.safety import interpolate_pose
from ...robot.sim_arm import FAULT_KINDS
from .ik import ARM_JOINTS, MujocoIK

LOG = logging.getLogger('cube_task.sim.mujoco.arm')

# Position servo gains per arm joint (Nm/rad, Nm s/rad) and torque limits (Nm),
# tuned for the menagerie PiPER (gravcomp bodies, frictionloss 0.3 Nm): measured
# 0.1-0.3 mm final TCP error and < 0.5 mm deviation from a linear path at 100 %
# speed with a 0.1-0.2 kg object held, no overshoot. The low torque limits make a
# collision stall the arm instead of crushing into the table (the real PiPER has
# collision protection). Overridable with config sim.mujoco.arm_kp / arm_kv /
# arm_force (6 numbers each).
_KP = (1500.0, 1500.0, 1500.0, 300.0, 400.0, 150.0)
_KV = (40.0, 45.0, 40.0, 6.0, 8.0, 3.0)
_FORCE = (10.0, 10.0, 10.0, 3.0, 3.0, 3.0)
# Gripper finger servo (N/m, N s/m); its force limit follows the commanded grip force.
_GRIP_KP = 1500.0
_GRIP_KV = 15.0
_GRIP_RATE = 0.10          # m/s change of the commanded width (full stroke ~0.7 s)
# Finger actuator force limit = grip force x this. The actuator drives one finger
# and the joint equality couples the other, so each finger presses on a held
# object with half the actuator force: 2.0 makes each finger press with the
# commanded grip force. Override: config sim.mujoco.grip_force_scale.
_GRIP_FORCE_SCALE = 2.0
_EQ_TIMECONST = 0.004      # s, stiffer finger coupling (menagerie default 0.02 lets it stretch)
_ACCEL_TIME_S = 0.25       # s, velocity ramp of the time scaling at full speed
_LIN_STEP_M = 0.003        # linear moves: IK waypoint every 3 mm ...
_LIN_STEP_RAD = math.radians(1.5)   # ... and every 1.5 deg of tool rotation
_MAX_WAYPOINTS = 3000
_MAX_JOINT_STEP = 0.2      # rad, larger jump between waypoints = branch switch / singularity
_JOINT_TOL = 0.01          # rad, move_joints "arrived" tolerance (as PyAgxArm)
_STILL_VEL = 0.02          # rad/s, every joint slower than this counts as stopped
_TRACK_STALL_RAD = 0.01    # rad tracking error held ...
# ... this long = blocked by a contact -> 'stalled'. 0.3 s, not 0.1 s: a move away
# from a contact (retreat after the fingers stalled on an object) lags 0.011 rad on
# joint4 (its 3 Nm limit = 0.01 rad x kp 300) for ~0.2 s while it breaks free.
_TRACK_STALL_S = 0.3
_WIDTH_EPS = 1e-4
_WIDTH_DIGITS = 6          # width feedback resolution 1 um (SDK unit 0.001 mm)
_GRIP_MOVE_EPS = 0.002     # m, width change that shows the fingers started moving
_GRIP_STABLE_EPS = 0.0005  # m, width change between polls counted as stable
_GRIP_OPEN_TOL = 0.005     # m, an opening command must reach target - this
_HOLD_MARGIN = 0.002       # m, measured width this far above the command = fingers blocked
_DROP_OPEN = 0.015         # m, drop fault: extra opening ...
_DROP_HOLD_S = 0.4         # ... kept this long before closing again
_MISS_CLEAR_DZ = 0.06      # m, grasp_miss: finger contacts return after lifting this far
_MOTION_FAULTS = ('unreachable', 'arm_error')
# SafeArm checks reachability with the SDK kinematics, which differ from the MuJoCo
# model by 1-10 mm. A target that IK misses by at most this much (typically at the
# joint5 limit) is approached as closely as the MuJoCo arm can, and the arrival
# check (robot.motion.pos_tol / rot_tol_deg) decides, instead of failing with
# 'unreachable' where the real arm would not.
_REACH_POS = 0.002         # m
_REACH_ROT = math.radians(1.0)


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


def _cfg_vec(cfg, key: str, default: Sequence[float]) -> np.ndarray:
    v = cfg.get(key, None)
    if v is None:
        return np.array(default, dtype=float)
    try:
        arr = np.array([float(x) for x in v], dtype=float)
    except (TypeError, ValueError):
        raise ValueError('config %s must be %d numbers, got %r' % (key, len(default), v))
    if arr.shape != (len(default),) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError('config %s must be %d positive numbers, got %r'
                         % (key, len(default), v))
    return arr


class _Fault:
    def __init__(self, kind: str, at_call: Optional[int], count: int):
        self.kind = kind
        self.at_call = at_call
        self.remaining = count


class _Profile:
    """
    Time scaling s(t) from 0 to 1: trapezoid in ds/dt with cosine-shaped ramps.

    v: peak ds/dt (1/s); ta: ramp time at that speed. Moves too short to
    reach v keep the same peak acceleration and a lower peak speed.
    """

    def __init__(self, v: float, ta: float):
        if not (v > 0 and math.isfinite(v)):
            self.v, self.ta, self.T = 0.0, 0.0, 0.0
            return
        ta = max(float(ta), 1e-6)
        if v * ta > 1.0:
            v2 = math.sqrt(v / ta)
            ta, v = ta * v2 / v, v2
        self.v, self.ta = v, ta
        self.T = ta + 1.0 / v

    def __call__(self, t: float):
        """Return (s, ds/dt, d2s/dt2) at time t after the start."""
        if self.T <= 0.0 or t >= self.T:
            return 1.0, 0.0, 0.0
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        v, ta = self.v, self.ta
        k = math.pi / ta
        if t < ta:
            return (0.5 * v * (t - math.sin(k * t) / k), 0.5 * v * (1.0 - math.cos(k * t)),
                    0.5 * v * k * math.sin(k * t))
        if t <= self.T - ta:
            return 0.5 * v * ta + v * (t - ta), v, 0.0
        r = self.T - t
        return (1.0 - 0.5 * v * (r - math.sin(k * r) / k), 0.5 * v * (1.0 - math.cos(k * r)),
                -0.5 * v * k * math.sin(k * r))


class _Trajectory:
    """Joint waypoints qs (n + 1, 6) uniform in s, timed by a _Profile from t0."""

    def __init__(self, t0: float, qs: np.ndarray, profile: Optional[_Profile] = None,
                 s_max: float = 1.0):
        self.t0 = float(t0)
        self.qs = np.array(qs, dtype=float).reshape(-1, 6)
        self.profile = profile if profile is not None else _Profile(0.0, 0.0)
        self.s_max = float(s_max)
        self.end_time = self.t0 + self.profile.T
        self.n = self.qs.shape[0] - 1
        self.q_end = self.qs[-1].copy()

    def eval(self, t: float):
        """Return (q, qd, qdd, s) at time t."""
        zero = np.zeros(6)
        if self.n == 0:
            return self.qs[0], zero, zero, 1.0
        s, sd, sdd = self.profile(t - self.t0)
        if s >= self.s_max:
            s, sd, sdd = self.s_max, 0.0, 0.0
        x = s * self.n
        k = min(int(x), self.n - 1)
        f = x - k
        dq = self.qs[k + 1] - self.qs[k]
        dqds = dq * self.n
        return self.qs[k] + f * dq, dqds * sd, dqds * sdd, s


class MujocoArm(ArmInterface):
    """
    Simulated PiPER on MuJoCo physics implementing ArmInterface.

    cfg: core.config.Config. clock: the SimClock that drives sim. sim:
    engine.MujocoSim whose meta names arm_joints (6), arm_actuators (6),
    gripper_actuator, finger_joint, flange_body, tcp_site (optional, only
    checked against robot.tcp_offset), max_width (m) and optionally
    base_body (default 'base_link'; poses are in its frame).

    Like the real arm, motions need connect() and enable() first. The arm
    holds the joint angles found in the sim at construction.
    """

    def __init__(self, cfg, clock: Clock, sim: Any):
        self._cfg = cfg
        self._clock = clock
        self._sim = sim
        m = sim.m
        self._mj = sim.mujoco
        meta = sim.meta
        self._T_ft = flange_to_tcp_from_cfg(cfg)
        self._lin_speed = _cfg_float(cfg, 'robot.motion.sim_lin_speed', 0.15)
        self._joint_speed = _cfg_float(cfg, 'sim.joint_speed', 2.0)
        self._pos_tol = _cfg_float(cfg, 'robot.motion.pos_tol', 0.006)
        self._rot_tol = math.radians(_cfg_float(cfg, 'robot.motion.rot_tol_deg', 4.0))
        self._settle_samples = max(1, int(_cfg_float(cfg, 'robot.motion.settle_samples', 5)))
        self._period = 1.0 / _cfg_float(cfg, 'robot.motion.poll_hz', 50.0)
        self._min_wait = _cfg_float(cfg, 'robot.motion.min_wait_s', 0.25, positive=False)
        self._stall_s = _cfg_float(cfg, 'robot.motion.stall_s', 1.0)
        self._timeout_base = _cfg_float(cfg, 'robot.motion.timeout_base_s', 6.0)
        self._timeout_per_m = _cfg_float(cfg, 'robot.motion.timeout_per_m_s', 15.0,
                                         positive=False)
        self._force = _cfg_float(cfg, 'robot.gripper.force', 1.5, positive=False)
        self._grip_settle = _cfg_float(cfg, 'robot.gripper.settle_s', 0.6, positive=False)
        self._grip_timeout = _cfg_float(cfg, 'robot.gripper.timeout_s', 3.0)
        self._grip_stable_s = _cfg_float(cfg, 'robot.gripper.stable_s', 0.15, positive=False)
        self._close_width = _cfg_float(cfg, 'robot.gripper.close_width', 0.0, positive=False)
        if cfg.get('robot.gripper.max_width', None) is None:
            raise ValueError('config robot.gripper.max_width is required (m)')
        self._max_width = min(_cfg_float(cfg, 'robot.gripper.max_width', 0.07),
                              float(meta.get('max_width', 0.07)))
        self._grip_rate = _cfg_float(cfg, 'sim.mujoco.gripper_speed', _GRIP_RATE)
        self._grip_scale = _cfg_float(cfg, 'sim.mujoco.grip_force_scale', _GRIP_FORCE_SCALE)
        self._speed = 100
        self.set_speed_percent(cfg.get('robot.speed_percent', 100))

        # ---- model lookup
        joints = tuple(meta.get('arm_joints', ARM_JOINTS))
        acts = tuple(meta.get('arm_actuators', joints))
        if len(joints) != 6 or len(acts) != 6:
            raise ValueError('meta arm_joints / arm_actuators must name 6 joints / actuators')
        jids = [sim.joint_id(n) for n in joints]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        self._dadr = np.array([m.jnt_dofadr[j] for j in jids], dtype=int)
        self._madr = np.array([m.dof_Madr[i] for i in self._dadr], dtype=int)
        self._act = np.array([sim.actuator_id(n) for n in acts], dtype=int)
        self._gact = sim.actuator_id(str(meta.get('gripper_actuator', 'gripper')))
        fj = sim.joint_id(str(meta.get('finger_joint', 'joint7')))
        self._fadr = int(m.jnt_qposadr[fj])
        self._flange = sim.body_id(str(meta.get('flange_body', 'link6')))
        base = str(meta.get('base_body', 'base_link'))
        try:
            self._T_wb = sim.body_pose(sim.body_id(base))
        except KeyError:
            LOG.warning('no base body %r: poses are in the MuJoCo world frame', base)
            self._T_wb = np.eye(4)
        self._T_bw = inv_T(self._T_wb)
        lim = np.array([m.jnt_range[j] for j in jids], dtype=float)
        sdk = JOINT_LIMITS.get(str(cfg.get('robot.model', 'piper')))
        if sdk is not None:   # never allow more than the real controller
            sdk = np.array(sdk, dtype=float)
            lim = np.column_stack([np.maximum(lim[:, 0], sdk[:, 0]),
                                   np.minimum(lim[:, 1], sdk[:, 1])])
        self.ik = MujocoIK(m, joints=joints, body=str(meta.get('flange_body', 'link6')),
                           T_body_tool=self._T_ft, base_body=base, limits=lim)
        self._check_tcp_site(meta.get('tcp_site'))
        self._finger_geoms = self._find_finger_geoms(meta.get('finger_geoms'))
        self._finger_contype = m.geom_contype[self._finger_geoms].copy()
        self._finger_conaff = m.geom_conaffinity[self._finger_geoms].copy()

        # ---- servo gains
        self._kp = _cfg_vec(cfg, 'sim.mujoco.arm_kp', _KP)
        self._kv = _cfg_vec(cfg, 'sim.mujoco.arm_kv', _KV)
        frc = _cfg_vec(cfg, 'sim.mujoco.arm_force', _FORCE)
        for i, a in enumerate(self._act):
            m.actuator_gainprm[a, 0] = self._kp[i]
            m.actuator_biasprm[a, 1] = -self._kp[i]
            m.actuator_biasprm[a, 2] = -self._kv[i]
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-frc[i], frc[i])
        self._gkp = _cfg_float(cfg, 'sim.mujoco.gripper_kp', _GRIP_KP)
        self._gkv = _cfg_float(cfg, 'sim.mujoco.gripper_kv', _GRIP_KV)
        m.actuator_gainprm[self._gact, 0] = self._gkp
        m.actuator_biasprm[self._gact, 1] = -self._gkp
        m.actuator_biasprm[self._gact, 2] = -self._gkv
        m.actuator_forcelimited[self._gact] = 1
        self._set_grip_force(self._force)
        self._stiffen_finger_coupling(fj)

        # ---- state
        self._lock = threading.RLock()
        q0 = self._q_meas()
        self._traj = _Trajectory(sim.time, q0[None, :])
        w0 = self._width_meas()
        self._grip_target = w0      # commanded width (m)
        self._grip_pos = w0         # speed-limited finger motor position (width, m)
        self._cmd_width = w0
        self._miss: Optional[Dict[str, float]] = None
        self._connected = False
        self._enabled = False
        self._estop = False
        self._comm_lost = False
        self._faults: List[_Fault] = []
        self.n_calls = 0
        self.fault_log: List[Dict[str, Any]] = []
        self.command_log: List[Dict[str, Any]] = []
        if getattr(sim, 'active_arm', None) is not None:
            LOG.warning('a second MujocoArm on the same sim: the new one takes control')
        sim.active_arm = self       # only the newest arm writes the actuators
        sim.add_controller(self._control)

    # ------------------------------------------------------------ setup helpers
    def _check_tcp_site(self, site: Any) -> None:
        if not site:
            return
        try:
            sid = self._sim.site_id(str(site))
        except KeyError:
            LOG.warning('meta tcp_site %r not in the model', site)
            return
        T_site = inv_T(self._sim.body_pose(self._flange)) @ self._sim.site_pose(sid)
        dp, dr = pose_error(T_site, self._T_ft)
        if dp > 0.001 or dr > math.radians(1.0):
            LOG.warning('tcp site %r is %.1f mm / %.1f deg from robot.tcp_offset; the arm uses '
                        'robot.tcp_offset', site, dp * 1e3, math.degrees(dr))

    def _find_finger_geoms(self, given: Any = None) -> np.ndarray:
        """Return meta finger_geoms (ids or names), else the collision geoms of the fingers."""
        m = self._sim.m
        if given:
            ids = []
            for g in given:
                i = self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_GEOM, g) \
                    if isinstance(g, str) else int(g)
                if not 0 <= i < m.ngeom:
                    raise KeyError('meta finger_geoms: no geom %r' % (g,))
                ids.append(int(i))
            return np.array(ids, dtype=int)
        below = set()
        for b in range(m.nbody):
            p = b
            while p > 0 and p != self._flange:
                p = int(m.body_parentid[p])
            if p == self._flange and b != self._flange and int(m.body_jntnum[b]) > 0:
                below.add(b)
        g = [i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in below
             and (m.geom_contype[i] or m.geom_conaffinity[i])]
        return np.array(g, dtype=int)

    def _set_grip_force(self, force: float) -> None:
        f = max(float(force) * self._grip_scale, 1e-3)
        self._sim.m.actuator_forcerange[self._gact] = (-f, f)

    def _stiffen_finger_coupling(self, finger_joint: int) -> None:
        """Make the joint equality that couples the second finger nearly rigid."""
        m, mj = self._sim.m, self._mj
        tc = max(_EQ_TIMECONST, 2.0 * float(m.opt.timestep))
        for e in range(m.neq):
            if (int(m.eq_type[e]) == int(mj.mjtEq.mjEQ_JOINT)
                    and finger_joint in (int(m.eq_obj1id[e]), int(m.eq_obj2id[e]))):
                m.eq_solref[e] = (min(float(m.eq_solref[e][0]), tc), 1.0)

    # ------------------------------------------------------------ physics side
    def _control(self, sim, t: float) -> None:
        """Write the actuator targets for the next physics step (controller hook)."""
        if getattr(sim, 'active_arm', self) is not self:
            return
        d = sim.d
        q, qd, qdd, _s = self._traj.eval(t)
        u = q + (self._kv * qd + np.asarray(d.qM[self._madr]) * qdd) / self._kp
        d.ctrl[self._act] = u
        # AGX finger motor: moves toward the target at _GRIP_RATE, never more than
        # `lead` ahead of the fingers, so a blocked close squeezes with the force
        # limit and the fingers never slam shut when an object slips out.
        wm = 2.0 * float(d.qpos[self._fadr])
        step = self._grip_rate * sim.dt
        p0 = self._grip_pos
        p = p0 + min(max(self._grip_target - p0, -step), step)
        lead = 2.0 * float(sim.m.actuator_forcerange[self._gact][1]) / self._gkp + 0.002
        p = min(max(p, wm - lead), wm + lead)
        v = min(max((p - p0) / sim.dt, -self._grip_rate), self._grip_rate)
        self._grip_pos = p
        d.ctrl[self._gact] = 0.5 * p + self._gkv * 0.5 * v / self._gkp
        miss = self._miss
        if miss is not None:
            if wm < miss['width'] - 0.005:
                miss['closed'] = 1.0
            if ((miss['closed'] and wm >= miss['width'] - 0.001)
                    or float(d.xpos[self._flange][2]) >= miss['z'] + _MISS_CLEAR_DZ):
                self._finger_contacts(True)

    def _finger_contacts(self, on: bool) -> None:
        m = self._sim.m
        if on:
            m.geom_contype[self._finger_geoms] = self._finger_contype
            m.geom_conaffinity[self._finger_geoms] = self._finger_conaff
            self._miss = None
        else:
            m.geom_contype[self._finger_geoms] = 0
            m.geom_conaffinity[self._finger_geoms] = 0

    def _q_meas(self) -> np.ndarray:
        with self._sim.lock:
            return np.array(self._sim.d.qpos[self._qadr], dtype=float)

    def _qd_meas(self) -> np.ndarray:
        with self._sim.lock:
            return np.array(self._sim.d.qvel[self._dadr], dtype=float)

    def _width_meas(self) -> float:
        """Return the measured width at the 1 um resolution of the gripper feedback."""
        with self._sim.lock:
            return round(2.0 * float(self._sim.d.qpos[self._fadr]), _WIDTH_DIGITS)

    def _flange_meas(self) -> np.ndarray:
        with self._sim.lock:
            return self._T_bw @ self._sim.body_pose(self._flange)

    def _tcp_meas(self) -> np.ndarray:
        return self._flange_meas() @ self._T_ft

    def _hold(self, q: Optional[np.ndarray] = None) -> None:
        """Hold q (default: the measured joints) from now on."""
        with self._lock:
            self._traj = _Trajectory(self._sim.time, (self._q_meas() if q is None
                                                      else np.asarray(q, float))[None, :])

    # ------------------------------------------------------------ test hooks
    def inject(self, kind: str, at_call: Optional[int] = None, count: int = 1) -> None:
        """
        Arm a fault (robot.sim_arm.FAULT_KINDS).

        at_call: 1-based index of the command (move_tcp / move_joints /
        gripper, counted in n_calls) from which the fault is armed; None = the
        next applicable command. count: how many applicable commands it hits.

        timeout      next motion stops half way and takes its timeout -> 'timeout'
        unreachable  next motion fails at once -> 'unreachable'
        comm_loss    health() not ok from now on; every command -> 'comm'
        grasp_miss   next closing gripper command grabs nothing: the finger
                     contacts are off while closing (width reads ~close width),
                     back on once the fingers reopen or the flange rose 6 cm
        drop         while holding, the fingers open 15 mm half way through the
                     next motion (the object falls), then close again (~0 width)
        arm_error    next motion fails at once -> 'arm_error'
        """
        if kind not in FAULT_KINDS:
            raise ValueError('unknown fault %r, expected one of %s' % (kind, FAULT_KINDS))
        if count < 1:
            raise ValueError('count must be >= 1')
        if at_call is not None and at_call < 1:
            raise ValueError('at_call is 1-based')
        self._faults.append(_Fault(kind, at_call, int(count)))

    def set_joints(self, q: Sequence[float]) -> None:
        """Teleport the arm and hold q (scenario setup and simulated drag-teach only)."""
        qt = np.asarray(q, dtype=float).reshape(-1)
        if not self.ik.within_limits(qt):
            raise ValueError('joints outside limits: %r' % (list(qt),))
        sim = self._sim
        with sim.lock:
            sim.d.qpos[self._qadr] = qt
            sim.d.qvel[self._dadr] = 0.0
            self._hold(qt)
            sim.d.ctrl[self._act] = qt
            self._mj.mj_forward(sim.m, sim.d)

    @property
    def sim(self) -> Any:
        return self._sim

    @property
    def world(self) -> Any:
        """No kinematic SimWorld behind this arm (the MuJoCo scene is the world)."""
        return None

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
    def _feedback_ok(self) -> bool:
        return self._connected and not self._comm_lost

    def connect(self) -> None:
        """Open the simulated link. Raises nothing (the sim is always there)."""
        self._connected = True

    def enable(self, timeout: float = 5.0) -> bool:
        """Enable the simulated motors; False without communication or after an e-stop."""
        if not self._feedback_ok() or self._estop:
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
        """Measured joint angles (rad), None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._q_meas(), self._clock.now())

    def get_flange_pose(self) -> Optional[Stamped]:
        """Measured T_base_flange, None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._flange_meas(), self._clock.now())

    def get_tcp_pose(self) -> Optional[Stamped]:
        """Measured T_base_tcp = T_base_flange @ T_flange_tcp, None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._tcp_meas(), self._clock.now())

    def gripper_width(self) -> Optional[Stamped]:
        """Measured opening (m, 2 x finger joint), None without communication."""
        if not self._feedback_ok():
            return None
        return Stamped(self._width_meas(), self._clock.now())

    def set_speed_percent(self, percent: int) -> None:
        """Set the speed scale of the trajectories (integer 1..100)."""
        try:
            p = int(percent)
        except (TypeError, ValueError):
            raise ValueError('speed percent must be an integer, got %r' % (percent,))
        if p != percent or not 1 <= p <= 100:
            raise ValueError('speed percent must be an integer in [1, 100], got %r' % (percent,))
        self._speed = p

    def stop(self) -> None:
        """Hold the measured joints (a running motion then returns 'stalled')."""
        self._hold()

    def emergency_stop(self) -> None:
        """Latch an e-stop: hold the measured joints; health() not ok, motions refused."""
        self._estop = True
        self._enabled = False
        self._hold()
        LOG.error('sim: electronic emergency stop')

    def close(self) -> None:
        """Disconnect (the motors keep holding, as on the real arm)."""
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

    def _limit(self, dist: float, timeout: Optional[float]) -> float:
        if timeout is None:
            return self._timeout_base + self._timeout_per_m * dist
        try:
            t = float(timeout)
        except (TypeError, ValueError):
            t = float('nan')
        if not (t > 0 and math.isfinite(t)):
            raise ValueError('timeout must be a positive number of seconds, got %r' % (timeout,))
        return t

    def _result(self, ok: bool, error: str, detail: str, t0: float,
                T_goal: np.ndarray) -> MotionResult:
        if error:
            LOG.warning('sim motion %s: %s', error, detail)
        T_now = self._tcp_meas()
        dp, dr = pose_error(T_now, T_goal)
        return MotionResult(ok=ok, error=error, detail=detail,
                            duration=self._clock.now() - t0, final_tcp=T_now,
                            pos_err=dp, rot_err_deg=math.degrees(dr))

    def _instant_failure(self, error: str, detail: str) -> MotionResult:
        t0 = self._clock.now()
        if self._min_wait > 0:
            self._clock.sleep(self._min_wait)
        T_now = self._tcp_meas()
        LOG.warning('sim motion %s: %s', error, detail)
        return MotionResult(ok=False, error=error, detail=detail,
                            duration=self._clock.now() - t0, final_tcp=T_now)

    def _holding(self) -> bool:
        """Return True when the last gripper close is blocked by something (an object is held)."""
        return self._cmd_width < self._width_meas() - _HOLD_MARGIN

    def _command_grip(self, target: float) -> None:
        self._grip_target = float(target)

    def _at_target(self, q: np.ndarray, T_goal: np.ndarray, q_goal: Optional[np.ndarray]):
        if q_goal is not None:
            e = float(np.max(np.abs(q - q_goal)))
            return e < _JOINT_TOL, 'max joint error %.4f rad' % e
        pe, re = pose_error(self._tcp_meas(), T_goal)
        return (pe <= self._pos_tol and re <= self._rot_tol,
                'tcp %.1f mm / %.2f deg from the target' % (pe * 1e3, math.degrees(re)))

    def _execute(self, qs: np.ndarray, v: float, T_goal: np.ndarray,
                 q_goal: Optional[np.ndarray], t0: float, limit: float) -> MotionResult:
        hang = self._take_fault(('timeout',)) is not None
        drop = (self._holding() and self._peek_fault('drop')
                and self._take_fault(('drop',)) == 'drop')
        traj = _Trajectory(self._sim.time, qs, _Profile(v, _ACCEL_TIME_S),
                           s_max=0.5 if hang else 1.0)
        with self._lock:
            self._traj = traj
        return self._wait_motion(traj, T_goal, q_goal, t0, limit, hang, drop)

    def _wait_motion(self, traj: _Trajectory, T_goal: np.ndarray, q_goal: Optional[np.ndarray],
                     t0: float, limit: float, hang: bool, drop: bool) -> MotionResult:
        """Poll until done, stalled, timed out, aborted or stopped (see the module doc)."""
        deadline = t0 + limit
        stable, still_since, track_since = 0, None, None
        drop_state, drop_t, drop_w = ('pending' if drop else ''), 0.0, 0.0
        while True:
            self._clock.sleep(self._period)
            now, ts = self._clock.now(), self._sim.time
            if self.abort_event.is_set():
                self._hold()
                return self._result(False, 'aborted', 'abort requested during motion', t0,
                                    T_goal)
            if self._traj is not traj:
                if self._estop:
                    return self._result(False, 'arm_error', 'emergency stop during motion',
                                        t0, T_goal)
                return self._result(False, 'stalled', 'motion stopped by stop()', t0, T_goal)
            q, qd = self._q_meas(), self._qd_meas()
            q_des, _qd, _qdd, s = traj.eval(ts)
            if drop_state == 'pending' and s >= 0.5:
                w = self._width_meas()
                self._command_grip(min(self._max_width, w + _DROP_OPEN))
                drop_state, drop_t = 'open', now
                LOG.info('sim fault drop: fingers open %.3f -> %.3f m', w,
                         min(self._max_width, w + _DROP_OPEN))
            elif drop_state == 'open' and now - drop_t >= _DROP_HOLD_S:
                self._command_grip(self._cmd_width)
                drop_state, drop_t, drop_w = 'closing', now, self._width_meas()
            elif drop_state == 'closing':
                w = self._width_meas()
                if abs(w - drop_w) > _GRIP_STABLE_EPS:
                    drop_t, drop_w = now, w
                elif now - drop_t >= self._grip_stable_s:
                    drop_state = 'done'
            err = np.abs(q_des - q)
            if float(np.max(err)) > _TRACK_STALL_RAD:
                track_since = now if track_since is None else track_since
                if now - track_since >= _TRACK_STALL_S:
                    self._hold()
                    j = int(np.argmax(err))
                    return self._result(False, 'stalled', 'blocked: joint%d %.3f rad behind the '
                                        'trajectory (contact?)' % (j + 1, float(err[j])), t0,
                                        T_goal)
            else:
                track_since = None
            if not hang and ts >= traj.end_time and drop_state in ('', 'done'):
                still = float(np.max(np.abs(qd))) < _STILL_VEL
                stable = stable + 1 if still else 0
                at, text = self._at_target(q, T_goal, q_goal)
                if at and stable >= self._settle_samples and now - t0 >= self._min_wait:
                    return self._result(True, '', '', t0, T_goal)
                if not at and still:
                    still_since = now if still_since is None else still_since
                    if now - still_since >= self._stall_s:
                        self._hold()
                        return self._result(False, 'stalled', 'arm stopped, %s' % text, t0,
                                            T_goal)
                else:
                    still_since = None
            if now >= deadline:
                self._hold()
                _at, text = self._at_target(q, T_goal, q_goal)
                return self._result(False, 'timeout', 'motion did not finish within %.2f s (%s)'
                                    % (limit, text), t0, T_goal)

    def _linear_waypoints(self, T0: np.ndarray, T1: np.ndarray, q0: np.ndarray):
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        n = int(math.ceil(max(dist / _LIN_STEP_M, ang / _LIN_STEP_RAD, 1.0) - 1e-9))
        if n > _MAX_WAYPOINTS:
            return None, 'linear move too long (%.2f m)' % dist
        poses = [interpolate_pose(T0, T1, k / float(n)) for k in range(n + 1)]
        poses[-1] = T1
        qs, why = self.ik.solve_path(poses, q0, max_joint_step=_MAX_JOINT_STEP,
                                     accept_pos=_REACH_POS, accept_rot=_REACH_ROT)
        if qs is None:
            return None, 'linear path: ' + why
        qs[0] = q0
        qs[-1] = self._polish(T1, qs[-1])
        return qs, ''

    def _polish(self, T: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Refine an IK solution; log when the target is just outside the MuJoCo arm's reach."""
        qr, ep, eo = self.ik.refine(T, q)
        if ep > 1e-4 or eo > 1e-3:
            LOG.info('sim: target %.2f mm / %.2f deg outside the MuJoCo arm reach (model vs '
                     'SDK kinematics); going to the closest pose', ep * 1e3, math.degrees(eo))
        return qr

    def move_tcp(self, T_base_tcp: np.ndarray, linear: bool = False,
                 timeout: Optional[float] = None) -> MotionResult:
        """Move the TCP to T_base_tcp (P2P in joint space, or a Cartesian line)."""
        gate = self._motion_gate()
        if gate is not None:
            return gate
        T = np.asarray(T_base_tcp, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not is_rotation(T[:3, :3], 1e-3):
            return MotionResult(ok=False, error='rejected', detail='invalid target pose')
        self.command_log.append({'cmd': 'move_tcp', 'linear': bool(linear), 'T': T.copy(),
                                 't': self._clock.now()})
        fault = self._take_fault(_MOTION_FAULTS)
        if fault:
            return self._instant_failure(fault, 'simulated %s' % fault)
        t0 = self._clock.now()
        q0, T0 = self._q_meas(), self._tcp_meas()
        dist = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
        limit = self._limit(dist, timeout)
        scale = self._speed / 100.0
        if linear:
            qs, why = self._linear_waypoints(T0, T, q0)
            if qs is None:
                return self._instant_failure('unreachable', why)
            ang = rotation_angle(T0[:3, :3].T @ T[:3, :3])
            n = qs.shape[0] - 1
            jrate = float(np.max(np.abs(np.diff(qs, axis=0)))) * n if n > 0 else 0.0
            rates = [r for r in (dist / (self._lin_speed * scale),
                                 ang / (self._joint_speed * scale),
                                 jrate / (self._joint_speed * scale)) if r > 0]
            v = 1.0 / max(rates) if rates else math.inf
        else:
            q_goal, ep, eo = self.ik.solve(T, q_seed=q0, accept_pos=_REACH_POS,
                                           accept_rot=_REACH_ROT)
            if q_goal is None:
                return self._instant_failure('unreachable', 'no IK solution (residual %.4f m, '
                                             '%.2f deg)' % (ep, math.degrees(eo)))
            q_goal = self._polish(T, q_goal)
            qs = np.vstack([q0, q_goal])
            dq = float(np.max(np.abs(q_goal - q0)))
            v = self._joint_speed * scale / dq if dq > 0 else math.inf
        return self._execute(qs, v, T, None, t0, limit)

    def move_joints(self, q: Sequence[float], timeout: Optional[float] = None) -> MotionResult:
        """Joint-space move to q (rad); arrived when every joint is within 0.01 rad."""
        gate = self._motion_gate()
        if gate is not None:
            return gate
        try:
            qt = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return MotionResult(ok=False, error='rejected', detail='joint target not numeric')
        if qt.shape[0] != 6 or not np.all(np.isfinite(qt)):
            return MotionResult(ok=False, error='rejected', detail='invalid joint target')
        self.command_log.append({'cmd': 'move_joints', 'q': qt.copy(), 't': self._clock.now()})
        if not self.ik.within_limits(qt):
            return self._instant_failure('unreachable', 'joint limits: '
                                         + ', '.join(self.ik.limit_violations(qt)))
        fault = self._take_fault(_MOTION_FAULTS)
        if fault:
            return self._instant_failure(fault, 'simulated %s' % fault)
        t0 = self._clock.now()
        q0 = self._q_meas()
        T_goal = self.ik.fk(qt)
        dist = float(np.linalg.norm(T_goal[:3, 3] - self._tcp_meas()[:3, 3]))
        limit = self._limit(dist, timeout)
        dq = float(np.max(np.abs(qt - q0)))
        v = self._joint_speed * (self._speed / 100.0) / dq if dq > 0 else math.inf
        return self._execute(np.vstack([q0, qt]), v, T_goal, qt, t0, limit)

    # ------------------------------------------------------------ gripper
    def gripper(self, width: float, force: Optional[float] = None,
                wait: bool = True) -> GripperResult:
        """
        Command the opening width (m, clipped to [0, max width]) like the AGX gripper.

        The fingers ramp to the width with the grip force limited to force (N,
        default robot.gripper.force, clipped to [0.5, 3.0]), so a close stops
        on the object. With wait, returns once the measured width has been
        stable for robot.gripper.stable_s after the fingers moved (or after
        settle_s when they did not need to move); blocked by an object = ok
        with the measured width. An opening stopped more than 5 mm short is
        error 'fault' ('opening stopped ...'); robot.gripper.timeout_s without
        a stable width is 'timeout'.
        """
        self.n_calls += 1
        if not self._connected:
            return GripperResult(ok=False, error='comm', detail='not connected')
        if self._comm_lost or self._take_fault(('comm_loss',)):
            self._comm_lost = True
            return GripperResult(ok=False, error='comm', detail='simulated communication loss')
        w0 = self._width_meas()
        if self.abort_event.is_set():
            return GripperResult(ok=False, width=w0, error='aborted', detail='abort requested')
        try:
            w = float(width)
            f = self._force if force is None else float(force)
        except (TypeError, ValueError):
            w = f = float('nan')
        if not (math.isfinite(w) and math.isfinite(f) and f >= 0):
            return GripperResult(ok=False, width=w0, error='fault',
                                 detail='invalid width / force %r / %r' % (width, force))
        target = min(max(w, 0.0), self._max_width)
        f_cmd = min(max(f, GRIPPER_FORCE_RANGE[0]), GRIPPER_FORCE_RANGE[1])
        self.command_log.append({'cmd': 'gripper', 'width': target, 'force': f_cmd,
                                 't': self._clock.now()})
        closing = target < w0 - _WIDTH_EPS and target <= self._close_width + 0.001
        if closing and self._miss is None and self._take_fault(('grasp_miss',)):
            with self._sim.lock:
                self._miss = {'width': w0, 'z': float(self._sim.d.xpos[self._flange][2]),
                              'closed': 0.0}
                self._finger_contacts(False)
        self._set_grip_force(f_cmd)
        self._cmd_width = target
        self._command_grip(target)
        if not wait:
            return GripperResult(ok=True, width=w0)
        return self._wait_gripper(target, w0)

    def _wait_gripper(self, target: float, w0: float) -> GripperResult:
        t0 = self._clock.now()
        moved = abs(target - w0) <= _GRIP_MOVE_EPS
        last_w, stable_since, w = None, t0, w0
        while True:
            self._clock.sleep(self._period)
            now = self._clock.now()
            if self.abort_event.is_set():
                return GripperResult(ok=False, width=w, error='aborted', detail='abort requested')
            w = self._width_meas()
            if abs(w - w0) > _GRIP_MOVE_EPS:
                moved = True
            if last_w is None or abs(w - last_w) > _GRIP_STABLE_EPS:
                stable_since = now
            last_w = w
            elapsed = now - t0
            settled = moved or elapsed >= self._grip_settle
            if settled and now - stable_since >= self._grip_stable_s:
                if target > w0 + _GRIP_MOVE_EPS and w < target - _GRIP_OPEN_TOL:
                    return GripperResult(ok=False, width=w, error='fault',
                                         detail='opening stopped at %.4f m, commanded %.4f m'
                                         % (w, target))
                return GripperResult(ok=True, width=w)
            if elapsed >= self._grip_timeout:
                return GripperResult(ok=False, width=w, error='timeout',
                                     detail='width not stable after %.1f s' % self._grip_timeout)

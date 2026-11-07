"""
Real PiPER arm + AgileX gripper through the pyAgxArm SDK (imported lazily).

Every SDK call made by this package is a method of ``_SdkPort`` below, so a
reviewer can audit them in one place. Nothing here calls ``disable()``; the
only ``reset()`` / ``move_js()`` calls are in ``exit_teach_mode()``, which
replicates the official agx_arm_ctrl node, is for on-site tools only and
refuses unless the arm is folded at joint zero. No joint / pose command is
sent while the controller reports drag-teach mode (except that sequence).

Teach mode and firmware (agx_arm_ros README + agx_arm_ctrl_single_node):
Piper firmware S-V1.8-5 and later switch from drag-teach to CAN control
seamlessly when a CAN control frame arrives (the node sends 0x151 through
set_speed_percent at start-up and never gates on ctrl_mode); below S-V1.8-5
only the exit_teach_mode reset sequence (or a power cycle) leaves it.

Frames: base frame of the arm; the SDK's move_p / move_l / get_flange_pose use
the FLANGE (link6). TCP = flange @ T_flange_tcp from config robot.tcp_offset;
the SDK's own TCP offset (set_tcp_offset) is never used.

Feedback freshness: pyAgxArm stores ``rx_can_frame.timestamp`` of python-can
in ``MessageAbstract.timestamp`` (drivers/core/table_driven.py). python-can
stamps socketcan frames with SO_TIMESTAMPNS (CLOCK_REALTIME) and virtual-bus
frames with time.time(), so the feedback age is ``time.time() - timestamp``.
This module therefore reads the wall clock through the injectable
``wall_time`` callable; all waiting uses the injected Clock. If the stamps are
not on the wall clock (other CAN backends), the age falls back to the
monotonic time since the stamp last changed.
"""
from __future__ import annotations

import logging
import math
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.interfaces import ArmError, ArmInterface, Clock
from ..core.transforms import inv_T, is_rotation, pose6_to_T, rotation_angle, T_to_pose6
from ..core.types import ArmHealth, GripperResult, MotionResult, Stamped
from .kinematics import flange_to_tcp_from_cfg, PiperKinematics

LOG = logging.getLogger('cube_task.robot.pyagxarm_arm')

# Values of pyAgxArm ArmMsgFeedbackStatusEnum (msgs/piper/default/feedback/
# arm_feedback_status.py).
ARM_STATUS_NAMES = {
    0x00: 'NORMAL', 0x01: 'EMERGENCY_STOP', 0x02: 'NO_SOLUTION', 0x03: 'SINGULARITY_POINT',
    0x04: 'TARGET_POS_EXCEEDS_LIMIT', 0x05: 'JOINT_COMMUNICATION_ERR',
    0x06: 'JOINT_BRAKE_NOT_RELEASED', 0x07: 'COLLISION_OCCURRED',
    0x08: 'OVERSPEED_DURING_TEACHING_DRAG', 0x09: 'JOINT_STATUS_ERR', 0x0A: 'OTHER_ERR',
    0x0B: 'TEACHING_RECORD', 0x0C: 'TEACHING_EXECUTION', 0x0D: 'TEACHING_PAUSE',
    0x0E: 'MAIN_CONTROLLER_NTC_OVER_TEMPERATURE',
    0x0F: 'RELEASE_RESISTOR_NTC_OVER_TEMPERATURE', 0xFF: 'UNKNOWN',
}
CTRL_MODE_NAMES = {
    0x00: 'STANDBY', 0x01: 'CAN_CTRL', 0x02: 'TEACHING_MODE', 0x03: 'ETHERNET_CONTROL_MODE',
    0x04: 'WIFI_CONTROL_MODE', 0x05: 'REMOTE_CONTROL_MODE',
    0x06: 'LINKAGE_TEACHING_INPUT_MODE', 0x07: 'OFFLINE_TRAJECTORY_MODE', 0xFF: 'UNKNOWN',
}
CTRL_TEACHING = 0x02
CTRL_USABLE = (0x00, 0x01)              # standby (before the first command), CAN control
# arm_status codes raised by a bad target; they clear with the next valid command,
# so they fail the motion that caused them but do not make health() fail.
STATUS_TARGET_ERRORS = (0x02, 0x03, 0x04)
STATUS_COMM_ERRORS = (0x05,)
MODELS = ('piper', 'piper_h', 'piper_l', 'piper_x')       # pyAgxArm ArmModel (Piper series)
FIRMWARES = ('default', 'v183', 'v188', 'v189')           # pyAgxArm PiperFW
GRIPPER_FORCE_RANGE = (0.5, 3.0)        # N, agx_arm_ctrl AgxGripperWrapper.FORCE_MIN / MAX
GRIPPER_FAULT_FLAGS = ('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                       'driver_overheating', 'sensor_status', 'driver_error_status')

_CAN_HINT = ("Check the CAN link: 'ip link show {ch}' must show state UP and "
             "'ip -details link show {ch}' bitrate 1000000 (bring it up with "
             "'sudo ip link set {ch} up type can bitrate 1000000'), the arm must be powered "
             "on, and no other program (agx_arm_ctrl node, another script) may use the arm.")
_CLOCK_SKEW_S = 1.0         # tolerated negative feedback age (wall clock slew)
_MAX_PLAUSIBLE_AGE_S = 3600.0
_JOINT_TOL = 0.01           # rad, joint-space "arrived" tolerance
_JOINT_STABLE_EPS = 0.002   # rad change between samples counted as stable
_ROT_STABLE_EPS = 0.003     # rad change between samples counted as stable
_GRIP_MOVE_EPS = 0.002      # m, width change that shows the fingers started moving
_GRIP_STABLE_EPS = 0.0005   # m, width change between polls counted as stable
_GRIP_OPEN_TOL = 0.005      # m, an opening command must reach target - this
_ENABLE_RETRY_S = 0.1
_RPY_WRAP_TOL = 1e-6        # rad, roll / yaw this close to +-pi are put on the feedback's side
#: exit_teach_mode() first jumps to joint zero in MIT mode (no smoothing): it is
#: only allowed when every joint is already this close to 0 (rad, ~2.9 deg).
EXIT_TEACH_MAX_JOINT_RAD = 0.05
#: First Piper firmware that leaves drag-teach mode seamlessly on a CAN control
#: frame (agx_arm_ctrl_single_node: not seamless = profile default, or v183 with
#: patch < 5; pyAgxArm maps >= 1.8.3 to v183, >= 1.8.8 to v188, >= 1.8.9 to v189).
SEAMLESS_TEACH_FIRMWARE = (1, 8, 5)
_PIPER_VERSION = re.compile(r'S-V(\d+)\.(\d+)-(\d+)')


class SdkCallError(ArmError):
    """An SDK call raised; the message names the call."""


def _import_sdk() -> SimpleNamespace:
    """Import pyAgxArm lazily. Tests replace this function with a fake SDK."""
    try:
        import pyAgxArm as sdk  # noqa: F401 (optional dependency)
        from pyAgxArm import (AgxArmFactory, ArmModel, create_agx_arm_config, PiperFW,
                              resolve_firmware_profile)
    except ImportError as exc:
        raise ArmError('pyAgxArm / python-can not importable (%s). Install them for the '
                       'Python that runs cube_task (see pyAgxArm README), do not upgrade '
                       'numpy.' % exc)
    return SimpleNamespace(create_agx_arm_config=create_agx_arm_config,
                           AgxArmFactory=AgxArmFactory, ArmModel=ArmModel, PiperFW=PiperFW,
                           resolve_firmware_profile=resolve_firmware_profile)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def piper_firmware_tuple(version: Any) -> Optional[Tuple[int, int, int]]:
    """Parse a Piper software version such as 'S-V1.8-5' into (1, 8, 5); None if not one."""
    m = _PIPER_VERSION.fullmatch(str(version or '').strip())
    return None if m is None else (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def seamless_teach_switch(version: Any) -> Optional[bool]:
    """
    Tell whether the firmware leaves drag-teach mode on a CAN control frame.

    True for S-V1.8-5 and later, False below, None when the version is unknown.
    """
    v = piper_firmware_tuple(version)
    return None if v is None else v >= SEAMLESS_TEACH_FIRMWARE


def teach_exit_hint(version: Any = None) -> str:
    """
    Return what to do when TEACHING_MODE stays after pressing the teach button again.

    version: controller software version (firmware_info['software_version'])
    or None. The text depends on seamless_teach_switch(version).
    """
    c_text = ('take CAN control with option c of calib_verify / color_calib --observe '
              '(enable + mode frame, no motion)')
    x_text = ('while drag mode is still on, drag the arm folded onto its support (every joint '
              'within %.1f deg of 0) and let go, then use option x (exit_teach_mode: jump to '
              'zero, e-stop, reset) or power-cycle the arm there'
              % math.degrees(EXIT_TEACH_MAX_JOINT_RAD))
    seamless = seamless_teach_switch(version)
    if seamless:
        return 'firmware %s (S-V1.8-5 or later) switches seamlessly: %s' % (version, c_text)
    if seamless is False:
        return ('firmware %s is below S-V1.8-5, only a reset ends TEACHING_MODE: %s'
                % (version, x_text))
    return 'firmware S-V1.8-5 and later: %s; older firmware: %s' % (c_text, x_text)


def zero_pose_problem(joints: Any, tol: float = EXIT_TEACH_MAX_JOINT_RAD) -> str:
    """Return why joints (rad) are not folded at joint zero within tol, '' when they are."""
    try:
        q = np.asarray(joints, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return 'joint angles not numeric'
    if q.shape[0] != 6 or not np.all(np.isfinite(q)):
        return 'need 6 finite joint angles'
    off = ['joint%d %.1f deg' % (i + 1, math.degrees(v)) for i, v in enumerate(q)
           if abs(v) > tol]
    if not off:
        return ''
    return ('not folded at joint zero: %s (limit %.1f deg)'
            % (', '.join(off), math.degrees(tol)))


def canonical_pose6(pose: Sequence[float], current: Optional[Sequence[float]]) -> List[float]:
    """
    Put a target roll / yaw that sits at +-pi on the side of the current feedback angle.

    pose, current: [x, y, z, roll, pitch, yaw] (m, rad). A vertical tool has
    roll = +pi or -pi depending on the sign of a floating-point zero; roll pi
    and -pi are the same rotation, but a controller that interpolates RPY
    componentwise would see a ~360 deg jump. Values within 1e-6 rad of +-pi
    whose sign differs from the current angle are replaced by pi with the
    current angle's sign. current None or malformed: pose unchanged.
    """
    out = [float(v) for v in pose]
    if current is None or len(current) != 6:
        return out
    for i in (3, 5):
        c = _num(current[i])
        if c is None or abs(abs(out[i]) - math.pi) > _RPY_WRAP_TOL:
            continue
        if math.copysign(1.0, out[i]) != math.copysign(1.0, c):
            out[i] = math.copysign(math.pi, c)
    return out


def linear_moves_enabled(cfg) -> bool:
    """
    Return config robot.motion.linear_moves (default True).

    False asks the skill layer to send the short lift / release / retreat
    moves as P2P (move_p) instead of move_l. Accepts booleans, 0 / 1 and
    the strings true / false / yes / no / on / off; anything else raises
    ValueError.
    """
    v = cfg.get('robot.motion.linear_moves', True)
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    s = str(v).strip().lower()
    if s in ('true', 'yes', 'on', '1'):
        return True
    if s in ('false', 'no', 'off', '0'):
        return False
    raise ValueError('config robot.motion.linear_moves must be true or false, got %r' % (v,))


class _SdkPort:
    """
    The complete list of pyAgxArm calls made by cube_task.

    Commands raise SdkCallError (the SDK raises RuntimeError / ValueError on
    send failures). Feedback reads return None when the SDK raises, and log
    the first failure of each read.
    """

    def __init__(self, sdk: SimpleNamespace, model: str, firmware: str, interface: str,
                 channel: str, effector: Optional[str]):
        self._failed_reads: Dict[str, bool] = {}
        self.sdk = sdk
        # The SDK keyword is spelled 'firmeware_version'.
        self.config = sdk.create_agx_arm_config(robot=model, firmeware_version=firmware,
                                                interface=interface, channel=channel)
        self.robot = sdk.AgxArmFactory.create_arm(self.config)
        self.effector = None
        if effector == 'agx_gripper':
            # Must happen before connect() (pyAgxArm docs, init_effector notes).
            self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def _cmd(self, name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            raise SdkCallError('pyAgxArm %s failed: %s' % (name, exc))

    def _read(self, name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            if not self._failed_reads.get(name):
                LOG.warning('pyAgxArm %s raised %r (reported once)', name, exc)
                self._failed_reads[name] = True
            return None

    # lifecycle
    def connect(self) -> None:
        self._cmd('connect', self.robot.connect)

    def disconnect(self) -> None:
        self._cmd('disconnect', self.robot.disconnect)

    # feedback (read-only)
    def is_ok(self) -> bool:
        return bool(self._read('is_ok', self.robot.is_ok))

    def comm_error(self) -> Optional[str]:
        """Return the SDK's recorded communication error as text, or None."""
        if not self._read('has_comm_error', self.robot.has_comm_error):
            return None
        return str(self._read('get_comm_error', self.robot.get_comm_error))

    def get_joint_angles(self):
        return self._read('get_joint_angles', self.robot.get_joint_angles)

    def get_flange_pose(self):
        return self._read('get_flange_pose', self.robot.get_flange_pose)

    def get_arm_status(self):
        return self._read('get_arm_status', self.robot.get_arm_status)

    def get_driver_states(self, joint_index: int):
        return self._read('get_driver_states', lambda: self.robot.get_driver_states(joint_index))

    def get_joints_enable_status_list(self) -> Optional[List[bool]]:
        v = self._read('get_joints_enable_status_list',
                       self.robot.get_joints_enable_status_list)
        return None if v is None else [bool(x) for x in v]

    def get_firmware(self, timeout: float) -> Optional[dict]:
        return self._read('get_firmware', lambda: self.robot.get_firmware(timeout=timeout,
                                                                          min_interval=0.0))

    def resolve_firmware_profile(self, model: str, version: str) -> Optional[str]:
        return self._read('resolve_firmware_profile',
                          lambda: self.sdk.resolve_firmware_profile(model, version))

    def gripper_is_ok(self) -> bool:
        return self.effector is not None and bool(self._read('gripper.is_ok',
                                                             self.effector.is_ok))

    def get_gripper_status(self):
        if self.effector is None:
            return None
        return self._read('get_gripper_status', self.effector.get_gripper_status)

    # commands
    def enable(self) -> bool:
        """Send the enable-all frame; True when every joint reports enabled."""
        return bool(self._cmd('enable', self.robot.enable))

    def set_speed_percent(self, percent: int) -> None:
        self._cmd('set_speed_percent', lambda: self.robot.set_speed_percent(int(percent)))

    def move_p(self, flange_pose6: Sequence[float]) -> None:
        pose = [float(v) for v in flange_pose6]       # the SDK validator wants a list
        self._cmd('move_p', lambda: self.robot.move_p(pose))

    def move_l(self, flange_pose6: Sequence[float]) -> None:
        pose = [float(v) for v in flange_pose6]
        self._cmd('move_l', lambda: self.robot.move_l(pose))

    def move_j(self, joints: Sequence[float]) -> None:
        q = [float(v) for v in joints]
        self._cmd('move_j', lambda: self.robot.move_j(q))

    def move_gripper_m(self, width: float, force: float) -> None:
        if self.effector is None:
            raise SdkCallError('no gripper effector initialized')
        self._cmd('move_gripper_m',
                  lambda: self.effector.move_gripper_m(value=float(width), force=float(force)))

    def electronic_emergency_stop(self) -> None:
        """Damped e-stop: the arm sinks slowly; needs reset() afterwards."""
        self._cmd('electronic_emergency_stop', self.robot.electronic_emergency_stop)

    # teach-mode exit only (PyAgxArm.exit_teach_mode); both can drop or jerk the arm
    def move_js(self, joints: Sequence[float]) -> None:
        q = [float(v) for v in joints]
        self._cmd('move_js', lambda: self.robot.move_js(q))

    def reset(self) -> None:
        """Reset the controller. The motors lose power at once: the arm drops."""
        self._cmd('reset', self.robot.reset)


class _Status:
    """Decoded 0x2A1 arm status feedback."""

    def __init__(self, msg: Any, age: float):
        self.age = age
        self.ctrl_mode = self._code(getattr(msg, 'ctrl_mode', None))
        self.arm_status = self._code(getattr(msg, 'arm_status', None))
        self.motion_status = self._code(getattr(msg, 'motion_status', None))
        self.teach_status = self._code(getattr(msg, 'teach_status', None))
        err = getattr(msg, 'err_status', None)
        self.limit_joints = [i for i in range(1, 7)
                             if bool(getattr(err, 'joint_%d_angle_limit' % i, False))]
        self.comm_joints = [i for i in range(1, 7)
                            if bool(getattr(err, 'communication_status_joint_%d' % i, False))]

    @staticmethod
    def _code(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0xFF

    def status_name(self) -> str:
        return ARM_STATUS_NAMES.get(self.arm_status, '0x%02X' % self.arm_status)

    def mode_name(self) -> str:
        return CTRL_MODE_NAMES.get(self.ctrl_mode, '0x%02X' % self.ctrl_mode)

    def motion_error(self) -> Tuple[str, str]:
        """(MotionResult error, detail) for this status during a motion; ('', '') if none."""
        if self.comm_joints:
            return 'comm', 'controller reports joint communication errors on joints %s' % (
                self.comm_joints,)
        code = self.arm_status
        if code == 0x00:
            # Angle-limit flags are informational: the folded rest pose sits on the
            # joint2 / joint3 limits, so failing on them could block every motion.
            # A real limit problem shows up as arm_status 0x04 or as a stall.
            return '', ''
        name = self.status_name()
        if code in STATUS_TARGET_ERRORS:
            return 'unreachable', 'controller arm_status %s' % name
        if code in STATUS_COMM_ERRORS:
            return 'comm', 'controller arm_status %s' % name
        return 'arm_error', 'controller arm_status %s' % name

    def health_problem(self) -> str:
        """Why the controller cannot be commanded now ('' when it can)."""
        if self.ctrl_mode not in CTRL_USABLE:
            hint = (' (drag-teach active: press the teach button on the arm again; if the mode '
                    'stays, check_setup --live prints the way out for this firmware)'
                    if self.ctrl_mode == CTRL_TEACHING else '')
            return 'controller in %s mode%s' % (self.mode_name(), hint)
        if self.comm_joints:
            return 'joint communication errors on joints %s' % (self.comm_joints,)
        if self.arm_status not in (0x00,) + STATUS_TARGET_ERRORS:
            return 'arm_status %s' % self.status_name()
        return ''


def _req(cfg, key: str, lo: float, hi: float) -> float:
    """Return a required numeric config value in [lo, hi]; never defaulted."""
    v = cfg.get(key, None)
    f = _num(v)
    if f is None or not lo <= f <= hi:
        raise ValueError('config %s must be a number in [%g, %g], got %r' % (key, lo, hi, v))
    return f


def _opt(cfg, key: str, default: float, lo: float, hi: float) -> float:
    v = cfg.get(key, default)
    f = _num(v)
    if f is None or not lo <= f <= hi:
        raise ValueError('config %s must be a number in [%g, %g], got %r' % (key, lo, hi, v))
    return f


class PyAgxArm(ArmInterface):
    """
    ArmInterface for the real PiPER via pyAgxArm (CAN).

    cfg keys (required unless a default is shown): robot.model, robot.firmware,
    robot.can_interface, robot.can_channel, robot.effector ('agx_gripper' or
    'none'), robot.speed_percent, robot.tcp_offset, robot.motion.{pos_tol,
    rot_tol_deg, settle_samples, stable_eps, min_wait_s, timeout_base_s,
    timeout_per_m_s, poll_hz, feedback_max_age_s}, robot.gripper.{max_width,
    force, settle_s, timeout_s}. Optional: robot.connect_timeout_s (5.0),
    robot.motion.stall_s (1.0), robot.motion.require_motion_status (true),
    robot.gripper.stable_s (0.15).

    clock: injected Clock used for every wait. wall_time: wall-clock source
    matching the SDK feedback stamps (default time.time; tests pass a fake).
    """

    def __init__(self, cfg, clock: Clock, wall_time: Optional[Callable[[], float]] = None):
        self._clock = clock
        self._wall = wall_time if wall_time is not None else time.time
        self.model = str(cfg.get('robot.model', None))
        if self.model not in MODELS:
            raise ValueError('config robot.model must be one of %s, got %r' % (MODELS, self.model))
        self.firmware = str(cfg.get('robot.firmware', None))
        if self.firmware not in FIRMWARES:
            raise ValueError('config robot.firmware must be one of %s, got %r'
                             % (FIRMWARES, self.firmware))
        self.interface = str(cfg.get('robot.can_interface', '') or '')
        self.channel = str(cfg.get('robot.can_channel', '') or '')
        if not self.interface or not self.channel:
            raise ValueError('config robot.can_interface and robot.can_channel are required')
        eff = cfg.get('robot.effector', None)
        eff = 'none' if eff in (None, '', 'none', False) else str(eff)
        if eff not in ('agx_gripper', 'none'):
            raise ValueError("config robot.effector must be 'agx_gripper' or 'none', got %r"
                             % (eff,))
        self.effector_kind = eff
        sp = cfg.get('robot.speed_percent', None)
        if _num(sp) is None or int(sp) != sp or not 1 <= int(sp) <= 100:
            raise ValueError('config robot.speed_percent must be an integer in [1, 100], got %r'
                             % (sp,))
        self._speed = int(sp)
        self._T_ft = flange_to_tcp_from_cfg(cfg)
        self._T_tf = inv_T(self._T_ft)
        self.kin = PiperKinematics(self.model)
        self._pos_tol = _req(cfg, 'robot.motion.pos_tol', 1e-4, 0.05)
        self._rot_tol = math.radians(_req(cfg, 'robot.motion.rot_tol_deg', 0.01, 30.0))
        self._settle_samples = int(_req(cfg, 'robot.motion.settle_samples', 1, 1000))
        self._stable_eps = _req(cfg, 'robot.motion.stable_eps', 1e-6, 0.05)
        self._min_wait = _req(cfg, 'robot.motion.min_wait_s', 0.0, 10.0)
        self._timeout_base = _req(cfg, 'robot.motion.timeout_base_s', 0.1, 600.0)
        self._timeout_per_m = _req(cfg, 'robot.motion.timeout_per_m_s', 0.0, 600.0)
        self._period = 1.0 / _req(cfg, 'robot.motion.poll_hz', 1.0, 1000.0)
        self._max_age = _req(cfg, 'robot.motion.feedback_max_age_s', 0.01, 10.0)
        self._stall_s = _opt(cfg, 'robot.motion.stall_s', 1.0, 0.05, 60.0)
        self._require_motion_status = bool(cfg.get('robot.motion.require_motion_status', True))
        self._connect_timeout = _opt(cfg, 'robot.connect_timeout_s', 5.0, 0.1, 120.0)
        self._max_width = _req(cfg, 'robot.gripper.max_width', 0.001, 0.2)
        self._force = _req(cfg, 'robot.gripper.force', 0.0, 3.0)
        self._grip_settle = _req(cfg, 'robot.gripper.settle_s', 0.0, 10.0)
        self._grip_timeout = _req(cfg, 'robot.gripper.timeout_s', 0.1, 60.0)
        self._grip_stable_s = _opt(cfg, 'robot.gripper.stable_s', 0.15, 0.0, 5.0)
        self._port: Optional[_SdkPort] = None
        self._connected = False
        self._seen: Dict[str, Tuple[float, float]] = {}
        self._clock_warned = False
        self.firmware_info: Optional[dict] = None

    # ------------------------------------------------------------ feedback
    def _age(self, key: str, m: Any) -> Optional[float]:
        """Age (s) of an SDK MessageAbstract, None when it carries no stamp."""
        ts = _num(getattr(m, 'timestamp', None))
        if ts is None or ts <= 0.0:
            return None
        now = self._clock.now()
        last = self._seen.get(key)
        if last is None or last[0] != ts:
            self._seen[key] = (ts, now)
        age = self._wall() - ts
        if -_CLOCK_SKEW_S <= age <= _MAX_PLAUSIBLE_AGE_S:
            return max(0.0, age)
        if not self._clock_warned:
            LOG.warning('feedback stamps are not on the wall clock (age %.1f s); using the '
                        'time since the stamp last changed instead', age)
            self._clock_warned = True
        return now - self._seen[key][1]

    def _read_joints(self) -> Optional[Tuple[np.ndarray, float]]:
        m = self._port.get_joint_angles() if self._port is not None else None
        if m is None:
            return None
        try:
            q = np.array([float(v) for v in m.msg], dtype=float)
        except (TypeError, ValueError):
            return None
        age = self._age('joints', m)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or age is None:
            return None
        return q, age

    def _read_flange(self) -> Optional[Tuple[np.ndarray, float]]:
        m = self._port.get_flange_pose() if self._port is not None else None
        if m is None:
            return None
        try:
            p = [float(v) for v in m.msg]
            T = pose6_to_T(p)
        except (TypeError, ValueError):
            return None
        age = self._age('flange', m)
        if not np.all(np.isfinite(T)) or age is None:
            return None
        return T, age

    def _read_flange_pose6(self) -> Optional[List[float]]:
        """Return the controller's own flange [x, y, z, roll, pitch, yaw], or None."""
        m = self._port.get_flange_pose() if self._port is not None else None
        try:
            p = [float(v) for v in m.msg]
        except (AttributeError, TypeError, ValueError):
            return None
        return p if len(p) == 6 and all(math.isfinite(v) for v in p) else None

    def _sdk_pose6(self, T_base_flange: np.ndarray) -> List[float]:
        """Flange pose6 for move_p / move_l, roll / yaw at +-pi on the feedback's side."""
        return canonical_pose6(T_to_pose6(T_base_flange), self._read_flange_pose6())

    def _read_status(self) -> Optional[_Status]:
        m = self._port.get_arm_status() if self._port is not None else None
        if m is None or getattr(m, 'msg', None) is None:
            return None
        age = self._age('status', m)
        if age is None:
            return None
        return _Status(m.msg, age)

    def _read_gripper(self) -> Optional[Dict[str, Any]]:
        m = self._port.get_gripper_status() if self._port is not None else None
        if m is None or getattr(m, 'msg', None) is None:
            return None
        age = self._age('gripper', m)
        width = _num(getattr(m.msg, 'value', None))
        if age is None or width is None:
            return None
        foc = getattr(m.msg, 'foc_status', None)
        faults = [f for f in GRIPPER_FAULT_FLAGS if bool(getattr(foc, f, False))]
        return {'width': width, 'age': age, 'faults': faults,
                'mode': getattr(m.msg, 'mode', 'width'),
                'enabled': bool(getattr(foc, 'driver_enable_status', False)),
                'homed': bool(getattr(foc, 'homing_status', False))}

    def _fresh(self, sample: Any) -> bool:
        if sample is None:
            return False
        age = sample.age if isinstance(sample, _Status) else (
            sample['age'] if isinstance(sample, dict) else sample[1])
        return age <= self._max_age

    def _comm_problem(self) -> str:
        """Why feedback cannot be trusted right now ('' when it can)."""
        if self._port is None or not self._connected:
            return 'not connected'
        err = self._port.comm_error()
        if err:
            return 'CAN communication error: %s' % err
        if not self._port.is_ok():
            return 'no CAN frames from the arm (pyAgxArm is_ok() is False)'
        for name, sample in (('joint angles', self._read_joints()),
                             ('flange pose', self._read_flange()),
                             ('arm status', self._read_status())):
            if sample is None:
                return 'no %s feedback' % name
            if not self._fresh(sample):
                age = sample.age if isinstance(sample, _Status) else sample[1]
                return '%s feedback is %.2f s old (limit %.2f s)' % (name, age, self._max_age)
        return ''

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        """
        Create the SDK driver, open CAN and wait for live feedback.

        Raises ArmError (with CAN troubleshooting hints) when the SDK is
        missing, the CAN device cannot be opened, or joint / flange / status
        feedback does not arrive within robot.connect_timeout_s.
        """
        if self._connected:
            return
        sdk = _import_sdk()
        hint = _CAN_HINT.format(ch=self.channel)
        port = None
        try:
            port = _SdkPort(sdk, self.model, self.firmware, self.interface, self.channel,
                            None if self.effector_kind == 'none' else self.effector_kind)
            port.connect()
        except Exception as exc:
            if port is not None:
                try:
                    port.disconnect()
                except SdkCallError:
                    pass
            raise ArmError('cannot open the PiPER on %s/%s: %s. %s'
                           % (self.interface, self.channel, exc, hint))
        self._port = port
        self._connected = True
        t0 = self._clock.now()
        while True:
            problem = self._comm_problem()
            if not problem:
                break
            if self._clock.now() - t0 > self._connect_timeout:
                self.close()
                raise ArmError('no live feedback from the PiPER on %s within %.1f s (%s). %s'
                               % (self.channel, self._connect_timeout, problem, hint))
            self._clock.sleep(0.05)
        LOG.info('PiPER connected on %s/%s (model %s, firmware profile %s)',
                 self.interface, self.channel, self.model, self.firmware)
        self._check_firmware()
        st = self._read_status()
        if st is not None and st.health_problem():
            LOG.warning('arm connected but not ready: %s', st.health_problem())

    def _check_firmware(self) -> None:
        info = None
        for _ in range(3):      # the SDK refuses the reply until its FPS counter is > 0
            info = self._port.get_firmware(timeout=0.5)
            if isinstance(info, dict):
                break
            self._clock.sleep(0.1)
        if not isinstance(info, dict):
            LOG.warning('could not read the firmware version (continuing with profile %s)',
                        self.firmware)
            return
        self.firmware_info = info
        sw = str(info.get('software_version', '')).strip()
        prof = self._port.resolve_firmware_profile(self.model, sw) if sw else None
        LOG.info('controller firmware %s (hardware %s)', sw, info.get('hardware_version'))
        if prof is not None and prof != self.firmware:
            LOG.warning('firmware %s maps to pyAgxArm profile %r but config robot.firmware is '
                        '%r; set robot.firmware: %s', sw, prof, self.firmware, prof)

    def enable(self, timeout: float = 5.0) -> bool:
        """
        Enable all joints (repeat the SDK enable until it reports True), then set speed.

        Like agx_arm_ctrl, first waits for the low-speed driver feedback that
        carries the enable flags. Returns False on timeout (seconds).
        """
        if not self._connected or self._port is None:
            return False
        t0 = self._clock.now()
        while self._port.get_driver_states(1) is None:
            if self._clock.now() - t0 > timeout:
                LOG.error('no driver-state feedback, cannot confirm enabling')
                return False
            self._clock.sleep(0.05)
        while True:
            try:
                if self._port.enable():
                    break
            except SdkCallError as exc:
                LOG.warning('%s', exc)
            if self._clock.now() - t0 > timeout:
                LOG.error('joints not all enabled after %.1f s: %s', timeout,
                          self._port.get_joints_enable_status_list())
                return False
            self._clock.sleep(_ENABLE_RETRY_S)
        try:
            self.set_speed_percent(self._speed)
        except SdkCallError as exc:
            LOG.error('enabled, but setting the speed failed: %s', exc)
            return False
        LOG.info('all joints enabled, speed %d %%', self._speed)
        return True

    def close(self) -> None:
        """Disconnect without disabling (disable / reset would drop the arm)."""
        port, self._port, self._connected = self._port, None, False
        if port is not None:
            try:
                port.disconnect()
            except SdkCallError as exc:
                LOG.warning('%s', exc)

    def supports_teach(self) -> bool:
        return True

    # ------------------------------------------------------------ state
    def health(self) -> ArmHealth:
        """Communication, controller mode / status, joint enable flags and gripper state."""
        if not self._connected or self._port is None:
            return ArmHealth(ok=False, reason='not connected', comm_ok=False)
        problem = self._comm_problem()
        st = self._read_status()
        en = self._port.get_joints_enable_status_list()
        detail: Dict[str, Any] = {'enable_status': en}
        if st is not None:
            detail.update(arm_status_name=st.status_name(), ctrl_mode_name=st.mode_name(),
                          motion_status=st.motion_status, teach_status=st.teach_status,
                          limit_flag_joints=st.limit_joints, status_age_s=st.age)
        g_ok, g_reason = self._gripper_health()
        detail.update(gripper_ok=g_ok, gripper_reason=g_reason)
        enabled = None if en is None else bool(en) and all(en)
        kw = dict(enabled=enabled, detail=detail,
                  ctrl_mode=None if st is None else st.ctrl_mode,
                  arm_status=None if st is None else st.arm_status)
        if problem or st is None:
            return ArmHealth(ok=False, reason=problem or 'no arm status', comm_ok=False, **kw)
        reason = st.health_problem()
        if st.comm_joints:
            return ArmHealth(ok=False, reason=reason, comm_ok=False, **kw)
        return ArmHealth(ok=not reason, reason=reason, comm_ok=True, **kw)

    def _gripper_health(self) -> Tuple[Optional[bool], str]:
        if self.effector_kind == 'none':
            return None, 'no gripper configured'
        g = self._read_gripper()
        if g is None or not self._fresh(g):
            return False, 'no fresh gripper feedback'
        if g['faults']:
            return False, 'gripper faults: %s' % ', '.join(g['faults'])
        if g['mode'] != 'width':
            return False, 'gripper in %s mode (expected width mode)' % g['mode']
        if not g['homed']:
            return True, 'gripper reports not homed (calibrate it if widths look wrong)'
        return True, ''

    def _stamped(self, value: Any, age: float) -> Stamped:
        return Stamped(value, self._clock.now() - age)

    def get_joints(self) -> Optional[Stamped]:
        """Joint angles (6,) rad; None when missing or older than feedback_max_age_s."""
        j = self._read_joints()
        return self._stamped(j[0], j[1]) if self._fresh(j) else None

    def get_flange_pose(self) -> Optional[Stamped]:
        """T_base_flange; None when missing or stale."""
        f = self._read_flange()
        return self._stamped(f[0], f[1]) if self._fresh(f) else None

    def get_tcp_pose(self) -> Optional[Stamped]:
        """T_base_tcp = T_base_flange @ T_flange_tcp; None when missing or stale."""
        f = self._read_flange()
        return self._stamped(f[0] @ self._T_ft, f[1]) if self._fresh(f) else None

    def gripper_width(self) -> Optional[Stamped]:
        """Measured gripper opening (m); None when missing or stale."""
        g = self._read_gripper()
        return self._stamped(g['width'], g['age']) if self._fresh(g) else None

    def set_speed_percent(self, percent: int) -> None:
        """Set the controller speed scale for move_p / move_l / move_j (integer 1..100)."""
        if _num(percent) is None or int(percent) != percent or not 1 <= int(percent) <= 100:
            raise ValueError('speed percent must be an integer in [1, 100], got %r' % (percent,))
        if self._port is None:
            raise ArmError('not connected')
        self._port.set_speed_percent(int(percent))
        self._speed = int(percent)

    # ------------------------------------------------------------ motion helpers
    def _pre_motion(self) -> Optional[MotionResult]:
        if not self._connected or self._port is None:
            return MotionResult(ok=False, error='comm', detail='not connected')
        if self.abort_event.is_set():
            return MotionResult(ok=False, error='aborted', detail='abort requested')
        problem = self._comm_problem()
        if problem:
            return MotionResult(ok=False, error='comm', detail=problem)
        st = self._read_status()
        if st is None:
            return MotionResult(ok=False, error='comm', detail='arm status feedback vanished')
        hp = st.health_problem()
        if hp:
            return MotionResult(ok=False, error='comm' if st.comm_joints else 'arm_error',
                                detail=hp)
        en = self._port.get_joints_enable_status_list()
        if not en or not all(en):
            return MotionResult(ok=False, error='arm_error',
                                detail='motors not enabled (status %s), call enable()' % (en,))
        return None

    def _hold(self, why: str) -> None:
        """
        Hold the current position.

        Sends move_j to the measured joints (what agx_arm_ctrl's soft stop does),
        or move_p to the measured flange pose when joint feedback is missing.
        """
        if self._port is None:
            return
        st = self._read_status()
        if st is not None and st.ctrl_mode == CTRL_TEACHING:
            # Nothing to stop: the controller does not run our commands in drag-teach
            # mode, and a joint command could switch it to CAN control under a hand.
            LOG.warning('not holding (%s): controller in drag-teach mode, no command sent', why)
            return
        try:
            j = self._read_joints()
            if self._fresh(j):
                self._port.move_j(j[0])
                LOG.warning('holding position (%s)', why)
                return
            f = self._read_flange()
            if self._fresh(f):
                self._port.move_p(self._sdk_pose6(f[0]))
                LOG.warning('holding position via move_p (%s)', why)
                return
        except SdkCallError as exc:
            LOG.error('could not hold position (%s): %s', why, exc)
            return
        LOG.error('could not hold position (%s): no fresh feedback', why)

    def _sleep_abortable(self, dt: float) -> bool:
        """Sleep dt seconds in poll steps; True when the abort event fired."""
        end = self._clock.now() + dt
        while True:
            if self.abort_event.is_set():
                return True
            left = end - self._clock.now()
            if left <= 0:
                return False
            self._clock.sleep(min(self._period, left))

    def _finish(self, ok: bool, error: str, detail: str, t0: float,
                T_goal_tcp: np.ndarray) -> MotionResult:
        if error:
            LOG.warning('motion %s: %s', error, detail)
        f = self._read_flange()
        if f is None:
            return MotionResult(ok=ok, error=error, detail=detail,
                                duration=self._clock.now() - t0)
        T_tcp = f[0] @ self._T_ft
        return MotionResult(ok=ok, error=error, detail=detail, duration=self._clock.now() - t0,
                            final_tcp=T_tcp,
                            pos_err=float(np.linalg.norm(T_tcp[:3, 3] - T_goal_tcp[:3, 3])),
                            rot_err_deg=math.degrees(rotation_angle(T_tcp[:3, :3].T
                                                                    @ T_goal_tcp[:3, :3])))

    def _measure(self, q_goal: Optional[np.ndarray], T_goal_tcp: np.ndarray):
        """(sample stamp, sample, at_target, error text) for the motion wait, or None."""
        if q_goal is not None:
            j = self._read_joints()
            if j is None:
                return None
            e = float(np.max(np.abs(j[0] - q_goal)))
            return self._seen['joints'][0], j[0], e < _JOINT_TOL, 'max joint error %.4f rad' % e
        f = self._read_flange()
        if f is None:
            return None
        T_tcp = f[0] @ self._T_ft
        pe = float(np.linalg.norm(T_tcp[:3, 3] - T_goal_tcp[:3, 3]))
        re = rotation_angle(T_tcp[:3, :3].T @ T_goal_tcp[:3, :3])
        return (self._seen['flange'][0], T_tcp, pe < self._pos_tol and re < self._rot_tol,
                'tcp %.1f mm / %.2f deg from the target' % (pe * 1e3, math.degrees(re)))

    def _moved(self, prev: np.ndarray, cur: np.ndarray, joints: bool) -> bool:
        if joints:
            return float(np.max(np.abs(cur - prev))) > _JOINT_STABLE_EPS
        return (float(np.linalg.norm(cur[:3, 3] - prev[:3, 3])) > self._stable_eps
                or rotation_angle(prev[:3, :3].T @ cur[:3, :3]) > _ROT_STABLE_EPS)

    def _wait_motion(self, t0: float, timeout: float, pre_status: int,
                     T_goal_tcp: np.ndarray, q_goal: Optional[np.ndarray] = None
                     ) -> MotionResult:
        """
        Block until the motion is verified complete, failed, stalled, timed out or aborted.

        Success needs motion_status == 0 (unless robot.motion.require_motion_status
        is false), the target reached within tolerance, and settle_samples
        consecutive new feedback samples without movement. A target error code
        (no solution / singularity / limit) that was already shown before the
        command is ignored until the arm stalls, since it may be left over from
        the previous command.
        """
        deadline = t0 + timeout
        if self._sleep_abortable(self._min_wait):
            self._hold('abort')
            return self._finish(False, 'aborted', 'abort requested', t0, T_goal_tcp)
        stable, prev, prev_key, stall_since = 0, None, None, None
        while True:
            now = self._clock.now()
            if self.abort_event.is_set():
                self._hold('abort')
                return self._finish(False, 'aborted', 'abort requested', t0, T_goal_tcp)
            problem = self._comm_problem()
            if problem:
                return self._finish(False, 'comm', problem, t0, T_goal_tcp)
            st = self._read_status()
            if st is None:
                return self._finish(False, 'comm', 'arm status feedback vanished', t0,
                                    T_goal_tcp)
            err, detail = st.motion_error()
            if err and not (err == 'unreachable' and st.arm_status == pre_status):
                return self._finish(False, err, detail, t0, T_goal_tcp)
            m = self._measure(q_goal, T_goal_tcp)
            if m is None:
                return self._finish(False, 'comm', 'pose feedback vanished', t0, T_goal_tcp)
            key, cur, at_target, text = m
            if key != prev_key:
                if prev is None or self._moved(prev, cur, q_goal is not None):
                    stable = 0
                else:
                    stable += 1
                prev, prev_key = cur, key
            done = st.motion_status == 0 or not self._require_motion_status
            if done and stable >= self._settle_samples:
                if at_target:
                    return self._finish(True, '', '', t0, T_goal_tcp)
                stall_since = now if stall_since is None else stall_since
                if now - stall_since >= self._stall_s:
                    self._hold('stalled')
                    if st.arm_status in STATUS_TARGET_ERRORS:
                        return self._finish(False, 'unreachable', 'controller arm_status %s, %s'
                                            % (st.status_name(), text), t0, T_goal_tcp)
                    return self._finish(False, 'stalled', 'arm stopped, %s' % text, t0,
                                        T_goal_tcp)
            else:
                stall_since = None
            if now >= deadline:
                self._hold('timeout')
                return self._finish(False, 'timeout', 'not done after %.1f s (motion_status %d, '
                                    '%s)' % (timeout, st.motion_status, text), t0, T_goal_tcp)
            self._clock.sleep(self._period)

    def _timeout_for(self, dist: float, timeout: Optional[float]) -> float:
        if timeout is None:
            return self._timeout_base + self._timeout_per_m * dist
        t = _num(timeout)
        if t is None or t <= 0:
            raise ValueError('timeout must be a positive number of seconds, got %r' % (timeout,))
        return t

    # ------------------------------------------------------------ motion
    def move_tcp(self, T_base_tcp: np.ndarray, linear: bool = False,
                 timeout: Optional[float] = None) -> MotionResult:
        """
        Move the TCP to T_base_tcp with move_l (linear) or move_p (P2P) and verify arrival.

        The SDK receives the flange pose T_base_tcp @ inv(T_flange_tcp) as
        [x, y, z, roll, pitch, yaw] (m, rad, R = Rz Ry Rx); a roll / yaw at
        +-pi is sent with the sign of the current feedback angle
        (canonical_pose6), so the RPY never jumps by ~360 deg. timeout (s)
        defaults to timeout_base_s + timeout_per_m_s * flange travel.
        """
        pre = self._pre_motion()
        if pre is not None:
            return pre
        T = np.asarray(T_base_tcp, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not is_rotation(T[:3, :3], 1e-3):
            return MotionResult(ok=False, error='rejected', detail='invalid target pose')
        T_fl = T @ self._T_tf
        pose = self._sdk_pose6(T_fl)
        if (not all(math.isfinite(v) for v in pose) or abs(pose[4]) > math.pi / 2 + 1e-12
                or abs(pose[3]) > math.pi + 1e-12 or abs(pose[5]) > math.pi + 1e-12):
            return MotionResult(ok=False, error='rejected',
                                detail='flange pose6 out of SDK range: %s' % (pose,))
        f, st = self._read_flange(), self._read_status()
        if f is None or st is None:
            return MotionResult(ok=False, error='comm', detail='feedback vanished')
        cur = self._read_flange_pose6()
        if linear and cur is not None and any(abs(pose[i] - cur[i]) > math.pi for i in (3, 5)):
            LOG.warning('move_l crosses the +-180 deg RPY wrap: rpy %s -> %s deg (the controller '
                        'is not known to take the short way)',
                        [round(math.degrees(v), 2) for v in cur[3:]],
                        [round(math.degrees(v), 2) for v in pose[3:]])
        dist = float(np.linalg.norm(T_fl[:3, 3] - f[0][:3, 3]))
        tmo = self._timeout_for(dist, timeout)
        pre_status = st.arm_status
        t0 = self._clock.now()
        try:
            if linear:
                self._port.move_l(pose)
            else:
                self._port.move_p(pose)
        except SdkCallError as exc:
            return MotionResult(ok=False, error='comm', detail=str(exc),
                                duration=self._clock.now() - t0)
        return self._wait_motion(t0, tmo, pre_status, T)

    def move_joints(self, q: Sequence[float], timeout: Optional[float] = None) -> MotionResult:
        """move_j to joint angles q (rad) and verify every joint within 0.01 rad."""
        pre = self._pre_motion()
        if pre is not None:
            return pre
        try:
            qt = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return MotionResult(ok=False, error='rejected', detail='joint target not numeric')
        if qt.shape[0] != 6 or not np.all(np.isfinite(qt)):
            return MotionResult(ok=False, error='rejected', detail='need 6 finite joint angles')
        if not self.kin.within_limits(qt):
            return MotionResult(ok=False, error='unreachable', detail='joint limits: '
                                + ', '.join(self.kin.limit_violations(qt)))
        T_goal = self.kin.fk(qt) @ self._T_ft
        f, st = self._read_flange(), self._read_status()
        if f is None or st is None:
            return MotionResult(ok=False, error='comm', detail='feedback vanished')
        dist = float(np.linalg.norm(T_goal[:3, 3] - (f[0] @ self._T_ft)[:3, 3]))
        tmo = self._timeout_for(dist, timeout)
        pre_status = st.arm_status
        t0 = self._clock.now()
        try:
            self._port.move_j(qt)
        except SdkCallError as exc:
            return MotionResult(ok=False, error='comm', detail=str(exc),
                                duration=self._clock.now() - t0)
        return self._wait_motion(t0, tmo, pre_status, T_goal, q_goal=qt)

    def stop(self) -> None:
        """Hold the current position (move_j to the measured joints)."""
        self._hold('stop()')

    def emergency_stop(self) -> None:
        """
        Trigger the electronic e-stop: the arm sinks slowly and needs a reset.

        Raises ArmError when the command cannot be sent.
        """
        if self._port is None:
            raise ArmError('emergency stop impossible: not connected')
        LOG.error('electronic emergency stop')
        self._port.electronic_emergency_stop()

    # ------------------------------------------------------------ gripper
    def gripper(self, width: float, force: Optional[float] = None,
                wait: bool = True) -> GripperResult:
        """
        Command the opening width (m) with move_gripper_m and wait until it settles.

        width is clipped to [0, robot.gripper.max_width]; force (N, default
        robot.gripper.force) to [0.5, 3.0]. With wait, returns once the
        measured width has been stable for robot.gripper.stable_s after the
        fingers moved (or after settle_s when they did not need to move).
        Closing on an object (width stable above the target) is ok; an opening
        that stops more than 5 mm short, a driver fault bit or a disabled
        driver is error 'fault'; no fresh feedback is 'comm'.
        """
        if not self._connected or self._port is None:
            return GripperResult(ok=False, error='comm', detail='not connected')
        if self._port.effector is None:
            return GripperResult(ok=False, error='fault', detail='no gripper (robot.effector)')
        if self.abort_event.is_set():
            return GripperResult(ok=False, error='aborted', detail='abort requested')
        w = _num(width)
        f = self._force if force is None else _num(force)
        if w is None or f is None or f < 0:
            return GripperResult(ok=False, error='fault',
                                 detail='invalid width / force %r / %r' % (width, force))
        target = min(max(w, 0.0), self._max_width)
        f_cmd = min(max(f, GRIPPER_FORCE_RANGE[0]), GRIPPER_FORCE_RANGE[1])
        if f_cmd != f:
            LOG.warning('gripper force %.2f N clipped to %.2f N', f, f_cmd)
        g0 = self._read_gripper()
        if not self._fresh(g0):
            return GripperResult(ok=False, error='comm', detail='no fresh gripper feedback')
        if g0['faults']:
            return GripperResult(ok=False, width=g0['width'], error='fault',
                                 detail='gripper faults: %s' % ', '.join(g0['faults']))
        try:
            self._port.move_gripper_m(target, f_cmd)
        except SdkCallError as exc:
            return GripperResult(ok=False, width=g0['width'], error='comm', detail=str(exc))
        if not wait:
            return GripperResult(ok=True, width=g0['width'])
        return self._wait_gripper(target, g0['width'])

    def _wait_gripper(self, target: float, w0: float) -> GripperResult:
        t0 = self._clock.now()
        moved = abs(target - w0) <= _GRIP_MOVE_EPS
        last_w, stable_since, w = None, t0, w0
        while True:
            now = self._clock.now()
            if self.abort_event.is_set():
                return GripperResult(ok=False, width=w, error='aborted', detail='abort requested')
            g = self._read_gripper()
            if not self._fresh(g):
                return GripperResult(ok=False, width=w, error='comm',
                                     detail='gripper feedback lost while moving')
            w = g['width']
            if g['faults']:
                return GripperResult(ok=False, width=w, error='fault',
                                     detail='gripper faults: %s' % ', '.join(g['faults']))
            if abs(w - w0) > _GRIP_MOVE_EPS:
                moved = True
            if last_w is None or abs(w - last_w) > _GRIP_STABLE_EPS:
                stable_since = now
            last_w = w
            elapsed = now - t0
            settled = moved or elapsed >= self._grip_settle
            if settled and now - stable_since >= self._grip_stable_s:
                if not g['enabled']:
                    return GripperResult(ok=False, width=w, error='fault',
                                         detail='gripper driver reports disabled')
                if target > w0 + _GRIP_MOVE_EPS and w < target - _GRIP_OPEN_TOL:
                    return GripperResult(ok=False, width=w, error='fault',
                                         detail='opening stopped at %.4f m, commanded %.4f m'
                                         % (w, target))
                return GripperResult(ok=True, width=w)
            if elapsed >= self._grip_timeout:
                return GripperResult(ok=False, width=w, error='timeout',
                                     detail='width not stable after %.1f s' % self._grip_timeout)
            self._clock.sleep(self._period)

    # ------------------------------------------------------------ teach mode (tools)
    def exit_teach_mode(self) -> bool:
        """
        Leave drag-teach mode with the official agx_arm_ctrl sequence.

        Order: (1) move_js([0]*6): MIT mode, no smoothing, no trajectory
        planning, the arm jumps to joint zero as fast as the motors allow,
        then 2 s wait; (2) electronic e-stop + move_j zero, 0.3 s; (3) reset():
        motor power is cut at once (a raised arm drops), 0.5 s; (4) enable;
        (5) move_j([0]*6) without any SafeArm check. Nobody may touch the arm
        during the sequence. Never call it from the task flow.

        Refuses (ArmError, nothing sent) unless fresh joint feedback shows
        every joint within EXIT_TEACH_MAX_JOINT_RAD of 0, i.e. the arm rests
        folded on its support, so the MIT jump and the power cut move nothing.
        Same steps and pauses as agx_arm_ctrl_single_node._exit_teach_mode_callback
        (the node's is_mit_mode condition is never true here: cube_task does
        not use MIT mode). Our enable() also re-applies robot.speed_percent.
        Only firmware below S-V1.8-5 needs it (seamless_teach_switch).
        Returns True when the sequence ran, False when the arm was not in
        teach mode. Raises ArmError when not connected, not folded at joint
        zero, or a command fails.
        """
        if self._port is None:
            raise ArmError('not connected')
        st = self._read_status()
        if st is None:
            raise ArmError('no arm status feedback')
        if st.ctrl_mode != CTRL_TEACHING:
            LOG.info('arm is not in teach mode (ctrl_mode %s)', st.mode_name())
            return False
        j = self._read_joints()
        if not self._fresh(j):
            raise ArmError('exit_teach_mode refused: no fresh joint feedback, so it cannot '
                           'check that the arm rests folded at joint zero')
        problem = zero_pose_problem(j[0])
        if problem:
            raise ArmError('exit_teach_mode refused: the arm is %s. Its first command jumps to '
                           'joint zero in MIT mode without smoothing, then reset() cuts motor '
                           'power. While drag mode is still on, drag the arm folded onto its '
                           'support, let go, then retry.' % problem)
        zeros = [0.0] * 6
        LOG.warning('exiting teach mode: move_js zero (MIT, unsmoothed), e-stop, reset (motors '
                    'lose power), enable, move_j zero')
        self._port.move_js(zeros)
        self._clock.sleep(2.0)
        self._port.electronic_emergency_stop()
        self._port.move_j(zeros)
        self._clock.sleep(0.3)
        self._port.reset()
        self._clock.sleep(0.5)
        if not self.enable(timeout=5.0):
            LOG.error('enable after leaving teach mode failed')
        self._port.move_j(zeros)
        return True

    def read_teach_pose(self) -> Optional[Dict[str, Any]]:
        """
        Return the current pose for teaching tools, or None without fresh feedback.

        Keys: joints (list, rad), T_base_flange and T_base_tcp (4x4 lists),
        ctrl_mode, teach_status (ints), age_s (feedback age, s).
        """
        j, f, st = self._read_joints(), self._read_flange(), self._read_status()
        if not (self._fresh(j) and self._fresh(f) and self._fresh(st)):
            return None
        return {'joints': [float(v) for v in j[0]],
                'T_base_flange': f[0].tolist(), 'T_base_tcp': (f[0] @ self._T_ft).tolist(),
                'ctrl_mode': st.ctrl_mode, 'teach_status': st.teach_status,
                'age_s': max(j[1], f[1], st.age)}

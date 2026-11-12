"""
Tests for robot/pyagxarm_arm.py against a scripted fake pyAgxArm SDK.

The fake mimics the SDK surface PyAgxArm uses (MessageAbstract-like feedback
with .msg / .timestamp / .hz, arm status enums as ints, move_p / move_l /
move_j, AgxGripper). Its state evolves with the SimClock, and PyAgxArm gets
wall_time=clock.now so feedback stamps and ages are consistent.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.core.interfaces import ArmError
from cube_task.robot import pyagxarm_arm as mod
from cube_task.robot.kinematics import PiperKinematics, tilt_outward

KIN = PiperKinematics()
T_FT = tf.pose6_to_T([0.0, 0.0, 0.1425, 0.0, 0.0, 0.0])
T_TF = tf.inv_T(T_FT)


def _td(x, y, z):
    return tf.top_down_T(x, y, z, tf.neutral_closing_angle(x, y))


Q_START = KIN.ik(_td(0.25, 0.0, 0.06) @ T_TF)[0]


class Ctl:
    """Scripted controller state shared by the fake robot and gripper."""

    def __init__(self, clock):
        self.clock = clock
        self.q = Q_START.copy()
        self.motion = None
        self.behavior = 'arrive'     # arrive | stall | nosolution | never_done | slow
        self.duration = 1.0          # s per motion
        self.latency = 0.05          # s before the controller reacts
        self.motion_status = 0
        self.arm_status = 0
        self.ctrl_mode = 1
        self.seamless = False        # set_speed_percent leaves teach mode (S-V1.8-5+)
        self.comm_joints = []
        self.error_at = None         # (seconds after command, arm_status code)
        self.enabled = [False] * 6
        self.enable_needs = 3        # enable() calls before all joints report enabled
        self.enable_calls = 0
        self.feedback = True
        self.driver_feedback = True
        self.comm_ok = True
        self.comm_error = None
        self.freeze_at = None        # stamps stop advancing from this time (stale feedback)
        self.stamp_offset = 0.0
        self.connect_raises = False
        self.sent = []
        # gripper
        self.g_w0 = 0.0
        self.g_motion = None
        self.g_speed = 0.1
        self.g_block = None          # object width: closing stops here
        self.g_open_block = None     # opening stops here
        self.g_enabled = True
        self.g_faults = set()

    def now(self):
        return self.clock.now()

    def stamp(self):
        t = self.now()
        if self.freeze_at is not None and t >= self.freeze_at:
            t = self.freeze_at
        return t + self.stamp_offset

    # ---- arm motion model
    def command(self, q_target, name):
        self.sent.append((name, self.now()))
        if self.behavior == 'nosolution':
            self.motion = None
            self.arm_status = 0x02
            return
        if self.arm_status in (0x02, 0x03, 0x04):
            self.arm_status = 0             # a valid command clears target errors
        stop = 0.5 if self.behavior == 'stall' else 1.0
        dur = 100.0 if self.behavior == 'slow' else self.duration
        self.motion = {'t0': self.now(), 'q0': self.q.copy(), 'q1': np.asarray(q_target, float),
                       'stop': stop, 'dur': dur}

    def update(self):
        m = self.motion
        if m is None:
            return
        t = self.now() - m['t0'] - self.latency
        if t < 0:
            return
        if self.error_at is not None and self.now() - m['t0'] >= self.error_at[0]:
            self.arm_status = self.error_at[1]
            self.motion = None
            self.motion_status = 0
            return
        s = min(t / m['dur'], m['stop'])
        self.q = m['q0'] + s * (m['q1'] - m['q0'])
        done = s >= m['stop']
        self.motion_status = 1 if (not done or self.behavior == 'never_done') else 0

    # ---- gripper model
    def g_width(self):
        m = self.g_motion
        if m is None:
            return self.g_w0
        t = self.now() - m['t0'] - self.latency
        if t <= 0:
            return m['w0']
        d = m['w1'] - m['w0']
        w = m['w0'] + math.copysign(min(abs(d), self.g_speed * t), d)
        if d < 0 and self.g_block is not None and w < self.g_block:
            w = self.g_block
        if d > 0 and self.g_open_block is not None and w > self.g_open_block:
            w = self.g_open_block
        return w


def _msg(value, stamp):
    return SimpleNamespace(msg=value, timestamp=stamp, hz=200.0)


class FakeEffector:
    def __init__(self, ctl):
        self.ctl = ctl

    def is_ok(self):
        return self.ctl.comm_ok

    def get_gripper_status(self):
        c = self.ctl
        foc = SimpleNamespace(driver_enable_status=c.g_enabled, homing_status=True,
                              **{f: (f in c.g_faults) for f in mod.GRIPPER_FAULT_FLAGS})
        return _msg(SimpleNamespace(value=c.g_width(), force=1.0, foc_status=foc, mode='width'),
                    c.stamp())

    def move_gripper_m(self, value=0.0, force=1.0):
        c = self.ctl
        assert isinstance(value, float) and isinstance(force, float)
        c.sent.append(('move_gripper_m', value, force))
        c.g_motion = {'t0': c.now(), 'w0': c.g_width(), 'w1': value}


class FakeRobot:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER='agx_gripper'))

    def __init__(self, ctl, config):
        self.ctl = ctl
        self.config = config

    def init_effector(self, kind):
        self.ctl.sent.append(('init_effector', kind))
        return FakeEffector(self.ctl)

    def connect(self):
        self.ctl.sent.append(('connect',))
        if self.ctl.connect_raises:
            raise ValueError("Device 'can0' does not exist.")

    def disconnect(self):
        self.ctl.sent.append(('disconnect',))

    def is_ok(self):
        return self.ctl.comm_ok

    def has_comm_error(self):
        return self.ctl.comm_error is not None

    def get_comm_error(self):
        return self.ctl.comm_error

    def get_joint_angles(self):
        self.ctl.update()
        return _msg(list(self.ctl.q), self.ctl.stamp()) if self.ctl.feedback else None

    def get_flange_pose(self):
        self.ctl.update()
        if not self.ctl.feedback:
            return None
        return _msg(tf.T_to_pose6(KIN.fk(self.ctl.q)), self.ctl.stamp())

    def get_arm_status(self):
        self.ctl.update()
        c = self.ctl
        if not c.feedback:
            return None
        err = SimpleNamespace(**{'joint_%d_angle_limit' % i: False for i in range(1, 7)})
        for i in range(1, 7):
            setattr(err, 'communication_status_joint_%d' % i, i in c.comm_joints)
        st = SimpleNamespace(ctrl_mode=c.ctrl_mode, arm_status=c.arm_status, teach_status=0,
                             motion_status=c.motion_status, err_status=err)
        return _msg(st, c.stamp())

    def get_driver_states(self, joint_index):
        return _msg(SimpleNamespace(), self.ctl.stamp()) if self.ctl.driver_feedback else None

    def get_joints_enable_status_list(self):
        return list(self.ctl.enabled)

    def get_firmware(self, timeout=1.0, min_interval=1.0):
        return {'software_version': 'S-V1.8-8', 'hardware_version': 'H-V1.2-1'}

    def enable(self):
        c = self.ctl
        c.enable_calls += 1
        c.sent.append(('enable',))
        if c.enable_calls >= c.enable_needs:
            c.enabled = [True] * 6
        return all(c.enabled)

    def set_speed_percent(self, percent=100):
        assert isinstance(percent, int)
        self.ctl.sent.append(('set_speed_percent', percent))
        if self.ctl.seamless and self.ctl.ctrl_mode == 2:
            self.ctl.ctrl_mode = 1          # 0x151 with ctrl_mode CAN (firmware >= S-V1.8-5)

    def move_p(self, pose):
        assert isinstance(pose, list) and abs(pose[4]) <= math.pi / 2
        self.ctl.update()
        q, _ep, _eo = KIN.ik(tf.pose6_to_T(pose), q_seed=self.ctl.q)
        self.ctl.last_pose = pose
        self.ctl.command(self.ctl.q if q is None else q, 'move_p')

    def move_l(self, pose):
        self.ctl.update()
        q, _ep, _eo = KIN.ik(tf.pose6_to_T(pose), q_seed=self.ctl.q)
        self.ctl.last_pose = pose
        self.ctl.command(self.ctl.q if q is None else q, 'move_l')

    def move_j(self, joints):
        assert isinstance(joints, list) and len(joints) == 6
        self.ctl.update()
        self.ctl.command(joints, 'move_j')

    def move_js(self, joints):
        self.ctl.sent.append(('move_js', self.ctl.now()))

    def electronic_emergency_stop(self):
        self.ctl.sent.append(('electronic_emergency_stop', self.ctl.now()))

    def reset(self):
        self.ctl.sent.append(('reset', self.ctl.now()))

    def disable(self):          # must never be called
        raise AssertionError('disable() called')


def _fake_sdk(ctl, record):
    def create_agx_arm_config(**kw):
        record.append(kw)
        return dict(kw)
    return SimpleNamespace(
        create_agx_arm_config=create_agx_arm_config,
        AgxArmFactory=SimpleNamespace(create_arm=lambda config: FakeRobot(ctl, config)),
        ArmModel=SimpleNamespace(PIPER='piper'), PiperFW=SimpleNamespace(DEFAULT='default'),
        resolve_firmware_profile=lambda model, version: 'v188')


@pytest.fixture()
def rig(monkeypatch):
    clock = SimClock(start=1000.0)
    ctl = Ctl(clock)
    record = []
    monkeypatch.setattr(mod, '_import_sdk', lambda: _fake_sdk(ctl, record))
    cfg = load_config(use_overlays=False)

    def make(connect=True, enable=True, **overrides):
        for k, v in overrides.items():
            cfg.set(k.replace('__', '.'), v)
        arm = mod.PyAgxArm(cfg, clock, wall_time=clock.now)
        if connect:
            arm.connect()
        if enable:
            assert arm.enable()
        return arm
    return SimpleNamespace(clock=clock, ctl=ctl, record=record, make=make, cfg=cfg)


def _names(ctl):
    return [s[0] for s in ctl.sent]


def _at(clock, dt, fn):
    """Run fn once when the SimClock passes now + dt."""
    t = clock.now() + dt
    state = {'done': False}

    def listener(_t_old, t_new):
        if not state['done'] and t_new >= t:
            state['done'] = True
            fn()
    clock.add_listener(listener)


# ---------------------------------------------------------------- connect / enable
def test_connect_creates_sdk_objects_in_order(rig):
    arm = rig.make(enable=False)
    assert rig.record == [{'robot': 'piper', 'firmeware_version': 'default',
                           'interface': 'socketcan', 'channel': 'can0'}]
    assert _names(rig.ctl)[:2] == ['init_effector', 'connect']
    assert arm.firmware_info['software_version'] == 'S-V1.8-8'
    assert arm.health().ok


def test_connect_without_feedback_raises_with_can_hint(rig):
    rig.ctl.feedback = False
    t0 = rig.clock.now()
    with pytest.raises(ArmError) as ei:
        rig.make(enable=False)
    msg = str(ei.value)
    assert 'ip link show can0' in msg and '1000000' in msg
    assert 5.0 <= rig.clock.now() - t0 < 5.5
    assert _names(rig.ctl)[-1] == 'disconnect'


def test_connect_failure_is_wrapped(rig):
    rig.ctl.connect_raises = True
    with pytest.raises(ArmError, match="does not exist.*ip link show can0"):
        rig.make(enable=False)


def test_connect_rejects_stale_feedback(rig):
    rig.ctl.freeze_at = rig.clock.now() - 2.0      # frames stopped two seconds ago
    with pytest.raises(ArmError, match='old'):
        rig.make(enable=False)


def test_enable_repeats_until_all_joints_enabled_then_sets_speed(rig):
    arm = rig.make(enable=False)
    assert arm.enable()
    names = _names(rig.ctl)
    assert names.count('enable') == 3
    assert rig.ctl.sent[-1] == ('set_speed_percent', 30)
    assert arm.health().enabled


def test_enable_timeout(rig):
    rig.ctl.enable_needs = 10 ** 6
    arm = rig.make(enable=False)
    t0 = rig.clock.now()
    assert not arm.enable(timeout=2.0)
    assert 2.0 <= rig.clock.now() - t0 < 2.3
    assert 'set_speed_percent' not in _names(rig.ctl)


def test_config_validation(rig):
    with pytest.raises(ValueError, match='robot.model'):
        rig.make(connect=False, enable=False, robot__model='ur5')
    rig.cfg.set('robot.model', 'piper')
    with pytest.raises(ValueError, match='speed_percent'):
        rig.make(connect=False, enable=False, robot__speed_percent=150)
    rig.cfg.set('robot.speed_percent', 30)
    rig.cfg.data['robot']['motion'].pop('feedback_max_age_s')
    with pytest.raises(ValueError, match='feedback_max_age_s'):
        rig.make(connect=False, enable=False)


# ---------------------------------------------------------------- motion outcomes
def test_move_tcp_arrives(rig):
    arm = rig.make()
    T = _td(0.30, 0.05, 0.05)
    res = arm.move_tcp(T)
    assert res.ok, res
    assert rig.ctl.sent[-1][0] == 'move_p'
    assert rig.ctl.last_pose == pytest.approx(tf.T_to_pose6(T @ T_TF))   # flange pose sent
    assert 1.05 < res.duration < 1.5
    assert res.pos_err < 0.006 and res.rot_err_deg < 4.0
    assert arm.move_tcp(_td(0.30, 0.05, 0.02), linear=True).ok
    assert rig.ctl.sent[-1][0] == 'move_l'


def test_move_tcp_stalled_holds(rig):
    arm = rig.make()
    rig.ctl.behavior = 'stall'
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'stalled' and 'from the target' in res.detail
    assert rig.ctl.sent[-1][0] == 'move_j'                    # hold at the measured joints
    assert 1.5 < res.duration < 2.0


def test_move_tcp_collision_status(rig):
    arm = rig.make()
    rig.ctl.error_at = (0.4, 0x07)
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'arm_error' and 'COLLISION_OCCURRED' in res.detail
    assert not arm.health().ok


def test_move_tcp_no_solution(rig):
    arm = rig.make()
    rig.ctl.behavior = 'nosolution'
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'unreachable' and 'NO_SOLUTION' in res.detail
    assert res.duration < 0.4
    assert arm.health().ok                                   # target errors do not block


def test_leftover_no_solution_status_is_not_blamed_on_the_next_move(rig):
    arm = rig.make()
    rig.ctl.arm_status = 0x02                                # left from an earlier command
    assert arm.move_tcp(_td(0.30, 0.05, 0.05)).ok            # a valid command clears it
    rig.ctl.behavior = 'nosolution'
    first = arm.move_tcp(_td(0.30, -0.05, 0.05))             # new error code: immediate
    assert first.error == 'unreachable' and first.duration < 0.4
    res = arm.move_tcp(_td(0.28, -0.05, 0.05))               # code already shown before
    assert res.error == 'unreachable' and res.duration > 1.0  # decided by the stall rule


def test_comm_loss_during_motion(rig):
    arm = rig.make()
    _at(rig.clock, 0.5, lambda: setattr(rig.ctl, 'comm_ok', False))
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'comm' and 'is_ok' in res.detail
    assert 0.5 <= res.duration < 0.6
    h = arm.health()
    assert not h.ok and not h.comm_ok


def test_can_error_refuses_motion(rig):
    arm = rig.make()
    rig.ctl.comm_error = 'ENETDOWN'
    n = len(rig.ctl.sent)
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'comm' and 'ENETDOWN' in res.detail
    assert len(rig.ctl.sent) == n


def test_stale_feedback_is_a_comm_problem(rig):
    arm = rig.make()
    rig.ctl.freeze_at = rig.clock.now() + 0.4
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'comm' and 'old' in res.detail
    assert 0.9 <= res.duration < 1.0                 # frozen at 0.4 s + 0.5 s max age
    assert arm.get_joints() is None and arm.get_tcp_pose() is None
    assert 'old' in arm.health().reason


def test_stamps_from_another_clock_fall_back_to_change_tracking(rig):
    rig.ctl.stamp_offset = 5.0e4
    arm = rig.make()
    assert arm.move_tcp(_td(0.30, 0.05, 0.05)).ok
    rig.ctl.freeze_at = rig.clock.now() + 0.3
    res = arm.move_tcp(_td(0.30, -0.05, 0.05))
    assert res.error == 'comm' and 'old' in res.detail


def test_abort_holds_position(rig):
    arm = rig.make()
    ev = threading.Event()
    arm.set_abort_event(ev)
    _at(rig.clock, 0.6, ev.set)
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'aborted' and 0.6 <= res.duration < 0.65
    assert rig.ctl.sent[-1][0] == 'move_j'
    n = len(rig.ctl.sent)
    assert arm.move_tcp(_td(0.30, 0.05, 0.05)).error == 'aborted'
    assert arm.gripper(0.05).error == 'aborted'
    assert len(rig.ctl.sent) == n                    # nothing sent while aborted


def test_timeout_holds_position(rig):
    arm = rig.make()
    rig.ctl.behavior = 'slow'
    T = _td(0.30, 0.05, 0.05)
    T0 = arm.get_flange_pose().value
    dist = float(np.linalg.norm((T @ T_TF)[:3, 3] - T0[:3, 3]))
    res = arm.move_tcp(T)
    assert res.error == 'timeout'
    assert res.duration == pytest.approx(6.0 + 15.0 * dist, abs=0.03)
    assert rig.ctl.sent[-1][0] == 'move_j'
    res = arm.move_tcp(T, timeout=0.5)
    assert res.error == 'timeout' and res.duration < 0.55


def test_motion_status_requirement_can_be_relaxed(rig):
    arm = rig.make()
    rig.ctl.behavior = 'never_done'
    assert arm.move_tcp(_td(0.30, 0.05, 0.05), timeout=3.0).error == 'timeout'
    arm2 = rig.make(robot__motion__require_motion_status=False)
    assert arm2.move_tcp(_td(0.30, 0.05, 0.05), timeout=3.0).ok


def test_motion_needs_enabled_motors_and_valid_pose(rig):
    arm = rig.make(enable=False)
    n = len(rig.ctl.sent)
    res = arm.move_tcp(_td(0.30, 0.05, 0.05))
    assert res.error == 'arm_error' and 'enable' in res.detail
    assert arm.enable()
    bad = _td(0.30, 0.05, 0.05)
    bad[1, 3] = float('nan')
    assert arm.move_tcp(bad).error == 'rejected'
    assert 'move_p' not in _names(rig.ctl)[n:]


def test_move_joints(rig):
    arm = rig.make()
    q = KIN.ik(_td(0.30, -0.05, 0.05) @ T_TF)[0]
    res = arm.move_joints(q)
    assert res.ok and rig.ctl.sent[-1][0] == 'move_j'
    assert np.max(np.abs(arm.get_joints().value - q)) < 0.01
    n = len(rig.ctl.sent)
    res = arm.move_joints([0.0, 0.0, 0.0, 0.0, 1.5, 0.0])
    assert res.error == 'unreachable' and 'joint5' in res.detail
    assert arm.move_joints([0.0] * 5).error == 'rejected'
    assert len(rig.ctl.sent) == n


def test_health_reports_controller_state(rig):
    arm = rig.make()
    h = arm.health()
    assert h.ok and h.comm_ok and h.enabled and h.ctrl_mode == 1 and h.arm_status == 0
    assert h.detail['gripper_ok'] is True
    rig.ctl.arm_status = 0x02
    assert arm.health().ok                           # leftover target error
    rig.ctl.arm_status = 0x01
    assert 'EMERGENCY_STOP' in arm.health().reason and not arm.health().ok
    rig.ctl.arm_status = 0
    rig.ctl.ctrl_mode = 2
    h = arm.health()
    assert not h.ok and 'teach' in h.reason
    assert arm.move_tcp(_td(0.30, 0.05, 0.05)).error == 'arm_error'
    rig.ctl.ctrl_mode = 1
    rig.ctl.comm_joints = [3]
    h = arm.health()
    assert not h.ok and not h.comm_ok and '3' in h.reason
    rig.ctl.comm_joints = []
    rig.ctl.g_faults = {'driver_overcurrent'}
    h = arm.health()
    assert h.ok and h.detail['gripper_ok'] is False
    arm.close()
    assert not arm.health().ok and _names(rig.ctl)[-1] == 'disconnect'


# ---------------------------------------------------------------- gripper
def test_gripper_open_and_close_on_object(rig):
    arm = rig.make()
    t0 = rig.clock.now()
    g = arm.gripper(0.05)
    assert g.ok and g.width == pytest.approx(0.05, abs=1e-6)
    assert rig.ctl.sent[-1] == ('move_gripper_m', 0.05, 1.5)          # robot.gripper.force
    assert 0.55 < rig.clock.now() - t0 < 0.9                           # 0.5 s travel + latency
    rig.ctl.g_block = 0.03
    g = arm.gripper(0.0)
    assert g.ok and g.width == pytest.approx(0.03)                     # blocked by the object
    rig.ctl.g_block = None
    arm.gripper(0.05)
    assert arm.gripper(0.0).width == pytest.approx(0.0)                # nothing held
    assert arm.gripper_width().value == pytest.approx(0.0)


def test_gripper_clips_width_and_force(rig):
    arm = rig.make()
    arm.gripper(0.5, force=9.0)
    assert rig.ctl.sent[-1] == ('move_gripper_m', 0.07, 3.0)
    arm.gripper(-0.1, force=0.1)
    assert rig.ctl.sent[-1] == ('move_gripper_m', 0.0, 0.5)
    assert arm.gripper(float('nan')).error == 'fault'


def test_gripper_same_width_returns_after_stable_time(rig):
    arm = rig.make()
    t0 = rig.clock.now()
    assert arm.gripper(0.0).ok
    assert rig.clock.now() - t0 < 0.3


def test_gripper_does_not_finish_before_the_fingers_move(rig):
    arm = rig.make()
    rig.ctl.latency = 0.4                     # longer than the 0.15 s stability window
    g = arm.gripper(0.05)
    assert g.ok and g.width == pytest.approx(0.05, abs=1e-6)


def test_gripper_faults(rig):
    arm = rig.make()
    rig.ctl.g_open_block = 0.02
    g = arm.gripper(0.07)
    assert g.error == 'fault' and 'opening stopped' in g.detail and g.width == pytest.approx(0.02)
    rig.ctl.g_open_block = None
    rig.ctl.g_enabled = False
    assert 'disabled' in arm.gripper(0.0).detail
    rig.ctl.g_enabled = True
    rig.ctl.g_faults = {'motor_overheating'}
    g = arm.gripper(0.03)
    assert g.error == 'fault' and 'motor_overheating' in g.detail
    rig.ctl.g_faults = set()
    rig.ctl.freeze_at = rig.clock.now() + 0.2
    assert arm.gripper(0.05).error == 'comm'


def test_gripper_no_wait_and_no_effector(rig):
    arm = rig.make()
    t0 = rig.clock.now()
    g = arm.gripper(0.05, wait=False)
    assert g.ok and g.width == pytest.approx(0.0) and rig.clock.now() == t0
    arm2 = rig.make(robot__effector='none')
    assert arm2.gripper(0.05).error == 'fault'
    assert arm2.health().detail['gripper_ok'] is None


# ---------------------------------------------------------------- teach / stop
def test_exit_teach_mode_replicates_official_sequence(rig):
    arm = rig.make()
    n = len(rig.ctl.sent)
    assert arm.exit_teach_mode() is False                # not in teach mode: nothing sent
    assert len(rig.ctl.sent) == n
    rig.ctl.ctrl_mode = 2
    rig.ctl.q = np.array([0.03, 0.0, -0.02, 0.0, 0.049, -0.045])   # folded on the support
    assert arm.exit_teach_mode() is True
    seq = rig.ctl.sent[n:]
    names = [s[0] for s in seq]
    assert names[:4] == ['move_js', 'electronic_emergency_stop', 'move_j', 'reset']
    assert names[-2:] == ['set_speed_percent', 'move_j'] and 'enable' in names[4:-2]
    t = {s[0]: s[1] for s in reversed(seq) if len(s) > 1 and isinstance(s[1], float)}
    assert t['electronic_emergency_stop'] - t['move_js'] == pytest.approx(2.0)
    assert t['reset'] - t['electronic_emergency_stop'] == pytest.approx(0.3)


def test_exit_teach_mode_refuses_unless_folded_at_joint_zero(rig):
    arm = rig.make()
    rig.ctl.ctrl_mode = 2
    n = len(rig.ctl.sent)
    with pytest.raises(ArmError, match='not folded at joint zero: joint2'):
        arm.exit_teach_mode()                            # Q_START: raised over the table
    rig.ctl.q = np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ArmError, match=r'joint1 3\.4 deg \(limit 2\.9 deg\)'):
        arm.exit_teach_mode()
    rig.ctl.q = np.zeros(6)
    rig.ctl.freeze_at = rig.clock.now()
    rig.clock.sleep(1.0)                                 # joint feedback now stale
    with pytest.raises(ArmError, match='no fresh joint feedback'):
        arm.exit_teach_mode()
    assert len(rig.ctl.sent) == n                        # nothing sent in any case


def test_no_joint_or_pose_command_while_in_teach_mode(rig):
    arm = rig.make()
    rig.ctl.ctrl_mode = 2
    n = len(rig.ctl.sent)
    assert arm.move_tcp(_td(0.30, 0.05, 0.05)).error == 'arm_error'
    assert arm.move_tcp(_td(0.30, 0.05, 0.05), linear=True).error == 'arm_error'
    assert arm.move_joints(Q_START).error == 'arm_error'
    arm.stop()                                           # nothing to hold in drag-teach
    assert len(rig.ctl.sent) == n
    assert 'check_setup --live' in arm.health().reason
    rig.ctl.ctrl_mode = 1
    arm.stop()
    assert rig.ctl.sent[-1][0] == 'move_j'


def test_take_can_control_option_sends_no_motion(rig):
    from cube_task.tools import onsite
    from cube_task.tools.ui import Ui
    arm = rig.make()
    rig.ctl.ctrl_mode = 2
    rig.ctl.seamless = True
    n = len(rig.ctl.sent)
    answers = iter(['c', 'CAN'])
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=lambda _p: next(answers), output=out)
    assert onsite.ensure_motion_ready(ui, arm) is True
    assert set(_names(rig.ctl)[n:]) == {'enable', 'set_speed_percent'}
    assert arm.health().ok and arm.health().ctrl_mode == 1
    text = out.getvalue()
    assert 'firmware S-V1.8-8 (S-V1.8-5 or later) switches seamlessly' in text
    assert 'CAN control taken' in text


def _wrap_start(da):
    """Joints of a 10 deg outward-tilted pose above (0.28, 0.05) and the vertical target below."""
    a = tf.neutral_closing_angle(0.28, 0.05) + da
    Ts = tilt_outward(tf.top_down_T(0.28, 0.05, 0.10, a), math.radians(10.0))
    return KIN.ik(Ts @ T_TF)[0], tf.top_down_T(0.28, 0.05, 0.06, a)


@pytest.mark.parametrize('da, roll_sent', [(0.3, -math.pi), (-0.3, math.pi)])
def test_linear_move_keeps_roll_on_the_feedback_side_of_the_wrap(rig, da, roll_sent):
    q0, T_goal = _wrap_start(da)
    rig.ctl.q = q0
    arm = rig.make()
    start_roll = arm._read_flange_pose6()[3]
    assert math.copysign(1.0, start_roll) == math.copysign(1.0, roll_sent)
    assert abs(start_roll) > math.radians(176.0)
    assert tf.T_to_pose6(T_goal @ T_TF)[3] == pytest.approx(math.pi)    # vertical: +pi
    res = arm.move_tcp(T_goal, linear=True)
    assert res.ok, res
    sent = rig.ctl.last_pose
    assert sent[3] == roll_sent                          # exactly +-pi, never clamped
    assert abs(sent[3] - start_roll) < math.radians(5.0)  # no ~360 deg RPY jump
    assert tf.pose6_to_T(sent) == pytest.approx(T_goal @ T_TF, abs=1e-9)


class _Records(logging.Handler):
    """Collect log records (caplog fails under the ROS launch_testing plugin)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_lift_across_the_wrap_is_logged(rig):
    """Start roll +179.99 deg (feedback), lift target -177 deg: cannot be fixed, only logged."""
    q_lift, T_low = _wrap_start(0.3)
    rig.ctl.q = KIN.ik(T_low @ T_TF)[0]
    arm = rig.make()
    T_lift = KIN.fk(q_lift) @ T_FT
    rec = _Records()
    mod.LOG.addHandler(rec)
    try:
        assert arm.move_tcp(T_lift, linear=True).ok
        lift_msgs, rec.messages = rec.messages, []
        assert arm.move_tcp(T_low, linear=True).ok           # descent: canonicalized
    finally:
        mod.LOG.removeHandler(rec)
    assert any('crosses the +-180 deg RPY wrap' in m for m in lift_msgs)
    assert not any('RPY wrap' in m for m in rec.messages)


def test_read_teach_pose_stop_and_estop(rig):
    arm = rig.make()
    p = arm.read_teach_pose()
    assert p['joints'] == pytest.approx(list(rig.ctl.q))
    assert np.asarray(p['T_base_tcp']) == pytest.approx(KIN.fk(rig.ctl.q) @ T_FT)
    assert arm.supports_teach()
    arm.stop()
    assert rig.ctl.sent[-1][0] == 'move_j'
    arm.emergency_stop()
    assert rig.ctl.sent[-1][0] == 'electronic_emergency_stop'
    assert 'reset' not in _names(rig.ctl)
    arm.close()
    with pytest.raises(ArmError):
        arm.emergency_stop()


def test_stamped_values_use_host_monotonic_time(rig):
    arm = rig.make()
    st = arm.get_tcp_pose()
    assert st.stamp == pytest.approx(rig.clock.now())
    assert st.value == pytest.approx(KIN.fk(rig.ctl.q) @ T_FT)

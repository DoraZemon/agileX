"""Tests for robot/safety.py: SafetyGuard limits and SafeArm rejection behavior."""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.core.config import load_config
from cube_task.core.interfaces import ArmInterface
from cube_task.core.types import ArmHealth, GripperResult, MotionResult, Stamped
from cube_task.robot.kinematics import flange_to_tcp_from_cfg, PiperKinematics
from cube_task.robot.safety import interpolate_pose, SafeArm, SafetyGuard


def _cfg():
    return load_config(use_overlays=False)


class RecordingArm(ArmInterface):
    """Minimal ArmInterface that records every command it receives."""

    def __init__(self, q, T_ft):
        self.kin = PiperKinematics()
        self.q = np.asarray(q, dtype=float)
        self.T_ft = T_ft
        self.calls = []
        self.health_value = ArmHealth(ok=True)

    def connect(self):
        self.calls.append('connect')

    def enable(self, timeout=5.0):
        self.calls.append('enable')
        return True

    def health(self):
        return self.health_value

    def get_tcp_pose(self):
        return Stamped(self.kin.fk(self.q) @ self.T_ft, 0.0)

    def get_flange_pose(self):
        return Stamped(self.kin.fk(self.q), 0.0)

    def get_joints(self):
        return Stamped(self.q.copy(), 0.0)

    def move_tcp(self, T_base_tcp, linear=False, timeout=None):
        self.calls.append(('move_tcp', linear))
        return MotionResult(ok=True)

    def move_joints(self, q, timeout=None):
        self.calls.append(('move_joints', list(q)))
        return MotionResult(ok=True)

    def gripper(self, width, force=None, wait=True):
        self.calls.append(('gripper', width))
        return GripperResult(ok=True, width=width)

    def gripper_width(self):
        return Stamped(0.0, 0.0)

    def set_speed_percent(self, percent):
        self.calls.append(('speed', percent))

    def stop(self):
        self.calls.append('stop')

    def emergency_stop(self):
        self.calls.append('estop')

    def close(self):
        self.calls.append('close')


# A joint configuration with the TCP straight down at about (0.30, 0, 0.05).
def _low_pose_joints(kin, T_ft):
    T = tf.top_down_T(0.30, 0.0, 0.05, tf.neutral_closing_angle(0.30, 0.0))
    q, _ep, _eo = kin.ik(T @ tf.inv_T(T_ft))
    assert q is not None
    return q


@pytest.fixture()
def setup():
    cfg = _cfg()
    T_ft = flange_to_tcp_from_cfg(cfg)
    kin = PiperKinematics()
    inner = RecordingArm(_low_pose_joints(kin, T_ft), T_ft)
    arm = SafeArm(inner, SafetyGuard(cfg), kin=kin, T_flange_tcp=T_ft)
    return cfg, inner, arm


def _td(x, y, z):
    return tf.top_down_T(x, y, z, tf.neutral_closing_angle(x, y))


# ---------------------------------------------------------------- SafetyGuard
def test_guard_accepts_nominal_pose():
    g = SafetyGuard(_cfg())
    assert g.check_tcp(_td(0.30, 0.0, 0.05)) == (True, '')
    assert g.check_tcp(_td(0.30, 0.0, 0.05), table_z=0.0) == (True, '')


@pytest.mark.parametrize('xyz,word', [
    ((0.07, 0.10, 0.05), 'x='),           # x below 0.08
    ((0.61, 0.0, 0.05), 'x='),
    ((0.30, 0.41, 0.05), 'y='),
    ((0.30, 0.0, -0.06), 'z='),
    ((0.30, 0.0, 0.51), 'z='),
    ((0.09, 0.0, 0.05), 'radius'),        # r 0.09 < r_min 0.10
    ((0.50, 0.38, 0.05), 'radius'),       # r 0.628 > r_max 0.62
])
def test_guard_box_and_radius(xyz, word):
    ok, reason = SafetyGuard(_cfg()).check_tcp(_td(*xyz))
    assert not ok and word in reason


def test_guard_boundaries_are_inclusive():
    g = SafetyGuard(_cfg())
    assert g.check_tcp(_td(0.60, 0.0, 0.05))[0]
    assert g.check_tcp(_td(0.30, 0.40, 0.05))[0]


def test_guard_table_clearance():
    g = SafetyGuard(_cfg())     # min_table_clearance 0.002
    assert g.check_tcp(_td(0.3, 0.0, 0.0125), table_z=0.010)[0]
    ok, reason = g.check_tcp(_td(0.3, 0.0, 0.0115), table_z=0.010)
    assert not ok and 'table' in reason
    assert not g.check_tcp(_td(0.3, 0.0, 0.05), table_z=float('nan'))[0]


def test_guard_tilt():
    g = SafetyGuard(_cfg())     # max_tilt_from_down_deg 30
    T = _td(0.3, 0.0, 0.1)
    from cube_task.robot.kinematics import tilt_outward
    assert g.check_tcp(tilt_outward(T, math.radians(29.9)))[0]
    ok, reason = g.check_tcp(tilt_outward(T, math.radians(30.5)))
    assert not ok and 'tilt' in reason
    assert not g.check_tcp(tf.make_T(np.eye(3), (0.3, 0.0, 0.1)))[0]   # tool pointing up
    assert g.check_tcp(tf.make_T(np.eye(3), (0.3, 0.0, 0.1)), check_tilt=False)[0]


def test_guard_rejects_malformed_poses():
    g = SafetyGuard(_cfg())
    T = _td(0.3, 0.0, 0.1)
    T[0, 3] = float('nan')
    assert g.check_tcp(T) == (False, 'pose contains NaN / inf')
    assert not g.check_tcp(np.eye(3))[0]
    S = _td(0.3, 0.0, 0.1)
    S[:3, :3] *= 1.1
    assert not g.check_tcp(S)[0]
    assert not g.check_tcp('not a pose')[0]


def test_guard_requires_config():
    cfg = _cfg()
    cfg.data['robot']['workspace'].pop('r_max')
    with pytest.raises(ValueError, match='r_max'):
        SafetyGuard(cfg)
    cfg = _cfg()
    cfg.set('robot.workspace.x', [0.6, 0.1])
    with pytest.raises(ValueError, match='workspace.x'):
        SafetyGuard(cfg)


# ---------------------------------------------------------------- SafeArm
def _moves(inner):
    return [c for c in inner.calls if isinstance(c, tuple) and c[0].startswith('move')]


def test_safe_arm_forwards_reachable_target(setup):
    _cfg_, inner, arm = setup
    assert arm.move_tcp(_td(0.30, 0.05, 0.04)).ok
    assert arm.move_tcp(_td(0.30, 0.0, 0.02), linear=True).ok
    assert _moves(inner) == [('move_tcp', False), ('move_tcp', True)]


@pytest.mark.parametrize('T', [
    _td(0.70, 0.0, 0.05),                         # outside the box
    _td(0.30, 0.0, 0.25),                         # vertical and high: joint5 cannot do it
    tf.make_T(np.eye(3), (0.3, 0.0, 0.1)),        # tool pointing up
])
def test_safe_arm_rejects_without_calling_inner(setup, T):
    _cfg_, inner, arm = setup
    res = arm.move_tcp(T)
    assert not res.ok and res.error == 'rejected' and res.detail
    assert _moves(inner) == []


def test_safe_arm_table_clearance(setup):
    _cfg_, inner, arm = setup
    arm.set_table_z(0.03)
    res = arm.move_tcp(_td(0.30, 0.0, 0.031))
    assert res.error == 'rejected' and 'table' in res.detail
    with pytest.raises(ValueError):
        arm.set_table_z(float('inf'))
    arm.set_table_z(None)
    assert arm.move_tcp(_td(0.30, 0.0, 0.031)).ok


def test_safe_arm_linear_path_is_checked(setup):
    _cfg_, inner, arm = setup
    inner.q = PiperKinematics().ik(_td(0.09, 0.12, 0.05) @ tf.inv_T(inner.T_ft))[0]
    assert inner.q is not None
    res = arm.move_tcp(_td(0.09, -0.12, 0.05), linear=True)   # passes r < r_min near the base
    assert res.error == 'rejected' and 'linear path' in res.detail
    assert _moves(inner) == []
    assert arm.move_tcp(_td(0.09, -0.12, 0.05), linear=False).ok


def test_safe_arm_refuses_when_aborted_or_unhealthy(setup):
    _cfg_, inner, arm = setup
    ev = threading.Event()
    arm.set_abort_event(ev)
    assert inner.abort_event is ev           # propagated to the wrapped arm
    ev.set()
    assert arm.move_tcp(_td(0.30, 0.0, 0.04)).error == 'aborted'
    assert arm.move_joints([0.0] * 6).error == 'aborted'
    assert arm.gripper(0.05).error == 'aborted'
    ev.clear()
    inner.health_value = ArmHealth(ok=False, reason='CAN down', comm_ok=False)
    res = arm.move_tcp(_td(0.30, 0.0, 0.04))
    assert res.error == 'comm' and 'CAN down' in res.detail
    inner.health_value = ArmHealth(ok=False, reason='arm_status COLLISION_OCCURRED')
    assert arm.move_joints([0.0] * 6).error == 'arm_error'
    assert inner.calls == []


def test_safe_arm_move_joints_checks(setup):
    _cfg_, inner, arm = setup
    assert arm.move_joints([0.0, 0.0, 0.0, 0.0, 1.3, 0.0]).error == 'rejected'
    assert arm.move_joints([0.0] * 5).error == 'rejected'
    assert arm.move_joints([0.0, float('nan'), 0, 0, 0, 0]).error == 'rejected'
    arm.set_table_z(0.10)
    res = arm.move_joints(inner.q)            # current pose has the TCP at z 0.05 < 0.102
    assert res.error == 'rejected' and 'below' in res.detail
    arm.set_table_z(0.0)
    assert arm.move_joints([0.0] * 6).ok
    assert _moves(inner) == [('move_joints', [0.0] * 6)]


def test_safe_arm_gripper_validation(setup):
    _cfg_, inner, arm = setup
    assert arm.gripper(float('nan')).error == 'fault'
    assert arm.gripper(-0.01).error == 'fault'
    assert arm.gripper(0.03, force=float('nan')).error == 'fault'
    assert arm.gripper(0.03).ok
    assert inner.calls == [('gripper', 0.03)]


def test_check_reachable_and_find_tilt(setup):
    _cfg_, inner, arm = setup
    assert arm.check_reachable(_td(0.30, 0.0, 0.04)) == (True, '')
    ok, reason = arm.check_reachable(_td(0.30, 0.0, 0.20))
    assert not ok and 'IK' in reason
    T, tilt, why = arm.find_reachable_tilt(_td(0.30, 0.0, 0.15))
    assert T is not None and 0 < tilt <= math.radians(30) and why == ''
    assert arm.check_reachable(T)[0]
    assert arm.move_tcp(T).ok
    T, tilt, why = arm.find_reachable_tilt(_td(0.30, 0.0, 0.35))   # needs > 30 deg
    assert T is None and why


def test_safe_arm_delegation(setup):
    _cfg_, inner, arm = setup
    assert arm.inner is inner
    arm.connect()
    assert arm.enable()
    arm.set_speed_percent(40)
    arm.stop()
    arm.emergency_stop()
    arm.close()
    assert inner.calls == ['connect', 'enable', ('speed', 40), 'stop', 'estop', 'close']
    assert arm.get_joints().value == pytest.approx(inner.q)
    assert not arm.supports_teach()
    with pytest.raises(NotImplementedError):
        arm.exit_teach_mode()
    with pytest.raises(ValueError):
        SafeArm(inner, arm.guard, kin=PiperKinematics())


def test_interpolate_pose_endpoints():
    A = _td(0.2, 0.0, 0.1)
    B = tf.make_T(tf.rot_z(0.5) @ A[:3, :3], (0.3, 0.1, 0.05))
    assert interpolate_pose(A, B, 0.0) == pytest.approx(A)
    assert interpolate_pose(A, B, 1.0) == pytest.approx(B)
    M = interpolate_pose(A, B, 0.5)
    assert tf.rotation_angle(A[:3, :3].T @ M[:3, :3]) == pytest.approx(0.25)

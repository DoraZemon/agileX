"""Tests for robot/sim_arm.py: timing model, grasp interaction and fault injection."""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.robot.kinematics import PiperKinematics
from cube_task.robot.sim_arm import SimArm

T_FT = tf.pose6_to_T([0.0, 0.0, 0.14, 0.0, 0.0, 0.0])


class FakeWorld:
    """One 30 mm block at (0.30, 0.0) on a table at z = 0; zone 'bin' covers x > 0.40."""

    def __init__(self):
        self.held = None
        self.block_xyz = np.array([0.30, 0.0, 0.0])
        self.width = 0.03
        self.height = 0.03
        self.updates = []
        self.released = []

    def try_grasp(self, T_base_tcp, open_width, close_width):
        p = T_base_tcp[:3, 3]
        inside = (np.linalg.norm(p[:2] - self.block_xyz[:2]) < 0.008
                  and self.block_xyz[2] <= p[2] <= self.block_xyz[2] + self.height)
        if inside and open_width > self.width and self.held is None:
            self.held = 'block'
            self._offset = p - self.block_xyz
            return self.width
        return None

    def update_tcp(self, T_base_tcp, width):
        self.updates.append((T_base_tcp[:3, 3].copy(), width))
        if self.held is not None:
            self.block_xyz = T_base_tcp[:3, 3] - self._offset

    def release(self, T_base_tcp):
        self.held = None
        self.block_xyz = np.array([T_base_tcp[0, 3], T_base_tcp[1, 3], 0.0])
        zone = 'bin' if self.block_xyz[0] > 0.40 else None
        self.released.append(zone)
        return zone


def _td(x, y, z):
    return tf.top_down_T(x, y, z, tf.neutral_closing_angle(x, y))


def _make(world=None, connect=True, **overrides):
    cfg = load_config(use_overlays=False)
    cfg.set('robot.backend', 'sim')
    for k, v in overrides.items():
        cfg.set(k.replace('__', '.'), v)
    clock = SimClock(start=100.0)
    arm = SimArm(cfg, clock, world=world)
    if connect:
        arm.connect()
        assert arm.enable()
    return cfg, clock, arm


def test_p2p_timing_and_exact_final_pose():
    cfg, clock, arm = _make()          # speed_percent 30, joint_speed 2.0, overhead 0.35
    T = _td(0.30, 0.05, 0.05)
    kin = PiperKinematics()
    q_goal, _ep, _eo = kin.ik(T @ tf.inv_T(T_FT), q_seed=np.zeros(6))
    q_goal, _ep, _eo = kin.refine(T @ tf.inv_T(T_FT), q_goal)
    expected = 0.35 + float(np.max(np.abs(q_goal))) / (2.0 * 0.30)
    t0 = clock.now()
    res = arm.move_tcp(T)
    assert res.ok and res.error == ''
    assert res.duration == pytest.approx(expected, abs=1e-9)
    assert clock.now() - t0 == pytest.approx(expected, abs=1e-9)
    assert res.pos_err < 1e-6 and res.rot_err_deg < 1e-4
    assert arm.get_tcp_pose().value == pytest.approx(res.final_tcp)
    assert arm.get_joints().value == pytest.approx(q_goal)


def test_linear_timing_scales_with_speed():
    cfg, clock, arm = _make()
    assert arm.move_tcp(_td(0.30, 0.0, 0.05)).ok
    res = arm.move_tcp(_td(0.30, 0.0, 0.02), linear=True)
    assert res.ok
    assert res.duration == pytest.approx(0.35 + 0.03 / (0.15 * 0.30), abs=1e-6)
    arm.set_speed_percent(60)
    res = arm.move_tcp(_td(0.30, 0.0, 0.05), linear=True)
    assert res.duration == pytest.approx(0.35 + 0.03 / (0.15 * 0.60), abs=1e-6)
    with pytest.raises(ValueError):
        arm.set_speed_percent(0)
    with pytest.raises(ValueError):
        arm.set_speed_percent(50.5)


def test_requires_connect_and_enable():
    cfg, clock, arm = _make(connect=False)
    assert arm.move_tcp(_td(0.3, 0.0, 0.05)).error == 'comm'
    assert arm.get_joints() is None and not arm.health().ok
    arm.connect()
    res = arm.move_joints([0.0] * 6)
    assert res.error == 'arm_error' and 'enable' in res.detail
    assert arm.enable()
    assert arm.move_joints([0.0] * 6).ok


def test_unreachable_and_joint_limits():
    cfg, clock, arm = _make()
    q0 = arm.get_joints().value
    res = arm.move_tcp(_td(0.30, 0.0, 0.30))      # vertical at z 0.30: joint5 limit
    assert res.error == 'unreachable'
    assert arm.get_joints().value == pytest.approx(q0)
    res = arm.move_joints([0.0, 0.0, 0.0, 0.0, 1.4, 0.0])
    assert res.error == 'unreachable' and 'joint5' in res.detail
    assert arm.move_tcp(np.full((4, 4), np.nan)).error == 'rejected'


def test_pick_and_place_moves_the_held_object():
    world = FakeWorld()
    cfg, clock, arm = _make(world=world)
    assert arm.gripper(0.05).width == pytest.approx(0.05)
    assert arm.move_tcp(_td(0.30, 0.0, 0.08)).ok
    assert arm.move_tcp(_td(0.30, 0.0, 0.015), linear=True).ok
    t0 = clock.now()
    g = arm.gripper(0.0)
    assert g.ok and g.width == pytest.approx(0.03) and world.held == 'block'
    assert clock.now() - t0 == pytest.approx(0.6)          # robot.gripper.settle_s
    n0 = len(world.updates)
    assert arm.move_tcp(_td(0.30, 0.0, 0.10), linear=True).ok
    assert len(world.updates) - n0 >= 10                   # 0.1 s steps over ~2 s
    assert world.block_xyz[2] == pytest.approx(0.085, abs=1e-3)
    assert arm.move_tcp(_td(0.41, 0.05, 0.05)).ok
    assert arm.gripper_width().value == pytest.approx(0.03)
    assert arm.gripper(0.07).ok
    assert world.held is None and world.released == ['bin']


def test_close_on_nothing_reads_commanded_width():
    world = FakeWorld()
    cfg, clock, arm = _make(world=world)
    arm.gripper(0.07)
    assert arm.move_tcp(_td(0.25, 0.10, 0.02)).ok
    assert arm.gripper(0.0).width == 0.0 and world.held is None


def test_width_is_clipped_and_validated():
    cfg, clock, arm = _make()
    assert arm.gripper(0.5).width == pytest.approx(0.07)
    assert arm.gripper(-1.0).width == 0.0
    assert arm.gripper(float('nan')).error == 'fault'


def test_fault_timeout_stops_part_way():
    cfg, clock, arm = _make()
    arm.inject('timeout')
    T = _td(0.30, 0.0, 0.05)
    t0 = clock.now()
    res = arm.move_tcp(T)
    T_home = PiperKinematics().fk(np.zeros(6)) @ T_FT
    dist = float(np.linalg.norm(T[:3, 3] - T_home[:3, 3]))
    assert res.error == 'timeout' and not res.ok
    # timeout = timeout_base_s + timeout_per_m_s * distance; the arm stops half way
    assert clock.now() - t0 == pytest.approx(6.0 + 15.0 * dist, abs=1e-6)
    assert res.pos_err > 0.02
    assert arm.move_tcp(T).ok                               # fault used up


def test_fault_unreachable_and_arm_error_with_at_call_and_count():
    cfg, clock, arm = _make()
    arm.inject('unreachable', at_call=3, count=2)
    arm.inject('arm_error', at_call=5)
    T1, T2 = _td(0.30, 0.0, 0.05), _td(0.25, 0.05, 0.05)
    results = [arm.move_tcp(T1 if i % 2 else T2).error for i in range(6)]
    assert results == ['', '', 'unreachable', 'unreachable', 'arm_error', '']
    assert [f['kind'] for f in arm.fault_log] == ['unreachable', 'unreachable', 'arm_error']
    with pytest.raises(ValueError):
        arm.inject('meteor')


def test_fault_comm_loss_is_permanent():
    cfg, clock, arm = _make()
    arm.inject('comm_loss')
    assert arm.move_joints([0.0] * 6).error == 'comm'
    h = arm.health()
    assert not h.ok and not h.comm_ok
    assert arm.gripper(0.05).error == 'comm'
    assert arm.get_joints() is None and arm.get_tcp_pose() is None
    assert not arm.enable()


def test_fault_grasp_miss():
    world = FakeWorld()
    cfg, clock, arm = _make(world=world)
    arm.gripper(0.05)
    assert arm.move_tcp(_td(0.30, 0.0, 0.015)).ok
    arm.inject('grasp_miss')
    g = arm.gripper(0.0)
    assert g.ok and g.width == 0.0 and world.held is None
    arm.gripper(0.05)
    assert arm.gripper(0.0).width == pytest.approx(0.03)    # next try works


def test_fault_drop_during_next_motion():
    world = FakeWorld()
    cfg, clock, arm = _make(world=world)
    arm.gripper(0.05)
    assert arm.move_tcp(_td(0.30, 0.0, 0.015)).ok
    arm.inject('drop')
    assert arm.gripper(0.0).width == pytest.approx(0.03)    # gripper call does not drop
    assert arm.move_tcp(_td(0.30, 0.0, 0.10), linear=True).ok
    assert world.held is None and world.released == [None]
    assert world.block_xyz[2] == 0.0
    assert arm.gripper_width().value == pytest.approx(0.0)


def test_abort_during_motion_stops_part_way():
    cfg, clock, arm = _make()
    ev = threading.Event()
    arm.set_abort_event(ev)
    t_abort = clock.now() + 0.8
    clock.add_listener(lambda t_old, t_new: ev.set() if t_new >= t_abort else None)
    T = _td(0.30, 0.0, 0.05)
    res = arm.move_tcp(T)
    assert res.error == 'aborted'
    assert 0.8 <= res.duration < 0.95 and res.pos_err > 0.01
    assert arm.move_tcp(T).error == 'aborted'


def test_emergency_stop_latches():
    cfg, clock, arm = _make()
    arm.emergency_stop()
    assert not arm.health().ok and arm.health().comm_ok
    assert arm.move_joints([0.0] * 6).error == 'arm_error'


def test_world_none_and_initial_state():
    cfg, clock, arm = _make()
    assert arm.gripper_width().value == 0.0                  # starts closed
    assert arm.get_joints().value == pytest.approx(np.zeros(6))
    assert arm.health().ok and arm.health().enabled
    assert not arm.supports_teach()
    arm.inject('drop')
    assert arm.move_tcp(_td(0.30, 0.0, 0.05)).ok             # nothing to drop
    assert math.isfinite(arm.get_tcp_pose().stamp)


def test_optional_joint_noise_is_deterministic():
    _c1, _k1, a1 = _make(sim__arm_joint_noise_std=0.001)
    _c2, _k2, a2 = _make(sim__arm_joint_noise_std=0.001)
    r1, r2 = a1.move_tcp(_td(0.3, 0.0, 0.05)), a2.move_tcp(_td(0.3, 0.0, 0.05))
    assert r1.ok and r1.pos_err > 1e-5
    assert r1.final_tcp == pytest.approx(r2.final_tcp)


def test_explicit_timeout_shorter_than_motion():
    cfg, clock, arm = _make()
    res = arm.move_tcp(_td(0.30, 0.0, 0.05), timeout=0.5)
    assert res.error == 'timeout' and res.duration == pytest.approx(0.5)

"""Tests for sim/mujoco/ik.py and sim/mujoco/arm.py on a minimal PiPER + table + box scene."""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.robot.kinematics import PiperKinematics, tilt_outward
from cube_task.sim.mujoco.arm import _Profile, MujocoArm
from cube_task.sim.mujoco.engine import MujocoSim
from cube_task.sim.mujoco.ik import MujocoIK

try:
    import mujoco
except ImportError:  # system Python: the tests below are skipped
    mujoco = None
# No import-time pytest.importorskip / pytest.skip: under the ROS launch_testing pytest
# plugin that aborts the collection of every test (see test_mujoco_scene.py).

PIPER_XML = os.path.expanduser('~/.cache/cube_task/mujoco/agilex_piper/piper.xml')
TCP_Z = 0.135                 # fingertip end in the link6 frame
BOX = 0.035                   # edge of the free box (m)
BOX_XY = (0.30, 0.0)
DOWN = math.pi / 2            # closing angle: tool y along base +y
META = {'arm_joints': ['joint%d' % i for i in range(1, 7)],
        'arm_actuators': ['joint%d' % i for i in range(1, 7)],
        'gripper_actuator': 'gripper', 'finger_joint': 'joint7', 'flange_body': 'link6',
        'tcp_site': 'tcp', 'max_width': 0.07, 'base_body': 'base_link'}

pytestmark = [
    pytest.mark.skipif(mujoco is None, reason='MuJoCo not installed (use ~/.venvs/piper-mujoco)'),
    pytest.mark.skipif(not os.path.exists(PIPER_XML), reason='PiPER MJCF not found at %s'
                       % PIPER_XML),
]

_MODELS = {}
BASE_MOVED = ((0.05, -0.1, 0.3), 0.5)     # base_link position (m) and yaw (rad) in the world


def _model(base=None):
    """Compile once: PiPER, a table with its top at world z = 0, a free 35 mm box, a TCP site."""
    if base not in _MODELS:
        mj = mujoco
        spec = mj.MjSpec.from_file(PIPER_XML)
        for k in list(spec.keys):
            k.delete()
        if base is not None:
            spec.body('base_link').pos = list(base[0])
            spec.body('base_link').quat = [math.cos(base[1] / 2), 0.0, 0.0, math.sin(base[1] / 2)]
        wb = spec.worldbody
        table = wb.add_body(name='table', pos=[0.42, 0.0, -0.02])
        table.add_geom(name='table_top', type=mj.mjtGeom.mjGEOM_BOX, size=[0.32, 0.45, 0.02])
        spec.add_exclude(bodyname1='base_link', bodyname2='table')
        box = wb.add_body(name='box', pos=[BOX_XY[0], BOX_XY[1], BOX / 2])
        box.add_freejoint(name='box_free')
        box.add_geom(name='box_geom', type=mj.mjtGeom.mjGEOM_BOX, size=[BOX / 2] * 3, mass=0.05)
        spec.body('link6').add_site(name='tcp', pos=[0.0, 0.0, TCP_Z])
        _MODELS[base] = spec.compile()
    return copy.deepcopy(_MODELS[base])


class Rig:
    """Fresh simulation: clock, sim, connected and enabled arm above the box."""

    def __init__(self, speed: int = 30, enable: bool = True, base=None):
        self.m = _model(base)
        self.d = mujoco.MjData(self.m)
        self.clock = SimClock()
        self.sim = MujocoSim(self.m, self.d, self.clock, META)
        self.cfg = load_config(use_overlays=False)
        self.cfg.set('robot.tcp_offset', [0.0, 0.0, TCP_Z, 0.0, 0.0, 0.0])
        self.cfg.set('robot.speed_percent', speed)
        self.arm = MujocoArm(self.cfg, self.clock, self.sim)
        self.arm.set_joints(self.arm.ik.solve(pose(BOX_XY[0], BOX_XY[1], 0.10))[0])
        self.box = self.sim.body_id('box')
        if enable:
            self.arm.connect()
            assert self.arm.enable()
        self.sim.settle(0.3)

    def tcp_site(self) -> np.ndarray:
        return self.sim.site_pose('tcp')     # base_link is at the world origin

    def box_pos(self) -> np.ndarray:
        return np.array(self.d.xpos[self.box], dtype=float)

    def at(self, t: float, fn) -> None:
        """Call fn() once when the virtual time passes t (seconds from now)."""
        t_fire = self.clock.now() + t
        fired = []

        def listener(_t0, t1):
            if not fired and t1 >= t_fire:
                fired.append(True)
                fn()
        self.clock.add_listener(listener)

    def grasp(self, lift: bool = True):
        a = self.arm
        assert a.gripper(0.06).ok
        r = a.move_tcp(pose(BOX_XY[0], BOX_XY[1], BOX / 2), linear=True)
        assert r.ok, r
        g = a.gripper(0.0)
        if lift:
            assert a.move_tcp(lift_pose(self), linear=True).ok
        return g


def pose(x: float, y: float, z: float, angle: float = DOWN) -> np.ndarray:
    return tf.top_down_T(x, y, z, angle)


def lift_pose(rig: Rig) -> np.ndarray:
    """TCP 10 cm above the grasp, leaned outward as little as the arm needs."""
    sol = rig.arm.ik.solve_with_tilt(pose(BOX_XY[0], BOX_XY[1], BOX / 2 + 0.10))
    assert sol is not None
    return sol[0]


@pytest.fixture
def rig():
    return Rig()


# ----------------------------------------------------------------------------- IK
def test_ik_grid_accuracy_and_speed():
    m = _model()
    ik = MujocoIK(m, site='tcp')
    times, worst_p, worst_r, n = [], 0.0, 0.0, 0
    for x in np.linspace(0.15, 0.45, 4):
        for y in np.linspace(-0.25, 0.25, 5):
            for z in np.linspace(0.01, 0.15, 4):
                T = pose(x, y, z, tf.neutral_closing_angle(x, y))
                sol = ik.solve_with_tilt(T)
                assert sol is not None, (x, y, z)
                Tc, q, tilt = sol
                assert tilt <= math.radians(30.0) + 1e-9
                assert ik.within_limits(q)
                t0 = time.perf_counter()
                q2, ep, eo = ik.solve(Tc)
                times.append(time.perf_counter() - t0)
                assert q2 is not None
                # independent check on a fresh MjData: the site really is at the target
                d = mujoco.MjData(m)
                d.qpos[:6] = q2
                mujoco.mj_kinematics(m, d)
                dp, dr = tf.pose_error(_site_T(m, d, 'tcp'), Tc)
                worst_p, worst_r = max(worst_p, dp), max(worst_r, math.degrees(dr))
                n += 1
    assert n == 80
    assert worst_p < 1e-3 and worst_r < 0.5
    assert float(np.mean(times)) < 0.01        # typically ~0.6 ms


def _site_T(m, d, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    return tf.make_T(np.array(d.site_xmat[i]).reshape(3, 3), d.site_xpos[i])


def test_ik_site_equals_body_offset_and_jacobian():
    m = _model()
    a = MujocoIK(m, site='tcp')
    b = MujocoIK(m, body='link6', T_body_tool=tf.make_T(np.eye(3), [0, 0, TCP_Z]))
    q = np.array([0.3, 1.2, -1.0, 0.2, 0.8, 0.5])
    assert np.allclose(a.fk(q), b.fk(q), atol=1e-12)
    assert np.allclose(a.fk_body(q)[:3, 3] + a.fk_body(q)[:3, 2] * TCP_Z, a.fk(q)[:3, 3])
    J = a.jacobian(q)
    eps = 1e-6
    T0 = a.fk(q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = a.fk(q + dq)
        assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-5)
    # MuJoCo and SDK kinematics agree on the flange rotation
    kin = PiperKinematics('piper')
    assert tf.rotation_angle(kin.fk(q)[:3, :3].T @ a.fk_body(q)[:3, :3]) < 1e-3


def test_ik_unreachable_and_limits():
    ik = MujocoIK(_model(), site='tcp')
    q, ep, _eo = ik.solve(pose(0.90, 0.0, 0.05))
    assert q is None and ep > 0.1
    lim = ik.limits
    assert ik.within_limits(lim.mean(axis=1))
    assert not ik.within_limits(lim[:, 1] + 0.01)
    assert ik.limit_violations(lim[:, 1] + 0.01)[0].startswith('joint1=')
    with pytest.raises(ValueError):
        ik.solve(np.full((4, 4), np.nan))


def test_ik_path_continuation():
    ik = MujocoIK(_model(), site='tcp')
    q0 = ik.solve(pose(0.25, -0.1, 0.06))[0]
    poses = [tf.make_T(pose(0.25, -0.1, 0.06)[:3, :3], [0.25, -0.1 + 0.2 * k / 50, 0.06])
             for k in range(51)]
    qs, why = ik.solve_path(poses, q0)
    assert qs is not None, why
    assert float(np.max(np.abs(np.diff(qs, axis=0)))) < 0.05
    bad, why = ik.solve_path([pose(0.25, 0.0, 0.06), pose(0.9, 0.0, 0.06)], q0)
    assert bad is None and 'no IK' in why


def test_profile():
    p = _Profile(1.0 / 2.0, 0.25)       # 2 s at full speed
    assert p.T == pytest.approx(2.25)
    ts = np.linspace(0.0, p.T, 400)
    s = np.array([p(t)[0] for t in ts])
    sd = np.array([p(t)[1] for t in ts])
    assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) >= -1e-12)
    assert sd.max() <= 0.5 + 1e-9
    short = _Profile(100.0, 0.25)       # cannot reach its speed: same peak accel, lower speed
    assert short.T < 0.25 and short(short.T / 2)[0] == pytest.approx(0.5)
    assert _Profile(math.inf, 0.25).T == 0.0


# ----------------------------------------------------------------------------- motion
def test_move_tcp_p2p_reaches_target_with_plausible_duration(rig):
    a = rig.arm
    q0 = a.get_joints().value
    T = pose(0.25, 0.15, 0.08, 0.3)
    r = a.move_tcp(T)
    assert r.ok, r
    q1 = a.get_joints().value
    move = float(np.max(np.abs(q1 - q0))) / (2.0 * 0.30)
    assert move + 0.25 <= r.duration <= move + 0.25 + 0.4
    assert r.pos_err < 0.001 and r.rot_err_deg < 0.5
    dp, dr = tf.pose_error(rig.tcp_site(), T)       # the MuJoCo site itself
    assert dp < 0.001 and math.degrees(dr) < 0.5
    assert np.allclose(a.get_tcp_pose().value, rig.tcp_site(), atol=1e-9)
    assert a.get_tcp_pose().stamp == rig.clock.now()
    # no ringing after "done": the TCP stays put
    track = []
    rig.sim.add_controller(lambda sim, _t: track.append(sim.site_pose('tcp')[:3, 3].copy()))
    rig.sim.settle(1.0)
    spread = np.ptp(np.array(track), axis=0)
    assert float(np.max(spread)) < 2e-4


def test_poses_are_in_the_base_frame():
    r = Rig(base=BASE_MOVED)
    a = r.arm
    T_wb = r.sim.body_pose('base_link')
    assert np.allclose(T_wb[:3, 3], BASE_MOVED[0])
    T = pose(0.25, 0.1, 0.06, 0.4)
    res = a.move_tcp(T, linear=True)
    assert res.ok, res
    dp, dr = tf.pose_error(tf.inv_T(T_wb) @ r.tcp_site(), T)
    assert dp < 0.001 and math.degrees(dr) < 0.5
    assert np.allclose(a.get_tcp_pose().value, tf.inv_T(T_wb) @ r.tcp_site(), atol=1e-9)
    Tf = a.get_flange_pose().value
    assert np.allclose(Tf @ tf.make_T(np.eye(3), [0, 0, TCP_Z]), a.get_tcp_pose().value)


def test_move_tcp_linear_straight_line_and_duration(rig):
    a = rig.arm
    start = a.get_tcp_pose().value[:3, 3].copy()
    T = pose(0.22, -0.12, 0.05)
    seg = T[:3, 3] - start
    L = float(np.linalg.norm(seg))
    dev = []

    def watch(sim, _t):
        p = sim.site_pose('tcp')[:3, 3] - start
        dev.append(float(np.linalg.norm(p - np.dot(p, seg) / L ** 2 * seg)))
    rig.sim.add_controller(watch)
    r = a.move_tcp(T, linear=True)
    assert r.ok, r
    move = L / (0.15 * 0.30)
    assert move + 0.25 <= r.duration <= move + 0.25 + 0.4
    assert max(dev) < 0.003
    dp, dr = tf.pose_error(rig.tcp_site(), T)
    assert dp < 0.001 and math.degrees(dr) < 0.5


def test_move_joints_and_rotation_only_linear(rig):
    a = rig.arm
    q = a.get_joints().value + np.array([0.2, 0.0, 0.0, 0.0, 0.0, -0.3])
    r = a.move_joints(q)
    assert r.ok, r
    assert np.max(np.abs(a.get_joints().value - q)) < 0.01
    T = a.get_tcp_pose().value
    T2 = tf.make_T(T[:3, :3] @ tf.rpy_to_rot(0.0, 0.0, 0.4), T[:3, 3])   # twist about tool z
    r = a.move_tcp(T2, linear=True)
    assert r.ok and r.duration > 0.4 / (2.0 * 0.30), r
    assert r.pos_err < 0.001


def test_unreachable_and_invalid_targets(rig):
    a = rig.arm
    q0 = a.get_joints().value
    r = a.move_tcp(pose(0.90, 0.0, 0.05))
    assert not r.ok and r.error == 'unreachable'
    r = a.move_tcp(pose(0.90, 0.0, 0.05), linear=True)
    assert not r.ok and r.error == 'unreachable'
    assert a.move_tcp(np.eye(3)).error == 'rejected'
    assert a.move_joints([0.0] * 5).error == 'rejected'
    r = a.move_joints([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    assert r.error == 'unreachable' and 'joint5' in r.detail
    assert np.max(np.abs(a.get_joints().value - q0)) < 1e-3


def test_blocked_by_table_reports_stalled(rig):
    a = rig.arm
    assert a.gripper(0.07).ok
    assert a.move_tcp(pose(0.22, 0.12, 0.05)).ok
    r = a.move_tcp(pose(0.22, 0.12, -0.03), linear=True)
    assert not r.ok and r.error == 'stalled', r
    assert r.final_tcp[2, 3] > -0.006       # stopped at the table, fingertips barely press
    rig.sim.settle(0.5)
    assert np.max(np.abs(a._qd_meas())) < 0.01
    assert a.move_tcp(pose(0.22, 0.12, 0.05), linear=True).ok


def test_requires_connect_and_enable():
    r = Rig(enable=False)
    a = r.arm
    assert a.move_tcp(pose(0.3, 0.0, 0.08)).error == 'comm'
    assert a.get_joints() is None and not a.health().ok
    assert a.gripper(0.05).error == 'comm'
    a.connect()
    assert a.move_tcp(pose(0.3, 0.0, 0.08)).error == 'arm_error'
    assert a.enable() and a.health().ok
    assert a.move_tcp(pose(0.3, 0.0, 0.08)).ok
    a.close()
    assert a.get_tcp_pose() is None


# ----------------------------------------------------------------------------- gripper
def test_gripper_open_and_close_on_free_box(rig):
    a = rig.arm
    g = a.gripper(0.06)
    assert g.ok and abs(g.width - 0.06) < 0.001
    assert a.move_tcp(pose(BOX_XY[0], BOX_XY[1], BOX / 2), linear=True).ok
    t0 = rig.clock.now()
    g = a.gripper(0.0)
    assert g.ok and abs(g.width - BOX) < 0.002, g
    assert rig.clock.now() - t0 < 1.0
    assert abs(a.gripper_width().value - g.width) < 1e-4
    # each finger presses with about the commanded grip force (1.5 N)
    f6 = np.zeros(6)
    per_finger = {}
    for i in range(rig.d.ncon):
        c = rig.d.contact[i]
        bodies = {int(rig.m.geom_bodyid[c.geom1]), int(rig.m.geom_bodyid[c.geom2])}
        if rig.box in bodies:
            other = (bodies - {rig.box}).pop()
            mujoco.mj_contactForce(rig.m, rig.d, i, f6)
            per_finger[other] = per_finger.get(other, 0.0) + f6[0]
    fingers = [per_finger.get(rig.sim.body_id(n), 0.0) for n in ('link7', 'link8')]
    assert sum(fingers) == pytest.approx(3.0, rel=0.1)
    assert abs(rig.box_pos()[0] - BOX_XY[0]) < 0.003   # not pushed away
    g = a.gripper(0.0, wait=False)
    assert g.ok
    assert a.gripper(float('nan')).error == 'fault'


def test_pick_lift_hold_while_moving_and_release(rig):
    a = rig.arm
    z0 = rig.box_pos()[2]
    g = rig.grasp()
    assert g.ok and abs(g.width - BOX) < 0.002
    assert rig.box_pos()[2] - z0 > 0.095
    t0 = rig.clock.now()
    for T in (pose(0.25, 0.15, 0.09, 0.3), pose(0.35, -0.12, 0.07, 2.0),
              pose(0.22, 0.05, 0.08)):
        r = a.move_tcp(T)
        assert r.ok, r
        tcp = rig.tcp_site()[:3, 3]
        assert np.linalg.norm(rig.box_pos() - tcp) < 0.01   # still between the fingers
    assert rig.clock.now() - t0 > 2.0
    assert rig.box_pos()[2] > 0.05
    assert abs(a.gripper_width().value - BOX) < 0.003
    xy = rig.tcp_site()[:2, 3].copy()
    g = a.gripper(0.07)
    assert g.ok and g.width > 0.065
    rig.sim.settle(1.0)
    p = rig.box_pos()
    assert p[2] < 0.03 and np.linalg.norm(p[:2] - xy) < 0.03


# ----------------------------------------------------------------------------- faults
def test_fault_timeout_stops_half_way(rig):
    a = rig.arm
    p0 = a.get_tcp_pose().value[:3, 3].copy()
    T = pose(0.25, 0.18, 0.06, 0.6)
    a.inject('timeout')
    r = a.move_tcp(T, timeout=3.0)
    assert not r.ok and r.error == 'timeout'
    assert r.duration == pytest.approx(3.0, abs=0.05)
    p = a.get_tcp_pose().value[:3, 3]
    assert np.linalg.norm(p - p0) > 0.02 and np.linalg.norm(p - T[:3, 3]) > 0.02
    assert a.move_tcp(T).ok


def test_fault_unreachable_arm_error_and_at_call(rig):
    a = rig.arm
    a.inject('unreachable')
    r = a.move_tcp(pose(0.25, 0.1, 0.06))
    assert r.error == 'unreachable' and r.duration <= 0.3
    a.inject('arm_error', at_call=a.n_calls + 2)
    assert a.move_tcp(pose(0.25, 0.1, 0.06)).ok
    assert a.move_joints(a.get_joints().value).error == 'arm_error'
    assert a.move_tcp(pose(0.30, 0.0, 0.08)).ok
    assert [f['kind'] for f in a.fault_log] == ['unreachable', 'arm_error']
    with pytest.raises(ValueError):
        a.inject('bogus')


def test_fault_comm_loss(rig):
    a = rig.arm
    a.inject('comm_loss')
    assert a.move_tcp(pose(0.25, 0.1, 0.06)).error == 'comm'
    assert not a.health().ok and not a.health().comm_ok
    assert a.get_joints() is None and a.gripper_width() is None
    assert a.gripper(0.05).error == 'comm'


def test_fault_grasp_miss_then_regrasp(rig):
    a = rig.arm
    assert a.gripper(0.06).ok
    assert a.move_tcp(pose(BOX_XY[0], BOX_XY[1], BOX / 2), linear=True).ok
    contype = rig.m.geom_contype.copy()
    a.inject('grasp_miss')
    g = a.gripper(0.0)
    assert g.ok and g.width < 0.004            # nothing held (skills: < empty_width)
    assert abs(rig.box_pos()[2] - BOX / 2) < 0.002
    assert not np.array_equal(rig.m.geom_contype, contype)
    pads = [i for i in range(rig.m.ngeom) if rig.m.geom_bodyid[i] in
            (rig.sim.body_id('link7'), rig.sim.body_id('link8')) and contype[i]]
    assert len(pads) == 4 and not rig.m.geom_contype[pads].any()
    assert a.gripper(0.06).ok                  # reopened: finger contacts are back
    assert np.array_equal(rig.m.geom_contype, contype)
    g = a.gripper(0.0)
    assert g.ok and abs(g.width - BOX) < 0.002
    assert a.move_tcp(lift_pose(rig), linear=True).ok
    assert rig.box_pos()[2] > 0.1


def test_fault_drop_during_motion(rig):
    a = rig.arm
    rig.grasp()
    assert rig.box_pos()[2] > 0.1
    a.inject('drop')
    r = a.move_tcp(pose(0.25, 0.12, 0.09, 0.3))
    assert r.ok, r
    assert a.gripper_width().value < 0.004     # fingers closed on nothing
    rig.sim.settle(0.5)
    assert rig.box_pos()[2] < 0.03             # the box fell to the table
    assert a.fault_log[-1]['kind'] == 'drop'


def test_drop_needs_a_held_object(rig):
    a = rig.arm
    a.inject('drop')
    assert a.move_tcp(pose(0.25, 0.1, 0.06)).ok
    assert a.fault_log == []                   # not consumed without an object


# ----------------------------------------------------------------------------- stop / e-stop
def test_stop_during_motion_holds(rig):
    a = rig.arm
    T = pose(0.25, 0.18, 0.06, 0.6)
    rig.at(0.6, a.stop)
    r = a.move_tcp(T)
    assert not r.ok and r.error == 'stalled' and 'stop' in r.detail
    q = a.get_joints().value
    rig.sim.settle(1.0)
    assert np.max(np.abs(a.get_joints().value - q)) < 0.02
    assert np.max(np.abs(a._qd_meas())) < 0.01
    assert a.move_tcp(T).ok


def test_abort_event_during_motion(rig):
    a = rig.arm
    rig.at(0.5, a.abort_event.set)
    r = a.move_tcp(pose(0.25, 0.18, 0.06, 0.6))
    assert r.error == 'aborted'
    assert a.move_tcp(pose(0.30, 0.0, 0.08)).error == 'aborted'
    assert a.gripper(0.05).error == 'aborted'


def test_emergency_stop_latches(rig):
    a = rig.arm
    rig.at(0.5, a.emergency_stop)
    r = a.move_tcp(pose(0.25, 0.18, 0.06, 0.6))
    assert not r.ok and r.error == 'arm_error'
    assert not a.health().ok and 'emergency' in a.health().reason
    assert not a.enable()
    assert a.move_joints(a.get_joints().value).error == 'arm_error'
    q = a.get_joints().value
    rig.sim.settle(1.0)
    assert np.max(np.abs(a.get_joints().value - q)) < 0.02


def test_speed_percent_and_determinism():
    durations, finals = [], []
    for speed in (100, 100, 30):
        r = Rig(speed=speed)
        res = r.arm.move_tcp(pose(0.25, 0.15, 0.08, 0.3))
        assert res.ok
        durations.append(res.duration)
        finals.append(np.array(r.d.qpos))
    assert durations[0] == durations[1] and np.array_equal(finals[0], finals[1])
    assert durations[2] > 2.0 * durations[0]
    with pytest.raises(ValueError):
        Rig().arm.set_speed_percent(0)


def test_tilted_lift_pose_matches_sdk_reachability():
    """A tilt SafeArm (PiperKinematics) picks is also solvable in MuJoCo."""
    m = _model()
    ik = MujocoIK(m, site='tcp')
    kin = PiperKinematics('piper')
    T_ft = tf.make_T(np.eye(3), [0, 0, TCP_Z])
    for x, y, z in ((0.2, 0.0, 0.15), (0.3, 0.15, 0.13), (0.4, -0.2, 0.12), (0.45, 0.0, 0.1)):
        sol = kin.solve_tcp_with_tilt(pose(x, y, z, tf.neutral_closing_angle(x, y)), T_ft)
        assert sol is not None
        q, _ep, _eo = ik.solve(sol[0])
        assert q is not None, (x, y, z, math.degrees(sol[2]))
    assert tilt_outward(pose(0.3, 0, 0.1), 0.0)[2, 2] == pytest.approx(-1.0)


# ----------------------------------------------------------------------------- skill layer
@pytest.mark.parametrize('fault, stage', [(None, 'done'), ('grasp_miss', 'grasp_miss'),
                                          ('drop', 'dropped')])
def test_pick_place_skill_through_safe_arm(fault, stage):
    """Drive the physics arm with PickPlaceSkill + SafeArm like the real one."""
    from cube_task.core.types import Detection
    from cube_task.core.zones import parse_zones
    from cube_task.perception.grasp import (GraspPlanner, held_offset_of, PlacePlanner,
                                            transit_height)
    from cube_task.robot.kinematics import flange_to_tcp_from_cfg
    from cube_task.robot.safety import SafeArm, SafetyGuard
    from cube_task.task.reach import adjust_plan_poses
    from cube_task.task.skills import PickPlaceSkill
    r = Rig()
    cfg = r.cfg
    cfg.set('zones', {'zone_red': {'center': [0.22, 0.20, 0.0], 'size': [0.12, 0.12]}})
    arm = SafeArm(r.arm, SafetyGuard(cfg), kin=PiperKinematics('piper'),
                  T_flange_tcp=flange_to_tcp_from_cfg(cfg))
    arm.set_table_z(0.0)
    det = Detection(label='red', score=1.0, center_base=np.array([BOX_XY[0], BOX_XY[1], BOX]),
                    top_z=BOX, height=BOX, yaw=0.0, length=BOX, width=BOX,
                    extra={'table_z': 0.0, 'cap_center_xy': list(BOX_XY)})
    zone = parse_zones(cfg)['zone_red']
    gp = GraspPlanner(cfg, 'L1').plan(det)
    held = held_offset_of(gp)
    pp = PlacePlanner(cfg).plan(zone, held, transit_height(cfg, [zone], 0.0, held))
    ok, why, _tilts = adjust_plan_poses(arm, gp, pp)
    assert gp.feasible and pp.feasible and ok, why
    if fault:
        r.arm.inject(fault)
    res = PickPlaceSkill(cfg, arm, r.clock).execute(gp, pp)
    assert res.stage == stage and res.ok == (fault is None), res
    r.sim.settle(1.0)
    p = r.box_pos()
    assert p[2] < 0.03
    assert zone.contains_xy(p[0], p[1]) == (fault is None)


def test_target_just_outside_mujoco_reach_goes_to_closest_pose(rig):
    """Reach a pose SafeArm (SDK kinematics) accepts but the MuJoCo model misses by < 2 mm."""
    ik = rig.arm.ik
    T = None
    for z in np.arange(0.09, 0.14, 0.0005):     # top-down poses above the joint5 limit
        cand = pose(0.30, 0.0, float(z))
        q, ep, eo = ik.solve(cand)
        if q is None:
            if ep < 0.0015 and eo < math.radians(0.8):
                T = cand
            break
    assert T is not None
    r = rig.arm.move_tcp(T)
    assert r.ok and r.pos_err < 0.002, r
    assert rig.arm.move_tcp(pose(0.30, 0.0, 0.25)).error == 'unreachable'


def test_second_arm_takes_over_and_meta_finger_geoms(rig):
    pads = [i for i in range(rig.m.ngeom) if rig.m.geom_contype[i] and rig.m.geom_bodyid[i]
            in (rig.sim.body_id('link7'), rig.sim.body_id('link8'))]
    rig.sim.meta['finger_geoms'] = pads
    arm2 = MujocoArm(rig.cfg, rig.clock, rig.sim)
    assert list(arm2._finger_geoms) == pads
    arm2.connect()
    assert arm2.enable()
    q = arm2.get_joints().value
    assert arm2.move_tcp(pose(0.25, 0.1, 0.06)).ok        # the old arm no longer holds q
    assert np.max(np.abs(arm2.get_joints().value - q)) > 0.05

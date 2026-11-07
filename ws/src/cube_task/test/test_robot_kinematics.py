"""Tests for robot/kinematics.py (FK against pyAgxArm, Jacobian, IK round trips)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.robot.kinematics import (JOINT_LIMITS, MDH_PRESETS, MODELS, PiperKinematics,
                                        rot_log, tilt_outward)

T_FT = tf.pose6_to_T([0.0, 0.0, 0.14, 0.0, 0.0, 0.0])   # default robot.tcp_offset
T_TF = tf.inv_T(T_FT)


def _sdk():
    try:
        from pyAgxArm.api import constants
        from pyAgxArm.utiles import mdh_kinematics
    except ImportError:
        pytest.skip('pyAgxArm not importable')
    return constants, mdh_kinematics


def test_tables_match_sdk_constants():
    constants, _ = _sdk()
    for model in MODELS:
        sdk_mdh = constants.ROBOT_MDH_PRESET[model]
        assert MDH_PRESETS[model] == tuple(tuple(link) for link in sdk_mdh)
        sdk_lim = constants.ROBOT_JOINT_LIMIT_PRESET_RAD[model]
        ours = JOINT_LIMITS[model]
        for i in range(6):
            assert list(ours[i]) == pytest.approx(sdk_lim['joint%d' % (i + 1)], abs=0.0)


@pytest.mark.parametrize('model', MODELS)
def test_fk_matches_pyagxarm(model):
    _, mdh_kinematics = _sdk()
    kin = PiperKinematics(model)
    mdh = mdh_kinematics.get_mdh(model)
    rng = np.random.RandomState(7)
    lim = kin.limits
    for _ in range(60):
        q = rng.uniform(lim[:, 0], lim[:, 1])
        T_sdk = tf.pose6_to_T(mdh_kinematics.fk_from_mdh(mdh, [float(v) for v in q]))
        assert np.abs(kin.fk(q) - T_sdk).max() < 1e-9


def test_fk_zero_pose_known_value():
    pose = tf.T_to_pose6(PiperKinematics('piper').fk([0.0] * 6))
    assert pose == pytest.approx([0.0561275, 0.0, 0.2132663, 0.0, 1.4835299, 0.0], abs=1e-6)


def test_jacobian_matches_finite_differences():
    kin = PiperKinematics()
    q = np.array([0.3, 1.2, -0.9, 0.2, 0.7, -0.4])
    J = kin.jacobian(q)
    T0 = kin.fk(q)
    h = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        T1 = kin.fk(q + dq)
        lin = (T1[:3, 3] - T0[:3, 3]) / h
        ang = rot_log(T1[:3, :3] @ T0[:3, :3].T) / h      # angular velocity, base frame
        assert lin == pytest.approx(J[:3, i], abs=1e-5)
        assert ang == pytest.approx(J[3:, i], abs=1e-5)


def _assert_solution(kin, T_flange, q, pos_tol=1e-4, rot_tol=1e-3):
    assert q is not None
    assert kin.within_limits(q)
    dp, dr = tf.pose_error(kin.fk(q), T_flange)
    assert dp < pos_tol and dr < rot_tol


def test_ik_vertical_top_down_grid():
    """Vertical grasps (tool straight down) over the area joint5 allows: TCP z 0.02-0.05."""
    kin = PiperKinematics()
    times = []
    for x in np.linspace(0.15, 0.38, 5):
        for y in np.linspace(-0.20, 0.20, 5):
            if math.hypot(x, y) > 0.40:     # beyond that a vertical tool needs some tilt
                continue
            for z in (0.02, 0.05):
                neutral = tf.neutral_closing_angle(x, y)
                for d in np.radians([-90.0, -45.0, 0.0, 45.0, 90.0]):
                    T = tf.top_down_T(x, y, z, neutral + d) @ T_TF
                    t0 = time.perf_counter()
                    q, _ep, _eo = kin.ik(T)
                    times.append(time.perf_counter() - t0)
                    _assert_solution(kin, T, q)
    assert np.mean(times) < 0.005, 'mean IK time %.2f ms' % (1e3 * np.mean(times))


def test_ik_any_closing_angle_within_joint6_range():
    kin = PiperKinematics()
    x, y, z = 0.28, 0.08, 0.03
    neutral = tf.neutral_closing_angle(x, y)
    for d in np.radians(np.arange(-165.0, 166.0, 15.0)):
        T = tf.top_down_T(x, y, z, neutral + d) @ T_TF
        q, _ep, _eo = kin.ik(T)
        _assert_solution(kin, T, q)


def test_ik_with_tilt_covers_pick_area_and_zone_ring():
    """Higher poses need the tool leaned outward; 30 deg covers TCP z <= 0.15 everywhere."""
    kin = PiperKinematics()
    targets = [(x, y, z) for x in (0.15, 0.30, 0.45) for y in (-0.25, 0.0, 0.25)
               for z in (0.02, 0.10, 0.15)]
    for az in np.radians([-60.0, 0.0, 60.0]):
        targets += [(0.52 * math.cos(az), 0.52 * math.sin(az), z) for z in (0.05, 0.10)]
    for x, y, z in targets:
        T = tf.top_down_T(x, y, z, tf.neutral_closing_angle(x, y))
        res = kin.solve_tcp_with_tilt(T, T_FT, max_tilt=math.radians(30.0))
        assert res is not None, (x, y, z)
        T_used, q, tilt = res
        assert T_used[:3, 3] == pytest.approx(T[:3, 3], abs=1e-12)
        _assert_solution(kin, T_used @ T_TF, q)


def test_vertical_high_pose_is_unreachable_on_standard_piper():
    # Documents the joint5 (+-70 deg) limit: a vertical tool at TCP z 0.25 cannot be reached.
    kin = PiperKinematics('piper')
    q, ep, _eo = kin.ik(tf.top_down_T(0.30, 0.0, 0.25, math.pi / 2) @ T_TF)
    assert q is None and ep > 1e-3


def test_ik_unreachable_and_invalid_input():
    kin = PiperKinematics()
    t0 = time.perf_counter()
    q, ep, _eo = kin.ik(tf.top_down_T(0.95, 0.0, 0.1, 0.0))
    assert q is None and ep > 0.1
    assert time.perf_counter() - t0 < 0.1
    bad = np.eye(4)
    bad[0, 3] = float('nan')
    with pytest.raises(ValueError):
        kin.ik(bad)
    with pytest.raises(ValueError):
        kin.ik(np.eye(3))
    skew = np.eye(4)
    skew[0, 1] = 0.5
    with pytest.raises(ValueError):
        kin.ik(skew)
    with pytest.raises(ValueError):
        kin.fk([0.0] * 5)
    with pytest.raises(ValueError):
        PiperKinematics('ur5')


def test_ik_prefers_the_seed_branch():
    kin = PiperKinematics()
    T = tf.top_down_T(0.30, -0.10, 0.04, tf.neutral_closing_angle(0.30, -0.10)) @ T_TF
    q1, _ep, _eo = kin.ik(T)
    q2, _ep, _eo = kin.ik(T, q_seed=q1 + 0.01)
    _assert_solution(kin, T, q2)
    assert np.max(np.abs(q2 - q1)) < 0.02


def test_within_limits_and_violations():
    kin = PiperKinematics()
    assert kin.within_limits([0.0] * 6)
    assert not kin.within_limits([0.0, 0.0, 0.0, 0.0, 1.3, 0.0])
    assert not kin.within_limits([0.0, 0.0, float('nan'), 0.0, 0.0, 0.0])
    assert not kin.within_limits([0.0] * 5)
    assert kin.limit_violations([0.0, -0.1, 0.0, 0.0, 1.3, 0.0]) == [
        'joint2=-0.100 < 0.000', 'joint5=1.300 > 1.222']


def test_tilt_outward_geometry():
    T = tf.top_down_T(0.3, 0.2, 0.1, tf.neutral_closing_angle(0.3, 0.2))
    Tt = tilt_outward(T, math.radians(20.0))
    assert Tt[:3, 3] == pytest.approx(T[:3, 3])
    assert math.degrees(math.acos(-Tt[2, 2])) == pytest.approx(20.0, abs=1e-9)
    radial = np.array([0.3, 0.2]) / math.hypot(0.3, 0.2)
    assert float(np.dot(Tt[:2, 2], radial)) > 0.3      # approach axis points away from the base
    assert abs(Tt[2, 1]) < 1e-12                       # closing axis stays horizontal

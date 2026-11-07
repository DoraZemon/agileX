"""Tests for sim/world.py: geometry, grasp, release and scoring."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.core.transforms import make_T, rot_x, rot_z, top_down_T
from cube_task.sim.world import SimObject, SimWorld, SimZone, point_in_polygon


def cube(name='c', label='red', edge=0.03, x=0.30, y=0.0, yaw=0.0):
    obj = SimObject(name, label, 'box', (edge, edge, edge), (40, 40, 200), np.eye(4))
    obj.place_upright(x, y, yaw, 0.0)
    return obj


def mats():
    return [SimZone('zone_red', (0.20, 0.30, 0.0), (0.10, 0.10)),
            SimZone('bin', (0.20, -0.30, 0.0), (0.14, 0.14), wall_height=0.05)]


def test_box_geometry_exact():
    obj = SimObject('b', 'x', 'box', (0.04, 0.02, 0.03), (0, 0, 0),
                    make_T(rot_z(math.pi / 2), (0.3, 0.1, 0.0)))
    lo, hi = obj.aabb()
    assert np.allclose(lo, [0.29, 0.08, 0.0]) and np.allclose(hi, [0.31, 0.12, 0.03])
    assert obj.top_z() == pytest.approx(0.03)
    assert obj.height() == pytest.approx(0.03)
    assert obj.footprint_extent((1.0, 0.0)) == pytest.approx(0.02)
    assert obj.footprint_extent((0.0, 1.0)) == pytest.approx(0.04)
    d = math.sqrt(0.5)
    assert obj.footprint_extent((d, d)) == pytest.approx((0.04 + 0.02) * d)
    c, r = obj.bounding_sphere()
    assert np.allclose(c, [0.3, 0.1, 0.015]) and r >= 0.025


@pytest.mark.parametrize('shape,size,height,width_x', [
    ('sphere', (0.03,), 0.06, 0.06),
    ('ellipsoid', (0.05, 0.02, 0.025), 0.05, 0.10),
    ('cylinder', (0.02, 0.07), 0.07, 0.04),
    ('capsule_chain', (0.015, [(-0.05, 0.0, 0.015), (0.05, 0.0, 0.015)]), 0.03, 0.13),
    ('sphere_cluster', (0.01, [(0.0, 0.0, 0.01), (0.02, 0.0, 0.01)]), 0.02, 0.04),
])
def test_shape_extents_match_surface_samples(shape, size, height, width_x):
    obj = SimObject('o', 'x', shape, size, (1, 2, 3), make_T(np.eye(3), (0.2, 0.0, 0.0)))
    assert obj.height() == pytest.approx(height, abs=1e-9)
    assert obj.footprint_extent((1.0, 0.0)) == pytest.approx(width_x, abs=1e-9)
    obj.T_base_obj = make_T(rot_z(0.4) @ rot_x(0.3), (0.2, 0.1, 0.05))
    pts = obj.surface_points()
    lo, hi = obj.aabb()
    assert np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9)
    assert np.allclose(pts.min(axis=0), lo, atol=0.0015)
    assert np.allclose(pts.max(axis=0), hi, atol=0.0015)


def test_capsule_chain_is_dense():
    obj = SimObject('k', 'carrot', 'capsule_chain',
                    (0.015, [(-0.06, 0.0, 0.015), (0.06, 0.0, 0.015)]), (0, 0, 0), np.eye(4))
    centers = obj.primitives()[0][2]
    gaps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    assert np.all(gaps <= 0.015 / 2.0 + 1e-12)


def test_invalid_objects_raise():
    with pytest.raises(ValueError):
        SimObject('a', 'x', 'pyramid', (0.1,), (0, 0, 0), np.eye(4))
    with pytest.raises(ValueError):
        SimObject('a', 'x', 'box', (0.03, float('nan'), 0.03), (0, 0, 0), np.eye(4))
    with pytest.raises(ValueError):
        SimObject('a', 'x', 'sphere', (0.03,), (0, 0, 300), np.eye(4))
    T = np.eye(4)
    T[0, 0] = 2.0
    with pytest.raises(ValueError):
        SimObject('a', 'x', 'sphere', (0.03,), (0, 0, 0), T)
    with pytest.raises(ValueError):
        SimWorld(objects=[cube('a'), cube('a')])


def test_grasp_success_attaches_and_follows():
    obj = cube(yaw=0.3, x=0.30, y=0.05)
    w = SimWorld(objects=[obj], zones=mats())
    T = top_down_T(0.302, 0.05, 0.015, 0.3 + math.pi / 2)
    width = w.try_grasp(T, 0.05, 0.0)
    assert width == pytest.approx(0.03, abs=1e-6)
    assert w.held is obj and w.last_grasp['ok']
    # Centered between the fingers after closing: moved along tool y only.
    yv = T[:2, 1]
    c0 = np.array([0.30, 0.05])
    expected = c0 + yv * float((T[:2, 3] - c0) @ yv)
    assert np.allclose(obj.center_xy(), expected, atol=1e-4)
    T2 = top_down_T(0.25, 0.10, 0.20, math.pi / 2)
    w.update_tcp(T2, width)
    assert obj.bottom_z() == pytest.approx(0.20 - 0.015, abs=1e-6)
    assert w.remaining() == []
    # Already holding: returns the held width and keeps the object.
    assert w.try_grasp(T2, 0.07, 0.0) == pytest.approx(width)


def test_grasp_close_width_floor():
    w = SimWorld(objects=[cube()])
    assert w.try_grasp(top_down_T(0.30, 0.0, 0.015, 0.0), 0.05, 0.035) == pytest.approx(0.035)


@pytest.mark.parametrize('T,open_w,reason', [
    (top_down_T(0.30, 0.0, 0.015, 0.0), 0.025, 'collision_finger'),      # opening too small
    (top_down_T(0.312, 0.0, 0.015, math.pi / 2), 0.06, 'no_object'),     # 12 mm off along tool x
    (top_down_T(0.30, 0.0075, 0.015, math.pi / 2), 0.06, 'no_object'),   # off along tool y
    (top_down_T(0.30, 0.0, 0.035, 0.0), 0.06, 'no_object'),              # above the top
    (top_down_T(0.30, 0.0, -0.001, 0.0), 0.06, 'no_object'),             # below the bottom
])
def test_grasp_failures(T, open_w, reason):
    obj = cube()
    w = SimWorld(objects=[obj])
    before = obj.T_base_obj
    assert w.try_grasp(T, open_w, 0.0) is None
    assert w.last_grasp['reason'] == reason
    assert w.held is None and np.allclose(obj.T_base_obj, before)


def test_too_wide_object_collides():
    big = SimObject('big', 'x', 'box', (0.08, 0.03, 0.03), (0, 0, 0), np.eye(4))
    big.place_upright(0.3, 0.0, 0.0, 0.0)
    w = SimWorld(objects=[big])
    assert w.try_grasp(top_down_T(0.3, 0.0, 0.015, 0.0), 0.07, 0.0) is None
    assert w.last_grasp['reason'] == 'collision_finger'
    # Across the narrow axis it works.
    assert w.try_grasp(top_down_T(0.3, 0.0, 0.015, math.pi / 2), 0.07, 0.0) == pytest.approx(0.03)


def test_neighbor_in_finger_path_blocks_grasp():
    a = cube('a', x=0.30, y=0.0)
    b = cube('b', x=0.30, y=0.045)          # 15 mm gap along base y
    w = SimWorld(objects=[a, b])
    # Closing along base y: the +y finger column (y 0.035..0.047) hits b.
    assert w.try_grasp(top_down_T(0.30, 0.0, 0.015, math.pi / 2), 0.07, 0.0) is None
    assert w.last_grasp['reason'] == 'collision_finger'
    assert w.last_grasp['collided'] == 'b'
    # Closing along base x avoids b.
    assert w.try_grasp(top_down_T(0.30, 0.0, 0.015, 0.0), 0.07, 0.0) == pytest.approx(0.03)


def test_touching_cubes_graspable_parallel_to_contact():
    a = cube('a', x=0.30, y=0.0)
    b = cube('b', x=0.33, y=0.0)            # touching along base x
    w = SimWorld(objects=[a, b])
    assert w.check_grasp(top_down_T(0.30, 0.0, 0.015, 0.0), 0.05, 0.0)[0] is False
    ok, info = w.check_grasp(top_down_T(0.30, 0.0, 0.015, math.pi / 2), 0.05, 0.0)
    assert ok and info['object'] == 'a'


def test_release_into_zone_table_and_bin():
    obj = cube(yaw=0.2)
    w = SimWorld(objects=[obj], zones=mats())
    assert w.release() is None and w.last_release['reason'] == 'nothing_held'
    w.try_grasp(top_down_T(0.30, 0.0, 0.015, 0.2), 0.05, 0.0)
    # Tilted TCP at release: object lands upright, keeps its yaw.
    T = top_down_T(0.20, 0.30, 0.10, 0.2 + 0.5)
    T[:3, :3] = T[:3, :3] @ rot_x(0.2)
    w.update_tcp(T, 0.03)
    c_before = obj.center_xy()
    assert w.release(T) == 'zone_red'
    assert np.allclose(obj.center_xy(), c_before)
    assert obj.bottom_z() == pytest.approx(0.0)
    assert np.allclose(obj.T_base_obj[:3, 2], [0, 0, 1])
    assert w.placements() == {'zone_red': ['red'], 'bin': []}
    assert w.score({'red': 'zone_red'}) == {'correct': 1, 'wrong': 0, 'remaining': 0}
    assert w.score({'red': 'bin'}) == {'correct': 0, 'wrong': 1, 'remaining': 0}
    # Pick it again, drop on the table.
    w.try_grasp(top_down_T(*obj.center_xy(), 0.015, obj.yaw()), 0.05, 0.0)
    assert w.release(top_down_T(0.35, 0.1, 0.2, 0.0)) is None
    assert w.last_release['ok'] and w.last_release['zone'] is None
    assert obj.bottom_z() == pytest.approx(0.0)
    assert [o.name for o in w.remaining()] == ['c']
    poly = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]
    assert len(w.remaining(poly)) == 1
    assert w.remaining([[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]) == []
    # Partly outside a walled bin still counts by the center; floor z used.
    w.zones[1] = SimZone('bin', (0.20, -0.30, 0.02), (0.14, 0.14), wall_height=0.05)
    w.try_grasp(top_down_T(*obj.center_xy(), 0.015, obj.yaw()), 0.05, 0.0)
    assert w.release(top_down_T(0.20 + 0.065, -0.30, 0.15, 0.0)) == 'bin'
    assert obj.bottom_z() == pytest.approx(0.02)
    assert w.score({'red': 'bin'})['correct'] == 1


def test_score_distractor_and_held():
    red = cube('r', 'red', x=0.25)
    yel = cube('y', 'yellow', x=0.35)
    w = SimWorld(objects=[red, yel], zones=mats())
    c2z = {'red': 'zone_red'}
    assert w.score(c2z) == {'correct': 0, 'wrong': 0, 'remaining': 1}
    w.try_grasp(top_down_T(0.35, 0.0, 0.015, 0.0), 0.05, 0.0)
    w.release(top_down_T(0.20, 0.30, 0.1, 0.0))
    assert w.score(c2z) == {'correct': 0, 'wrong': 1, 'remaining': 1}
    w.try_grasp(top_down_T(0.25, 0.0, 0.015, 0.0), 0.05, 0.0)
    assert w.score(c2z)['remaining'] == 1          # held target is not placed


def test_copy_is_deep_and_to_dict():
    w = SimWorld(objects=[cube()], zones=mats())
    w2 = w.copy()
    w2.try_grasp(top_down_T(0.30, 0.0, 0.015, 0.0), 0.05, 0.0)
    w2.release(top_down_T(0.20, 0.30, 0.1, 0.0))
    assert w.placements()['zone_red'] == [] and w2.placements()['zone_red'] == ['red']
    d = w2.to_dict()
    assert d['objects'][0]['name'] == 'c' and d['placements']['zone_red'] == ['red']


def test_invalid_gripper_inputs():
    w = SimWorld(objects=[cube()])
    with pytest.raises(ValueError):
        w.try_grasp(np.full((4, 4), np.nan), 0.05, 0.0)
    with pytest.raises(ValueError):
        w.try_grasp(top_down_T(0.3, 0, 0.015, 0), float('nan'), 0.0)
    with pytest.raises(ValueError):
        w.update_tcp(top_down_T(0.3, 0, 0.015, 0), -0.01)


def test_zone_matches_core_semantics():
    z = SimZone('z', (0.2, 0.3, 0.01), (0.10, 0.06), yaw=0.5, label='red')
    core = z.to_zone()
    for x, y in [(0.2, 0.3), (0.24, 0.32), (0.26, 0.3), (0.2, 0.33)]:
        assert z.contains_xy(x, y) == core.contains_xy(x, y)
    assert z.to_config()['center'] == [0.2, 0.3, 0.01]
    assert point_in_polygon(0.2, 0.3, z.corners_xy())

"""Tests for sim/scenarios.py: determinism, layout, graspability, views and overlay."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.core.config import deep_merge, load_config
from cube_task.core.transforms import inv_T, pose6_to_T, rotation_angle
from cube_task.core.zones import parse_zones
from cube_task.sim import scenarios as sc
from cube_task.sim.render import camera_rays, render_rgbd
from cube_task.sim.world import WALL_THICKNESS, point_in_polygon

POLY = np.array(sc.DEFAULT_PICK_POLYGON)
COMBOS = [(t, d) for t in sc.TASKS for d in sc.DIFFICULTIES]


def find_grasp(world, obj):
    """Return (closing angle, width) of the narrowest working top-down grasp, or None."""
    return sc.find_top_down_grasp(world, obj, steps=36)


def table_xy(T_base_camera, intr):
    rays = camera_rays(intr).reshape(-1, 3) @ T_base_camera[:3, :3].T
    t = -T_base_camera[2, 3] / np.where(rays[:, 2] < -1e-9, rays[:, 2], np.nan)
    return (T_base_camera[:3, 3][None, :2] + t[:, None] * rays[:, :2]).reshape(
        intr.height, intr.width, 2)


def in_poly(xy, poly):
    x, y = xy[..., 0], xy[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        with np.errstate(divide='ignore', invalid='ignore'):
            cross = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        inside ^= cross
    return inside


def test_invalid_arguments():
    with pytest.raises(ValueError):
        sc.make_scenario('L4', 0)
    with pytest.raises(ValueError):
        sc.make_scenario('L1', 0, mount='ceiling')
    with pytest.raises(ValueError):
        sc.make_scenario('L1', 0, difficulty='easy')


@pytest.mark.parametrize('task,diff', COMBOS)
def test_deterministic(task, diff):
    w1, i1 = sc.make_scenario(task, 4, difficulty=diff)
    w2, i2 = sc.make_scenario(task, 4, difficulty=diff)
    w3, _ = sc.make_scenario(task, 5, difficulty=diff)
    assert [o.name for o in w1.objects] == [o.name for o in w2.objects]
    assert all(np.array_equal(a.T_base_obj, b.T_base_obj) for a, b in zip(w1.objects, w2.objects))
    assert i1['labels'] == i2['labels'] and i1.get('instructions') == i2.get('instructions')
    assert not all(np.allclose(a.T_base_obj, b.T_base_obj) for a, b in zip(w1.objects, w3.objects))


@pytest.mark.parametrize('task,diff', COMBOS)
def test_layout_and_graspability(task, diff):
    for seed in range(3):
        w, info = sc.make_scenario(task, seed, difficulty=diff)
        assert w.table_z == 0.0 and set(info['zones']) == {z.name for z in w.zones}
        for z in w.zones:
            assert sc.top_down_reachable(z.center[0], z.center[1], z.floor_z + 0.06)
            assert z.center[0] >= 0.08
            # No zone point (with walls and a 2 cm margin) inside the pick polygon.
            g = np.linspace(-1.0, 1.0, 9)
            m = (WALL_THICKNESS if z.wall_height > 0 else 0.0) + 0.02 - 1e-6
            for gx in g:
                for gy in g:
                    lx, ly = gx * (z.size[0] / 2 + m), gy * (z.size[1] / 2 + m)
                    c, s = math.cos(z.yaw), math.sin(z.yaw)
                    p = z.center[:2] + np.array([c * lx - s * ly, s * lx + c * ly])
                    assert not point_in_polygon(p[0], p[1], POLY), (z.name, p)
        for obj in w.objects:
            c = obj.center_xy()
            lo, hi = obj.aabb()
            assert lo[2] == pytest.approx(0.0, abs=1e-9)
            for corner in sc._footprint(obj):
                assert point_in_polygon(corner[0], corner[1], POLY)
            assert w.zone_of(obj) is None
            g = find_grasp(w, obj)
            assert g is not None, (task, diff, seed, obj.name)
            assert g[1] <= sc.GRIPPER_MAX_WIDTH - 0.008 + 5e-4
            assert sc.REACH_R[0] <= math.hypot(c[0], c[1]) <= sc.REACH_R[1]
        if diff == 'normal':
            for i, a in enumerate(w.objects):
                for b in w.objects[i + 1:]:
                    assert sc._gap_between(sc._footprint(a), sc._footprint(b)) > 0.01
        assert w.score(info['class_to_zone'])['remaining'] == sum(
            1 for o in w.objects if o.label in info['class_to_zone'])


def test_l1_content():
    w, info = sc.make_scenario('L1', 2)
    assert info['class_to_zone'] == {'red': 'zone_red', 'green': 'zone_green',
                                     'blue': 'zone_blue'}
    assert sorted(o.label for o in w.objects) == ['blue', 'green', 'red']
    for o in w.objects:
        assert o.shape == 'box' and 0.03 <= o.size[0] <= 0.04
    cs = [o.center_xy() for o in w.objects]
    assert min(np.linalg.norm(a - b) for i, a in enumerate(cs) for b in cs[i + 1:]) >= 0.06
    w, info = sc.make_scenario('L1', 2, difficulty='hard')
    assert 'yellow' in info['labels'] and 'yellow' not in info['class_to_zone']
    a, b = w.object('red_cube'), w.object('green_cube')
    assert abs(sc._gap_between(sc._footprint(a), sc._footprint(b))) < 1e-6   # touching
    assert info['required'] == 3


def test_l2_l3_content():
    w, info = sc.make_scenario('L2', 3)
    assert set(info['class_to_zone']) == set(info['labels']) == {o.label for o in w.objects}
    cats = [sc.CATALOG[o.label][0] for o in w.objects]
    assert cats.count('fruit') == 3 and cats.count('vegetable') == 3
    for lab, zone in info['class_to_zone'].items():
        assert zone == ('fruit_bin' if lab in sc.FRUITS else 'veg_bin')
    _, info = sc.make_scenario('L2', 3, difficulty='hard')
    assert info['labels'] == sorted(sc.CATALOG)
    w, info = sc.make_scenario('L3', 3)
    assert len(info['instructions']) == 5 and info['class_to_zone'] == {}
    apples = [o for o in w.objects if o.label == 'apple']
    left = max(apples, key=lambda o: o.center_xy()[1])
    spec = info['instruction_specs'][0]
    assert spec['selector'] == 'leftmost' and spec['object'] == left.name
    for s in info['instruction_specs']:
        assert w.object(s['object']).label == s['label'] and s['zone'] in info['zones']
        assert s['zone'] in info['zone_aliases']


def test_ik_round_trip_and_limits():
    q = [0.3, 1.2, -1.0, 0.1, 0.9, -0.2]
    T = sc.piper_fk(q)
    q2, ep, er = sc.piper_ik(T)
    assert q2 is not None and ep < 1e-4 and er < 1e-3
    assert np.allclose(sc.piper_fk(q2), T, atol=1e-4)
    assert sc.piper_fk([0.0] * 6)[:3, 3] == pytest.approx([0.0561, 0.0, 0.2133], abs=2e-4)
    assert not sc.top_down_reachable(0.30, 0.0, 0.30)       # documented reach limit
    assert sc.top_down_reachable(0.30, 0.0, 0.05)


def test_eye_in_hand_observe_sees_objects():
    for task in ('L1', 'L2'):
        for seed in range(3):
            w, info = sc.make_scenario(task, seed, mount='eye_in_hand')
            q = info['observe_joints']
            assert sc.within_limits(q, 0.02)
            T_bc = sc.piper_fk(q) @ info['T_flange_camera']
            assert np.allclose(T_bc, info['T_base_camera_observe'], atol=1e-9)
            assert rotation_angle(T_bc[:3, :3].T @ sc.look_at(
                T_bc[:3, 3], (0.30, 0.0, 0.0))[:3, :3]) < math.radians(0.5)
            intr = info['intrinsics']
            Tcb = inv_T(T_bc)
            for obj in w.objects:
                lo, hi = obj.aabb()
                for x in (lo[0], hi[0]):
                    for y in (lo[1], hi[1]):
                        for z in (lo[2], hi[2]):
                            p = Tcb[:3, :3] @ np.array([x, y, z]) + Tcb[:3, 3]
                            u = intr.fx * p[0] / p[2] + intr.cx
                            v = intr.fy * p[1] / p[2] + intr.cy
                            assert 0 <= u < intr.width and 0 <= v < intr.height, obj.name


@pytest.mark.parametrize('mount', sc.MOUNTS)
def test_observe_pose_gripper_leaves_pick_area_unoccluded(mount):
    w, info = sc.make_scenario('L2', 1, mount=mount)
    q = info['observe_joints']
    T_tcp = sc.piper_fk(q) @ pose6_to_T(sc.DEFAULT_TCP_OFFSET)
    T_bc = info['T_base_camera'] if mount == 'eye_to_hand' else info['T_base_camera_observe']
    intr = info['intrinsics']
    _, d_free = render_rgbd(w, intr, T_bc, gripper=False)
    _, d_grip = render_rgbd(w, intr, T_bc, gripper=(T_tcp, 0.07))
    pick = in_poly(table_xy(T_bc, intr), POLY)
    _, d_empty = render_rgbd(type(w)(zones=w.zones), intr, T_bc, gripper=False)
    objects = np.abs(d_free - d_empty) > 1e-4
    changed = np.abs(d_grip - d_free) > 1e-6
    assert pick.sum() > 20000
    assert not np.any(changed & (pick | objects))
    if mount == 'eye_to_hand':
        assert changed.any()                                 # gripper is in the image
        c = T_tcp[:2, 3]
        assert not point_in_polygon(c[0], c[1], POLY)
        assert T_tcp[0, 3] >= 0.08 and 0.10 <= math.hypot(*T_tcp[:2, 3]) <= 0.62
        Tcb = inv_T(T_bc)
        for x, y in POLY:
            p = Tcb[:3, :3] @ np.array([x, y, 0.0]) + Tcb[:3, 3]
            assert 0 <= intr.fx * p[0] / p[2] + intr.cx < intr.width
            assert 0 <= intr.fy * p[1] / p[2] + intr.cy < intr.height


@pytest.mark.parametrize('task,mount', [('L1', 'eye_to_hand'), ('L3', 'eye_in_hand')])
def test_config_overlay(task, mount):
    _, info = sc.make_scenario(task, 6, mount=mount)
    ov = sc.scenario_config_overlay(info, extrinsics_error=(0.004, 1.5))
    cfg = load_config(use_overlays=False)
    merged = deep_merge(cfg.data, ov)
    assert merged['camera']['backend'] == 'sim' and merged['robot']['backend'] == 'sim'
    assert merged['camera']['mount'] == mount
    assert merged['perception']['roi']['pick_polygon'] == sc.DEFAULT_PICK_POLYGON
    assert merged['perception']['roi']['pick_polygon'] == cfg.get('perception.roi.pick_polygon')
    assert merged['tasks'][task]['labels'] == info['labels']
    assert merged['tasks'][task]['class_to_zone'] == info['class_to_zone']
    assert merged['tasks'][task]['time_limit_s'] == cfg.get('tasks.%s.time_limit_s' % task)
    assert merged['robot']['poses']['observe_joints'] == pytest.approx(info['observe_joints'])
    zones = parse_zones(merged)
    assert set(zones) == set(info['zones'])
    for name, z in zones.items():
        ref = info['zones'][name]
        assert np.allclose(z.center, ref.center) and z.size == pytest.approx(ref.size)
        assert z.wall_height == ref.wall_height and z.label == ref.label
    key = 'T_base_camera' if mount == 'eye_to_hand' else 'T_flange_camera'
    other = 'T_flange_camera' if mount == 'eye_to_hand' else 'T_base_camera'
    T_true = np.array(merged['sim'][key])
    assert merged['sim'][other] is None
    assert np.allclose(T_true, info[key])
    inline = merged['camera']['extrinsics_inline']
    assert set(inline) == {'mount', 'matrix'} and inline['mount'] == mount
    T_est = np.array(inline['matrix'])
    assert np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) == pytest.approx(0.004, abs=1e-9)
    assert math.degrees(rotation_angle(T_est[:3, :3].T @ T_true[:3, :3])) == pytest.approx(1.5)
    exact = sc.scenario_config_overlay(info)
    assert np.allclose(np.array(exact['camera']['extrinsics_inline']['matrix']), info[key])
    assert merged['sim']['intrinsics']['fx'] == 380.0
    assert merged['sim']['intrinsics']['width'] == 640
    if task == 'L3':
        assert merged['tasks']['L3']['zone_aliases']['zone_blue'][0] == '蓝色托盘'
    with pytest.raises(ValueError):
        sc.scenario_config_overlay(info, extrinsics_error=(-0.01, 0.0))

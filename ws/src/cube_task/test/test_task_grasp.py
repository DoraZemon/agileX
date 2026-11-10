"""Tests for perception/grasp.py (geometry of grasp and place plans)."""
from __future__ import annotations

import math

from cube_task.core.transforms import neutral_closing_angle, wrap_pi
from cube_task.core.types import Zone
from cube_task.perception.grasp import (GraspPlanner, held_offset_of, PlacePlanner,
                                        transit_height)
import numpy as np
import pytest
from test_task_fakes import make_cfg, make_det


def _ang_mod(a, period):
    """Distance of angle a to the nearest multiple of period."""
    r = a % period
    return min(r, period - r)


def test_basic_block_plan_heights_and_orientation():
    cfg = make_cfg()
    p = GraspPlanner(cfg, 'L1').plan(make_det('red', 0.30, 0.05, size=0.03, height=0.03))
    assert p.feasible, p.reason
    top = 0.03
    assert p.T_pre[2, 3] == pytest.approx(top + 0.08)
    assert p.T_grasp[2, 3] == pytest.approx(top - 0.015)       # 0.5 * height
    assert p.T_lift[2, 3] == pytest.approx(p.T_grasp[2, 3] + 0.10)
    for T in (p.T_pre, p.T_grasp, p.T_lift):
        assert np.allclose(T[:3, 2], [0, 0, -1])
        assert np.allclose(T[:2, 3], [0.30, 0.05])
    assert p.expected_width == pytest.approx(0.03)
    assert p.open_width == pytest.approx(0.05)                  # width + open_margin
    assert p.close_width == pytest.approx(0.0)


def test_grasp_depth_clipped_and_z_clamped_above_table():
    cfg = make_cfg()
    gp = GraspPlanner(cfg, 'L1')
    tall = gp.plan(make_det('red', 0.3, 0.0, height=0.10, table_z=0.02))
    assert tall.T_grasp[2, 3] == pytest.approx(0.12 - 0.030)     # max_grasp_depth
    flat = gp.plan(make_det('red', 0.3, 0.0, height=0.010, table_z=0.02))
    floor = 0.02 + 0.002 + 0.001
    assert flat.feasible
    assert flat.T_grasp[2, 3] == pytest.approx(floor)             # clamped, not top - 0.008
    assert flat.T_grasp[2, 3] < 0.03
    too_flat = gp.plan(make_det('red', 0.3, 0.0, height=0.002, table_z=0.02))
    assert not too_flat.feasible and too_flat.reason.startswith('too_flat')
    assert too_flat.T_grasp[2, 3] >= floor - 1e-12


def test_too_wide_is_infeasible():
    cfg = make_cfg()
    p = GraspPlanner(cfg, 'L2').plan(make_det('apple', 0.3, 0.0, size=0.068, length=0.09))
    assert not p.feasible and p.reason.startswith('too_wide')
    cfg.set('grasp.max_object_width', 0.07)
    p2 = GraspPlanner(cfg, 'L2').plan(make_det('apple', 0.3, 0.0, size=0.068, length=0.09))
    assert p2.feasible
    assert p2.open_width == pytest.approx(0.07)                   # capped at max_width


def test_square_block_symmetry_keeps_wrist_near_neutral():
    gp = GraspPlanner(make_cfg(), 'L1')
    for x, y in ((0.30, 0.0), (0.25, 0.18), (0.40, -0.20), (0.18, 0.10)):
        neutral = neutral_closing_angle(x, y)
        for yaw in np.linspace(-math.pi / 2, math.pi / 2, 37, endpoint=False):
            p = gp.plan(make_det('blue', x, y, size=0.03, yaw=float(yaw)))
            assert p.feasible
            assert abs(wrap_pi(p.closing_angle - neutral)) <= math.pi / 4 + 1e-9
            # Fingers close across two faces: closing axis is a face normal.
            assert _ang_mod(p.closing_angle - yaw, math.pi / 2) < 1e-9


def test_elongated_object_closes_across_minor_axis():
    gp = GraspPlanner(make_cfg(), 'L2')
    for yaw in (-1.2, -0.3, 0.0, 0.7, 1.5):
        d = make_det('banana', 0.32, 0.1, size=0.035, length=0.15, yaw=yaw)
        p = gp.plan(d)
        assert p.feasible
        assert p.expected_width == pytest.approx(0.035)
        assert _ang_mod(p.closing_angle - (yaw + math.pi / 2), math.pi) < 1e-9
        assert abs(wrap_pi(p.closing_angle - neutral_closing_angle(0.32, 0.1))) \
            <= math.pi / 2 + 1e-9


def test_round_objects_use_neutral_angle_and_max_extent():
    gp = GraspPlanner(make_cfg(), 'L2')
    d = make_det('orange', 0.3, -0.1, size=0.050, length=0.052, yaw=0.9)
    p = gp.plan(d)
    assert p.closing_angle == pytest.approx(neutral_closing_angle(0.3, -0.1))
    assert p.expected_width == pytest.approx(0.052)
    # Slightly elongated (ratio 1.1): close across the minor axis, but never open
    # less than the height suggests for a partly hidden round cross-section.
    e = gp.plan(make_det('lemon', 0.3, -0.1, size=0.050, length=0.055, yaw=0.9))
    assert e.expected_width == pytest.approx(0.050)
    # In L2 a square-ish box is not treated as a 90-degree symmetric block.
    q = GraspPlanner(make_cfg(), 'L2').plan(make_det('corn', 0.3, -0.1, size=0.03, yaw=0.4))
    assert q.closing_angle == pytest.approx(neutral_closing_angle(0.3, -0.1))


@pytest.mark.parametrize('field,value', [('center_base', None), ('height', None),
                                         ('yaw', float('nan')), ('width', None),
                                         ('top_z', None)])
def test_missing_geometry_is_infeasible_with_reason(field, value):
    d = make_det('red', 0.3, 0.0)
    setattr(d, field, value)
    if field == 'top_z':
        d.center_base = None
    p = GraspPlanner(make_cfg(), 'L1').plan(d)
    assert not p.feasible
    assert p.reason.startswith('missing_geometry')


def test_missing_table_z_and_bad_values():
    d = make_det('red', 0.3, 0.0)
    d.extra = {}
    d.height = None
    p = GraspPlanner(make_cfg(), 'L1').plan(d)
    assert not p.feasible and 'table_z' in p.reason
    d2 = make_det('red', 0.3, 0.0)
    d2.width = 0.0
    assert GraspPlanner(make_cfg(), 'L1').plan(d2).reason.startswith('bad_geometry')


def test_missing_max_width_raises():
    cfg = make_cfg()
    cfg.set('robot.gripper.max_width', None)
    with pytest.raises(ValueError):
        GraspPlanner(cfg, 'L1')
    with pytest.raises(ValueError):
        PlacePlanner(cfg)


def test_transit_height_clears_walls_and_held_object():
    cfg = make_cfg()
    zones = [Zone('a', np.array([0.2, 0.3, 0.0]), wall_height=0.10),
             Zone('b', np.array([0.2, -0.3, 0.0]))]
    tz = transit_height(cfg, zones, 0.0, held_offset=0.04)
    assert tz >= 0.10 + 0.08
    assert tz >= 0.10 + 0.04 + 0.05
    assert tz >= 0.15 and tz >= 0.12
    assert transit_height(cfg, [], 0.0, 0.0) == pytest.approx(0.15)
    assert transit_height(cfg, [], 0.0, 0.03, obstacles_top_z=0.2) == pytest.approx(0.28)
    with pytest.raises(ValueError):
        transit_height(cfg, [], float('nan'), 0.0)


def test_place_plan_flat_and_walled():
    cfg = make_cfg()
    pp = PlacePlanner(cfg)
    flat = Zone('flat', np.array([0.2, 0.3, 0.01]))
    p = pp.plan(flat, held_offset=0.015, transit_z=0.2)
    assert p.feasible
    assert p.T_release[2, 3] == pytest.approx(0.01 + 0.015 + 0.015)   # release_height_flat
    assert p.T_pre[2, 3] == pytest.approx(0.2)
    assert p.T_retreat[2, 3] == pytest.approx(p.T_release[2, 3] + 0.10)
    assert p.open_width == pytest.approx(0.07)
    assert np.allclose(p.T_release[:2, 3], [0.2, 0.3])
    assert np.allclose(p.T_release[:3, 2], [0, 0, -1])
    walled = Zone('bin', np.array([0.2, 0.3, 0.0]), wall_height=0.08)
    q = pp.plan(walled, held_offset=0.02, transit_z=0.1)
    assert q.T_release[2, 3] == pytest.approx(0.08 + 0.02 + 0.02)
    assert q.T_pre[2, 3] >= q.T_release[2, 3]
    over = Zone('o', np.array([0.2, 0.3, 0.0]), release_height=0.02)
    assert pp.plan(over, 0.01, 0.2).T_release[2, 3] == pytest.approx(0.03)
    assert not pp.plan(None, 0.01, 0.2).feasible
    assert not pp.plan(flat, float('nan'), 0.2).feasible


def test_held_offset_of_plan():
    gp = GraspPlanner(make_cfg(), 'L1')
    p = gp.plan(make_det('red', 0.3, 0.0, height=0.04, table_z=0.01))
    assert held_offset_of(p) == pytest.approx(0.04 - 0.02)


def test_l3_block_labels_use_square_rule():
    p = GraspPlanner(make_cfg(), 'L3').plan(make_det('red', 0.3, 0.1, size=0.03, yaw=0.6))
    assert _ang_mod(p.closing_angle - 0.6, math.pi / 2) < 1e-9
    q = GraspPlanner(make_cfg(), 'L3').plan(make_det('apple', 0.3, 0.1, size=0.05, yaw=0.6))
    assert q.closing_angle == pytest.approx(neutral_closing_angle(0.3, 0.1))

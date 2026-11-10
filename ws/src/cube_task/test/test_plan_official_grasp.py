"""Height-aware grasps of the official L1 / L2 objects with the 70 mm and 100 mm grippers."""
from __future__ import annotations

import math

from cube_task.perception.grasp import GraspPlanner, section_profile, section_width
import numpy as np
import pytest
from test_plan_official_shapes import make_object_det
from test_task_fakes import make_cfg

#: (class, gripper max_width) -> feasible. With 70 mm only the pear neck, the cube and the
#: long objects fit (fingers need max_width - 2 * finger_clearance = 60 mm).
FEASIBLE = {
    0.07: {'cube': True, 'cucumber': True, 'pear': True, 'banana': True, 'potato': False,
           'persimmon': False, 'bell_pepper': False},
    0.10: {'cube': True, 'cucumber': True, 'pear': True, 'banana': True, 'potato': True,
           'persimmon': True, 'bell_pepper': True},
}


def _plan(kind, max_width, yaw=0.3, with_points=True):
    cfg = make_cfg('L1' if kind == 'cube' else 'L2')
    cfg.set('robot.gripper.max_width', max_width)
    planner = GraspPlanner(cfg, 'L1' if kind == 'cube' else 'L2')
    det = make_object_det(kind, yaw=yaw, with_points=with_points)
    return planner, det, planner.plan(det)


@pytest.mark.parametrize('max_width', [0.07, 0.10])
@pytest.mark.parametrize('kind', sorted(FEASIBLE[0.07]))
@pytest.mark.parametrize('yaw', [0.0, 0.7, 1.3])
def test_feasibility_per_class_and_gripper(kind, max_width, yaw):
    planner, det, p = _plan(kind, max_width, yaw)
    info = det.extra['grasp']
    assert info['mode'] == 'profile'
    assert p.feasible == FEASIBLE[max_width][kind], p.reason
    if not p.feasible:
        assert p.reason.startswith('too_wide') and kind in p.reason
        assert '%.3f' % max_width in p.reason
        return
    top = det.top_z
    z = float(p.T_grasp[2, 3])
    # (a) finger reach, min grasp depth and table clearance.
    assert top - planner.max_depth - 1e-9 <= z <= top - planner.min_depth + 1e-9
    assert z >= 0.0 + planner.min_grasp_height - 1e-9
    # (b) every section the open fingers pass fits between them.
    prof = section_profile(det.extra['_points_base'], p.T_grasp[:2, 3], p.closing_angle, 0.0)
    fit = max_width - 2.0 * planner.finger_clearance
    assert section_width(prof, z - planner.pad_half, float(prof[0][-1])) <= fit + 1e-9
    assert p.expected_width <= fit + 1e-9
    assert p.expected_width + planner.finger_clearance <= p.open_width <= max_width + 1e-9


def test_pear_is_held_at_its_neck_with_the_70_mm_gripper():
    _planner, det, p = _plan('pear', 0.07)
    assert p.feasible, p.reason
    z_rel = det.extra['grasp']['z_rel']
    assert z_rel > 0.07                          # above the 85 mm bulb (at ~40 mm)
    assert p.expected_width < 0.06


@pytest.mark.parametrize('kind,equator', [('persimmon', 0.025), ('potato', 0.0275),
                                          ('bell_pepper', 0.05)])
def test_convex_bodies_are_held_at_or_below_the_equator(kind, equator):
    _planner, det, p = _plan(kind, 0.10)
    assert p.feasible, p.reason
    assert det.extra['grasp']['z_rel'] <= equator + 0.003
    assert not det.extra['grasp'].get('neck')


def test_dome_above_a_too_wide_equator_is_not_pinched():
    # 70 mm gripper: the persimmon narrows above its equator, but as a dome.
    _planner, det, p = _plan('persimmon', 0.07)
    assert not p.feasible and 'persimmon' in p.reason and 'max_width 0.070' in p.reason


def test_cube_grasp_for_50_mm_blocks():
    for mw in (0.07, 0.10):
        _planner, det, p = _plan('cube', mw)
        assert p.feasible, p.reason
        assert p.expected_width == pytest.approx(0.05, abs=0.004)
        assert 0.05 - 0.03 - 1e-9 <= p.T_grasp[2, 3] <= 0.05 - 0.008
        assert p.open_width <= mw


def test_without_points_the_footprint_rule_is_unchanged():
    _planner, det, p = _plan('cucumber', 0.07, with_points=False)
    assert det.extra['grasp']['mode'] == 'footprint'
    assert p.feasible
    assert p.expected_width == pytest.approx(max(det.width, min(det.length, det.height)))
    _planner, det, p = _plan('persimmon', 0.07, with_points=False)
    assert not p.feasible and p.reason.startswith('too_wide')


def test_section_profile_switch_off():
    cfg = make_cfg('L2')
    cfg.set('grasp.section_profile', False)
    det = make_object_det('pear')
    GraspPlanner(cfg, 'L2').plan(det)
    assert det.extra['grasp']['mode'] == 'footprint'


def test_report_table_for_both_grippers():
    rows = []
    for mw in (0.07, 0.10):
        for kind in sorted(FEASIBLE[mw]):
            _planner, det, p = _plan(kind, mw)
            rows.append((mw, kind, det.extra['grasp']['z_rel'], p.expected_width, p.open_width,
                         p.feasible))
    assert len(rows) == 14 and all(r[5] == FEASIBLE[r[0]][r[1]] for r in rows)


@pytest.mark.parametrize('yaw', [0.0, 0.7, 1.3, 2.2])
def test_curved_banana_is_gripped_on_its_axis_not_inside_its_bow(yaw):
    _planner, det, p = _plan('banana', 0.07, yaw)
    bow = 0.03                                   # sagitta of the synthetic banana
    axis = np.array([0.30 - math.sin(yaw) * bow, math.cos(yaw) * bow])
    assert np.linalg.norm(np.asarray(det.extra['cap_center_xy']) - axis) > 0.008
    assert np.linalg.norm(p.T_grasp[:2, 3] - axis) < 0.003


def test_small_crest_offsets_do_not_move_the_grasp():
    # Cucumber (bow 12 mm): the cap center is ~4 mm off the axis, below SHIFT_MIN.
    _planner, det, p = _plan('cucumber', 0.07, 0.7)
    assert p.T_grasp[:2, 3] == pytest.approx(det.extra['cap_center_xy'], abs=1e-9)

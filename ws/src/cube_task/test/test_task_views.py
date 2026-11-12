"""Several wrist observe poses: task/views.py and the runner's multi-view loop."""
from __future__ import annotations

import copy

from cube_task.core.types import CameraIntrinsics, PerceptionResult
from cube_task.task.views import (drop_near, merge_view_detections, nearest_view,
                                  observe_views, polygon_points, tag_view, view_coverage,
                                  view_of, views_near)
import numpy as np
import pytest
from test_task_fakes import FakeExt, FakePerception, make_cfg, make_det, Rig

VIEWS = [[-0.5, 0.4, -0.5, 0.0, 0.6, 0.0], [0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
         [0.5, 0.4, -0.5, 0.0, 0.6, 0.0]]
#: base y range each view sees in the fake world (view 0 right, 2 left)
SEES = [(-1.0, -0.06), (-0.09, 0.09), (0.06, 1.0)]


def _det(label, x, y, n=200):
    d = make_det(label, x, y)
    d.n_points = n
    return d


# ------------------------------------------------------------------ views.py
def test_observe_views_fall_back_to_observe_joints_and_validate():
    cfg = make_cfg()
    assert observe_views(cfg) == [[0.0, 0.3, -0.5, 0.0, 0.6, 0.0]]
    cfg.set('robot.poses.observe_views', VIEWS)
    assert observe_views(cfg) == VIEWS
    cfg.set('robot.poses.observe_views', [[0.0] * 5])
    with pytest.raises(ValueError):
        observe_views(cfg)
    cfg.set('robot.poses.observe_views', [])
    cfg.set('robot.poses.observe_joints', None)
    assert observe_views(cfg) == []


def test_merge_keeps_one_detection_per_object():
    a0 = tag_view([_det('red', 0.30, 0.05, n=100)], 0)
    a1 = tag_view([_det('red', 0.31, 0.06, n=300), _det('green', 0.30, -0.10)], 1)
    merged = merge_view_detections({0: a0, 1: a1})
    assert sorted(d.label for d in merged) == ['green', 'red']
    red = [d for d in merged if d.label == 'red'][0]
    assert view_of(red) == 1                          # more depth points
    red_stale = [d for d in merge_view_detections({0: a0, 1: a1}, stale={1})
                 if d.label == 'red'][0]
    assert view_of(red_stale) == 0                    # a fresh view beats a stale one
    far = merge_view_detections({0: a0, 1: tag_view([_det('red', 0.40, 0.05)], 1)})
    assert len(far) == 2


def test_merge_prefers_a_whole_view_even_over_a_fresh_partial_one():
    whole = tag_view([_det('cucumber', 0.30, 0.05, n=200)], 0)
    whole[0].length = 0.24
    cut = tag_view([_det('cucumber', 0.31, 0.05, n=900)], 1)
    cut[0].length = 0.08                                        # under the gripper
    partial = (lambda d: d.length < 0.15)
    kept = merge_view_detections({0: whole, 1: cut}, stale={0}, partial=partial)
    assert len(kept) == 1 and view_of(kept[0]) == 0
    assert view_of(merge_view_detections({0: whole, 1: cut}, stale={0})[0]) == 1


def test_views_near():
    scene = {0: [_det('red', 0.30, 0.0)], 1: [_det('blue', 0.30, 0.20)], 2: []}
    assert views_near(scene, (0.32, 0.05), 0.12) == {0}
    assert views_near(scene, (0.30, 0.10), 0.12) == {0, 1}
    assert views_near(scene, (0.60, 0.0), 0.12) == set()


def test_drop_near_and_nearest_view():
    scene = {0: [_det('red', 0.3, 0.0)], 1: [_det('red', 0.31, 0.0), _det('blue', 0.3, 0.2)]}
    drop_near(scene, (0.3, 0.0), 0.04)
    assert scene[0] == [] and [d.label for d in scene[1]] == ['blue']
    assert nearest_view(VIEWS, {0, 2}, [0.45, 0.4, -0.5, 0.0, 0.6, 0.0]) == 2
    assert nearest_view(VIEWS, {2, 1}, None) == 1


def test_polygon_points_and_coverage_of_two_downward_views():
    pts = polygon_points([[0.1, -0.2], [0.4, -0.2], [0.4, 0.2], [0.1, 0.2]], 0.02)
    assert len(pts) == 15 * 20
    intr = CameraIntrinsics(fx=400.0, fy=400.0, cx=319.5, cy=239.5, width=640, height=480)

    def cam(y):                                       # 0.3 m above (0.25, y), looking down
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        T[:3, 3] = [0.25, y, 0.3]
        return T

    poly = [[0.15, -0.30], [0.35, -0.30], [0.35, 0.30], [0.15, 0.30]]
    one, per1, missing = view_coverage(poly, 0.0, [cam(0.0)], intr)
    two, per2, _ = view_coverage(poly, 0.0, [cam(-0.2), cam(0.0), cam(0.2)], intr)
    assert 0.3 < one < 0.7 and len(missing) > 0
    assert two > 0.99 and len(per2) == 3


# ---------------------------------------------------------- runner multi-view
class ViewPerception(FakePerception):
    """Return only the objects the view of the last observe move sees (SEES)."""

    def __init__(self, clock, world, arm):
        super().__init__(clock, world)
        self.arm = arm
        self.views_seen = []

    def process(self, frame, T_base_camera, zones=None):
        self.clock.sleep(self.process_time)
        q = [c[1] for c in self.arm.commands if c[0] == 'move_joints'][-1]
        v = int(np.argmin([np.max(np.abs(np.asarray(w) - q)) for w in VIEWS]))
        self.views_seen.append(v)
        lo, hi = SEES[v]
        dets = [copy.deepcopy(d) for d in self.world.objects
                if lo <= float(d.center_base[1]) <= hi]
        return PerceptionResult(detections=dets, frame_stamp=frame.stamp)


def _rig(blocks):
    rig = Rig('L1', camera__mount='eye_in_hand', robot__poses__observe_views=VIEWS)
    rig.ext = FakeExt(np.eye(4), 'eye_in_hand')
    rig.perception = ViewPerception(rig.clock, rig.world, rig.arm)
    for lbl, x, y in blocks:
        rig.world.add(make_det(lbl, x, y))
    return rig


def _actions(rig):
    return [e for e in rig.logger.kinds('decision') if e['action'] in ('pick', 'view_scanned')]


def test_objects_seen_only_from_other_views_are_all_placed():
    rig = _rig([('red', 0.25, -0.15), ('green', 0.30, 0.0), ('blue', 0.30, 0.16)])
    rig.cfg.set('runner.scan_all_first', True)
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 3 and rep.stop_reason == 'done'
    acts = _actions(rig)
    first_pick = [i for i, e in enumerate(acts) if e['action'] == 'pick'][0]
    assert sorted(e['view'] for e in acts[:first_pick]) == [0, 1, 2]   # full scan first
    assert sorted(rig.world.placed) == ['zone_blue', 'zone_green', 'zone_red']


def test_an_object_seen_from_two_views_is_picked_once():
    rig = _rig([('red', 0.25, -0.07), ('green', 0.30, 0.16)])   # red in views 0 and 1
    rig.cfg.set('tasks.L1.required', 2)
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 2 and rep.attempts == 2


def test_the_target_view_is_captured_again_after_a_pick():
    rig = _rig([('red', 0.25, -0.15), ('blue', 0.30, -0.12)])   # both only in view 0
    rig.cfg.set('tasks.L1.required', 2)
    rep = rig.runner().run()
    assert rep.ok
    acts = _actions(rig)
    picks = [i for i, e in enumerate(acts) if e['action'] == 'pick']
    between = acts[picks[0] + 1:picks[1]]
    assert [e['view'] for e in between] == [0]     # fresh view of the next target only


@pytest.mark.parametrize('radius,rescans', [(0.12, []), (-1.0, [2])])
def test_a_target_far_from_the_last_pick_is_picked_without_a_new_view(radius, rescans):
    rig = _rig([('red', 0.25, -0.15), ('blue', 0.30, 0.16)])   # views 0 and 2, far apart
    rig.cfg.set('tasks.L1.required', 2)
    rig.cfg.set('runner.rescan_radius_m', radius)
    rig.cfg.set('runner.scan_all_first', True)
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 2
    acts = _actions(rig)
    picks = [i for i, e in enumerate(acts) if e['action'] == 'pick']
    assert [e['view'] for e in acts[picks[0] + 1:picks[1]]] == rescans


def test_a_partial_view_of_a_long_object_is_not_a_target():
    runner = _rig([]).runner()
    cuc = _det('cucumber', 0.30, 0.0)
    for length, partial in ((0.08, True), (0.22, False)):
        cuc.length = length
        assert runner._partial(cuc) is partial
    cube = _det('red', 0.30, 0.0)
    cube.length = 0.01
    assert not runner._partial(cube)                    # not an elongated class


def test_nothing_left_after_full_scans_ends_the_round():
    rig = _rig([('red', 0.25, -0.15)])
    rig.cfg.set('tasks.L1.required', 2)
    rig.cfg.set('runner.max_empty_observations', 2)
    rep = rig.runner().run()
    assert rep.placed == 1 and rep.stop_reason == 'no_targets'


def test_one_view_keeps_the_single_view_loop():
    rig = Rig('L1', camera__mount='eye_in_hand')
    rig.ext = FakeExt(np.eye(4), 'eye_in_hand')
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.cfg.set('tasks.L1.required', 1)
    rep = rig.runner().run()
    assert rep.ok and not [e for e in rig.logger.kinds('decision')
                           if e['action'] == 'view_scanned']


def test_the_first_pick_does_not_wait_for_a_full_scan():
    rig = _rig([('red', 0.25, -0.15), ('green', 0.30, 0.0), ('blue', 0.30, 0.16)])
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 3
    acts = _actions(rig)
    first_pick = [i for i, e in enumerate(acts) if e['action'] == 'pick'][0]
    assert len(acts[:first_pick]) == 1                  # one view, then the first pick

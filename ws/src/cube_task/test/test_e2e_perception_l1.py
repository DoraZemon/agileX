"""End-to-end simulation checks of touching L1 blocks (perception, grasp, full runs)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.camera.factory import make_camera
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.extrinsics import camera_pose_in_base, extrinsics_from_cfg
from cube_task.core.logger import RunLogger
from cube_task.perception.factory import make_perception
from cube_task.robot.factory import make_arm
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from cube_task.sim.world import SimObject, SimWorld
from cube_task.task.runner import TaskRunner

BASE = load_config(use_overlays=False)


def _setup(world, info):
    cfg = Config(deep_merge(BASE.data, scenario_config_overlay(info)), BASE.root, BASE.sources)
    clock = SimClock()
    arm = make_arm(cfg, clock, sim_world=world)
    arm.connect()
    arm.enable()
    cam = make_camera(cfg, clock, sim_world=world, arm=arm)
    cam.start()
    return cfg, clock, arm, cam


def _observe(world, info):
    cfg, clock, arm, cam = _setup(world, info)
    arm.move_joints(cfg.get('robot.poses.observe_joints'))
    frame = cam.get_frame()
    T_bf = arm.get_flange_pose().value if info['mount'] == 'eye_in_hand' else None
    T = camera_pose_in_base(extrinsics_from_cfg(cfg), T_bf)
    return make_perception(cfg, 'L1').process(frame, T)


def _same_color(world, info):
    """Recolor the touching green cube red: two red blocks touching flush."""
    objs = []
    for o in world.objects:
        if o.label == 'green':
            red = next(q for q in world.objects if q.label == 'red')
            o = SimObject('red2_cube', 'red', 'box', o.size, red.color_bgr, o.T_base_obj)
        objs.append(o)
    return SimWorld(table_z=0.0, objects=objs, zones=world.zones, seed=0), dict(info)


def _errors(res, world):
    """Return {object name: (center error mm, label)} of the nearest valid detection."""
    out = {}
    for o in world.objects:
        best = None
        for d in res.valid_detections():
            e = 1000.0 * math.hypot(*(np.asarray(d.center_base[:2]) - o.center_xy()))
            if best is None or e < best[0]:
                best = (e, d.label)
        out[o.name] = best
    return out


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
@pytest.mark.parametrize('seed', [0, 4])
def test_touching_blocks_are_localized(mount, seed):
    world, info = make_scenario('L1', seed, mount=mount, difficulty='hard')
    res = _observe(world, info)
    assert len(res.valid_detections()) == 4
    for o in world.objects:
        err, label = _errors(res, world)[o.name]
        assert err < 5.0, (o.name, err)
        assert label == o.label
    assert sum(1 for d in res.detections if d.extra.get('split') == 'color') == 2
    same_world, same_info = _same_color(world, info)
    res = _observe(same_world, same_info)
    for o in same_world.objects:
        err, label = _errors(res, same_world)[o.name]
        assert err < 5.0 and label == o.label, (o.name, err, label)
    assert res.timings['total'] < 150.0


@pytest.mark.parametrize('mount,seed',
                         [('eye_to_hand', 4), ('eye_to_hand', 23), ('eye_in_hand', 4)])
def test_hard_l1_run_places_all_blocks(mount, seed, tmp_path):
    world, info = make_scenario('L1', seed, mount=mount, difficulty='hard')
    cfg, clock, arm, cam = _setup(world, info)
    log = RunLogger(str(tmp_path), 'L1', clock, save_images=False)
    rep = TaskRunner(cfg, 'L1', arm, cam, make_perception(cfg, 'L1'), clock, log).run()
    assert rep.ok and rep.placed == 3 and rep.attempts == 3, rep.stop_reason
    assert world.score(info['class_to_zone']) == {'correct': 3, 'wrong': 0, 'remaining': 0}
    assert any(o.label == 'yellow' for o in world.remaining(info['pick_polygon']))

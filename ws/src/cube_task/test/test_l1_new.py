"""perception.backend: l1_new (piper_vision L1 detector behind the Perception interface)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.camera.factory import make_camera
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.extrinsics import camera_pose_in_base, extrinsics_from_cfg
from cube_task.perception.factory import make_perception
from cube_task.perception.pipeline import Perception
from cube_task.robot.factory import make_arm
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay

try:
    from cube_task.perception.l1_new import _import_piper_vision
    _import_piper_vision()
except ImportError as exc:  # pragma: no cover - only without ws/src/piper_vision
    pytest.skip('piper_vision not available: %s' % exc, allow_module_level=True)

BASE = load_config(use_overlays=False)


def _cfg(info, backend):
    cfg = Config(deep_merge(BASE.data, scenario_config_overlay(info)), BASE.root, BASE.sources)
    cfg.set('perception.backend', backend)
    cfg.set('perception.l1_block_size', 0.05)
    return cfg


def _observe(world, info, cfg):
    clock = SimClock()
    arm = make_arm(cfg, clock, sim_world=world)
    arm.connect()
    arm.enable()
    cam = make_camera(cfg, clock, sim_world=world, arm=arm)
    cam.start()
    arm.move_joints(cfg.get('robot.poses.observe_joints'))
    frame = cam.get_frame()
    T_bf = arm.get_flange_pose().value if info['mount'] == 'eye_in_hand' else None
    return make_perception(cfg, 'L1').process(frame, camera_pose_in_base(
        extrinsics_from_cfg(cfg), T_bf))


def test_backend_switch():
    _, info = make_scenario('L1', 0)
    assert isinstance(make_perception(_cfg(info, 'pipeline'), 'L1'), Perception)
    p = make_perception(_cfg(info, 'l1_new'), 'L1')
    assert type(p).__name__ == 'L1NewPerception'
    # other tasks keep the pipeline even with the switch set
    cfg = _cfg(info, 'l1_new')
    cfg.set('tasks.L2.classifier', 'color')
    assert isinstance(make_perception(cfg, 'L2'), Perception)


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_detections_have_planner_fields(mount):
    world, info = make_scenario('L1', 1, mount=mount)
    res = _observe(world, info, _cfg(info, 'l1_new'))
    valid = res.valid_detections()
    targets = [o for o in world.objects if o.label in ('red', 'green', 'yellow')]
    assert len(valid) == len(targets)
    assert res.table is not None and res.table.frame == 'base'
    for o in targets:
        d = min(valid, key=lambda d: math.hypot(*(d.center_base[:2] - o.center_xy())))
        assert d.label == o.label
        assert math.hypot(*(d.center_base[:2] - o.center_xy())) < 0.004
        assert d.extra['table_z'] == pytest.approx(0.0, abs=0.004)
        assert d.extra['_points_base'].shape[1] == 3
        assert -math.pi / 2 <= d.yaw < math.pi / 2
        assert d.length >= d.width


def test_never_raises_on_bad_frame():
    _, info = make_scenario('L1', 0)
    p = make_perception(_cfg(info, 'l1_new'), 'L1')

    class F:
        color = None
        depth = None
        intrinsics = None
        stamp = 0.0
    assert p.process(F(), np.eye(4)).warnings

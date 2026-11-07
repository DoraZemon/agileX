"""Tests for task/selfcheck.py."""
from __future__ import annotations

from cube_task.core.clock import SimClock
from cube_task.task.selfcheck import self_check
import pytest
from test_task_fakes import (default_camera_T, FakeCamera, FakeExt, FakeSafeArm,
                             install_fake_modules, make_cfg)


@pytest.fixture(autouse=True)
def fake_modules(monkeypatch):
    install_fake_modules(monkeypatch)


def _fails(msgs):
    return [m for m in msgs if m.startswith('[FAIL]')]


def _has(msgs, text):
    return any(text in m for m in _fails(msgs))


def test_good_sim_config_passes():
    ok, msgs = self_check(make_cfg(), 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert ok, msgs
    assert any(m.startswith('[ OK ]') for m in msgs)


def test_lists_every_problem_at_once(monkeypatch):
    install_fake_modules(monkeypatch, ext_from_cfg=lambda cfg: None)
    cfg = make_cfg()
    cfg.set('zones', {})
    cfg.set('robot.poses.observe_joints', None)
    cfg.set('robot.poses.observe_tcp', None)
    cfg.set('robot.tcp_offset', [0, 0, 0.3, 0, 0, 0])
    ok, msgs = self_check(cfg, 'L1', mode='real')
    assert not ok
    assert _has(msgs, 'no zones configured')
    assert _has(msgs, "label red -> zone 'zone_red' which is not in zones")
    assert _has(msgs, 'no observe pose')
    assert _has(msgs, 'tcp_offset z = 0.300')
    assert _has(msgs, 'no extrinsics')


def test_missing_zone_for_mapped_label_and_unmapped_label_warning():
    cfg = make_cfg()
    cfg.set('tasks.L1.class_to_zone', {'red': 'zone_red', 'green': 'zone_nowhere'})
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert not ok and _has(msgs, 'zone_nowhere')
    assert any(m.startswith('[WARN]') and 'blue has no zone' in m for m in msgs)


def test_extrinsics_mount_mismatch_and_loader_error(monkeypatch):
    cfg = make_cfg()
    ok, msgs = self_check(cfg, 'L1', mode='sim',
                          extrinsics=FakeExt(default_camera_T(), 'eye_in_hand'))
    assert not ok and _has(msgs, 'mount')

    def broken(cfg):
        raise ValueError('bad yaml')
    install_fake_modules(monkeypatch, ext_from_cfg=broken)
    ok, msgs = self_check(cfg, 'L1', mode='sim')
    assert not ok and _has(msgs, 'bad yaml')


def test_observe_tcp_over_pick_area_fails_for_eye_to_hand_real():
    cfg = make_cfg()
    cfg.set('robot.poses.observe_joints', None)        # default observe_tcp is over the area
    ok, msgs = self_check(cfg, 'L1', mode='real', extrinsics=FakeExt(default_camera_T()))
    assert not ok and _has(msgs, 'camera view')
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert ok, msgs


def test_zone_outside_workspace_and_classifier_assets():
    cfg = make_cfg()
    cfg.set('zones.zone_far', {'center': [0.9, 0.0, 0.0]})
    cfg.set('perception.classifier.knn.model_file', 'models/does_not_exist.npz')
    ok, msgs = self_check(cfg, 'L2', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert _has(msgs, 'zone zone_far center outside the workspace')
    assert _has(msgs, 'knn model file missing')
    cfg2 = make_cfg()
    cfg2.set('tasks.L1.labels', ['red', 'pink'])
    ok, msgs = self_check(cfg2, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert _has(msgs, 'no HSV ranges for: pink')


def test_hardware_checks_in_real_mode():
    clock = SimClock()
    arm = FakeSafeArm(clock)
    cam = FakeCamera(clock)
    ext = FakeExt(default_camera_T())
    ok, msgs = self_check(make_cfg(), 'L1', arm=arm, camera=cam, mode='real', extrinsics=ext,
                          clock=clock)
    assert ok, msgs
    assert any('valid depth in the pick area' in m for m in msgs)
    arm.healthy = False
    arm.no_width = True
    cam.depth_value = 0.0
    ok, msgs = self_check(make_cfg(), 'L1', arm=arm, camera=cam, mode='real', extrinsics=ext)
    assert not ok
    assert _has(msgs, 'arm not healthy') and _has(msgs, 'gripper width not readable')
    assert _has(msgs, 'valid depth')
    cam.none_frames = 5
    ok, msgs = self_check(make_cfg(), 'L1', arm=arm, camera=cam, mode='real', extrinsics=ext)
    assert _has(msgs, 'no camera frame')
    # sim mode skips hardware checks
    ok, msgs = self_check(make_cfg(), 'L1', arm=arm, camera=cam, mode='sim', extrinsics=ext)
    assert ok, msgs


def test_unreachable_zone_fails():
    clock = SimClock()
    arm = FakeSafeArm(clock)
    arm.reach_fn = lambda T: (T[1, 3] < 0.25, 'out of reach')
    ok, msgs = self_check(make_cfg(), 'L1', arm=arm, mode='sim',
                          extrinsics=FakeExt(default_camera_T()))
    assert not ok and _has(msgs, 'zone zone_red release not reachable')


def test_unknown_task():
    ok, msgs = self_check(make_cfg(), 'L9', mode='sim')
    assert not ok and _has(msgs, 'unknown task')

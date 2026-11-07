"""Tests for sim/sim_camera.py: stamping, eye-in-hand pose and fault injection."""
from __future__ import annotations

import numpy as np
import pytest

from cube_task.core.clock import SimClock
from cube_task.core.interfaces import CameraSource
from cube_task.core.transforms import make_T, top_down_T
from cube_task.core.types import CameraIntrinsics, Stamped
from cube_task.sim.scenarios import look_at
from cube_task.sim.sim_camera import FRAME_PERIOD_S, SimCamera
from cube_task.sim.world import SimObject, SimWorld

INTR = CameraIntrinsics(95.0, 95.0, 79.5, 59.5, 160, 120)
T_CAM = look_at((0.30, 0.0, 0.50), (0.30, 0.0, 0.0), down_hint=(-1.0, 0.0, 0.0))


class FakeArm:
    """Minimal stand-in exposing get_flange_pose()."""

    def __init__(self, clock, T=None):
        self.clock = clock
        self.T = T

    def get_flange_pose(self):
        if self.T is None:
            return None
        return Stamped(self.T.copy(), self.clock.now())


def world():
    c = SimObject('c', 'red', 'box', (0.04, 0.04, 0.04), (40, 40, 200), np.eye(4))
    c.place_upright(0.30, 0.0, 0.0, 0.0)
    return SimWorld(objects=[c])


def make(clock=None, **kw):
    clock = clock or SimClock()
    cam = SimCamera(world(), clock, INTR, T_base_camera=T_CAM, **kw)
    cam.start()
    return cam, clock


def test_is_camera_source_and_meta():
    cam, clock = make()
    assert isinstance(cam, CameraSource)
    f = cam.get_frame()
    assert f.source == 'sim' and f.stamp == clock.now()
    assert f.color.shape == (120, 160, 3) and f.depth.dtype == np.float32
    assert np.allclose(f.meta['T_base_camera_true'], T_CAM)
    assert cam.intrinsics() is INTR


def test_newer_than_advances_clock():
    cam, clock = make()
    f1 = cam.get_frame()
    f2 = cam.get_frame(newer_than=f1.stamp)
    assert f2.stamp > f1.stamp
    assert f2.stamp - f1.stamp == pytest.approx(FRAME_PERIOD_S)
    clock.advance(0.5)
    t = clock.now()
    f3 = cam.get_frame(newer_than=f2.stamp)
    assert f3.stamp == t                                  # already newer: no wait
    f4 = cam.get_frame(newer_than=t + 0.1, timeout=1.0)
    assert f4.stamp > t + 0.1 and f4.stamp - t < 0.1 + 2 * FRAME_PERIOD_S


def test_newer_than_beyond_timeout_returns_none():
    cam, clock = make()
    t = clock.now()
    assert cam.get_frame(timeout=0.2, newer_than=t + 5.0) is None
    assert clock.now() == pytest.approx(t + 0.2)


def test_not_started_returns_none():
    cam = SimCamera(world(), SimClock(), INTR, T_base_camera=T_CAM)
    assert cam.get_frame() is None
    cam.start()
    assert cam.get_frame() is not None
    cam.stop()
    cam.stop()
    assert cam.get_frame() is None


def test_fault_injection():
    cam, clock = make()
    f1 = cam.get_frame()
    cam.inject('none_frames', 2)
    cam.inject('stale', 1)
    cam.inject('no_depth', 1)
    t = clock.now()
    assert cam.get_frame(timeout=0.3) is None
    assert cam.get_frame(timeout=0.3) is None
    assert clock.now() == pytest.approx(t + 0.6)
    stale = cam.get_frame(newer_than=f1.stamp)
    assert stale is f1 and stale.stamp == f1.stamp
    nd = cam.get_frame(newer_than=f1.stamp)
    assert nd.depth is None and nd.stamp > f1.stamp and nd.meta['fault'] == 'no_depth'
    ok = cam.get_frame(newer_than=nd.stamp)
    assert ok.depth is not None and ok.meta['fault'] == ''
    cam.inject('none_frames', 3)
    cam.clear_faults()
    assert cam.get_frame() is not None
    with pytest.raises(ValueError):
        cam.inject('explode', 1)
    with pytest.raises(ValueError):
        cam.inject('stale', 0)


def test_eye_in_hand_follows_flange():
    clock = SimClock()
    T_flange = top_down_T(0.30, 0.0, 0.40, 0.0)
    arm = FakeArm(clock, T_flange)
    T_fc = make_T(np.eye(3), (0.0, 0.0, 0.03))
    cam = SimCamera(world(), clock, INTR, T_flange_camera=T_fc, arm=arm, noise_std=0.0,
                    dropout=0.0)
    cam.start()
    f = cam.get_frame()
    assert np.allclose(f.meta['T_base_camera_true'], T_flange @ T_fc)
    # Camera at z = 0.37 looking down; the cube top (z 0.04) is at 0.33 m.
    assert f.depth[59, 79] == pytest.approx(0.33, abs=1e-3)
    arm.T = top_down_T(0.30, 0.0, 0.50, 0.0)
    f2 = cam.get_frame(newer_than=f.stamp)
    assert f2.depth[59, 79] == pytest.approx(0.43, abs=1e-3)
    arm.T = None
    assert cam.get_frame(newer_than=f2.stamp) is None
    arm.T = np.full((4, 4), np.nan)
    assert cam.get_frame() is None


def test_constructor_validation_and_determinism():
    with pytest.raises(ValueError):
        SimCamera(world(), SimClock(), INTR)
    with pytest.raises(ValueError):
        SimCamera(world(), SimClock(), INTR, T_base_camera=T_CAM, T_flange_camera=np.eye(4))
    with pytest.raises(ValueError):
        SimCamera(world(), SimClock(), INTR, T_flange_camera=np.eye(4))
    with pytest.raises(ValueError):
        SimCamera(world(), SimClock(), INTR, T_base_camera=T_CAM, dropout=2.0)
    a, _ = make(seed=3)
    b, _ = make(seed=3)
    fa, fb = a.get_frame(), b.get_frame()
    assert np.array_equal(fa.depth, fb.depth) and np.array_equal(fa.color, fb.color)

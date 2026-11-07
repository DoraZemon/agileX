"""Tests for camera/utils.py (median depth, capture_averaged)."""
from __future__ import annotations

import numpy as np
import pytest

from cube_task.camera.utils import capture_averaged, median_depth, valid_depth_mask
from cube_task.core.clock import SimClock
from cube_task.core.interfaces import CameraSource
from cube_task.core.types import CameraIntrinsics, Frame

INTR = CameraIntrinsics(fx=500.0, fy=500.0, cx=3.5, cy=2.5, width=8, height=6)


class FakeCamera(CameraSource):
    """Serve a fixed list of depth images; each frame is stamped 1/30 s apart."""

    name = 'fake'

    def __init__(self, depths, clock, fail_after=None):
        self.depths = list(depths)
        self.clock = clock
        self.calls = []
        self.i = 0
        self.fail_after = fail_after

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self, timeout=1.0, newer_than=None):
        self.calls.append((timeout, newer_than))
        if self.fail_after is not None and self.i >= self.fail_after:
            return None
        self.clock.advance(1.0 / 30.0)
        d = self.depths[self.i % len(self.depths)]
        color = np.full((6, 8, 3), self.i, np.uint8)
        self.i += 1
        return Frame(color=color, depth=None if d is None else np.array(d, np.float32),
                     intrinsics=INTR, stamp=self.clock.now(), source='fake',
                     meta={'i': self.i - 1})


def _d(v):
    return np.full((6, 8), v, np.float32)


def test_median_depth_rules():
    a, b, c = _d(0.50), _d(0.52), _d(0.90)
    a[0, 0] = 0.0
    b[0, 0] = np.nan
    a[1, 1] = np.inf
    b[1, 1] = 0.0
    c[1, 1] = 0.0                             # no valid sample at all
    c[2, 2] = -1.0                            # two valid samples -> mean of the two
    m = median_depth([a, b, c])
    assert m.dtype == np.float32
    assert m[3, 3] == pytest.approx(0.52)     # median of three
    assert m[0, 0] == 0.0                     # one valid of three < ceil(3/2) = 2
    assert m[1, 1] == 0.0
    assert m[2, 2] == pytest.approx(0.51)
    assert np.array_equal(median_depth([a]), np.where(valid_depth_mask(a), a, 0))
    assert median_depth([a, b, c], min_valid=1)[0, 0] == pytest.approx(0.90)
    with pytest.raises(ValueError):
        median_depth([])


def test_capture_averaged_median_and_freshness():
    clock = SimClock(start=10.0)
    d1, d2, d3 = _d(0.60), _d(0.61), _d(0.75)
    d2[2, 3] = 0.0
    d3[2, 3] = np.nan                         # only one valid sample of three
    cam = FakeCamera([d1, d2, d3], clock)
    f = capture_averaged(cam, 3, newer_than=10.0, timeout=0.5)
    assert f is not None
    assert f.meta['frames_averaged'] == 3 and f.meta['i'] == 2
    assert f.color[0, 0, 0] == 2              # color of the last frame
    assert f.stamp == pytest.approx(10.1)
    assert np.allclose(np.delete(f.depth.ravel(), 2 * 8 + 3), 0.61)
    assert f.depth[2, 3] == 0.0
    # Each request asks for a frame newer than the previous one, with the same timeout.
    assert cam.calls[0] == (0.5, 10.0)
    assert cam.calls[1][1] == pytest.approx(10.0 + 1 / 30.0)
    assert cam.calls[2][1] == pytest.approx(10.0 + 2 / 30.0)
    assert f.meta['averaged_stamps'] == pytest.approx([10 + 1 / 30, 10 + 2 / 30, 10.1])


def test_capture_averaged_single_partial_and_none():
    clock = SimClock()
    cam = FakeCamera([_d(0.5)], clock)
    f = capture_averaged(cam, 1)
    assert f is not None and 'frames_averaged' not in f.meta and len(cam.calls) == 1
    cam = FakeCamera([_d(0.5), _d(0.7)], clock, fail_after=2)
    f = capture_averaged(cam, 5)
    assert f.meta['frames_averaged'] == 2
    assert np.allclose(f.depth, 0.6)
    cam = FakeCamera([_d(0.5)], clock, fail_after=0)
    assert capture_averaged(cam, 3) is None


def test_capture_averaged_without_depth():
    cam = FakeCamera([None], SimClock())
    f = capture_averaged(cam, 3)
    assert f.depth is None and f.meta['frames_averaged'] == 3


def test_capture_averaged_rejects_stale_frames():
    class Stale(FakeCamera):
        def get_frame(self, timeout=1.0, newer_than=None):
            f = super().get_frame(timeout, newer_than)
            f.stamp = 1.0                    # violates the newer_than contract
            return f

    cam = Stale([_d(0.5)], SimClock())
    f = capture_averaged(cam, 3, newer_than=None)
    assert f.meta['frames_averaged'] == 1

"""Tests for camera/registration.py on a ray-cast synthetic scene."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from cube_task.camera.registration import register_depth_to_color
from cube_task.core.transforms import make_T, rot_y
from cube_task.core.types import CameraIntrinsics


def _intr(w, h, f):
    return CameraIntrinsics(fx=f, fy=f, cx=(w - 1) / 2.0 + 0.3, cy=(h - 1) / 2.0 - 0.2,
                            width=w, height=h)


def _render(intr, T_color_cam):
    """
    Depth (z in the camera frame) of a tilted table plane plus a floating square plate.

    The scene is defined in the color camera frame: table z = 0.65 + 0.1 y,
    plate z = 0.5 for |x|, |y| < 5 cm.
    """
    u, v = np.meshgrid(np.arange(intr.width, dtype=float), np.arange(intr.height, dtype=float))
    d_cam = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], -1)
    R, o = T_color_cam[:3, :3], T_color_cam[:3, 3]
    d = d_cam @ R.T
    best = np.full(u.shape, np.inf)
    # Table: -0.1 y + z = 0.65.
    n = np.array([0.0, -0.1, 1.0])
    s = (0.65 - n @ o) / (d @ n)
    best = np.where(s > 0, np.minimum(best, s), best)
    # Plate: z = 0.5 within the square.
    s = (0.5 - o[2]) / d[..., 2]
    p = o + s[..., None] * d
    hit = (s > 0) & (np.abs(p[..., 0]) < 0.05) & (np.abs(p[..., 1]) < 0.05)
    best = np.where(hit, np.minimum(best, s), best)
    return np.where(np.isfinite(best), best, 0.0).astype(np.float32)


def _edge_mask(gt, px=3):
    gy, gx = np.gradient(gt.astype(float))
    edges = (np.abs(gx) > 0.005) | (np.abs(gy) > 0.005)
    return cv2.dilate(edges.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


# Depth camera 2.5 cm to the side of the color camera, slightly rotated.
T_COLOR_DEPTH = make_T(rot_y(math.radians(0.7)), [0.025, 0.001, -0.002])


def _check(reg, gt, border=30):
    edges = _edge_mask(gt)
    inner = np.zeros(gt.shape, bool)
    inner[border:-border, border:-border] = True
    sel = inner & ~edges
    valid = reg > 0
    err = np.abs(reg - gt)
    assert np.mean(valid[sel]) > 0.995, 'coverage %.4f' % np.mean(valid[sel])
    assert np.max(err[sel & valid]) < 0.002, np.max(err[sel & valid])
    # Even at edges and borders, a valid pixel never lies behind the true surface by much
    # in the plate interior (no background bleeding into the object).
    plate = (gt > 0) & (gt < 0.55) & ~edges
    assert np.all(err[plate & valid] < 0.002)


def test_same_resolution_baseline():
    intr = _intr(320, 240, 300.0)
    depth_d = _render(intr, T_COLOR_DEPTH)
    gt = _render(intr, np.eye(4))
    reg = register_depth_to_color(depth_d, intr, intr, T_COLOR_DEPTH)
    assert reg.shape == (240, 320) and reg.dtype == np.float32
    _check(reg, gt)
    # Without registration the plate would be shifted by ~15 px: the test is meaningful.
    sel = ~_edge_mask(gt)
    assert np.mean(np.abs(depth_d - gt)[sel] > 0.01) > 0.02


def test_occlusion_shadow_stays_invalid_or_correct():
    intr = _intr(320, 240, 300.0)
    depth_d = _render(intr, T_COLOR_DEPTH)
    gt = _render(intr, np.eye(4))
    reg = register_depth_to_color(depth_d, intr, intr, T_COLOR_DEPTH)
    valid = reg > 0
    # Wrong values are only allowed next to depth edges (z-buffer + hole fill).
    bad = valid & (np.abs(reg - gt) > 0.002)
    assert not np.any(bad & ~_edge_mask(gt, px=2))


def test_lower_resolution_depth_is_upsampled():
    intr_c = _intr(320, 240, 300.0)
    intr_d = _intr(160, 120, 150.0)
    depth_d = _render(intr_d, T_COLOR_DEPTH)
    gt = _render(intr_c, np.eye(4))
    reg = register_depth_to_color(depth_d, intr_d, intr_c, T_COLOR_DEPTH)
    assert reg.shape == (240, 320)
    edges = _edge_mask(gt, px=4)
    sel = np.zeros(gt.shape, bool)
    sel[30:-30, 30:-30] = True
    sel &= ~edges
    assert np.mean(reg[sel] > 0) > 0.99
    assert np.max(np.abs(reg - gt)[sel & (reg > 0)]) < 0.002


def test_distortion_identity_round_trip():
    intr = _intr(320, 240, 300.0)
    intr.dist = np.array([0.08, -0.03, 0.001, -0.0007, 0.0])
    yy, xx = np.mgrid[0:240, 0:320]
    depth = (0.6 + 0.0005 * xx + 0.0003 * yy).astype(np.float32)
    depth[100:110, 100:110] = 0.0
    reg = register_depth_to_color(depth, intr, intr, np.eye(4), fill_holes=False)
    inner = np.zeros(depth.shape, bool)
    inner[5:-5, 5:-5] = True
    assert np.mean(reg[inner] == depth[inner]) > 0.999
    assert np.all(reg[100:110, 100:110] == 0.0)


def test_hole_fill_and_invalid_input():
    intr = _intr(80, 60, 80.0)
    depth = np.full((60, 80), 0.7, np.float32)
    depth[30, 40] = np.nan
    depth[10, 10] = 0.0
    depth[:, :3] = np.inf
    reg = register_depth_to_color(depth, intr, intr, np.eye(4))
    assert reg[30, 40] == pytest.approx(0.7) and reg[10, 10] == pytest.approx(0.7)
    assert np.all(reg[:, :2] == 0.0)
    assert np.all(register_depth_to_color(np.zeros((60, 80)), intr, intr, np.eye(4)) == 0)


def test_errors():
    intr = _intr(80, 60, 80.0)
    with pytest.raises(ValueError, match='does not match'):
        register_depth_to_color(np.zeros((61, 80)), intr, intr, np.eye(4))
    with pytest.raises(ValueError, match='translation'):
        register_depth_to_color(np.zeros((60, 80)), intr, intr, make_T(np.eye(3), [0.5, 0, 0]))
    bad = _intr(80, 60, 80.0)
    bad.fx = 0.0
    with pytest.raises(ValueError, match='intrinsics invalid'):
        register_depth_to_color(np.zeros((60, 80)), bad, intr, np.eye(4))

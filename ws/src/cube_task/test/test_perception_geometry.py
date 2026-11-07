"""Tests for cube_task.perception.geometry."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.core.transforms import make_T, rpy_to_rot, transform_points
from cube_task.core.types import CameraIntrinsics, Zone
from cube_task.perception import geometry as G

INTR = CameraIntrinsics(fx=600.0, fy=605.0, cx=320.0, cy=240.0, width=640, height=480)
DIST = np.array([0.12, -0.25, 0.001, -0.0015, 0.08])


def _intr_dist() -> CameraIntrinsics:
    return CameraIntrinsics(600.0, 605.0, 320.0, 240.0, 640, 480, DIST.copy())


def test_valid_depth_mask_handles_invalid_values():
    d = np.array([[0.5, 0.0, np.nan], [np.inf, 0.1, 3.5], [0.15, 3.0, -1.0]], np.float32)
    m = G.valid_depth_mask(d, 0.15, 3.0)
    assert m.tolist() == [[True, False, False], [False, False, False], [True, True, False]]


def test_backproject_plane_and_invalid_pixels():
    depth = np.full((480, 640), 0.8, np.float32)
    depth[10, 20] = 0.0
    depth[11, 21] = np.nan
    depth[12, 22] = np.inf
    pts = G.backproject(depth, INTR)
    assert pts.dtype == np.float32 and pts.shape == (480, 640, 3)
    for v, u in ((10, 20), (11, 21), (12, 22)):
        assert np.all(np.isnan(pts[v, u]))
    u, v = 100, 300
    exp = [(u - INTR.cx) / INTR.fx * 0.8, (v - INTR.cy) / INTR.fy * 0.8, 0.8]
    assert np.allclose(pts[v, u], exp, atol=1e-6)


def test_backproject_rejects_bad_inputs():
    with pytest.raises(ValueError):
        G.backproject(np.zeros((480, 640), np.uint16), INTR)
    with pytest.raises(ValueError):
        G.backproject(np.zeros((240, 320), np.float32), INTR)
    with pytest.raises(ValueError):
        G.backproject(np.zeros((480, 640, 1), np.float32), INTR)


def test_distortion_round_trip_and_cache():
    intr = _intr_dist()
    rng = np.random.default_rng(3)
    depth = rng.uniform(0.3, 1.5, size=(480, 640)).astype(np.float32)
    pts = G.backproject(depth, intr)
    T = np.eye(4)
    v = rng.integers(0, 480, 400)
    u = rng.integers(0, 640, 400)
    uv = G.project_points(pts[v, u].astype(np.float64), T, intr)
    err = np.hypot(uv[:, 0] - u, uv[:, 1] - v)
    assert float(err.max()) < 0.02
    a = G.normalized_grid(intr)
    b = G.normalized_grid(_intr_dist())
    assert a[0] is b[0]
    assert not a[0].flags.writeable
    # Undistorted rays differ from the pinhole rays near the corner.
    xn0 = (0 - intr.cx) / intr.fx
    assert abs(float(a[0][0, 0]) - xn0) > 0.005


def test_project_points_with_pose_and_behind_camera():
    T = make_T(rpy_to_rot(math.pi, 0.0, 0.3), (0.3, 0.1, 0.6))  # looking down
    p_cam = np.array([[0.02, -0.03, 0.5], [0.0, 0.0, -0.2], [np.nan, 0.0, 1.0]])
    p_base = transform_points(T, p_cam)
    uv = G.project_points(p_base, T, INTR)
    assert np.allclose(uv[0], [600 * 0.04 + 320, 605 * -0.06 + 240], atol=1e-6)
    assert np.all(np.isnan(uv[1])) and np.all(np.isnan(uv[2]))
    assert G.project_points(p_base[0], T, INTR).shape == (1, 2)


def test_transform_cloud_matches_core_and_keeps_nan():
    rng = np.random.default_rng(0)
    T = make_T(rpy_to_rot(0.3, -0.2, 1.0), (0.1, -0.2, 0.5))
    p = rng.normal(size=(5, 7, 3)).astype(np.float32)
    p[1, 2] = np.nan
    out = G.transform_cloud(p, T)
    assert out.shape == p.shape and out.dtype == np.float32
    ref = transform_points(T, p.reshape(-1, 3).astype(np.float64)).reshape(p.shape)
    ok = np.isfinite(ref)
    assert np.allclose(out[ok], ref[ok], atol=1e-5)
    assert np.all(np.isnan(out[1, 2]))
    with pytest.raises(ValueError):
        G.transform_cloud(p, np.eye(3))


def test_points_in_polygon_convex_concave_and_margin():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    xy = np.array([[0.5, 0.5], [1.05, 0.5], [-0.2, 0.5], [0.97, 0.5], [np.nan, 0.5]])
    assert G.points_in_polygon(xy, sq).tolist() == [True, False, False, True, False]
    assert G.points_in_polygon(xy, sq, margin=0.1).tolist() == [True, True, False, True, False]
    inner = G.points_in_polygon(xy, sq, margin=-0.05)
    assert inner.tolist() == [True, False, False, False, False]
    # Clockwise order and a concave L shape.
    ell = np.array([[0, 0], [0, 2], [1, 2], [1, 1], [2, 1], [2, 0]], float)
    q = np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 0.5], [0.5, 0.5]])
    assert G.points_in_polygon(q, ell).tolist() == [True, False, True, True]


def test_points_in_zones_matches_zone_contains():
    zones = {'a': Zone('a', np.array([0.3, 0.2, 0.0]), (0.12, 0.08), yaw=0.6),
             'b': Zone('b', np.array([0.3, -0.2, 0.0]), (0.10, 0.10))}
    rng = np.random.default_rng(1)
    xy = rng.uniform([0.15, -0.35], [0.45, 0.35], size=(2000, 2))
    for margin in (0.0, 0.02):
        got = G.points_in_zones(xy, zones, margin)
        exp = [any(z.contains_xy(x, y, margin) for z in zones.values()) for x, y in xy]
        assert got.tolist() == exp
    assert not G.points_in_zones(xy, None).any()


def test_parse_polygon_validation():
    assert G.parse_polygon(None) is None
    assert G.parse_polygon([]) is None
    assert G.parse_polygon([[0, 0], [1, 0], [1, 1]]).shape == (3, 2)
    for bad in ([[0, 0], [1, 1]], [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [float('nan'), 1]],
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]]):
        with pytest.raises(ValueError):
            G.parse_polygon(bad)


def test_densify_polyline_closes_loop():
    sq = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
    d = G.densify_polyline(sq, step=0.01)
    assert np.allclose(d[0], d[-1])
    assert d.shape[0] == 41

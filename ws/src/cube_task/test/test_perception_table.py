"""Tests for cube_task.perception.table (RANSAC table plane)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.types import Zone
from cube_task.perception.table import fit_table_plane, fit_table_plane_detailed, TableParams

POLY = np.array([[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]])


def _cfg(**over):
    cfg = load_config(use_overlays=False)
    cfg.set('perception.roi.pick_polygon', POLY.tolist())
    for k, v in over.items():
        cfg.set('perception.table.' + k, v)
    return cfg


def _plane_z(x, y, tilt_x_deg, z0):
    return z0 + math.tan(math.radians(tilt_x_deg)) * (x - 0.3)


def _scene(seed=0, tilt_deg=2.0, z0=0.012, occl=0.4, outliers=0.1, n=30000, noise=0.001):
    """Table points, 40 % object tops 3-8 cm above, 10 % outliers, wider than the ROI."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0.05, -0.35], [0.55, 0.35], size=(n, 2))
    z = _plane_z(xy[:, 0], xy[:, 1], tilt_deg, z0) + rng.normal(0, noise, n)
    k = rng.random(n)
    obj = k < occl
    # Object tops at a few discrete heights (coplanar tops are the hard case).
    z[obj] += rng.choice([0.03, 0.045, 0.08], size=int(obj.sum()))
    out = (k >= occl) & (k < occl + outliers)
    z[out] = rng.uniform(-0.2, 0.4, size=int(out.sum()))
    return np.column_stack([xy, z])


def _true_normal(tilt_deg):
    t = math.tan(math.radians(tilt_deg))
    n = np.array([-t, 0.0, 1.0])
    return n / np.linalg.norm(n)


def test_fit_with_occlusion_and_outliers():
    for seed in range(3):
        pts = _scene(seed)
        fit = fit_table_plane_detailed(pts, _cfg(), np.random.default_rng(seed))
        p = fit.plane
        assert p is not None, fit.reason
        ang = math.degrees(math.acos(float(np.dot(p.normal, _true_normal(2.0)))))
        assert ang < 0.3
        assert abs(p.z_at(0.3, 0.0) - 0.012) < 0.001
        assert p.normal[2] > 0 and p.frame == 'base'
        assert 0.4 < p.inlier_ratio < 0.6
        assert p.rms < 0.002


def test_points_outside_roi_and_in_zones_are_ignored():
    pts = _scene(1)
    # A large step outside the ROI (> 5 cm margin) and a raised tray inside a zone.
    far = pts[:, 0] > 0.51
    pts[far, 2] += 0.10
    zone = Zone('tray', np.array([0.30, 0.15, 0.0]), (0.12, 0.12))
    tray = np.array([zone.contains_xy(x, y, 0.02) for x, y in pts[:, :2]])
    pts[tray, 2] = 0.06
    assert tray.sum() > 1000
    p = fit_table_plane(pts, _cfg(), np.random.default_rng(0), zones={'tray': zone})
    assert p is not None
    assert abs(p.z_at(0.3, 0.0) - 0.012) < 0.001


def test_deterministic_with_seed():
    pts = _scene(2)
    a = fit_table_plane(pts, _cfg(), np.random.default_rng(5))
    b = fit_table_plane(pts, _cfg(), np.random.default_rng(5))
    assert np.array_equal(a.normal, b.normal) and a.d == b.d


def test_rejects_tilted_plane():
    pts = _scene(0, tilt_deg=30.0, occl=0.0, outliers=0.0)
    fit = fit_table_plane_detailed(pts, _cfg(max_tilt_deg=20.0), np.random.default_rng(0))
    assert fit.plane is None and fit.reason == 'tilt'
    fit = fit_table_plane_detailed(pts, _cfg(max_tilt_deg=40.0), np.random.default_rng(0))
    assert fit.plane is not None


def test_rejects_unexpected_z_and_low_inlier_ratio():
    pts = _scene(0)
    fit = fit_table_plane_detailed(pts, _cfg(expected_z=0.08, expected_z_tol=0.02),
                                   np.random.default_rng(0))
    assert fit.plane is None and fit.reason == 'unexpected_z'
    assert fit.candidate is not None and abs(fit.z_ref - 0.012) < 0.002
    ok = fit_table_plane(pts, _cfg(expected_z=0.0, expected_z_tol=0.02),
                         np.random.default_rng(0))
    assert ok is not None
    noisy = _scene(0, occl=0.0, outliers=0.85)
    fit = fit_table_plane_detailed(noisy, _cfg(min_inlier_ratio=0.3), np.random.default_rng(0))
    assert fit.plane is None and fit.reason == 'low_inlier_ratio'


def test_too_few_points_and_nan_rows():
    pts = _scene(0, n=150)
    fit = fit_table_plane_detailed(pts, _cfg(), np.random.default_rng(0))
    assert fit.plane is None and fit.reason == 'too_few_points'
    big = _scene(0)
    big[::3] = np.nan
    assert fit_table_plane(big, _cfg(), np.random.default_rng(0)) is not None
    assert fit_table_plane(np.zeros((0, 3)), _cfg()) is None


def test_params_validation():
    with pytest.raises(ValueError):
        TableParams.from_cfg(_cfg(inlier_thresh=-1.0))
    with pytest.raises(ValueError):
        TableParams.from_cfg(_cfg(ransac_iters=0))
    p = TableParams(roi_margin=0.05)
    pts = _scene(0)
    assert fit_table_plane(pts, p, np.random.default_rng(0), pick_polygon=POLY) is not None

"""Segmentation and base-frame geometry tests on analytic RGB-D scenes."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from cube_task.core.config import load_config
from cube_task.core.types import Zone
from cube_task.perception.pipeline import Perception
from cube_task.perception.segment import depth_edges, rect_geometry, wrap_half_pi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_perception_synth as S  # noqa: E402
from test_perception_synth import Box, Sphere  # noqa: E402

CAMS = {'overhead': S.overhead_camera(), 'oblique': S.oblique_camera()}


def _perception(**over):
    cfg = load_config(use_overlays=False)
    for k, v in over.items():
        cfg.set(k, v)
    return Perception(cfg)


def _run(objs, T, zones=None, perception=None, **kw):
    fr = S.make_frame(objs, T, zones=list((zones or {}).values()), **kw)
    return (perception or _perception()).process(fr, T, zones=zones or {})


def _match(det, objs):
    return min(objs, key=lambda o: math.hypot(o.x - det.center_base[0], o.y - det.center_base[1]))


def _box_errors(det, b):
    exy = math.hypot(b.x - det.center_base[0], b.y - det.center_base[1])
    e3 = math.sqrt(exy ** 2 + (det.center_base[2] - b.sz) ** 2)
    L, W = max(b.sx, b.sy), min(b.sx, b.sy)
    major = b.yaw if b.sx >= b.sy else b.yaw + math.pi / 2
    period = math.pi / 2 if abs(b.sx - b.sy) < 1e-9 else math.pi
    eyaw = math.degrees(S.angle_diff_mod(det.yaw, major, period))
    return e3, eyaw, det.length - L, det.width - W


def _check_scene(res, objs, max_center, max_yaw, max_lw):
    assert not res.warnings, res.warnings
    assert len(res.detections) == len(objs)
    for d in res.detections:
        assert d.valid, d.reject_reason
        b = _match(d, objs)
        e3, eyaw, el, ew = _box_errors(d, b)
        assert e3 < max_center, (e3, b)
        assert eyaw < max_yaw, (eyaw, b)
        assert abs(el) < max_lw and abs(ew) < max_lw, (el, ew, b)
        assert abs(d.height - b.sz) < max_center
        assert abs(d.extra['table_z']) < 0.001
        assert -math.pi / 2 <= d.yaw < math.pi / 2 and d.length >= d.width
        assert d.mask is not None and d.mask.dtype == bool and d.mask.shape == (480, 640)
        assert d.n_points >= 40


def test_localization_without_noise():
    objs = S.scene_boxes()
    for name, T in CAMS.items():
        _check_scene(_run(objs, T), objs, max_center=0.003, max_yaw=3.0, max_lw=0.004)


def test_localization_with_depth_noise():
    objs = S.scene_boxes()
    for name, T in CAMS.items():
        for seed in (1, 2):
            res = _run(objs, T, noise_std=0.0015, seed=seed)
            _check_scene(res, objs, max_center=0.006, max_yaw=4.0, max_lw=0.006)


def test_flying_pixels_do_not_inflate_or_add_objects():
    objs = S.scene_boxes()
    for name, T in CAMS.items():
        res = _run(objs, T, flying=0.8, seed=4)
        _check_scene(res, objs, max_center=0.004, max_yaw=3.0, max_lw=0.005)


def test_dropout_and_raised_tilted_table_reference():
    objs = [Box(0.30, 0.0, 0.5, 0.04, 0.03, 0.03, S.RED)]
    for name, T in CAMS.items():
        res = _run(objs, T, dropout=0.03, seed=2, table_z=0.02)
        assert len(res.detections) == 1 and res.detections[0].valid
        d = res.detections[0]
        assert abs(d.extra['table_z'] - 0.02) < 0.001
        assert abs(d.height - 0.03) < 0.003
        assert abs(d.top_z - 0.05) < 0.003


def test_center_px_is_projection_of_center():
    from cube_task.perception.geometry import project_points
    T = CAMS['oblique']
    res = _run(S.scene_boxes(), T)
    for d in res.detections:
        uv = project_points(d.center_base, T, S.INTR)[0]
        assert np.allclose(d.center_px, uv, atol=1e-6)
        x0, y0, x1, y1 = d.bbox
        assert x0 <= uv[0] < x1 and y0 <= uv[1] < y1


def _pairs():
    a = Box(0.30, 0.0, 0.0, 0.04, 0.04, 0.04, S.RED)
    rot = Box(0.30 + 0.02 + 0.02 * math.sqrt(2), 0.0, math.pi / 4, 0.04, 0.04, 0.04, S.BLUE)
    return {
        'rotated_corner': [a, rot],
        'staggered': [Box(0.30, -0.01, 0, 0.04, 0.04, 0.04), Box(0.34, 0.01, 0, 0.04, 0.04, 0.04)],
        'different_height': [Box(0.30, 0.0, 0, 0.04, 0.04, 0.03),
                             Box(0.30, 0.04, 0, 0.04, 0.04, 0.05)],
        'spheres': [Sphere(0.30, -0.035, 0.035), Sphere(0.30, 0.035, 0.035)],
    }


def test_touching_objects_are_split():
    for name, T in CAMS.items():
        for case, objs in _pairs().items():
            res = _run(objs, T)
            assert len(res.detections) == 2, (name, case, len(res.detections))
            matched = set()
            for d in res.detections:
                # Spheres: the footprint center is biased towards an oblique camera
                # (hidden far side); the cap center is the reliable grasp point.
                cx, cy = (d.extra['cap_center_xy'] if case == 'spheres'
                          else d.center_base[:2])
                b = min(objs, key=lambda o: math.hypot(o.x - cx, o.y - cy))
                matched.add(id(b))
                assert math.hypot(b.x - cx, b.y - cy) < 0.006, (name, case)
            assert len(matched) == 2, (name, case)


def test_long_single_object_is_not_split():
    for name, T in CAMS.items():
        for yaw in (0.0, 0.4, 1.2):
            res = _run([Box(0.30, 0.0, yaw, 0.15, 0.03, 0.03, S.GREEN)], T)
            assert len(res.detections) == 1, (name, yaw)
            d = res.detections[0]
            assert abs(d.length - 0.15) < 0.006 and abs(d.width - 0.03) < 0.004


def test_tiny_and_huge_blobs_are_rejected_with_reasons():
    objs = [Box(0.22, -0.12, 0, 0.008, 0.008, 0.02), Box(0.32, 0.05, 0, 0.25, 0.20, 0.02)]
    for name, T in CAMS.items():
        res = _run(objs, T)
        reasons = sorted(d.reject_reason for d in res.detections)
        assert reasons == ['too_large', 'too_small'], (name, reasons)
        assert not res.valid_detections()
        for d in res.detections:
            assert d.extra['reject_reasons'][0] == d.reject_reason


def test_roi_border_image_border_and_tall_objects():
    for name, T in CAMS.items():
        res = _run([Box(0.45, 0.0, 0, 0.04, 0.04, 0.03)], T)
        assert [d.reject_reason for d in res.detections] == ['touches_roi_border'], name
        res = _run([Box(0.30, 0.0, 0, 0.04, 0.04, 0.30)], T)
        assert [d.reject_reason for d in res.detections] == ['too_tall'], name
    big = [[0.0, -0.5], [0.8, -0.5], [0.8, 0.5], [0.0, 0.5]]
    p = _perception(**{'perception.roi.pick_polygon': big})
    res = _run([Box(0.54, 0.0, 0, 0.04, 0.04, 0.03)], CAMS['overhead'], perception=p)
    assert [d.reject_reason for d in res.detections] == ['touches_image_border']


def test_zone_contents_are_excluded():
    zones = {'bin': Zone('bin', np.array([0.30, 0.15, 0.0]), (0.10, 0.10))}
    objs = [Box(0.30, 0.15, 0.3, 0.04, 0.04, 0.03, S.RED), Box(0.30, -0.1, 0, 0.04, 0.04, 0.03)]
    for name, T in CAMS.items():
        res = _run(objs, T, zones=zones)
        assert len(res.detections) == 1, name
        assert abs(res.detections[0].center_base[1] + 0.1) < 0.003
        # An object straddling the zone margin is reported, not silently cut.
        res = _run([Box(0.30, 0.085, 0.0, 0.05, 0.05, 0.03)], T, zones=zones)
        assert [d.reject_reason for d in res.detections] == ['touches_zone'], name
    p = _perception(**{'perception.roi.exclude_zones': False})
    res = _run(objs, CAMS['overhead'], zones=zones, perception=p)
    assert len(res.detections) == 2


def test_occluded_object_in_oblique_view_is_flagged():
    objs = [Sphere(0.27, 0.0, 0.035), Sphere(0.34, 0.0, 0.035)]
    res = _run(objs, CAMS['oblique'])
    by_x = sorted(res.detections, key=lambda d: d.center_base[0])
    assert by_x[0].valid and by_x[0].extra['occlusion_ratio'] < 0.05
    assert not by_x[1].valid and by_x[1].reject_reason == 'occluded'
    res = _run(objs, CAMS['overhead'])
    assert all(d.valid for d in res.detections)


def test_sphere_cap_center_is_reported():
    s = Sphere(0.30, 0.05, 0.035)
    for name, T in CAMS.items():
        d = _run([s], T).detections[0]
        cx, cy = d.extra['cap_center_xy']
        assert math.hypot(cx - s.x, cy - s.y) < 0.004, name
        assert abs(d.top_z - 0.07) < 0.004


# --------------------------------------------------------------------------- units


def test_rect_geometry_rotated_rectangle():
    rng = np.random.default_rng(0)
    loc = rng.uniform([-0.03, -0.01], [0.03, 0.01], size=(4000, 2))
    yaw = 2.5
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    xy = loc @ R.T + [0.3, -0.1]
    c, y, L, W = rect_geometry(xy, trim_pct=0.0)
    assert np.allclose(c, [0.3, -0.1], atol=5e-4)
    assert abs(wrap_half_pi(y - yaw)) < math.radians(1.0)
    assert abs(L - 0.06) < 1e-3 and abs(W - 0.02) < 1e-3
    # The extension set may push sides out by at most max_extend.
    far = np.vstack([xy, [[0.3 + 0.2 * math.cos(yaw), -0.1 + 0.2 * math.sin(yaw)]] * 50])
    _, _, L2, _ = rect_geometry(xy, far, max_extend=0.002, trim_pct=0.5)
    _, _, L1, _ = rect_geometry(xy, None, trim_pct=0.5)
    assert 0.0 <= L2 - L1 <= 0.002 + 1e-9


def test_wrap_half_pi_range():
    for a in np.linspace(-7, 7, 101):
        w = wrap_half_pi(a)
        assert -math.pi / 2 <= w < math.pi / 2
        assert abs(math.sin(2 * (w - a))) < 1e-9


def test_depth_edges_marks_step_not_slope():
    z = np.tile(np.linspace(0.5, 0.6, 100, dtype=np.float32), (50, 1))
    z[:, 60:] += 0.05
    valid = np.ones_like(z, dtype=bool)
    e = depth_edges(z, valid, 0.006, 0.01)
    assert e[:, 59:61].all()
    assert not e[:, :58].any() and not e[:, 62:].any()

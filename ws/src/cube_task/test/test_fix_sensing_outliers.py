"""Persistent depth outliers nearer the camera do not move grasp points or footprints."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from cube_task.camera.utils import capture_averaged
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.perception.factory import make_classifier
from cube_task.perception.pipeline import Perception
from cube_task.perception.segment import cap_band, depth_spikes
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from test_task_fakes import still_wrist_camera


def test_depth_spikes_marks_isolated_near_pixels_only():
    z = np.full((60, 80), 0.70, np.float32)
    z[20:40, 30:50] = 0.65                       # a box top: straight borders, 4 corners
    valid = np.ones(z.shape, bool)
    valid[5, 5] = False
    z[5, 5] = 0.0
    spikes = [(10, 10), (50, 60), (25, 35), (6, 5)]            # (25, 35) on the box top
    for y, x in spikes:
        z[y, x] -= 0.02
    z[45:47, 10:12] -= 0.03                       # 2 x 2 cluster
    z[50, 20] += 0.03                             # a farther outlier is not a spike
    got = depth_spikes(z, valid, 0.006, 0.01)
    want = np.zeros(z.shape, bool)
    for y, x in spikes:
        want[y, x] = True
    want[45:47, 10:12] = True
    corners = [(20, 30), (20, 49), (39, 30), (39, 49)]
    for y, x in corners:
        want[y, x] = True
    assert np.array_equal(got, want), np.argwhere(got != want)
    assert not np.any(got & ~valid)


def test_cap_band_is_bounded_above():
    lo, hi = cap_band(0.035, 0.035, 0.001)
    assert lo == pytest.approx(0.033) and hi == pytest.approx(0.038)
    lo, hi = cap_band(0.08, 0.08, 0.0)            # round top: band depth 4 mm both ways
    assert lo == pytest.approx(0.076) and hi == pytest.approx(0.084)
    assert cap_band(0.035, 0.035, 0.003)[1] == pytest.approx(0.044)
    assert cap_band(0.035, 0.035, float('nan')) == cap_band(0.035, 0.035, None)


def _grasp_xy(d):
    x, y = float(d.center_base[0]), float(d.center_base[1])
    cap = d.extra.get('cap_center_xy')
    if cap is not None and math.hypot(cap[0] - x, cap[1] - y) <= 0.5 * float(d.width):
        return cap[0], cap[1]
    return x, y


def _errors(task, seeds, frac, **over):
    """Return ({label: [grasp-point error mm]}, n missing, n extra fragments)."""
    rng = np.random.default_rng(123)
    base = load_config(use_overlays=False)
    errs, missing, extra = {}, 0, 0
    for s in seeds:
        world, info = make_scenario(task, s)
        cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root,
                     base.sources)
        for k, v in over.items():
            cfg.set(k, v)
        T = np.array(info['T_base_camera_observe'], dtype=float)
        cam = still_wrist_camera(world, SimClock(), info['intrinsics'], T, seed=s)
        cam.start()
        fr = capture_averaged(cam, 3)
        clf = make_classifier(cfg, task) if task == 'L1' else None
        clean = Perception(cfg, classifier=clf).process(fr, T)
        depth = fr.depth.copy()
        for d in clean.detections:
            # Persistent (in every averaged frame) outliers 5-50 mm nearer the camera.
            ys, xs = np.nonzero(d.mask & (depth > 0))
            idx = rng.choice(ys.size, int(round(frac * ys.size)), replace=False)
            depth[ys[idx], xs[idx]] -= rng.uniform(0.005, 0.05, idx.size).astype(np.float32)
        res = Perception(cfg, classifier=clf).process(dataclasses.replace(fr, depth=depth), T)
        dets = res.valid_detections()
        for o in world.remaining(info['pick_polygon']):
            c = o.T_base_obj[:2, 3]
            near = [d for d in dets if np.hypot(*(np.asarray(d.center_base[:2]) - c)) < 0.05]
            if not near:
                missing += 1
                continue
            extra += len(near) - 1
            d = min(near, key=lambda q: np.hypot(*(np.asarray(q.center_base[:2]) - c)))
            gx, gy = _grasp_xy(d)
            key = 'block' if task == 'L1' else o.label
            errs.setdefault(key, []).append(1000.0 * math.hypot(gx - c[0], gy - c[1]))
    return errs, missing, extra


def _p90(errs, labels=None):
    v = [e for k, es in errs.items() if labels is None or k in labels for e in es]
    return float(np.percentile(v, 90))


def test_l1_blocks_with_persistent_outliers():
    clean, m0, x0 = _errors('L1', range(6), 0.0)
    errs, missing, extra = _errors('L1', range(6), 0.015)
    assert m0 == missing == 0 and x0 == extra == 0
    assert _p90(errs) < 2.0 and _p90(errs) < _p90(clean) + 0.5, (errs, clean)
    # Without despiking (cap bound and edge-free geometry only) the outliers hurt more.
    off, _, _ = _errors('L1', range(6), 0.015, **{'perception.segmentation.despike': False})
    assert _p90(off) > _p90(errs)


ROUND = ('apple', 'orange', 'lemon', 'tomato', 'green_pepper', 'eggplant', 'grape')


def test_l2_objects_with_persistent_outliers():
    clean, m0, x0 = _errors('L2', range(6), 0.0)
    errs, missing, extra = _errors('L2', range(6), 0.015)
    assert m0 == missing == 0 and extra <= x0        # x0 = 1 already in the clean frames
    assert _p90(errs) < 3.0 and _p90(errs, ROUND) < _p90(clean, ROUND) + 1.0, (errs, clean)
    off, _, _ = _errors('L2', range(6), 0.015, **{'perception.segmentation.despike': False})
    assert _p90(off, ROUND) > 2.0 * _p90(errs, ROUND), (off, errs)

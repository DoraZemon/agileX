"""Tests for cube_task.perception.pipeline and visualize (robustness, labels, timing)."""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.interfaces import Detector2D, InstanceClassifier
from cube_task.core.transforms import make_T, rot_x
from cube_task.core.types import Frame, PerceptionResult, Zone
from cube_task.perception.classify_color import ColorClassifier
from cube_task.perception.pipeline import mask_in_box_overlap, Perception
from cube_task.perception.visualize import draw_result, label_color

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_perception_synth as S  # noqa: E402
from test_perception_synth import Box, Sphere  # noqa: E402

T = S.overhead_camera()
OBJS = S.scene_boxes()


def _cfg(**over):
    cfg = load_config(use_overlays=False)
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def _frame(objs=OBJS, stamp=1.0, **kw):
    return S.make_frame(objs, T, stamp=stamp, **kw)


class FakeDetector(Detector2D):
    labels = ['ball', 'thing']

    def __init__(self, boxes, fail=False):
        self.boxes = boxes
        self.fail = fail
        self.calls = 0

    def detect(self, color):
        self.calls += 1
        if self.fail:
            raise RuntimeError('boom')
        return self.boxes


class FixedClassifier(InstanceClassifier):
    labels = ['x']

    def __init__(self, score=0.9, fail=False):
        self.score = score
        self.fail = fail
        self.seen = 0

    def classify(self, frame, detections):
        if self.fail:
            raise RuntimeError('boom')
        for d in detections:
            self.seen += 1
            d.label, d.score, d.score_kind = 'x', self.score, 'knn'


# --------------------------------------------------------------------------- robustness


def test_missing_depth_extrinsics_and_bad_inputs_never_raise():
    p = Perception(_cfg())
    fr = _frame()
    no_depth = Frame(fr.color, None, fr.intrinsics, 1.0)
    r = p.process(no_depth, T)
    assert r.warnings == ['no_depth'] and r.detections == [] and r.debug_image is not None
    zero = Frame(fr.color, np.zeros_like(fr.depth), fr.intrinsics, 1.0)
    assert p.process(zero, T).warnings == ['no_depth']
    nan = Frame(fr.color, np.full_like(fr.depth, np.nan), fr.intrinsics, 1.0)
    assert p.process(nan, T).warnings == ['no_depth']
    r = p.process(fr, None)
    assert r.warnings == ['no_extrinsics'] and r.detections == []
    bad = T.copy()
    bad[:3, :3] *= 2.0
    assert p.process(fr, bad).warnings == ['bad_extrinsics']
    assert p.process(fr, np.eye(3)).warnings == ['bad_extrinsics']
    mm = Frame(fr.color, (fr.depth * 1000).astype(np.uint16), fr.intrinsics, 1.0)
    assert p.process(mm, T).warnings == ['depth_mismatch']
    small = Frame(fr.color[::2, ::2], fr.depth[::2, ::2], fr.intrinsics, 1.0)
    assert p.process(small, T).warnings == ['depth_mismatch']
    wrong_color = Frame(fr.color[:, :320], fr.depth, fr.intrinsics, 1.0)
    assert p.process(wrong_color, T).warnings == ['color_mismatch']


def test_no_table_frames_return_warnings():
    p = Perception(_cfg())
    # Only a floating object, no table in view.
    fr = S.make_frame([Box(0.30, 0.0, 0.0, 0.10, 0.10, 0.05)], T, table=False)
    r = p.process(fr, T)
    assert 'no_table' in r.warnings and r.detections == [] and r.table is None
    assert any(w.startswith('table:') for w in r.warnings)
    # Badly wrong extrinsics: the table appears tilted by 35 degrees.
    tilted = make_T(rot_x(math.radians(35.0)), (0, 0, 0)) @ T
    r = p.process(_frame(), tilted)
    assert 'no_table' in r.warnings and 'table:tilt' in r.warnings
    # Expected table height check.
    p2 = Perception(_cfg(**{'perception.table.expected_z': 0.10}))
    r = p2.process(_frame(), T)
    assert 'table:unexpected_z' in r.warnings


def test_unexpected_exception_is_reported():
    p = Perception(_cfg())
    fr = _frame()
    fr.intrinsics = S.INTR.scaled(640, 480)
    fr.intrinsics.fx = 0.0
    r = p.process(fr, T)
    assert r.detections == [] and any(w.startswith('error:') for w in r.warnings)


def test_no_color_still_localizes():
    p = Perception(_cfg(), classifier=FixedClassifier())
    fr = _frame()
    fr.color = None
    r = p.process(fr, T)
    assert r.warnings == ['no_color']
    assert len(r.valid_detections()) == 4
    assert all(d.label == 'unknown' for d in r.detections)


def test_construction_validates_config():
    with pytest.raises(ValueError):
        Perception(_cfg(**{'perception.roi.pick_polygon': [[0, 0], [1, 1]]}))
    with pytest.raises(ValueError):
        Perception(_cfg(**{'perception.segmentation.min_height': 0.3}))
    with pytest.raises(ValueError):
        Perception(_cfg(**{'perception.classifier.min_score': 2.0}))
    with pytest.raises(ValueError):
        Perception(_cfg(**{'zones': {'a': {'size': [0.1, 0.1]}}}))


# --------------------------------------------------------------------------- labels


def _nearest(res, x, y):
    return min(res.detections,
               key=lambda q: math.hypot(q.center_base[0] - x, q.center_base[1] - y))


def _box_around(res, x, y, pad=4):
    d = _nearest(res, x, y)
    x0, y0, x1, y1 = d.bbox
    return d, (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def test_detector_association_label_map_and_fallback():
    fr = _frame()
    base = Perception(_cfg()).process(fr, T)
    a, box_a = _box_around(base, OBJS[0].x, OBJS[0].y)
    b, box_b = _box_around(base, OBJS[1].x, OBJS[1].y)
    # Box b is too small to hold enough of the mask.
    x0, y0, x1, y1 = box_b
    tiny_b = (x0, y0, x0 + (x1 - x0) * 0.3, y1)
    det = FakeDetector([(box_a, 'sports ball', 0.8), (tiny_b, 'thing', 0.99),
                        ((1, 2, 3), 'broken', 0.5), (box_a, 'thing', float('nan'))])
    cfg = _cfg(**{'perception.classifier.onnx.label_map': {'sports ball': 'apple'}})
    clf = FixedClassifier(score=0.9)
    r = Perception(cfg, classifier=clf, detector=det).process(fr, T)
    assert det.calls == 1 and not r.warnings
    da = _nearest(r, OBJS[0].x, OBJS[0].y)
    assert da.label == 'apple' and da.score_kind == 'prob' and abs(da.score - 0.8) < 1e-9
    assert da.extra['detector_label'] == 'sports ball' and da.extra['detector_overlap'] == 1.0
    others = [d for d in r.detections if d is not da]
    assert all(d.label == 'x' and d.score_kind == 'knn' for d in others)
    assert clf.seen == 3
    # Without a classifier unmatched instances stay unknown.
    r = Perception(cfg, detector=det).process(fr, T)
    assert sorted(d.label for d in r.detections) == ['apple', 'unknown', 'unknown', 'unknown']
    assert 'detector' in r.timings


def test_detector_and_classifier_failures_are_contained():
    fr = _frame()
    r = Perception(_cfg(), classifier=FixedClassifier(),
                   detector=FakeDetector([], fail=True)).process(fr, T)
    assert r.warnings == ['detector_error']
    assert all(d.label == 'x' for d in r.detections)
    r = Perception(_cfg(), classifier=FixedClassifier(fail=True)).process(fr, T)
    assert r.warnings == ['classifier_error']
    assert all(d.label == 'unknown' and d.score == 0.0 for d in r.detections)
    assert len(r.valid_detections()) == 4


def test_min_score_turns_labels_unknown():
    fr = _frame()
    r = Perception(_cfg(**{'perception.classifier.min_score': 0.55}),
                   classifier=FixedClassifier(score=0.4)).process(fr, T)
    for d in r.detections:
        assert d.label == 'unknown' and d.extra['raw_label'] == 'x'
        assert abs(d.extra['raw_score'] - 0.4) < 1e-9 and abs(d.score - 0.4) < 1e-9


def test_mask_in_box_overlap():
    m = np.zeros((10, 10), bool)
    m[2:6, 2:6] = True
    assert mask_in_box_overlap(m, (0, 0, 10, 10)) == 1.0
    assert abs(mask_in_box_overlap(m, (2, 2, 4, 6)) - 0.5) < 1e-9
    assert mask_in_box_overlap(m, (20, 20, 30, 30)) == 0.0
    assert mask_in_box_overlap(m, (6, 6, 2, 2)) == 1.0           # swapped corners
    assert mask_in_box_overlap(np.zeros((4, 4), bool), (0, 0, 4, 4)) == 0.0


# --------------------------------------------------------------------------- table reuse


def test_table_reuse_uses_frame_stamps():
    p = Perception(_cfg(**{'perception.table.reuse_s': 5.0}))
    r1 = p.process(_frame(stamp=10.0), T)
    r2 = p.process(_frame(stamp=12.0), T)
    assert r2.table is r1.table
    r3 = p.process(_frame(stamp=20.0), T)
    assert r3.table is not r1.table
    p.reset_table()
    assert p.process(_frame(stamp=20.5), T).table is not r3.table
    q = Perception(_cfg())
    assert q.process(_frame(stamp=1.0), T).table is not q.process(_frame(stamp=1.1), T).table


# --------------------------------------------------------------------------- performance


def test_processing_time_bound():
    objs = OBJS + [Sphere(0.42, -0.17, 0.03), Sphere(0.20, 0.17, 0.035, S.RED)]
    cfg = _cfg()
    clf = ColorClassifier(cfg.get('perception.classifier.color.classes'), 0.35)
    p = Perception(cfg, classifier=clf)
    times = []
    for k, T_cam in enumerate((S.overhead_camera(), S.oblique_camera()) * 2):
        fr = S.make_frame(objs, T_cam, noise_std=0.0015, seed=k)
        t0 = time.perf_counter()
        r = p.process(fr, T_cam)
        times.append(time.perf_counter() - t0)
        assert len(r.valid_detections()) == 6
    assert sorted(times)[len(times) // 2] < 0.150, times


# --------------------------------------------------------------------------- drawing


def test_draw_result_overlay():
    zones = {'bin': Zone('bin', np.array([0.30, 0.20, 0.0]), (0.08, 0.08))}
    objs = OBJS + [Box(0.22, -0.18, 0, 0.008, 0.008, 0.02)]
    fr = S.make_frame(objs, T, zones=list(zones.values()))
    orig = fr.color.copy()
    p = Perception(_cfg(), classifier=ColorClassifier(
        _cfg().get('perception.classifier.color.classes'), 0.35))
    r = p.process(fr, T, zones=zones)
    assert any(not d.valid for d in r.detections)
    img = draw_result(fr, r, T, zones, p.pick_polygon)
    assert img.shape == fr.color.shape and img.dtype == np.uint8
    assert np.array_equal(fr.color, orig)
    assert np.count_nonzero(np.any(img != orig, axis=2)) > 5000
    assert r.debug_image is not None and r.debug_image.shape == fr.color.shape
    # Missing pieces must not raise.
    draw_result(fr, PerceptionResult([], warnings=['no_table']))
    fr.color = None
    assert draw_result(fr, r, None, None, None).shape == (480, 640, 3)
    p_off = Perception(_cfg(**{'perception.debug_image': False}))
    assert p_off.process(_frame(), T).debug_image is None
    assert label_color('red') == (40, 40, 230) and label_color('apple') == label_color('apple')

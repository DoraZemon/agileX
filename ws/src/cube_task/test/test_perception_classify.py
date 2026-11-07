"""Tests for features, the color / KNN classifiers and the perception factory."""
from __future__ import annotations

import math
import os
import sys
import types

import cv2
import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.types import Detection
from cube_task.perception.classify_color import (ColorClassifier, hsv_ranges_from_samples,
                                                 in_ranges)
from cube_task.perception.classify_knn import KnnClassifier
from cube_task.perception.factory import make_classifier, make_detector, make_perception
from cube_task.perception.features import (FEATURE_NAMES, feature_vector, hue_histogram,
                                           instance_pixels)
from cube_task.perception.pipeline import Perception

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_perception_synth as S  # noqa: E402
from test_perception_synth import Box, Sphere  # noqa: E402

T_OVER = S.overhead_camera()
T_OBL = S.oblique_camera()
RED_WRAP = (60, 20, 200)      # hue ~ 173 (wraps below 180)
RED_LOW = (20, 40, 200)       # hue ~ 4


def _cfg():
    return load_config(use_overlays=False)


def _hue(bgr):
    return int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def _detect(objs, T, classifier=None, seed=0, **kw):
    fr = S.make_frame(objs, T, seed=seed, **kw)
    res = Perception(_cfg(), classifier=classifier).process(fr, T, zones={})
    return fr, res


def _nearest(res, x, y):
    return min(res.detections,
               key=lambda d: math.hypot(d.center_base[0] - x, d.center_base[1] - y))


# --------------------------------------------------------------------------- color


def test_color_classifier_blocks_including_red_wrap():
    assert _hue(RED_WRAP) > 165 and _hue(RED_LOW) < 8
    cfg = _cfg()
    clf = ColorClassifier(cfg.get('perception.classifier.color.classes'),
                          cfg.get('perception.classifier.color.min_fraction'))
    objs = [Box(0.24, -0.12, 0.2, 0.04, 0.04, 0.04, RED_WRAP),
            Box(0.36, -0.10, 0.5, 0.04, 0.04, 0.04, S.GREEN),
            Box(0.24, 0.08, 0.9, 0.04, 0.04, 0.04, S.BLUE),
            Box(0.36, 0.10, 0.0, 0.04, 0.04, 0.04, S.YELLOW),
            Box(0.30, -0.01, 0.0, 0.035, 0.035, 0.03, RED_LOW)]
    expect = ['red', 'green', 'blue', 'yellow', 'red']
    for T in (T_OVER, T_OBL):
        _, res = _detect(objs, T, clf)
        assert len(res.detections) == 5
        for o, lab in zip(objs, expect):
            d = _nearest(res, o.x, o.y)
            assert d.label == lab, (lab, d.label, d.extra.get('color_fractions'))
            assert d.score_kind == 'color' and d.score > 0.9


def test_color_classifier_unknown_and_restriction():
    cfg = _cfg()
    classes = cfg.get('perception.classifier.color.classes')
    clf = ColorClassifier(classes, 0.35, allowed_labels=['green', 'blue'])
    assert clf.labels == ['green', 'blue']
    _, res = _detect([Box(0.30, 0.0, 0.0, 0.04, 0.04, 0.04, S.RED)], T_OVER, clf)
    d = res.detections[0]
    assert d.label == 'unknown' and d.score < 0.35
    assert set(d.extra['color_fractions']) == {'green', 'blue'}
    with pytest.raises(ValueError):
        ColorClassifier(classes, 0.35, allowed_labels=['red', 'pink'])
    with pytest.raises(ValueError):
        ColorClassifier({'red': [[[0, 0, 0], [10, 300, 255]]]})
    with pytest.raises(ValueError):
        ColorClassifier({})


def test_color_classifier_empty_mask():
    clf = ColorClassifier({'red': [[[0, 90, 60], [8, 255, 255]]]})
    fr = S.make_frame([], T_OVER)
    det = Detection('unknown', 0.0, mask=None)
    clf.classify(fr, [det])
    assert det.label == 'unknown' and det.score == 0.0
    assert det.extra['classify_reason'] == 'empty_mask'


def test_in_ranges_wraparound_range():
    hsv = np.array([[175, 200, 200], [3, 200, 200], [90, 200, 200], [3, 20, 200]])
    assert in_ranges(hsv, np.array([[[170, 90, 60], [8, 255, 255]]])).tolist() == \
        [True, True, False, False]


def test_hsv_ranges_from_samples():
    rng = np.random.default_rng(0)

    def px(h_center, n=2000, s=(150, 240), v=(90, 230)):
        h = np.round(h_center + rng.normal(0, 2.5, n)) % 180
        return np.column_stack([h, rng.integers(*s, n), rng.integers(*v, n)]).astype(np.uint8)

    samples = {'red': px(178), 'green': px(60), 'gray': px(100, s=(0, 25))}
    rr = hsv_ranges_from_samples(samples)
    assert len(rr['red']) == 2                       # split at the 0 / 180 wrap
    assert len(rr['green']) == 1
    assert rr['gray'][0][0][0] == 0 and rr['gray'][0][1][0] == 180
    for row in rr['red'] + rr['green']:
        for v in row[0] + row[1]:
            assert isinstance(v, int)
    clf = ColorClassifier({k: rr[k] for k in ('red', 'green')}, 0.5)
    fr_red = clf.fractions(px(178))
    fr_green = clf.fractions(px(60))
    assert fr_red['red'] > 0.9 and fr_red['green'] < 0.01
    assert fr_green['green'] > 0.9 and fr_green['red'] < 0.01
    with pytest.raises(ValueError):
        hsv_ranges_from_samples({'x': np.zeros((3, 3), np.uint8)})


# --------------------------------------------------------------------------- features


def test_feature_vector_properties():
    objs = [Box(0.25, -0.08, 0.3, 0.05, 0.05, 0.04, S.BLUE), Sphere(0.34, 0.08, 0.035)]
    fr, res = _detect(objs, T_OVER)
    box = _nearest(res, 0.25, -0.08)
    ball = _nearest(res, 0.34, 0.08)
    fb, names = feature_vector(fr, box)
    fs, _ = feature_vector(fr, ball)
    assert names == FEATURE_NAMES and fb.dtype == np.float32 and fb.shape == (len(names),)
    assert np.all(np.isfinite(fb)) and np.all(np.isfinite(fs))
    assert np.array_equal(fb, feature_vector(fr, box)[0])        # deterministic
    f = dict(zip(names, fb))
    g = dict(zip(names, fs))
    assert abs(sum(fb[:12]) - 1.0) < 1e-5
    assert abs(f['length'] - 0.05) < 0.004 and abs(f['height'] - 0.04) < 0.003
    assert f['fill'] > 0.9 and 0.65 < g['fill'] < 0.9
    assert g['circularity'] > f['circularity']
    # Upper pixels exclude the table and the dark sides.
    bgr = instance_pixels(fr, box)
    assert bgr.shape[0] > 100 and np.abs(bgr.mean(axis=0) - S.BLUE).max() < 10


def test_feature_vector_missing_geometry_is_nan():
    fr = S.make_frame([], T_OVER)
    m = np.zeros((480, 640), bool)
    m[100:120, 100:120] = True
    det = Detection('unknown', 0.0, bbox=(100, 100, 120, 120), mask=m)
    f, names = feature_vector(fr, det)
    assert np.all(np.isfinite(f[:18]))
    assert np.all(np.isnan(f[18:]))


def test_hue_histogram_circular_soft_binning():
    h = hue_histogram(np.array([[179, 255, 255], [1, 255, 255]]))
    assert abs(h.sum() - 1.0) < 1e-9
    assert h[0] + h[11] > 0.99
    assert np.all(hue_histogram(np.array([[30, 0, 200]])) == 0)


# --------------------------------------------------------------------------- KNN


def _training_scenes():
    """(X, y) from rendered scenes of four fruit-like classes at varied poses."""
    rng = np.random.default_rng(7)
    X, y = [], []
    cls = {
        'apple': lambda x, yy, a: Sphere(x, yy, 0.036, (40, 30, 190)),
        'orange': lambda x, yy, a: Sphere(x, yy, 0.030, (20, 130, 240)),
        'banana': lambda x, yy, a: Box(x, yy, a, 0.15, 0.035, 0.03, (40, 210, 225)),
        'cube': lambda x, yy, a: Box(x, yy, a, 0.04, 0.04, 0.04, (190, 90, 20)),
    }
    slots = [(0.24, -0.12), (0.36, -0.10), (0.25, 0.10), (0.37, 0.11)]
    for k in range(4):
        objs, labels = [], []
        for (sx, sy), (lab, make) in zip(slots, cls.items()):
            jx, jy = rng.uniform(-0.01, 0.01, 2)
            objs.append(make(sx + jx, sy + jy, rng.uniform(-1.5, 1.5)))
            labels.append(lab)
        slots = slots[1:] + slots[:1]
        T = T_OVER if k % 2 == 0 else T_OBL
        fr, res = _detect(objs, T, seed=k)
        assert len(res.detections) == 4
        for o, lab in zip(objs, labels):
            d = _nearest(res, o.x, o.y)
            X.append(feature_vector(fr, d)[0])
            y.append(lab)
    return np.vstack(X), y, cls


def test_knn_train_save_load_predict_and_unknown(tmp_path):
    X, y, cls = _training_scenes()
    knn = KnnClassifier(k=3, max_distance=3.0).fit(X, y, FEATURE_NAMES)
    assert knn.labels == ['apple', 'banana', 'cube', 'orange']
    rep = knn.leave_one_out()
    assert rep['n'] == 16 and rep['accuracy'] == 1.0, rep['confusion']
    path = knn.save(tmp_path / 'm' / 'knn.npz', created='2026-01-01T00:00:00')
    assert path.is_file() and not (tmp_path / 'm' / 'knn.npz.tmp').exists()
    z = np.load(path, allow_pickle=False)
    assert set(z.files) >= {'X', 'y', 'mean', 'std', 'feature_names', 'version', 'created'}
    knn2 = KnnClassifier.load(path, k=3, max_distance=3.0)
    assert knn2.created == '2026-01-01T00:00:00' and knn2.n_samples == 16
    assert np.allclose(knn2.std, knn.std) and knn2.predict(X)[0] == knn.predict(X)[0]

    # New poses, through the pipeline with the loaded model.
    objs = [cls['cube'](0.26, -0.05, 0.7), cls['apple'](0.36, 0.05, 0.0),
            cls['banana'](0.27, 0.12, 0.2)]
    for T in (T_OVER, T_OBL):
        _, res = _detect(objs, T, knn2, seed=11)
        assert len(res.detections) == 3
        got = [_nearest(res, o.x, o.y).label for o in objs]
        assert got == ['cube', 'apple', 'banana'], got
        assert all(d.score_kind == 'knn' and d.score > 0.55 for d in res.detections)

    # A very different object is rejected as unknown.
    _, res = _detect([Box(0.30, 0.0, 0.0, 0.10, 0.08, 0.12, (200, 40, 160))], T_OVER, knn2)
    d = res.detections[0]
    assert d.label == 'unknown' and d.extra['classify_reason'] == 'too_far'
    labels, scores, dists = knn2.predict(np.full(len(FEATURE_NAMES), np.nan))
    assert labels == ['unknown'] and scores[0] == 0.0 and np.isinf(dists[0])


def test_knn_add_samples_and_validation(tmp_path):
    rng = np.random.default_rng(0)
    D = len(FEATURE_NAMES)
    A = rng.normal(0.0, 1.0, (10, D))
    B = rng.normal(5.0, 1.0, (10, D))
    knn = KnnClassifier(k=3, max_distance=3.0).fit(A, ['a'] * 10)
    assert knn.predict(B[:1])[0] == ['unknown']
    knn.add_samples(B, ['b'] * 10)
    assert knn.n_samples == 20 and knn.labels == ['a', 'b']
    assert knn.predict(B[:2] + 0.1)[0] == ['b', 'b']
    with pytest.raises(ValueError):
        knn.add_samples(np.full((1, D), np.nan), ['a'])
    with pytest.raises(ValueError):
        knn.add_samples(A[:2], ['a'])
    with pytest.raises(ValueError):
        KnnClassifier().fit(A, ['unknown'] * 10)
    with pytest.raises(FileNotFoundError):
        KnnClassifier.load(tmp_path / 'missing.npz')
    # A model trained with other feature names must not load silently.
    bad = tmp_path / 'bad.npz'
    KnnClassifier().fit(A[:, :3], ['a'] * 10, ['f1', 'f2', 'f3']).save(bad)
    with pytest.raises(ValueError):
        KnnClassifier.load(bad)


# --------------------------------------------------------------------------- factory


def test_factory_color_knn_onnx(tmp_path, monkeypatch):
    cfg = _cfg()
    c = make_classifier(cfg, 'L1')
    assert isinstance(c, ColorClassifier) and c.labels == ['red', 'green', 'blue']
    cfg.set('tasks.L1.labels', ['red', 'pink'])
    with pytest.raises(ValueError):
        make_classifier(cfg, 'L1')
    cfg = _cfg()
    cfg.set('perception.classifier.knn.model_file', str(tmp_path / 'none.npz'))
    with pytest.raises(FileNotFoundError, match='teach_classes'):
        make_classifier(cfg, 'L2')
    D = len(FEATURE_NAMES)
    KnnClassifier().fit(np.random.default_rng(0).normal(size=(5, D)), ['a'] * 5).save(
        tmp_path / 'none.npz')
    assert isinstance(make_classifier(cfg, 'L2'), KnnClassifier)
    cfg.set('perception.classifier.knn.k', 1)
    assert make_classifier(cfg, 'L2').k == 1
    assert make_detector(cfg, 'L1') is None

    class FakeOnnx:
        labels = ['apple']

        def __init__(self, cfg):
            self.cfg = cfg

        def detect(self, color):
            return []

    mod = types.ModuleType('cube_task.perception.detector_onnx')
    mod.OnnxDetector = FakeOnnx
    monkeypatch.setitem(sys.modules, 'cube_task.perception.detector_onnx', mod)
    cfg.set('tasks.L2.classifier', 'onnx')
    assert make_classifier(cfg, 'L2') is None
    assert isinstance(make_detector(cfg, 'L2'), FakeOnnx)
    cfg.set('tasks.L2.classifier', 'onnx+knn')
    p = make_perception(cfg, 'L2')
    assert isinstance(p.classifier, KnnClassifier) and isinstance(p.detector, FakeOnnx)
    cfg.set('tasks.L2.classifier', None)
    cfg.set('perception.classifier.backend', 'color')
    cfg.set('tasks.L2.labels', [])
    p = make_perception(cfg, 'L2')
    assert isinstance(p.classifier, ColorClassifier) and p.detector is None
    assert len(p.classifier.labels) == 6
    cfg.set('perception.classifier.backend', 'magic')
    with pytest.raises(ValueError):
        make_classifier(cfg, 'L2')

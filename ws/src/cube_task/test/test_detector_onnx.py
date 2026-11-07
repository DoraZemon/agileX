"""Tests for perception/detector_onnx.py (system python, no torch, no network)."""
from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cube_task.core.config import Config
from cube_task.perception import detector_onnx as dx
from cube_task.perception.detector_onnx import DetectorError, OnnxDetector

ROOT = Path(__file__).resolve().parents[1]
COCO_MODEL = ROOT / 'models' / 'yolo11n_coco.onnx'
COCO_LABELS = ROOT / 'models' / 'yolo11n_coco_labels.txt'
COCO_RAW = ROOT / 'models' / 'yolo11n_coco.raw.onnx'


# --------------------------------------------------------------------------
# Helpers: synthetic model output and a fake inference runner
# --------------------------------------------------------------------------

def lb_params(h, w, size):
    """Letterbox scale and padding, computed independently of the module."""
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    return r, (int(round((size - nw) / 2 - 0.1)), int(round((size - nh) / 2 - 0.1)))


def make_output(objs, nc, img_hw, size=640, n=8400, transposed=False, seed=0):
    """
    Build a [1, 4+nc, n] tensor holding objs = [(x0, y0, x1, y1, cls, prob)].

    Boxes are given in image pixels and converted to letterboxed model-input
    cx, cy, w, h. The other rows get small random scores.
    """
    rng = np.random.RandomState(seed)
    out = np.zeros((4 + nc, n), np.float32)
    out[0:2] = rng.uniform(0, size, (2, n))
    out[2:4] = rng.uniform(5, 50, (2, n))
    out[4:] = rng.uniform(0, 0.05, (nc, n))
    r, (pl, pt) = lb_params(img_hw[0], img_hw[1], size)
    for k, (x0, y0, x1, y1, c, p) in enumerate(objs):
        col = 100 + 7 * k
        out[0, col] = (x0 + x1) / 2 * r + pl
        out[1, col] = (y0 + y1) / 2 * r + pt
        out[2, col] = (x1 - x0) * r
        out[3, col] = (y1 - y0) * r
        out[4:, col] = 0.0
        out[4 + c, col] = p
    out = out[None]
    return np.ascontiguousarray(out.transpose(0, 2, 1)) if transposed else out


class FakeRunner:
    """Stands in for cv2.dnn / onnxruntime; returns a fixed output."""

    name = 'fake'

    def __init__(self, output):
        self.output = output
        self.blobs = []

    def run(self, blob):
        self.blobs.append(blob)
        return self.output


def write_labels(tmp_path, labels, name='labels.txt'):
    p = tmp_path / name
    p.write_text('\n'.join(labels) + '\n', encoding='utf-8')
    return p


def make_cfg(tmp_path, labels, model_bytes=b'not an onnx file', **onnx):
    model = tmp_path / 'model.onnx'
    model.write_bytes(model_bytes)
    lab = write_labels(tmp_path, labels)
    sec = {'model_file': str(model), 'labels_file': str(lab), 'input_size': 640,
           'conf_threshold': 0.35, 'nms_threshold': 0.45, 'backend': 'opencv',
           'label_map': {}}
    sec.update(onnx)
    return Config({'perception': {'classifier': {'onnx': sec}}}, tmp_path, ['test'])


def make_detector(tmp_path, monkeypatch, labels, output, **onnx):
    runner = FakeRunner(output)
    monkeypatch.setattr(OnnxDetector, '_make_runner', lambda self, backend, info: runner)
    det = OnnxDetector(make_cfg(tmp_path, labels, **onnx))
    return det, runner


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u


LABELS = ['apple', 'banana', 'carrot']


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------

@pytest.mark.parametrize('h,w', [(480, 640), (720, 1280), (1080, 810), (100, 100), (481, 643)])
def test_letterbox_shape_padding_and_mapping(h, w):
    img = np.full((h, w, 3), 200, np.uint8)
    img[h // 2, w // 3] = (0, 0, 255)
    out, r, pad = dx.letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    er, epad = lb_params(h, w, 640)
    assert r == pytest.approx(er) and pad == epad
    nw, nh = int(round(w * r)), int(round(h * r))
    if pad[0] > 0:
        assert (out[:, :pad[0]] == 114).all()
    if pad[1] > 0:
        assert (out[:pad[1]] == 114).all()
    assert (out[pad[1]:pad[1] + nh, pad[0]:pad[0] + nw] != 114).any()
    # Unletterboxing a box around the mapped point returns the original point.
    u, v = w // 3 * r + pad[0], h // 2 * r + pad[1]
    back = dx.unletterbox_boxes(np.array([[u, v, u + r, v + r]]), r, pad, w, h)[0]
    assert back[0] == pytest.approx(w // 3, abs=1e-6)
    assert back[1] == pytest.approx(h // 2, abs=1e-6)


@pytest.mark.parametrize('transposed', [False, True])
def test_decode_both_layouts(transposed):
    out = make_output([(100, 50, 200, 150, 1, 0.9)], 3, (480, 640), transposed=transposed)
    boxes, scores, cls = dx.decode_output(out, 3, 0.35)
    assert boxes.shape == (1, 4) and scores.shape == (1,) and cls.tolist() == [1]
    r, (pl, pt) = lb_params(480, 640, 640)
    assert boxes[0] == pytest.approx([100 * r + pl, 50 * r + pt, 200 * r + pl, 150 * r + pt],
                                     abs=1e-3)
    assert scores[0] == pytest.approx(0.9)


def test_decode_drops_nan_inf_and_degenerate_rows():
    out = make_output([(10, 10, 60, 60, 0, 0.9), (100, 100, 160, 160, 2, 0.8),
                       (300, 300, 340, 340, 1, 0.7)], 3, (640, 640))
    a = out[0]
    a[0, 100] = np.nan                       # first object: NaN center
    a[2, 107] = 0.0                          # second object: zero width
    a[5, 200] = np.inf                       # background row: inf score
    boxes, scores, cls = dx.decode_output(out, 3, 0.35)
    assert cls.tolist() == [1] and scores[0] == pytest.approx(0.7)
    assert np.isfinite(boxes).all()


def test_decode_class_count_mismatch_messages():
    out = make_output([], 5, (480, 640))
    with pytest.raises(DetectorError, match='model has 5 classes'):
        dx.decode_output(out, 3, 0.35)
    with pytest.raises(DetectorError, match='end-to-end'):
        dx.decode_output(np.zeros((1, 300, 6), np.float32), 80, 0.35)
    with pytest.raises(DetectorError, match='batch size 2'):
        dx.decode_output(np.zeros((2, 7, 10), np.float32), 3, 0.35)
    with pytest.raises(DetectorError, match='unexpected model output shape'):
        dx.decode_output(np.zeros((7,), np.float32), 3, 0.35)


def test_decode_empty_output():
    boxes, scores, cls = dx.decode_output(np.zeros((1, 7, 0), np.float32), 3, 0.35)
    assert boxes.shape == (0, 4) and scores.shape == (0,) and cls.shape == (0,)


def test_nms_per_label():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [0, 0, 100, 100],
                      [300, 300, 350, 350]], float)
    scores = np.array([0.8, 0.9, 0.7, 0.6])
    keep = dx.nms_per_label(boxes, scores, ['a', 'a', 'b', 'a'], 0.45)
    assert keep.tolist() == [1, 2, 3]        # sorted by score; 'b' is not suppressed by 'a'
    assert dx.nms_per_label(np.zeros((0, 4)), np.zeros(0), [], 0.45).tolist() == []
    keep = dx.nms_per_label(boxes, scores, ['a'] * 4, 0.45, max_det=2)
    assert keep.tolist() == [1, 2] or keep.tolist() == [1, 3]
    with pytest.raises(ValueError):
        dx.nms_per_label(boxes, scores, ['a'], 0.45)


def test_unletterbox_clips_to_image():
    b = dx.unletterbox_boxes(np.array([[-50.0, -10.0, 700.0, 650.0]]), 1.0, (0, 80), 640, 480)
    assert b[0].tolist() == [0.0, 0.0, 640.0, 480.0]
    with pytest.raises(ValueError):
        dx.unletterbox_boxes(np.zeros((1, 4)), 0.0, (0, 0), 10, 10)


def test_read_labels(tmp_path):
    p = tmp_path / 'l.txt'
    p.write_bytes('﻿apple\n香蕉 \nsports ball\n\n\n'.encode('utf-8'))
    assert dx.read_labels(p) == ['apple', '香蕉', 'sports ball']
    p.write_text('apple\n\nbanana\n')
    with pytest.raises(DetectorError, match='line 2 is blank'):
        dx.read_labels(p)
    p.write_text('\n\n')
    with pytest.raises(DetectorError, match='empty'):
        dx.read_labels(p)
    with pytest.raises(DetectorError, match='not found'):
        dx.read_labels(tmp_path / 'missing.txt')


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self, logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_label_mapping():
    labels = ['apple', 'orange', 'sports ball', 'person']
    mapped, out = dx.build_label_mapping(labels, {'sports ball': 'orange', 'person': ''})
    assert mapped == ['apple', 'orange', 'orange', '']
    assert out == ['apple', 'orange']
    mapped, out = dx.build_label_mapping(labels, {'apple': 'apple', 'orange': 'orange',
                                                  '*': ''})
    assert mapped == ['apple', 'orange', '', ''] and out == ['apple', 'orange']
    handler = _ListHandler()
    dx.LOG.addHandler(handler)
    try:
        dx.build_label_mapping(labels, {'aple': 'apple'})
    finally:
        dx.LOG.removeHandler(handler)
    assert any("'aple'" in m and 'typo' in m for m in handler.messages)
    with pytest.raises(DetectorError, match='swaps and chains'):
        dx.build_label_mapping(labels, {'apple': 'orange', 'orange': 'apple'})
    with pytest.raises(DetectorError, match='null'):
        dx.build_label_mapping(labels, {'person': None})
    with pytest.raises(DetectorError, match='mapping'):
        dx.build_label_mapping(labels, ['apple'])
    assert dx.build_label_mapping(labels, None)[1] == labels


# --------------------------------------------------------------------------
# Hand-made ONNX protobuf (the system python has no onnx package)
# --------------------------------------------------------------------------

def _varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _f(field, payload):
    """Encode one protobuf field: int -> varint, bytes / str -> length-delimited."""
    if isinstance(payload, int):
        return _varint(field << 3) + _varint(payload)
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def fake_onnx(names=('apple', 'banana', 'carrot'), shape=(1, 3, 640, 640), sub_const_first=False,
              task='detect', end2end='False'):
    dims = b''.join(_f(1, _f(1, d) if isinstance(d, int) else _f(2, d)) for d in shape)
    tensor_type = _f(1, 1) + _f(2, dims)
    graph_input = _f(1, 'images') + _f(2, _f(1, tensor_type))
    init = _f(1, 4) + _f(2, 1) + _f(8, 'c') + _f(9, struct.pack('<4f', 1, 2, 3, 4))
    a, b = ('c', 'x') if sub_const_first else ('x', 'c')
    node = _f(1, a) + _f(1, b) + _f(2, 'y') + _f(3, '/model.23/Sub') + _f(4, 'Sub')
    graph = (_f(1, node) + _f(2, 'g') + _f(5, init) + _f(11, graph_input)
             + _f(12, _f(1, 'output0')))
    meta = {'names': repr({i: n for i, n in enumerate(names)}), 'task': task,
            'end2end': end2end, 'imgsz': repr(list(shape[2:]))}
    props = b''.join(_f(14, _f(1, k) + _f(2, v)) for k, v in meta.items())
    return _f(1, 7) + _f(2, 'pytorch') + _f(7, graph) + _f(8, _f(1, '') + _f(2, 12)) + props


def test_read_onnx_info(tmp_path):
    p = tmp_path / 'm.onnx'
    p.write_bytes(fake_onnx(sub_const_first=True))
    info = dx.read_onnx_info(p)
    assert info.ir_version == 7 and info.opsets == {'': 12}
    assert info.input_name == 'images' and info.input_shape == [1, 3, 640, 640]
    assert info.output_names == ['output0']
    assert info.names() == ['apple', 'banana', 'carrot']
    assert info.metadata['task'] == 'detect'
    assert info.const_first_ops == ['/model.23/Sub']
    p.write_bytes(fake_onnx(shape=(1, 3, 'h', 'w')))
    info = dx.read_onnx_info(p)
    assert info.input_shape == [1, 3, 'h', 'w'] and info.const_first_ops == []
    p.write_bytes(b'\x0a\xff\xff')
    with pytest.raises(ValueError):
        dx.read_onnx_info(p)


def test_opencv_runner_rejects_const_first_sub(tmp_path):
    p = tmp_path / 'm.onnx'
    p.write_bytes(fake_onnx(sub_const_first=True))
    with pytest.raises(DetectorError, match='constant first operand'):
        dx._OpenCVRunner(p, dx.read_onnx_info(p))


def test_onnxruntime_missing_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)    # import raises ImportError
    assert not dx._onnxruntime_available()
    with pytest.raises(DetectorError, match='onnxruntime cannot be imported'):
        dx._OrtRunner(tmp_path / 'm.onnx', None)


def test_backend_auto_falls_back_to_opencv(tmp_path, monkeypatch):
    out = make_output([], 3, (480, 640))
    monkeypatch.setattr(dx, '_onnxruntime_available', lambda: False)

    class FakeCv(FakeRunner):
        name = 'opencv'

        def __init__(self, path, info):
            FakeRunner.__init__(self, out)

    monkeypatch.setattr(dx, '_OpenCVRunner', FakeCv)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(), backend='auto')
    assert OnnxDetector(cfg).backend == 'opencv'
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(), backend='cuda')
    with pytest.raises(DetectorError, match='backend must be one of'):
        OnnxDetector(cfg)


# --------------------------------------------------------------------------
# OnnxDetector with a fake runner
# --------------------------------------------------------------------------

OBJS = [(100, 50, 200, 150, 0, 0.90),     # apple
        (104, 52, 204, 152, 0, 0.60),     # duplicate apple -> NMS
        (300, 200, 380, 330, 1, 0.80),    # banana
        (102, 50, 202, 150, 2, 0.50),     # carrot on the apple: other class, kept
        (600, 400, 700, 520, 2, 0.70),    # carrot partly outside a 640x480 image
        (10, 10, 40, 40, 1, 0.20)]        # below conf_threshold


@pytest.mark.parametrize('transposed', [False, True])
def test_detect_end_to_end(tmp_path, monkeypatch, transposed):
    out = make_output(OBJS, 3, (480, 640), transposed=transposed)
    det, runner = make_detector(tmp_path, monkeypatch, LABELS, out, model_bytes=fake_onnx())
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (255, 0, 0)                                    # pure blue in BGR
    res = det.detect(img)
    labels = [lab for _b, lab, _p in res]
    probs = [p for _b, _l, p in res]
    assert labels == ['apple', 'banana', 'carrot', 'carrot']
    assert probs == sorted(probs, reverse=True)
    assert all(isinstance(v, int) for b, _l, _p in res for v in b)
    assert all(isinstance(p, float) and 0 <= p <= 1 for p in probs)
    assert box_iou(res[0][0], (100, 50, 200, 150)) > 0.97
    assert box_iou(res[1][0], (300, 200, 380, 330)) > 0.97
    assert res[2][0] == (600, 400, 640, 480)                # clipped to the image
    blob = runner.blobs[-1]
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert blob[0, 2, 320, 320] == pytest.approx(1.0)       # RGB order: blue is channel 2
    assert blob[0, 0, 320, 320] == pytest.approx(0.0)
    assert blob[0, 0, 0, 0] == pytest.approx(114 / 255.0)   # letterbox padding
    assert det.last_latency_ms > 0
    assert set(det.last_timings_ms) == {'pre', 'infer', 'post'}
    assert det.labels == LABELS and det.input_size == 640


def test_detect_label_map(tmp_path, monkeypatch):
    out = make_output(OBJS, 3, (480, 640))
    det, _r = make_detector(tmp_path, monkeypatch, LABELS, out, model_bytes=fake_onnx(),
                            label_map={'banana': '', 'carrot': 'apple'})
    assert det.labels == ['apple']
    res = det.detect(np.zeros((480, 640, 3), np.uint8))
    # The carrot on the apple now has the same task label: suppressed by class-wise NMS.
    assert [lab for _b, lab, _p in res] == ['apple', 'apple']
    assert [round(p, 2) for _b, _l, p in res] == [0.9, 0.7]


def test_detect_nothing_and_bad_images(tmp_path, monkeypatch):
    out = make_output([], 3, (480, 640))
    det, _r = make_detector(tmp_path, monkeypatch, LABELS, out, model_bytes=fake_onnx())
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []
    for bad in (None, np.zeros((480, 640), np.uint8), np.zeros((480, 640, 3), np.float32),
                np.zeros((480, 640, 4), np.uint8), np.zeros((0, 640, 3), np.uint8)):
        with pytest.raises(ValueError):
            det.detect(bad)


def test_constructor_errors(tmp_path, monkeypatch):
    out = make_output([], 3, (480, 640))
    monkeypatch.setattr(OnnxDetector, '_make_runner', lambda self, b, i: FakeRunner(out))
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx())
    cfg.set('perception.classifier.onnx.model_file', str(tmp_path / 'nope.onnx'))
    with pytest.raises(DetectorError, match='ONNX model not found'):
        OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx())
    cfg.set('perception.classifier.onnx.model_file', '')
    with pytest.raises(DetectorError, match='model_file is not set'):
        OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx())
    cfg.set('perception.classifier.onnx.labels_file', str(tmp_path / 'nope.txt'))
    with pytest.raises(DetectorError, match='labels file not found.*--write-labels'):
        OnnxDetector(cfg)
    for key, value, msg in (('conf_threshold', 0.0, r'\(0, 1\]'),
                            ('nms_threshold', 1.5, r'\(0, 1\]'),
                            ('conf_threshold', 'high', 'number'),
                            ('input_size', 600, 'multiple of 32'),
                            ('input_size', 320, 'exported for 640 x 640'),
                            ('input_size', '640', 'integer')):
        cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(), **{key: value})
        with pytest.raises(DetectorError, match=msg):
            OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(), input_size=None)
    assert OnnxDetector(cfg).input_size == 640


def test_model_and_labels_must_agree(tmp_path, monkeypatch):
    out = make_output([], 3, (480, 640))
    monkeypatch.setattr(OnnxDetector, '_make_runner', lambda self, b, i: FakeRunner(out))
    cfg = make_cfg(tmp_path, ['apple', 'carrot', 'banana'], model_bytes=fake_onnx())
    with pytest.raises(DetectorError, match="index 1 is 'carrot' in the file but 'banana'"):
        OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, ['apple', 'banana'], model_bytes=fake_onnx())
    with pytest.raises(DetectorError, match='has 2 names but model.onnx has 3 classes'):
        OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(task='segment'))
    with pytest.raises(DetectorError, match="'segment' model"):
        OnnxDetector(cfg)
    cfg = make_cfg(tmp_path, LABELS, model_bytes=fake_onnx(end2end='True'))
    with pytest.raises(DetectorError, match='end-to-end'):
        OnnxDetector(cfg)


def test_warmup_detects_class_count_mismatch(tmp_path, monkeypatch):
    out = make_output([], 5, (480, 640))                  # model with 5 classes
    with pytest.raises(DetectorError, match='model has 5 classes'):
        make_detector(tmp_path, monkeypatch, LABELS, out)  # unparsable model: no metadata


def test_from_paths(tmp_path, monkeypatch):
    out = make_output([(10, 10, 50, 50, 1, 0.9)], 3, (100, 100))
    monkeypatch.setattr(OnnxDetector, '_make_runner', lambda self, b, i: FakeRunner(out))
    m = tmp_path / 'm.onnx'
    m.write_bytes(fake_onnx())
    lab = write_labels(tmp_path, LABELS)
    det = OnnxDetector.from_paths(str(m), str(lab), conf_threshold=0.5)
    assert det.conf_threshold == 0.5
    assert [x[1] for x in det.detect(np.zeros((100, 100, 3), np.uint8))] == ['banana']


# --------------------------------------------------------------------------
# Real model (models/yolo11n_coco.onnx, written by tools/yolo/export_onnx.py)
# --------------------------------------------------------------------------

needs_coco = pytest.mark.skipif(not (COCO_MODEL.is_file() and COCO_LABELS.is_file()),
                                reason='models/yolo11n_coco.onnx not exported (docs/yolo.md)')


@needs_coco
def test_coco_model_opencv_backend():
    det = OnnxDetector.from_paths(str(COCO_MODEL), str(COCO_LABELS), backend='opencv')
    assert det.backend == 'opencv' and det.input_size == 640 and len(det.model_labels) == 80
    rng = np.random.RandomState(0)
    img = np.full((480, 640, 3), (170, 180, 190), np.uint8)
    for _ in range(5):
        cv2.circle(img, (int(rng.randint(60, 580)), int(rng.randint(60, 420))),
                   int(rng.randint(20, 50)), tuple(int(v) for v in rng.randint(0, 255, 3)), -1)
    res = det.detect(img)
    for (x0, y0, x1, y1), lab, p in res:
        assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480
        assert lab in det.model_labels and 0.35 <= p <= 1.0
    assert 0 < det.last_latency_ms < 2000


@needs_coco
def test_coco_model_label_map_wildcard():
    det = OnnxDetector.from_paths(str(COCO_MODEL), str(COCO_LABELS), backend='opencv',
                                  label_map={'*': '', 'apple': 'apple', 'orange': 'orange',
                                             'sports ball': 'orange'})
    assert det.labels == ['orange', 'apple']   # model-class order: sports ball (32) first


@pytest.mark.skipif(not COCO_RAW.is_file(), reason='models/yolo11n_coco.raw.onnx not exported')
def test_unpatched_export_is_rejected_by_opencv_backend():
    with pytest.raises(DetectorError, match='constant first operand'):
        OnnxDetector.from_paths(str(COCO_RAW), str(COCO_LABELS), backend='opencv')


BUS = Path.home() / '.venvs/piper-yolo/lib/python3.10/site-packages/ultralytics/assets/bus.jpg'


@needs_coco
@pytest.mark.skipif(not BUS.is_file(), reason='ultralytics sample image not available')
def test_coco_model_finds_bus_and_people():
    det = OnnxDetector.from_paths(str(COCO_MODEL), str(COCO_LABELS), backend='opencv')
    labels = [lab for _b, lab, _p in det.detect(cv2.imread(str(BUS)))]
    assert 'bus' in labels and labels.count('person') >= 3

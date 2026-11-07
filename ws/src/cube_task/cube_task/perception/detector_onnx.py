"""
YOLOv8 / YOLO11 ONNX object detector (L2 fruit / vegetable recognition).

Runs a fixed-size, single-image ultralytics ONNX export with OpenCV DNN (CPU,
works with the system OpenCV 4.5.4) or with onnxruntime (optional, imported
lazily). Expected model output: [1, 4 + nc, N] (ultralytics default) or the
transposed [1, N, 4 + nc]; rows are (cx, cy, w, h) in model-input pixels
followed by nc class probabilities (already sigmoid, no objectness, no NMS
inside the model). docs/yolo.md lists the verified export settings.

Config (perception.classifier.onnx.*):
    model_file, labels_file   paths relative to the package root
    input_size                square model input in pixels (multiple of 32);
                              null = take it from the model's static input shape
    conf_threshold            minimum class probability, (0, 1]
    nms_threshold             IoU above which a lower-scored box of the same
                              output label is suppressed, (0, 1]
    backend                   auto | opencv | onnxruntime
    label_map                 {model label: task label}. '' drops the label.
                              Key '*' maps every label not listed explicitly.

Output boxes are (x0, y0, x1, y1) integer pixels of the input color image,
x1 / y1 exclusive (usable as slice ends), clipped to [0, W] x [0, H].
"""
from __future__ import annotations

import ast
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.interfaces import Detector2D

LOG = logging.getLogger('cube_task.perception.detector_onnx')

CFG_KEY = 'perception.classifier.onnx'
BACKENDS = ('auto', 'opencv', 'onnxruntime')
PAD_VALUE = 114                 # ultralytics letterbox padding gray
MAX_DET = 100                   # detections kept per image after NMS
STRIDES = (8, 16, 32)           # YOLOv8 / YOLO11 P3-P5 detection strides
DEFAULT_CONF = 0.35             # same values as config/default.yaml
DEFAULT_NMS = 0.45

Box = Tuple[int, int, int, int]
DetectionTuple = Tuple[Box, str, float]


class DetectorError(RuntimeError):
    """Raised when the model, the labels file or the detector config is unusable."""


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def read_labels(path: Path) -> List[str]:
    """
    Read class names (UTF-8, one per line, in class-index order).

    Trailing blank lines are ignored. A blank line before the last name is an
    error because it would shift every following class index.
    """
    p = Path(path)
    if not p.is_file():
        raise DetectorError('labels file not found: %s' % p)
    lines = p.read_text(encoding='utf-8-sig').splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    labels = [ln.strip() for ln in lines]
    if not labels:
        raise DetectorError('labels file is empty: %s' % p)
    for i, name in enumerate(labels):
        if not name:
            raise DetectorError('%s: line %d is blank; write one class name per line '
                                'in class-index order' % (p, i + 1))
    dup = sorted({n for n in labels if labels.count(n) > 1})
    if dup:
        LOG.warning('%s: duplicate class names %s', p, dup)
    return labels


def build_label_mapping(model_labels: Sequence[str],
                        label_map: Optional[Dict[Any, Any]]
                        ) -> Tuple[List[str], List[str]]:
    """
    Apply label_map to the model labels.

    Returns (mapped, out_labels): mapped[i] is the task label of model class i
    ('' = dropped); out_labels lists the distinct non-empty task labels in
    model-class order. Keys not listed keep their model name unless the key
    '*' is present, whose value then applies to them.
    """
    if label_map is None:
        label_map = {}
    if not isinstance(label_map, dict):
        raise DetectorError('label_map must be a mapping {model label: task label}, got %r'
                            % (label_map,))
    lm: Dict[str, str] = {}
    for k, v in label_map.items():
        if v is None:
            raise DetectorError("label_map[%r] is null; use '' to drop a label" % (k,))
        lm[str(k).strip()] = str(v).strip()
    known = set(model_labels)
    unknown = sorted(k for k in lm if k != '*' and k not in known)
    if unknown:
        LOG.warning('label_map keys %s are not model labels (typo?); model labels: %s',
                    unknown, list(model_labels))
    default = lm.get('*')
    mapped = []
    for name in model_labels:
        if name in lm:
            mapped.append(lm[name])
        elif default is not None:
            mapped.append(default)
        else:
            mapped.append(name)
    # perception/pipeline.py applies label_map once more to the labels returned by
    # detect(), so the map must be idempotent: no swaps or chains (a -> b, b -> c).
    for m in mapped:
        if m and m in lm and lm[m] != m:
            raise DetectorError('label_map maps to %r, which is itself mapped to %r; swaps and '
                                'chains are not allowed (the label would be renamed twice)'
                                % (m, lm[m]))
    out: List[str] = []
    for m in mapped:
        if m and m not in out:
            out.append(m)
    return mapped, out


# --------------------------------------------------------------------------
# Pre- / post-processing (pure functions, unit tested)
# --------------------------------------------------------------------------

def letterbox(img: np.ndarray, size: int, pad_value: int = PAD_VALUE
              ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize keeping the aspect ratio, then pad to size x size (ultralytics LetterBox).

    Returns (image, r, (pad_left, pad_top)). An input pixel coordinate u maps
    to u * r + pad_left in the letterboxed image (same for v with pad_top).
    """
    h, w = img.shape[:2]
    r = min(size / float(h), size / float(w))
    nw = max(1, int(round(w * r)))
    nh = max(1, int(round(h * r)))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (size - nw) / 2.0
    dh = (size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(pad_value, pad_value, pad_value))
    if out.shape[0] != size or out.shape[1] != size:
        raise AssertionError('letterbox produced %s for size %d' % (out.shape, size))
    return out, r, (left, top)


def decode_output(output: np.ndarray, num_classes: int, conf_threshold: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode a YOLOv8 / YOLO11 output tensor (no NMS).

    output: [1, 4 + nc, N] or [1, N, 4 + nc] (batch dim optional). Box rows
    are (cx, cy, w, h) in model-input pixels; class scores are probabilities.
    Returns (boxes Kx4 xyxy float32 in model-input pixels, scores K float32,
    class_ids K int64) for rows whose best class score >= conf_threshold.
    Rows with non-finite values or non-positive w / h are dropped.
    """
    a = np.asarray(output)
    shape = tuple(a.shape)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise DetectorError('model output %s has batch size %d; export with batch=1'
                                % (shape, a.shape[0]))
        a = a[0]
    if a.ndim != 2:
        raise DetectorError('unexpected model output shape %s; expected [1, 4+nc, N]' % (shape,))
    c = 4 + int(num_classes)
    if a.shape[0] == c:
        pred = a.T                      # ultralytics default [4+nc, N]
    elif a.shape[1] == c:
        pred = a                        # transposed [N, 4+nc]
    else:
        raise DetectorError(_shape_mismatch_message(shape, num_classes))
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape[0] == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64))
    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(pred.shape[0]), class_ids]
    finite = np.isfinite(pred).all(axis=1)
    keep = finite & (scores >= conf_threshold) & (pred[:, 2] > 0) & (pred[:, 3] > 0)
    p = pred[keep]
    half_w = p[:, 2] / 2.0
    half_h = p[:, 3] / 2.0
    boxes = np.stack([p[:, 0] - half_w, p[:, 1] - half_h,
                      p[:, 0] + half_w, p[:, 1] + half_h], axis=1).astype(np.float32)
    return boxes, scores[keep].astype(np.float32), class_ids[keep].astype(np.int64)


def _shape_mismatch_message(shape: Tuple[int, ...], num_classes: int) -> str:
    dims = [d for d in shape if d != 1] or list(shape)
    small = min(dims) if dims else 0
    msg = ('model output shape %s does not match the labels file: %d labels need a '
           'dimension of 4 + %d = %d' % (shape, num_classes, num_classes, num_classes + 4))
    if small == 6 and max(dims) <= 1000:
        msg += ('. The shape looks like an end-to-end (NMS inside) export; re-export a '
                'YOLO11 / YOLOv8 model with nms=False (see docs/yolo.md)')
    elif small - 5 == num_classes:
        msg += ('. The shape looks like a YOLOv5-style output with objectness; use a '
                'YOLO11 / YOLOv8 model')
    elif small > 4:
        msg += ('. The model has %d classes; use the labels file written together with this '
                'model (one name per class)' % (small - 4))
    return msg


def nms_per_label(boxes: np.ndarray, scores: np.ndarray, groups: Sequence[Any],
                  nms_threshold: float, max_det: int = MAX_DET) -> np.ndarray:
    """
    Class-wise NMS with cv2.dnn.NMSBoxes.

    boxes: Kx4 xyxy (any pixel frame), scores: K, groups: K hashable labels;
    boxes only suppress boxes of the same group. Returns indices into the
    inputs, sorted by descending score, at most max_det of them.
    """
    n = int(len(scores))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(n, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(n)
    group_arr = list(groups)
    if len(group_arr) != n:
        raise ValueError('nms_per_label: %d groups for %d boxes' % (len(group_arr), n))
    kept: List[int] = []
    by_group: Dict[Any, List[int]] = {}
    for i, g in enumerate(group_arr):
        by_group.setdefault(g, []).append(i)
    for idx in by_group.values():
        b = boxes[idx]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
        # score_threshold 0: the confidence filter ran before NMS (NMSBoxes keeps
        # score > threshold and asserts threshold >= 0).
        res = cv2.dnn.NMSBoxes(xywh.tolist(), scores[idx].tolist(), 0.0,
                               float(nms_threshold))
        for j in np.asarray(res, dtype=np.int64).reshape(-1):
            kept.append(idx[int(j)])
    order = sorted(kept, key=lambda i: (-scores[i], i))
    return np.asarray(order[:max_det], dtype=np.int64)


def unletterbox_boxes(boxes: np.ndarray, r: float, pad: Tuple[int, int],
                      width: int, height: int) -> np.ndarray:
    """
    Map Kx4 xyxy boxes from letterboxed model input to image pixels, clipped.

    Result is float, x in [0, width], y in [0, height].
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if r <= 0:
        raise ValueError('letterbox scale must be positive, got %r' % r)
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / r
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(width))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(height))
    return b


def to_pixel_box(box: Sequence[float]) -> Box:
    """Round a float xyxy box to integer pixels."""
    return (int(round(float(box[0]))), int(round(float(box[1]))),
            int(round(float(box[2]))), int(round(float(box[3]))))


# --------------------------------------------------------------------------
# Minimal ONNX protobuf reader (no onnx package at runtime)
# --------------------------------------------------------------------------

class OnnxInfo:
    """
    Facts read from an ONNX file without the onnx package.

    input_shape: list of int (static) or str (symbolic) dims of the first
    graph input; metadata: ModelProto.metadata_props (ultralytics stores
    names, imgsz, task, end2end there); const_first_ops: names of Sub / Div
    nodes whose first operand is a constant and second one is not (OpenCV
    < 4.7 computes those with the operands swapped).
    """

    def __init__(self) -> None:
        self.ir_version = 0
        self.opsets: Dict[str, int] = {}
        self.input_name = ''
        self.input_shape: List[Any] = []
        self.output_names: List[str] = []
        self.metadata: Dict[str, str] = {}
        self.const_first_ops: List[str] = []

    def names(self) -> Optional[List[str]]:
        """Class names embedded by ultralytics (index order), or None."""
        raw = self.metadata.get('names')
        if not raw:
            return None
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
        if isinstance(val, dict):
            try:
                keys = sorted(int(k) for k in val)
            except (TypeError, ValueError):
                return None
            if keys != list(range(len(keys))):
                return None
            return [str(val[k]).strip() for k in keys]
        if isinstance(val, (list, tuple)):
            return [str(v).strip() for v in val]
        return None


def _varint(buf: memoryview, pos: int, end: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= end:
            raise ValueError('truncated varint')
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError('varint too long')


def _fields(buf: memoryview, start: int, end: int):
    """Yield (field, wire_type, value) of one message; wire type 2 gives (start, end)."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos, end)
        field, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = _varint(buf, pos, end)
            yield field, wt, val
        elif wt == 1:
            pos += 8
        elif wt == 2:
            n, pos = _varint(buf, pos, end)
            if pos + n > end:
                raise ValueError('truncated field %d' % field)
            yield field, wt, (pos, pos + n)
            pos += n
        elif wt == 5:
            pos += 4
        else:
            raise ValueError('unsupported protobuf wire type %d' % wt)
    if pos != end:
        raise ValueError('message overruns its length')


def _text(buf: memoryview, span: Tuple[int, int]) -> str:
    return bytes(buf[span[0]:span[1]]).decode('utf-8', errors='replace')


def _value_info(buf: memoryview, span: Tuple[int, int]) -> Tuple[str, List[Any]]:
    name, dims = '', []
    for f, wt, v in _fields(buf, *span):
        if f == 1 and wt == 2:
            name = _text(buf, v)
        elif f == 2 and wt == 2:                          # TypeProto
            for f2, wt2, v2 in _fields(buf, *v):
                if f2 != 1 or wt2 != 2:                   # tensor_type
                    continue
                for f3, wt3, v3 in _fields(buf, *v2):
                    if f3 != 2 or wt3 != 2:               # shape
                        continue
                    for f4, wt4, v4 in _fields(buf, *v3):
                        if f4 != 1 or wt4 != 2:           # dim
                            continue
                        dim: Any = '?'
                        for f5, wt5, v5 in _fields(buf, *v4):
                            if f5 == 1 and wt5 == 0:
                                dim = int(v5)
                            elif f5 == 2 and wt5 == 2:
                                dim = _text(buf, v5)
                        dims.append(dim)
    return name, dims


def read_onnx_info(path: Path) -> OnnxInfo:
    """Parse the parts of an ONNX ModelProto the detector checks. Raise ValueError if malformed."""
    data = memoryview(Path(path).read_bytes())
    info = OnnxInfo()
    graph_span = None
    for f, wt, v in _fields(data, 0, len(data)):
        if f == 1 and wt == 0:
            info.ir_version = int(v)
        elif f == 7 and wt == 2:
            graph_span = v
        elif f == 8 and wt == 2:
            dom, ver = '', 0
            for f2, wt2, v2 in _fields(data, *v):
                if f2 == 1 and wt2 == 2:
                    dom = _text(data, v2)
                elif f2 == 2 and wt2 == 0:
                    ver = int(v2)
            info.opsets[dom] = ver
        elif f == 14 and wt == 2:
            key, val = '', ''
            for f2, wt2, v2 in _fields(data, *v):
                if f2 == 1 and wt2 == 2:
                    key = _text(data, v2)
                elif f2 == 2 and wt2 == 2:
                    val = _text(data, v2)
            info.metadata[key] = val
    if graph_span is None:
        raise ValueError('no graph in %s' % path)
    consts = set()
    nodes: List[Tuple[str, str, List[str]]] = []
    inputs: List[Tuple[str, List[Any]]] = []
    for f, wt, v in _fields(data, *graph_span):
        if wt != 2:
            continue
        if f == 1:                                        # NodeProto
            ins: List[str] = []
            outs: List[str] = []
            name = op = ''
            for f2, wt2, v2 in _fields(data, *v):
                if wt2 != 2:
                    continue
                if f2 == 1:
                    ins.append(_text(data, v2))
                elif f2 == 2:
                    outs.append(_text(data, v2))
                elif f2 == 3:
                    name = _text(data, v2)
                elif f2 == 4:
                    op = _text(data, v2)
            if op == 'Constant':
                consts.update(outs)
            nodes.append((name, op, ins))
        elif f == 5:                                      # TensorProto initializer
            for f2, wt2, v2 in _fields(data, *v):
                if f2 == 8 and wt2 == 2:
                    consts.add(_text(data, v2))
        elif f == 11:
            inputs.append(_value_info(data, v))
        elif f == 12:
            info.output_names.append(_value_info(data, v)[0])
    for name, dims in inputs:
        if name not in consts:
            info.input_name, info.input_shape = name, dims
            break
    for name, op, ins in nodes:
        if op in ('Sub', 'Div') and len(ins) == 2 and ins[0] in consts and ins[1] not in consts:
            info.const_first_ops.append(name or op)
    return info


# --------------------------------------------------------------------------
# Inference backends
# --------------------------------------------------------------------------

def opencv_version() -> Tuple[int, int]:
    """(major, minor) of the imported cv2."""
    parts = cv2.__version__.split('.')
    return int(parts[0]), int(parts[1])


class _OpenCVRunner:
    """cv2.dnn on the CPU (OpenCV 4.5.4 from Ubuntu has no CUDA DNN backend)."""

    name = 'opencv'

    def __init__(self, path: Path, info: Optional[OnnxInfo]):
        if info is not None and info.const_first_ops:
            raise DetectorError(
                '%s contains %s with a constant first operand (const - x); OpenCV %s '
                'computes x - const there, so the boxes would be wrong. Re-export with '
                'cube_task/tools/yolo/export_onnx.py (it patches the graph), or use '
                'backend: onnxruntime. See docs/yolo.md'
                % (path.name, info.const_first_ops, cv2.__version__))
        try:
            net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as e:
            raise DetectorError(
                'OpenCV %s cannot load %s: %s. Export with simplify=True and the patch in '
                'cube_task/tools/yolo/export_onnx.py (docs/yolo.md)'
                % (cv2.__version__, path, _first_line(e)))
        if net.empty():
            raise DetectorError('OpenCV loaded an empty network from %s' % path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        outs = list(net.getUnconnectedOutLayersNames())
        if len(outs) != 1:
            raise DetectorError('%s has %d outputs %s; only single-output detection '
                                'models are supported (not -seg / -pose)'
                                % (path.name, len(outs), outs))
        self._net = net
        self._out = outs[0]

    def run(self, blob: np.ndarray) -> np.ndarray:
        self._net.setInput(blob)
        try:
            return np.asarray(self._net.forward(self._out))
        except cv2.error as e:
            raise DetectorError(
                'OpenCV %s failed to run the model: %s. Export it with '
                'cube_task/tools/yolo/export_onnx.py (verified settings in docs/yolo.md)'
                % (cv2.__version__, _first_line(e)))


class _OrtRunner:
    """onnxruntime CPU session (optional dependency, imported here)."""

    name = 'onnxruntime'

    def __init__(self, path: Path, info: Optional[OnnxInfo]):
        try:
            import onnxruntime as ort  # noqa: F401 (optional dependency)
        except ImportError as e:
            raise DetectorError('backend onnxruntime requested but onnxruntime cannot be '
                                'imported (%s). Use backend: opencv, or see docs/yolo.md '
                                'for the version that keeps numpy 1.21.5' % e)
        try:
            sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        except Exception as e:  # onnxruntime raises its own exception types
            raise DetectorError('onnxruntime %s cannot load %s: %s'
                                % (getattr(ort, '__version__', '?'), path, _first_line(e)))
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise DetectorError('%s has %d inputs / %d outputs; expected 1 / 1 '
                                '(detection model, not -seg / -pose)'
                                % (path.name, len(inputs), len(outputs)))
        self._sess = sess
        self._in = inputs[0].name
        self._out = outputs[0].name
        self._fp16 = 'float16' in str(inputs[0].type)
        self.input_shape = list(inputs[0].shape)

    def run(self, blob: np.ndarray) -> np.ndarray:
        x = blob.astype(np.float16) if self._fp16 else blob
        return np.asarray(self._sess.run([self._out], {self._in: x})[0], dtype=np.float32)


def _first_line(e: BaseException, limit: int = 400) -> str:
    """Collapse an exception message (OpenCV ones span several lines) to one line."""
    text = ' '.join(ln.strip() for ln in str(e).splitlines() if ln.strip()) or repr(e)
    return text if len(text) <= limit else text[:limit] + '...'


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

class OnnxDetector(Detector2D):
    """
    YOLOv8 / YOLO11 ONNX detector (implements core.interfaces.Detector2D).

    detect(color) -> [((x0, y0, x1, y1), label, prob), ...] sorted by
    descending prob: integer pixel boxes in the color image (x1 / y1
    exclusive, clipped), task labels after label_map, prob in [0, 1].

    Public attributes:
        labels          task labels this detector can output (after label_map)
        model_labels    class names of the model, index order (labels file)
        backend         'opencv' or 'onnxruntime' (after resolving 'auto')
        input_size      square model input size (pixels)
        last_latency_ms wall time of the last detect() call (0.0 before the
                        first one); diagnostics only, not used for control
        last_timings_ms {'pre', 'infer', 'post'} of the last call
        warmup_ms       time of the warm-up inference run by the constructor

    The constructor validates everything up front (files, class count, class
    names embedded by ultralytics, input size, backend) and runs one warm-up
    inference, so a bad deployment fails at startup, never mid-run. Calls to
    detect() are serialized by a lock; the detector is meant for one thread.
    """

    def __init__(self, cfg: Any, section: str = CFG_KEY):
        self.section = section
        key = section + '.'
        model_path = cfg.path(key + 'model_file')
        labels_path = cfg.path(key + 'labels_file')
        if model_path is None:
            raise DetectorError('%smodel_file is not set' % key)
        if labels_path is None:
            raise DetectorError('%slabels_file is not set' % key)
        if not model_path.is_file():
            raise DetectorError('ONNX model not found: %s (%smodel_file). Train or export one '
                                '(docs/yolo.md) and copy it there' % (model_path, key))
        self.model_path = model_path
        self.labels_path = labels_path
        self.conf_threshold = _probability(cfg.get(key + 'conf_threshold', DEFAULT_CONF),
                                           key + 'conf_threshold')
        self.nms_threshold = _probability(cfg.get(key + 'nms_threshold', DEFAULT_NMS),
                                          key + 'nms_threshold')

        info: Optional[OnnxInfo] = None
        try:
            info = read_onnx_info(model_path)
        except (ValueError, OSError) as e:
            LOG.warning('cannot parse %s (%s); class-name / input-size checks skipped',
                        model_path, e)
        self.info = info
        if info is not None:
            _check_metadata(info, model_path)

        if not labels_path.is_file():
            hint = ''
            if info is not None and info.names():
                hint = (' The model embeds the names; write them with: python3 -m '
                        'cube_task.tools.yolo.check_onnx --model %s --write-labels %s'
                        % (model_path, labels_path))
            raise DetectorError('labels file not found: %s (%slabels_file).%s'
                                % (labels_path, key, hint))
        self.model_labels = read_labels(labels_path)
        if info is not None:
            _check_names(info, self.model_labels, model_path, labels_path)
        self.num_classes = len(self.model_labels)

        self.input_size = _resolve_input_size(cfg.get(key + 'input_size', None), info, key)
        self._mapped, self.labels = build_label_mapping(self.model_labels,
                                                        cfg.get(key + 'label_map', {}))
        self._runner = self._make_runner(str(cfg.get(key + 'backend', 'auto')), info)
        self.backend = self._runner.name
        ort_shape = getattr(self._runner, 'input_shape', None)
        if ort_shape and all(isinstance(d, int) for d in ort_shape[2:4]):
            if list(ort_shape[2:4]) != [self.input_size, self.input_size]:
                raise DetectorError('%s expects input %s but %sinput_size is %d'
                                    % (model_path.name, ort_shape, key, self.input_size))
        self._lock = threading.Lock()
        self.last_latency_ms = 0.0
        self.last_timings_ms: Dict[str, float] = {}
        self.warmup_ms = self._warmup()
        LOG.info('ONNX detector %s: backend %s (cv2 %s), input %d, %d classes -> labels %s, '
                 'conf %.2f, nms %.2f, warm-up %.0f ms', model_path.name, self.backend,
                 cv2.__version__, self.input_size, self.num_classes, self.labels,
                 self.conf_threshold, self.nms_threshold, self.warmup_ms)

    @classmethod
    def from_paths(cls, model_file: str, labels_file: str, **options: Any) -> 'OnnxDetector':
        """
        Build a detector without a full config (tools, tests).

        options: any perception.classifier.onnx key (input_size,
        conf_threshold, nms_threshold, backend, label_map).
        """
        from ..core.config import Config
        sec = dict(options)
        sec['model_file'] = str(Path(model_file).expanduser().resolve())
        sec['labels_file'] = str(Path(labels_file).expanduser().resolve())
        data = {'perception': {'classifier': {'onnx': sec}}}
        return cls(Config(data, Path.cwd(), ['from_paths']))

    def _make_runner(self, backend: str, info: Optional[OnnxInfo]):
        b = backend.strip().lower()
        if b not in BACKENDS:
            raise DetectorError('%s.backend must be one of %s, got %r'
                                % (self.section, '|'.join(BACKENDS), backend))
        if b == 'opencv':
            return _OpenCVRunner(self.model_path, info)
        if b == 'onnxruntime':
            return _OrtRunner(self.model_path, info)
        if _onnxruntime_available():
            try:
                return _OrtRunner(self.model_path, info)
            except DetectorError as e:
                LOG.warning('backend auto: onnxruntime failed (%s); trying opencv', e)
        return _OpenCVRunner(self.model_path, info)

    def _warmup(self) -> float:
        blob = np.full((1, 3, self.input_size, self.input_size), PAD_VALUE / 255.0,
                       dtype=np.float32)
        t0 = time.perf_counter()
        out = self._runner.run(blob)
        dt = (time.perf_counter() - t0) * 1e3
        decode_output(out, self.num_classes, 1.0)        # raises on a shape mismatch
        n = max(int(d) for d in np.asarray(out).shape)
        expected = sum((self.input_size // s) ** 2 for s in STRIDES)
        if n != expected:
            LOG.warning('%s: %d candidate boxes, a P3-P5 YOLOv8 / YOLO11 head at %d px '
                        'gives %d; check input_size', self.model_path.name, n,
                        self.input_size, expected)
        return dt

    def detect(self, color: np.ndarray) -> List[DetectionTuple]:
        """Detect objects in a BGR uint8 HxWx3 image (see the class docstring)."""
        _check_image(color)
        h, w = color.shape[:2]
        with self._lock:
            t0 = time.perf_counter()
            lb, r, pad = letterbox(color, self.input_size)
            blob = cv2.dnn.blobFromImage(lb, scalefactor=1.0 / 255.0, swapRB=True, crop=False)
            t1 = time.perf_counter()
            out = self._runner.run(blob)
            t2 = time.perf_counter()
            dets = self._postprocess(out, r, pad, w, h)
            t3 = time.perf_counter()
            self.last_timings_ms = {'pre': (t1 - t0) * 1e3, 'infer': (t2 - t1) * 1e3,
                                    'post': (t3 - t2) * 1e3}
            self.last_latency_ms = (t3 - t0) * 1e3
        return dets

    def _postprocess(self, out: np.ndarray, r: float, pad: Tuple[int, int],
                     width: int, height: int) -> List[DetectionTuple]:
        boxes, scores, cls = decode_output(out, self.num_classes, self.conf_threshold)
        labels = [self._mapped[int(c)] for c in cls]
        keep = [i for i, lab in enumerate(labels) if lab]
        if not keep:
            return []
        boxes, scores = boxes[keep], scores[keep]
        labels = [labels[i] for i in keep]
        order = nms_per_label(boxes, scores, labels, self.nms_threshold, MAX_DET)
        img_boxes = unletterbox_boxes(boxes[order], r, pad, width, height)
        result: List[DetectionTuple] = []
        for k, i in enumerate(order):
            b = to_pixel_box(img_boxes[k])
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            result.append((b, labels[int(i)], float(scores[int(i)])))
        return result


def _probability(value: Any, key: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise DetectorError('%s must be a number in (0, 1], got %r' % (key, value))
    if not (0.0 < v <= 1.0) or math.isnan(v):
        raise DetectorError('%s must be in (0, 1], got %r' % (key, value))
    return v


def _resolve_input_size(value: Any, info: Optional[OnnxInfo], key: str) -> int:
    model_hw = None
    if info is not None and len(info.input_shape) == 4:
        hw = info.input_shape[2:4]
        if all(isinstance(d, int) and d > 0 for d in hw):
            model_hw = (int(hw[0]), int(hw[1]))
    if model_hw is not None and model_hw[0] != model_hw[1]:
        raise DetectorError('model input %s is not square; export with a single imgsz'
                            % (info.input_shape,))
    if value is None:
        if model_hw is None:
            raise DetectorError('%sinput_size is null and the model input size is unknown '
                                '(dynamic or unreadable); set input_size' % key)
        return model_hw[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise DetectorError('%sinput_size must be an integer (pixels), got %r' % (key, value))
    size = int(value)
    if size < 32 or size % 32:
        raise DetectorError('%sinput_size must be a positive multiple of 32, got %d'
                            % (key, size))
    if model_hw is not None and model_hw[0] != size:
        raise DetectorError('%sinput_size is %d but the model was exported for %d x %d; set '
                            'input_size: %d (or null)'
                            % (key, size, model_hw[0], model_hw[1], model_hw[0]))
    return size


def _check_metadata(info: OnnxInfo, path: Path) -> None:
    task = info.metadata.get('task', '').strip()
    if task and task != 'detect':
        raise DetectorError('%s is a %r model; only detection models are supported'
                            % (path.name, task))
    if info.metadata.get('end2end', '').strip().lower() == 'true':
        raise DetectorError('%s is an end-to-end export (NMS inside the model, output '
                            '[1, 300, 6]); re-export with end2end=False / nms=False'
                            % path.name)
    if info.output_names and len(info.output_names) != 1:
        raise DetectorError('%s has %d outputs %s; only single-output detection models are '
                            'supported' % (path.name, len(info.output_names),
                                           info.output_names))


def _check_names(info: OnnxInfo, labels: List[str], model_path: Path,
                 labels_path: Path) -> None:
    names = info.names()
    if names is None or names == labels:
        return
    if len(names) != len(labels):
        raise DetectorError('%s has %d names but %s has %d classes %s; use the labels file '
                            'exported with this model'
                            % (labels_path, len(labels), model_path.name, len(names), names))
    i = next(k for k in range(len(names)) if names[k] != labels[k])
    raise DetectorError('%s does not match the class names stored in %s: index %d is %r in '
                        'the file but %r in the model. Use the labels file exported with the '
                        'model; rename classes with label_map instead'
                        % (labels_path, model_path.name, i, labels[i], names[i]))


def _check_image(color: Any) -> None:
    if not isinstance(color, np.ndarray):
        raise ValueError('color must be a numpy array, got %s' % type(color).__name__)
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError('color must be HxWx3 uint8 BGR, got %s %s' % (color.dtype, color.shape))
    if color.shape[0] == 0 or color.shape[1] == 0:
        raise ValueError('color image is empty: %s' % (color.shape,))


__all__: List[str] = ['OnnxDetector', 'DetectorError', 'read_labels', 'build_label_mapping',
                      'letterbox', 'decode_output', 'nms_per_label', 'unletterbox_boxes',
                      'read_onnx_info', 'OnnxInfo']

"""
Build a YOLO-format detection dataset from capture sessions of ONE class each.

Every session is a directory recorded by the capture tool (camera/recorder.py:
<idx:06d>_color.png / _depth.png / _depth.npy / _meta.json), or a plain folder
of images (.png / .jpg) for --boxes model. All objects in a session have the
class given on the command line, so only boxes are needed:

  --boxes plane       (default) depth points above the dominant plane, in the
                      camera frame; needs depth, no extrinsics, no arm.
  --boxes perception  instances from perception.pipeline.Perception (pick
                      polygon, zone exclusion); needs depth and extrinsics.
  --boxes model       class-agnostic boxes from an ONNX detector (--model,
                      --labels), e.g. for RGB-only image folders.

Train / val split is by session (frames of one session are near-duplicates):
--val-session forces a session into val; otherwise --val-fraction of each
class's sessions (at least one when a class has two or more) go to val.

Example (system python3, from the package root):
    python3 -m cube_task.tools.yolo.make_dataset --out ~/vision_runs/yolo/datasets/own \
        --classes apple,banana,carrot \
        --session apple=~/captures/apple_1 --session apple=~/captures/apple_2 \
        --session banana=~/captures/banana_1 --session banana=~/captures/banana_2 \
        --roi 80,40,600,460 --preview
Check every image in <out>/preview/ before training; delete bad frames from
images/ and labels/ (same stem) or re-run with a tighter --roi.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOG = logging.getLogger('cube_task.tools.yolo.make_dataset')

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
_META_RE = re.compile(r'^(\d{6,})_meta\.json$')

Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Plane segmentation (camera frame, no extrinsics)
# --------------------------------------------------------------------------

class PlaneParams:
    """
    Parameters of the 'above the dominant plane' box finder.

    Units: meters, pixels. Heights are measured along the plane normal toward
    the camera.
    """

    def __init__(self, min_height: float = 0.008, max_height: float = 0.25,
                 min_area_m2: float = 0.00012, max_area_m2: float = 0.040,
                 min_pixels: int = 40, z_range: Tuple[float, float] = (0.15, 3.0),
                 inlier_thresh: float = 0.006, min_inlier_ratio: float = 0.3,
                 ransac_iters: int = 200, sample_points: int = 20000, morph_px: int = 3,
                 roi: Optional[Box] = None, drop_border: bool = True, pad_px: int = 0):
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.min_area_m2 = float(min_area_m2)
        self.max_area_m2 = float(max_area_m2)
        self.min_pixels = int(min_pixels)
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.inlier_thresh = float(inlier_thresh)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ransac_iters = int(ransac_iters)
        self.sample_points = int(sample_points)
        self.morph_px = int(morph_px)
        self.roi = roi
        self.drop_border = bool(drop_border)
        self.pad_px = int(pad_px)
        if not (0.0 < self.min_height < self.max_height):
            raise ValueError('need 0 < min_height < max_height')
        if not (0.0 <= self.min_area_m2 < self.max_area_m2):
            raise ValueError('need 0 <= min_area_m2 < max_area_m2')
        if not (0.0 < self.z_range[0] < self.z_range[1]):
            raise ValueError('need 0 < z_min < z_max')


def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int,
                     rng: np.random.RandomState) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Fit the dominant plane n.p + d = 0 to Nx3 camera-frame points (m).

    The normal is oriented toward the camera origin (d > 0), so n.p + d is the
    height above the plane on the camera side. Returns (n, d, inlier_ratio)
    after a least-squares refit on the inliers, or None if no plane is found.
    """
    n_pts = pts.shape[0]
    if n_pts < 3:
        return None
    best_count, best = 0, None
    # Repeated indices in a triple give a zero normal and are skipped below.
    for i in rng.randint(0, n_pts, size=(iters, 3)):
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        count = int(np.count_nonzero(np.abs(pts @ n + d) < thresh))
        if count > best_count:
            best_count, best = count, (n, d)
    if best is None:
        return None
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    if inl.shape[0] >= 3:
        c = inl.mean(axis=0)
        _u, _s, vt = np.linalg.svd(inl - c, full_matrices=False)
        n2 = vt[2]
        if float(n2 @ n) < 0:
            n2 = -n2
        n, d = n2, -float(n2 @ c)
    if d < 0:
        n, d = -n, -d
    ratio = float(np.count_nonzero(np.abs(pts @ n + d) < thresh)) / float(n_pts)
    return n, float(d), ratio


def plane_boxes(depth: np.ndarray, K: np.ndarray, params: PlaneParams,
                rng: np.random.RandomState) -> Tuple[List[Box], Dict[str, Any]]:
    """
    Return boxes of blobs above the dominant plane, plus diagnostics.

    depth: HxW float32 meters aligned to the color image (0 / NaN / inf =
    invalid). K: 3x3 color intrinsics. Boxes are (x0, y0, x1, y1) pixels with
    x1 / y1 exclusive. diag['reason'] is set when no plane could be fitted.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError('depth must be HxW, got %s' % (d.shape,))
    h, w = d.shape
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(d) & (d > params.z_range[0]) & (d < params.z_range[1])
    diag: Dict[str, Any] = {'valid_ratio': float(valid.mean()), 'reason': ''}
    if np.count_nonzero(valid) < 500:
        diag['reason'] = 'too few valid depth pixels'
        return [], diag
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if not (fx > 0 and fy > 0):
        raise ValueError('invalid intrinsics fx=%r fy=%r' % (fx, fy))
    vv, uu = np.nonzero(valid)
    z = d[vv, uu].astype(np.float64)
    pts = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=1)
    if pts.shape[0] > params.sample_points:
        sample = pts[rng.choice(pts.shape[0], params.sample_points, replace=False)]
    else:
        sample = pts
    fit = fit_plane_ransac(sample, params.inlier_thresh, params.ransac_iters, rng)
    if fit is None:
        diag['reason'] = 'no plane found'
        return [], diag
    n, dd, ratio = fit
    diag.update(plane_normal=[round(float(v), 4) for v in n], plane_d=round(dd, 4),
                inlier_ratio=round(ratio, 3))
    if ratio < params.min_inlier_ratio:
        diag['reason'] = 'dominant plane covers only %.0f %% of the points' % (100 * ratio)
        return [], diag
    height = np.zeros((h, w), np.float64)
    height[vv, uu] = pts @ n + dd
    mask = valid & (height > params.min_height) & (height < params.max_height)
    if params.roi is not None:
        x0, y0, x1, y1 = params.roi
        roi = np.zeros_like(mask)
        roi[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        mask &= roi
    m8 = mask.astype(np.uint8)
    if params.morph_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_px, params.morph_px))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    n_lab, lab, stats, _c = cv2.connectedComponentsWithStats(m8, connectivity=8)
    # Metric footprint of each pixel at its depth (camera-facing approximation).
    px_area = np.where(valid, (d.astype(np.float64) / fx) * (d.astype(np.float64) / fy), 0.0)
    areas = np.bincount(lab.ravel(), weights=px_area.ravel(), minlength=n_lab)
    boxes: List[Box] = []
    rejected: Dict[str, int] = {}
    for i in range(1, n_lab):
        x, y, bw, bh, npx = (int(v) for v in stats[i])
        why = ''
        if npx < params.min_pixels:
            why = 'pixels'
        elif not (params.min_area_m2 <= areas[i] <= params.max_area_m2):
            why = 'area'
        elif params.drop_border and (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            why = 'border'
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        p = params.pad_px
        boxes.append((max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p)))
    diag['rejected'] = rejected
    return boxes, diag


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

class SessionFrame:
    """One input image: color BGR, optional depth (m) and intrinsics, source file."""

    def __init__(self, idx: int, color: np.ndarray, depth: Optional[np.ndarray],
                 K: Optional[np.ndarray], color_path: Optional[Path], frame: Any = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.idx = idx
        self.color = color
        self.depth = depth
        self.K = K
        self.color_path = color_path
        self.frame = frame
        self.extra = extra or {}


def _load_capture_frame(d: Path, idx: int) -> SessionFrame:
    """Load one capture frame with camera/recorder.load_frame (lazy import)."""
    from cube_task.camera.recorder import load_frame
    frame, extra = load_frame(d, idx)
    color_path = d / ('%06d_color.png' % idx)
    return SessionFrame(idx, frame.color, frame.depth, frame.intrinsics.K(),
                        color_path if color_path.is_file() else None, frame, extra)


def iter_session(d: Path, every: int = 1) -> Iterator[Tuple[int, Any]]:
    """
    Yield (idx, SessionFrame or error string) for every every-th frame of d.

    Capture directories (with *_meta.json) are read through the recorder;
    other directories are read as plain image folders (no depth).
    """
    metas = sorted(int(m.group(1)) for m in (_META_RE.match(p.name) for p in d.iterdir()) if m)
    if metas:
        for k, idx in enumerate(metas):
            if k % every:
                continue
            try:
                yield idx, _load_capture_frame(d, idx)
            except (OSError, ValueError, KeyError) as e:
                yield idx, 'cannot load frame %d: %s' % (idx, e)
        return
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for k, p in enumerate(files):
        if k % every:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            yield k, 'cannot decode %s' % p.name
            continue
        yield k, SessionFrame(k, img, None, None, p)


# --------------------------------------------------------------------------
# Box sources
# --------------------------------------------------------------------------

class PerceptionBoxes:
    """Boxes of valid instances from perception.pipeline.Perception (needs extrinsics)."""

    def __init__(self, config_files: Sequence[str]):
        from cube_task.core.config import load_config
        from cube_task.core.extrinsics import extrinsics_from_cfg
        from cube_task.core.transforms import pose6_to_T
        from cube_task.perception.pipeline import Perception
        self.cfg = load_config(extra=list(config_files))
        self.ext = extrinsics_from_cfg(self.cfg)
        if self.ext is None:
            raise ValueError('--boxes perception needs camera extrinsics (camera.extrinsics_file)')
        self.T_flange_tcp = pose6_to_T(self.cfg.get('robot.tcp_offset'))
        self.perception = Perception(self.cfg)

    def __call__(self, sf: SessionFrame) -> Tuple[List[Box], Dict[str, Any]]:
        from cube_task.core.extrinsics import camera_pose_in_base
        if sf.frame is None or sf.depth is None:
            return [], {'reason': 'no depth'}
        T_bf = sf.extra.get('T_base_flange')
        T_bc = camera_pose_in_base(self.ext, None if T_bf is None else np.asarray(T_bf, float),
                                   self.T_flange_tcp)
        res = self.perception.process(sf.frame, T_bc)
        boxes = [tuple(int(v) for v in det.bbox) for det in res.valid_detections()]
        return boxes, {'warnings': list(res.warnings)}


class ModelBoxes:
    """Class-agnostic boxes from an ONNX detector (perception/detector_onnx.py)."""

    def __init__(self, model: str, labels: str, conf: float, iou: float):
        from cube_task.perception.detector_onnx import OnnxDetector
        self.det = OnnxDetector.from_paths(model, labels, conf_threshold=conf,
                                           nms_threshold=iou)
        self.iou = iou

    def __call__(self, sf: SessionFrame) -> Tuple[List[Box], Dict[str, Any]]:
        from cube_task.perception.detector_onnx import nms_per_label
        dets = self.det.detect(sf.color)
        if not dets:
            return [], {}
        boxes = np.array([b for b, _l, _p in dets], dtype=np.float64)
        scores = np.array([p for _b, _l, p in dets], dtype=np.float64)
        keep = nms_per_label(boxes, scores, [0] * len(dets), self.iou)
        return ([tuple(int(v) for v in boxes[i]) for i in keep],
                {'model_labels': [dets[int(i)][1] for i in keep]})


# --------------------------------------------------------------------------
# Dataset writing
# --------------------------------------------------------------------------

def parse_session(text: str) -> Tuple[str, Path]:
    """Parse 'LABEL=DIR' (label may contain spaces, not '=')."""
    if '=' not in text:
        raise ValueError('expected LABEL=DIR, got %r' % text)
    label, path = text.split('=', 1)
    label = label.strip()
    if not label:
        raise ValueError('empty label in %r' % text)
    return label, Path(path.strip()).expanduser()


def split_sessions(sessions: Sequence[Tuple[str, Path]], forced_val: Sequence[Tuple[str, Path]],
                   val_fraction: float, seed: int) -> List[Tuple[str, Path, str]]:
    """
    Assign 'train' / 'val' per session (never per frame).

    forced_val sessions are val. For the rest, per class: none to val when it
    has one session, else round(n * val_fraction) clipped to [1, n - 1],
    chosen with a seeded permutation. Returns [(label, dir, split)].
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError('val_fraction must be in [0, 1)')
    rng = np.random.RandomState(seed)
    out: List[Tuple[str, Path, str]] = [(lab, p, 'val') for lab, p in forced_val]
    by_class: Dict[str, List[Path]] = {}
    for lab, p in sessions:
        by_class.setdefault(lab, []).append(p)
    for lab, dirs in by_class.items():
        n = len(dirs)
        n_val = 0
        if n >= 2 and val_fraction > 0:
            n_val = int(min(n - 1, max(1, round(n * val_fraction))))
        val_idx = set(int(i) for i in rng.permutation(n)[:n_val])
        out.extend((lab, p, 'val' if i in val_idx else 'train') for i, p in enumerate(dirs))
    return out


def yolo_line(cls: int, box: Box, width: int, height: int) -> str:
    """Return 'cls cx cy w h' normalized to [0, 1] for an x0,y0,x1,y1 pixel box."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0 / width
    cy = (y0 + y1) / 2.0 / height
    return '%d %.6f %.6f %.6f %.6f' % (cls, cx, cy, (x1 - x0) / float(width),
                                       (y1 - y0) / float(height))


def _safe(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_') or 'x'


def build_dataset(out: Path, plan: Sequence[Tuple[str, Path, str]], classes: List[str],
                  box_fn: Any, every: int = 1, keep_empty: bool = False,
                  max_boxes: int = 0, preview: bool = False) -> Dict[str, Any]:
    """
    Write images/, labels/, data.yaml and manifest.json under out; return the manifest.

    box_fn(SessionFrame) -> (boxes, diag). Frames without boxes are skipped
    unless keep_empty (pure background sessions); frames with more than
    max_boxes boxes (> 0) are skipped as suspicious.
    """
    import yaml
    out = Path(out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('%s exists and is not empty; choose a new --out' % out)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    if preview:
        (out / 'preview').mkdir(parents=True, exist_ok=True)
    cls_index = {c: i for i, c in enumerate(classes)}
    manifest: Dict[str, Any] = {'classes': classes, 'sessions': [], 'counts': {}}
    used_names: Dict[str, int] = {}
    totals = {'train': [0, 0], 'val': [0, 0]}
    for label, sdir, split in plan:
        if label not in cls_index:
            raise ValueError('session label %r is not in --classes %s' % (label, classes))
        if not sdir.is_dir():
            raise FileNotFoundError('session directory not found: %s' % sdir)
        base = '%s_%s' % (_safe(label), _safe(sdir.name))
        used_names[base] = used_names.get(base, 0) + 1
        if used_names[base] > 1:
            base = '%s_%d' % (base, used_names[base])
        info: Dict[str, Any] = {'label': label, 'dir': str(sdir), 'split': split, 'name': base,
                                'frames': 0, 'boxes': 0, 'skipped': {}}
        for idx, sf in iter_session(sdir, every):
            if isinstance(sf, str):
                info['skipped'][sf] = info['skipped'].get(sf, 0) + 1
                continue
            try:
                boxes, diag = box_fn(sf)
            except (ValueError, RuntimeError) as e:
                key = 'box error: %s' % e
                info['skipped'][key] = info['skipped'].get(key, 0) + 1
                continue
            reason = diag.get('reason', '')
            if not boxes and not keep_empty:
                key = 'no boxes' + (' (%s)' % reason if reason else '')
                info['skipped'][key] = info['skipped'].get(key, 0) + 1
                continue
            if max_boxes > 0 and len(boxes) > max_boxes:
                key = 'more than %d boxes' % max_boxes
                info['skipped'][key] = info['skipped'].get(key, 0) + 1
                continue
            h, w = sf.color.shape[:2]
            stem = '%s_%06d' % (base, idx)
            img_out = out / 'images' / split / (stem + '.png')
            if sf.color_path is not None and sf.color_path.suffix.lower() == '.png':
                shutil.copy2(str(sf.color_path), str(img_out))
            elif not cv2.imwrite(str(img_out), sf.color):
                raise OSError('cannot write %s' % img_out)
            lines = [yolo_line(cls_index[label], b, w, h) for b in boxes]
            (out / 'labels' / split / (stem + '.txt')).write_text(
                ''.join(ln + '\n' for ln in lines), encoding='utf-8')
            if preview:
                vis = sf.color.copy()
                for b in boxes:
                    cv2.rectangle(vis, (b[0], b[1]), (b[2] - 1, b[3] - 1), (0, 255, 0), 2)
                cv2.putText(vis, '%s %s %d boxes' % (split, label, len(boxes)), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(out / 'preview' / (stem + '.jpg')), vis)
            info['frames'] += 1
            info['boxes'] += len(boxes)
        totals[split][0] += info['frames']
        totals[split][1] += info['boxes']
        manifest['sessions'].append(info)
        LOG.info('%-5s %-12s %s: %d frames, %d boxes, skipped %s', split, label, sdir,
                 info['frames'], info['boxes'], info['skipped'] or '-')
    manifest['counts'] = {k: {'images': v[0], 'boxes': v[1]} for k, v in totals.items()}
    val_dir = 'images/val'
    if totals['val'][0] == 0:
        LOG.warning('no validation images: data.yaml uses the train images for val, so the '
                    'reported mAP is optimistic. Record two or more sessions per class.')
        val_dir = 'images/train'
        manifest['val_is_train'] = True
    data = {'path': str(out), 'train': 'images/train', 'val': val_dir, 'nc': len(classes),
            'names': {i: c for i, c in enumerate(classes)}}
    with open(out / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write('# written by cube_task/tools/yolo/make_dataset.py\n')
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding='utf-8')
    return manifest


def _parse_roi(text: Optional[str]) -> Optional[Box]:
    if not text:
        return None
    vals = [int(v) for v in text.split(',')]
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise ValueError('--roi needs x0,y0,x1,y1 with x1 > x0 and y1 > y0')
    return vals[0], vals[1], vals[2], vals[3]


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('--out', required=True, help='new dataset directory')
    p.add_argument('--session', action='append', default=[], metavar='LABEL=DIR',
                   help='capture session or image folder holding objects of class LABEL')
    p.add_argument('--val-session', action='append', default=[], metavar='LABEL=DIR',
                   help='session that always goes to val')
    p.add_argument('--classes', default=None,
                   help='comma-separated class names in index order (must match any other '
                        'dataset you train with); default: labels in order of appearance')
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--every', type=int, default=1, help='use every N-th frame')
    p.add_argument('--boxes', choices=['plane', 'perception', 'model'], default='plane')
    p.add_argument('--roi', default=None, metavar='x0,y0,x1,y1',
                   help='plane: only pixels inside this rectangle (exclude bins, arm)')
    p.add_argument('--min-height', type=float, default=0.008, help='plane: m above the table')
    p.add_argument('--max-height', type=float, default=0.25)
    p.add_argument('--min-area-m2', type=float, default=0.00012)
    p.add_argument('--max-area-m2', type=float, default=0.040)
    p.add_argument('--keep-border', action='store_true',
                   help='plane: keep blobs touching the image border')
    p.add_argument('--pad-px', type=int, default=0,
                   help='plane: grow boxes by N px (only if previews show cut-off objects)')
    p.add_argument('--config', action='append', default=[],
                   help='perception: extra config files merged over default.yaml')
    p.add_argument('--model', default=None, help='model: ONNX file')
    p.add_argument('--labels', default=None, help='model: labels file of --model')
    p.add_argument('--model-conf', type=float, default=0.25)
    p.add_argument('--max-boxes', type=int, default=0,
                   help='skip frames with more boxes than this (0 = no limit)')
    p.add_argument('--keep-empty', action='store_true',
                   help='keep frames without boxes (background-only sessions)')
    p.add_argument('--preview', action='store_true', help='write preview/*.jpg with boxes')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    try:
        sessions = [parse_session(s) for s in args.session]
        forced = [parse_session(s) for s in args.val_session]
        if not sessions and not forced:
            raise ValueError('give at least one --session LABEL=DIR')
        if args.every < 1:
            raise ValueError('--every must be >= 1')
        if args.classes:
            classes = [c.strip() for c in args.classes.split(',') if c.strip()]
        else:
            classes = []
            for lab, _p in sessions + forced:
                if lab not in classes:
                    classes.append(lab)
        if len(set(classes)) != len(classes):
            raise ValueError('duplicate names in --classes')
        if args.boxes == 'plane':
            params = PlaneParams(min_height=args.min_height, max_height=args.max_height,
                                 min_area_m2=args.min_area_m2, max_area_m2=args.max_area_m2,
                                 roi=_parse_roi(args.roi), drop_border=not args.keep_border,
                                 pad_px=args.pad_px)
            rng = np.random.RandomState(args.seed)

            def box_fn(sf: SessionFrame) -> Tuple[List[Box], Dict[str, Any]]:
                if sf.depth is None or sf.K is None:
                    raise ValueError('no depth / intrinsics (use --boxes model for image '
                                     'folders)')
                return plane_boxes(sf.depth, sf.K, params, rng)
        elif args.boxes == 'perception':
            box_fn = PerceptionBoxes(args.config)
        else:
            if not args.model or not args.labels:
                raise ValueError('--boxes model needs --model and --labels')
            box_fn = ModelBoxes(args.model, args.labels, args.model_conf, 0.5)
        plan = split_sessions(sessions, forced, args.val_fraction, args.seed)
        manifest = build_dataset(Path(args.out), plan, classes, box_fn, every=args.every,
                                 keep_empty=args.keep_empty, max_boxes=args.max_boxes,
                                 preview=args.preview)
    except (ValueError, OSError) as e:
        LOG.error('%s', e)
        return 1
    c = manifest['counts']
    print('dataset %s: train %d images / %d boxes, val %d images / %d boxes, classes %s'
          % (Path(args.out).expanduser().resolve(), c['train']['images'], c['train']['boxes'],
             c['val']['images'], c['val']['boxes'], classes))
    print('next: check preview/, then train with cube_task/tools/yolo/train_yolo.py --data '
          '%s/data.yaml' % Path(args.out).expanduser().resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())

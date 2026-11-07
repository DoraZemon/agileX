"""
Save and load RGB-D frames as files (capture tool, replay with FileCamera).

Files per frame index i (6-digit, zero padded)::

    <i>_color.png   8-bit BGR
    <i>_depth.png   16-bit depth in millimeters, 0 = invalid, clipped at 65.535 m
    <i>_depth.npy   optional float32 depth in meters (save_depth_npy=True)
    <i>_meta.json   intrinsics, stamp, frame_id, source, depth_unit, frame.meta,
                    extra (for example T_base_flange as a 4x4 list)

The meta file is written last, so a frame counts as recorded (list_frames)
only when all its files are complete.
"""
from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.logger import atomic_write_text, to_jsonable
from ..core.types import CameraIntrinsics, Frame
from .utils import valid_depth_mask

FORMAT = 'cube_task_frame_v1'
_META_RE = re.compile(r'^(\d{6,})_meta\.json$')


def _name(idx: int, what: str) -> str:
    return '%06d_%s' % (idx, what)


def _check_idx(idx: Any) -> int:
    if isinstance(idx, bool) or int(idx) != idx or int(idx) < 0:
        raise ValueError('frame index must be an integer >= 0, got %r' % (idx,))
    return int(idx)


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    """Convert float meters to uint16 millimeters (rounded; invalid -> 0, max 65535)."""
    d = np.asarray(depth, dtype=np.float64)
    valid = valid_depth_mask(d)
    mm = np.zeros(d.shape, dtype=np.uint16)
    mm[valid] = np.clip(np.rint(d[valid] * 1000.0), 1, 65535).astype(np.uint16)
    return mm


def _imwrite_atomic(path: Path, img: np.ndarray) -> None:
    import cv2
    tmp = path.with_name('.' + path.stem + '.tmp' + path.suffix)
    if not cv2.imwrite(str(tmp), np.ascontiguousarray(img)):
        raise IOError('cv2.imwrite failed for %s' % tmp)
    os.replace(str(tmp), str(path))


def save_frame(dir: Union[str, Path], idx: int, frame: Frame,
               extra: Optional[Dict[str, Any]] = None,
               save_depth_npy: bool = False) -> Path:
    """
    Write one frame as files (see module doc) and return the meta file path.

    extra: JSON-able data stored with the frame, for example
    ``{'T_base_flange': T.tolist(), 'joints': q}``; numpy values are converted.
    Raises ValueError for a malformed frame and IOError when writing fails.
    """
    idx = _check_idx(idx)
    d = Path(dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    color = np.asarray(frame.color)
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError('frame.color must be HxWx3 uint8, got %s %s'
                         % (color.shape, color.dtype))
    h, w = color.shape[:2]
    color_name = _name(idx, 'color.png')
    _imwrite_atomic(d / color_name, color)
    depth_name = None
    npy_name = None
    if frame.depth is not None:
        depth = np.asarray(frame.depth)
        if depth.shape != (h, w):
            raise ValueError('frame.depth shape %s does not match color %dx%d'
                             % (depth.shape, w, h))
        depth_name = _name(idx, 'depth.png')
        _imwrite_atomic(d / depth_name, depth_to_mm(depth))
        if save_depth_npy:
            npy_name = _name(idx, 'depth.npy')
            clean = np.where(valid_depth_mask(depth), depth, 0.0).astype(np.float32)
            tmp = d / ('.' + npy_name + '.tmp')
            with open(tmp, 'wb') as fh:
                np.save(fh, clean)
            os.replace(str(tmp), str(d / npy_name))
    meta = {
        'format': FORMAT,
        'index': idx,
        'color_file': color_name,
        'depth_file': depth_name,
        'depth_npy_file': npy_name,
        'depth_unit': 'mm',
        'depth_scale': 0.001,
        'width': int(w),
        'height': int(h),
        'intrinsics': frame.intrinsics.to_dict(),
        'stamp': float(frame.stamp),
        'frame_id': frame.frame_id,
        'source': frame.source,
        'wall': datetime.datetime.now().isoformat(timespec='milliseconds'),
        'meta': to_jsonable(frame.meta),
        'extra': to_jsonable(extra or {}),
    }
    meta_path = d / _name(idx, 'meta.json')
    atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2,
                                            allow_nan=False) + '\n')
    return meta_path


def list_frames(dir: Union[str, Path]) -> List[int]:
    """Return the sorted indices of complete frames in dir ([] if dir is missing)."""
    d = Path(dir).expanduser()
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = _META_RE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def next_index(dir: Union[str, Path]) -> int:
    """Return the first unused frame index in dir."""
    idx = list_frames(dir)
    return idx[-1] + 1 if idx else 0


def load_frame(dir: Union[str, Path], idx: int) -> Tuple[Frame, Dict[str, Any]]:
    """
    Load frame idx; return (Frame, extra).

    Depth comes from the .npy file when present, else from the 16-bit PNG
    (meters = value * depth_scale, 0 stays invalid). Frame.source /
    stamp / frame_id / meta are the recorded values. Raises
    FileNotFoundError for missing files and ValueError for inconsistent ones.
    """
    import cv2
    idx = _check_idx(idx)
    d = Path(dir).expanduser()
    meta_path = d / _name(idx, 'meta.json')
    if not meta_path.is_file():
        raise FileNotFoundError('frame meta file not found: %s' % meta_path)
    with open(meta_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict) or 'intrinsics' not in meta:
        raise ValueError('%s: not a cube_task frame meta file' % meta_path)
    intr = CameraIntrinsics.from_dict(meta['intrinsics'])
    color_path = d / meta.get('color_file', _name(idx, 'color.png'))
    if not color_path.is_file():
        raise FileNotFoundError('color image not found: %s' % color_path)
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError('cannot decode %s' % color_path)
    h, w = color.shape[:2]
    if (meta.get('width', w), meta.get('height', h)) != (w, h):
        raise ValueError('%s: image is %dx%d but meta says %sx%s'
                         % (color_path, w, h, meta.get('width'), meta.get('height')))
    depth = None
    if meta.get('depth_npy_file') and (d / meta['depth_npy_file']).is_file():
        depth = np.load(str(d / meta['depth_npy_file']), allow_pickle=False)
        depth = np.where(valid_depth_mask(depth), depth, 0.0).astype(np.float32)
    elif meta.get('depth_file'):
        depth_path = d / meta['depth_file']
        if not depth_path.is_file():
            raise FileNotFoundError('depth image not found: %s' % depth_path)
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError('%s: expected a 16-bit single-channel PNG' % depth_path)
        scale = meta.get('depth_scale')
        if scale is None:
            if meta.get('depth_unit') != 'mm':
                raise ValueError('%s: no depth_scale and depth_unit is %r'
                                 % (meta_path, meta.get('depth_unit')))
            scale = 0.001
        depth = raw.astype(np.float32) * np.float32(scale)
    if depth is not None and depth.shape != (h, w):
        raise ValueError('%s: depth shape %s does not match color %dx%d'
                         % (meta_path, depth.shape, w, h))
    frame = Frame(color=color, depth=depth, intrinsics=intr,
                  stamp=float(meta.get('stamp', 0.0)),
                  frame_id=str(meta.get('frame_id', 'camera_color_optical_frame')),
                  source=str(meta.get('source', 'file')),
                  meta=dict(meta.get('meta') or {}))
    extra = meta.get('extra') or {}
    return frame, dict(extra)

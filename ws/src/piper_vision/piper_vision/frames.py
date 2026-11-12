"""
Read RGB-D frames from disk.

Reads the cube_task capture format without importing cube_task::

    NNNNNN_color.png   8-bit BGR
    NNNNNN_depth.png   16-bit depth in mm (0 = invalid)
    NNNNNN_depth.npy   optional float32 depth in m (preferred)
    NNNNNN_meta.json   intrinsics, extra.T_base_flange (4x4) for wrist cameras

so frames recorded with ``python3 -m cube_task capture`` (including drag-teach
poses for hand-eye calibration) can be used directly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .camera import Intrinsics

_META_RE = re.compile(r'^(\d{6,})_meta\.json$')


@dataclass
class RgbdFrame:
    """One frame with the arm pose recorded at capture (if any)."""

    color: np.ndarray
    depth: Optional[np.ndarray]
    intrinsics: Optional[Intrinsics]
    T_base_flange: Optional[np.ndarray] = None
    name: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)


def list_frames(directory: str) -> List[int]:
    """Return the frame indices of a capture directory, sorted."""
    d = Path(directory)
    if not d.is_dir():
        raise ValueError('not a directory: %s' % directory)
    return sorted(int(m.group(1)) for m in (_META_RE.match(p.name) for p in d.iterdir()) if m)


def load_frame(directory: str, idx: int) -> RgbdFrame:
    """Load one frame of a capture directory."""
    d = Path(directory)
    stem = '%06d' % idx
    meta = json.loads((d / ('%s_meta.json' % stem)).read_text(encoding='utf-8'))
    color = cv2.imread(str(d / ('%s_color.png' % stem)), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError('cannot read %s_color.png in %s' % (stem, directory))
    depth = None
    npy = d / ('%s_depth.npy' % stem)
    png = d / ('%s_depth.png' % stem)
    if npy.is_file():
        depth = np.load(str(npy)).astype(np.float32)
    elif png.is_file():
        raw = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            depth = raw.astype(np.float32) / 1000.0
    intr = Intrinsics.from_dict(meta['intrinsics']) if meta.get('intrinsics') else None
    extra = meta.get('extra') or {}
    T = extra.get('T_base_flange')
    return RgbdFrame(color, depth, intr, None if T is None else np.asarray(T, dtype=float),
                     name=stem, meta=meta)


def load_image_pair(color_path: str, depth_path: Optional[str] = None,
                    depth_scale: float = 0.001) -> RgbdFrame:
    """Load a color image and an optional depth image (.png in depth units or .npy in m)."""
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError('cannot read %s' % color_path)
    depth = None
    if depth_path:
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        else:
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError('cannot read %s' % depth_path)
            depth = raw.astype(np.float32) * float(depth_scale)
    return RgbdFrame(color, depth, None, name=Path(color_path).stem)

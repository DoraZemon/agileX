"""Camera helpers: depth validity, per-pixel median depth, averaged capture."""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Optional, Sequence

import numpy as np

from ..core.interfaces import CameraSource
from ..core.types import Frame

_LOG = logging.getLogger('cube_task.camera.utils')


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """Return HxW bool: True where depth (meters) is finite and > 0."""
    d = np.asarray(depth)
    with np.errstate(invalid='ignore'):
        return np.isfinite(d) & (d > 0)


def median_depth(depths: Sequence[np.ndarray], min_valid: Optional[int] = None) -> np.ndarray:
    """
    Per-pixel median of several HxW depth images (meters) over valid samples.

    A pixel is valid in the result when at least ``min_valid`` samples are
    valid (default ceil(k / 2) for k images); invalid pixels are 0. With an
    even number of valid samples the two middle values are averaged.
    Returns float32 HxW.
    """
    if len(depths) == 0:
        raise ValueError('median_depth needs at least one depth image')
    stack = np.stack([np.asarray(d, dtype=np.float32) for d in depths])
    if stack.ndim != 3:
        raise ValueError('median_depth needs HxW depth images, got shape %s'
                         % (stack.shape[1:],))
    k = stack.shape[0]
    need = int(math.ceil(k / 2.0)) if min_valid is None else int(min_valid)
    need = max(1, need)
    valid = valid_depth_mask(stack)
    work = np.where(valid, stack, np.float32(np.inf))
    work.sort(axis=0)
    cnt = valid.sum(axis=0)
    lo_idx = np.maximum((cnt - 1) // 2, 0)[None]
    hi_idx = np.maximum(cnt // 2, 0)[None]
    hi_idx = np.minimum(hi_idx, k - 1)
    lo = np.take_along_axis(work, lo_idx, axis=0)[0]
    hi = np.take_along_axis(work, hi_idx, axis=0)[0]
    ok = cnt >= need
    out = np.zeros(stack.shape[1:], dtype=np.float32)
    out[ok] = 0.5 * (lo[ok] + hi[ok])
    return out


def _compatible(a: Frame, b: Frame) -> bool:
    if a.color.shape != b.color.shape:
        return False
    if (a.depth is None) != (b.depth is None):
        return False
    if a.depth is not None and a.depth.shape != b.depth.shape:
        return False
    ia, ib = a.intrinsics, b.intrinsics
    return (ia.width, ia.height) == (ib.width, ib.height) and abs(ia.fx - ib.fx) < 1e-6


def capture_averaged(camera: CameraSource, n: int, newer_than: Optional[float] = None,
                     timeout: float = 2.0) -> Optional[Frame]:
    """
    Grab n consecutive new frames and return one with median depth.

    Each frame must be newer than the previous one (the first newer than
    ``newer_than``, Clock seconds). ``timeout`` (s) applies to each
    get_frame() call. The result has the color image, intrinsics and stamp
    of the last frame and the per-pixel median depth over valid samples (a
    pixel needs ceil(k / 2) valid samples of the k frames used, else 0).
    ``meta['frames_averaged']`` = k. When a later frame times out, the frames
    already captured are used (warning). Returns None when no frame arrives.
    n <= 1 returns a single get_frame() result. The caller must keep the
    camera (and an eye-in-hand arm) still during the capture.
    """
    if n is None or int(n) <= 1:
        return camera.get_frame(timeout=timeout, newer_than=newer_than)
    n = int(n)
    frames = []
    last = newer_than
    for _ in range(2 * n + 2):
        if len(frames) >= n:
            break
        f = camera.get_frame(timeout=timeout, newer_than=last)
        if f is None:
            break
        if last is not None and not f.stamp > last:
            _LOG.warning('camera %s returned a frame (stamp %.4f) not newer than %.4f',
                         getattr(camera, 'name', '?'), f.stamp, last)
            break
        last = f.stamp
        if frames and not _compatible(frames[0], f):
            _LOG.warning('frame size / intrinsics changed during averaging; restarting')
            frames = []
        frames.append(f)
    if not frames:
        return None
    if len(frames) < n:
        _LOG.warning('capture_averaged: got %d of %d frames; using those', len(frames), n)
    ref = frames[-1]
    meta = dict(ref.meta)
    meta['frames_averaged'] = len(frames)
    meta['averaged_stamps'] = [float(f.stamp) for f in frames]
    if ref.depth is None:
        return dataclasses.replace(ref, meta=meta)
    depth = median_depth([f.depth for f in frames])
    return dataclasses.replace(ref, depth=depth, meta=meta)

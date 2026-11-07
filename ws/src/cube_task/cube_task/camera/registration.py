"""
Software depth-to-color registration (used when the driver does not align depth).

Prefer the driver's own alignment (Orbbec ``depth_registration:=true``,
RealSense ``align_depth.enable:=true``); this module is the fallback for an
unaligned depth stream with a known depth -> color transform.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.extrinsics import validate_transform
from ..core.types import CameraIntrinsics
from .utils import valid_depth_mask

#: Largest accepted depth -> color translation (m); RGB-D baselines are a few cm.
MAX_BASELINE_M = 0.3
_FAR = np.float32(1e9)


def _has_dist(intr: CameraIntrinsics) -> bool:
    d = np.asarray(intr.dist, dtype=float).ravel()
    return d.size > 0 and bool(np.any(d != 0.0))


def _check_intr(intr: CameraIntrinsics, what: str) -> None:
    vals = (intr.fx, intr.fy, intr.cx, intr.cy)
    if not all(math.isfinite(float(v)) for v in vals) or intr.fx <= 0 or intr.fy <= 0:
        raise ValueError('%s intrinsics invalid: fx=%r fy=%r cx=%r cy=%r'
                         % ((what,) + vals))
    if intr.width <= 0 or intr.height <= 0:
        raise ValueError('%s intrinsics have no image size' % what)


def register_depth_to_color(depth_d: np.ndarray, intr_d: CameraIntrinsics,
                            intr_c: CameraIntrinsics, T_color_depth: np.ndarray,
                            fill_holes: bool = True) -> np.ndarray:
    """
    Reproject a depth image into the color camera.

    Parameters
    ----------
    depth_d
        HdxWd depth in meters in the depth camera image (0 / NaN / inf =
        invalid). Must match intr_d width / height.
    intr_d, intr_c
        Depth and color intrinsics. Distortion is applied
        (cv2.undistortPoints / cv2.projectPoints) only when non-zero.
    T_color_depth
        4x4 mapping depth optical frame points into the color optical frame
        (meters).
    fill_holes
        Fill isolated holes (an invalid pixel with at least 5 of 8 valid
        neighbors) with the nearest neighbor depth.

    Returns
    -------
    numpy.ndarray
        HcxWc float32 depth (meters, z in the color frame), 0 = invalid. Each
        valid depth pixel is back-projected, transformed, projected to the
        nearest color pixel, and the smallest z per pixel wins (z-buffer).
        When the color image is >= 1.4x denser than the depth image, the
        splat runs on a coarser grid (factor round(fx_c / fx_d)) and is
        upsampled with nearest neighbor so the result has no splat gaps.

    """
    import cv2

    _check_intr(intr_d, 'depth')
    _check_intr(intr_c, 'color')
    depth = np.asarray(depth_d, dtype=np.float32)
    if depth.ndim != 2 or depth.shape != (intr_d.height, intr_d.width):
        raise ValueError('depth image shape %s does not match depth intrinsics %dx%d'
                         % (depth.shape, intr_d.width, intr_d.height))
    T = validate_transform(T_color_depth, 'T_color_depth', max_translation=MAX_BASELINE_M)
    Wc, Hc = int(intr_c.width), int(intr_c.height)

    ratio = min(intr_c.fx / intr_d.fx, intr_c.fy / intr_d.fy)
    f = int(round(ratio)) if ratio >= 1.4 else 1
    Wg, Hg = (Wc + f - 1) // f, (Hc + f - 1) // f

    valid = valid_depth_mask(depth)
    vs, us = np.nonzero(valid)
    grid = np.full(Hg * Wg, _FAR, dtype=np.float32)
    if us.size:
        z = depth[vs, us].astype(np.float64)
        if _has_dist(intr_d):
            pts = np.stack([us, vs], axis=-1).astype(np.float64).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, intr_d.K(),
                                      np.asarray(intr_d.dist, dtype=np.float64)).reshape(-1, 2)
            x, y = und[:, 0], und[:, 1]
        else:
            x = (us - intr_d.cx) / intr_d.fx
            y = (vs - intr_d.cy) / intr_d.fy
        P = np.stack([x * z, y * z, z], axis=-1) @ T[:3, :3].T + T[:3, 3]
        P = P[P[:, 2] > 1e-6]
        if P.shape[0]:
            if _has_dist(intr_c):
                uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                          intr_c.K(),
                                          np.asarray(intr_c.dist, dtype=np.float64))
                u, v = uv[:, 0, 0], uv[:, 0, 1]
            else:
                u = intr_c.fx * P[:, 0] / P[:, 2] + intr_c.cx
                v = intr_c.fy * P[:, 1] / P[:, 2] + intr_c.cy
            ok = np.isfinite(u) & np.isfinite(v)
            iu = np.floor(u[ok] + 0.5)
            iv = np.floor(v[ok] + 0.5)
            zc = P[ok, 2].astype(np.float32)
            inb = (iu >= 0) & (iu < Wc) & (iv >= 0) & (iv < Hc)
            gu = iu[inb].astype(np.int64) // f
            gv = iv[inb].astype(np.int64) // f
            np.minimum.at(grid, gv * Wg + gu, zc[inb])
    grid = grid.reshape(Hg, Wg)

    if fill_holes:
        holes = grid >= _FAR
        if np.any(holes) and not np.all(holes):
            mn = cv2.erode(grid, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                           borderValue=float(_FAR))
            cnt = cv2.boxFilter((~holes).astype(np.float32), -1, (3, 3), normalize=False,
                                borderType=cv2.BORDER_CONSTANT)
            fill = holes & (cnt >= 4.5) & (mn < _FAR)
            grid[fill] = mn[fill]
    grid[grid >= _FAR] = 0.0

    if f == 1:
        return np.ascontiguousarray(grid[:Hc, :Wc])
    rows = np.arange(Hc) // f
    cols = np.arange(Wc) // f
    return np.ascontiguousarray(grid[rows[:, None], cols[None, :]])

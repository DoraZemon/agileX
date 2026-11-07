"""
Per-instance feature vectors for the instance classifiers.

Color features use the instance's upper pixels (height above the table >
0.4 * object height) so table shadows and the dark lower sides weigh less.
Geometric features are in meters and come from the base-frame geometry the
pipeline fills before classification. The pipeline stores two arrays in
Detection.extra that this module reads:

- ``'_height_crop'``: float32 height above the table (m) of each pixel of the
  bbox crop (NaN outside the instance or where depth is invalid).
- ``'_points_xy'``: Nx2 float32 base XY (m) of the instance's reliable points.

Everything is deterministic for a given frame and detection.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.types import Detection, Frame

HUE_BINS = 12
UPPER_FRACTION = 0.4        # upper pixels: height above table > this * object height
GRID_CELL_M = 0.002         # footprint raster cell size (m)
MIN_UPPER_PIXELS = 10       # fewer upper pixels -> use the whole mask
_MAX_GRID = 400             # cells per side cap (0.8 m)

FEATURE_NAMES: List[str] = (['hue_%02d' % i for i in range(HUE_BINS)]
                            + ['s_mean', 's_std', 'v_mean', 'v_std', 'lab_a', 'lab_b',
                               'length', 'width', 'height', 'aspect', 'fill',
                               'circularity'])


def _crop_mask(det: Detection) -> Optional[np.ndarray]:
    if det.mask is None:
        return None
    x0, y0, x1, y1 = det.bbox
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray(det.mask[y0:y1, x0:x1], dtype=bool)


def upper_mask(det: Detection, height_crop: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Return the bbox-crop bool mask of the instance's upper pixels.

    Upper pixels have a height above the table greater than UPPER_FRACTION
    times det.height. Falls back to the whole instance mask when the height
    data is missing or fewer than MIN_UPPER_PIXELS pixels qualify. Returns
    None when the detection has no mask.
    """
    m = _crop_mask(det)
    if m is None:
        return None
    hc = height_crop if height_crop is not None else det.extra.get('_height_crop')
    h = det.height
    if hc is not None and h is not None and math.isfinite(h) and h > 0:
        hc = np.asarray(hc)
        if hc.shape == m.shape:
            with np.errstate(invalid='ignore'):
                up = m & (hc > UPPER_FRACTION * float(h))
            if np.count_nonzero(up) >= MIN_UPPER_PIXELS:
                return up
    return m


def instance_pixels(frame: Frame, det: Detection, upper: bool = True,
                    height_crop: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return the Nx3 uint8 BGR pixels of the instance (upper pixels by default).

    Returns an empty (0, 3) array when the detection has no mask or the
    frame has no usable color image.
    """
    empty = np.zeros((0, 3), dtype=np.uint8)
    color = frame.color
    if color is None or color.ndim != 3 or color.shape[2] != 3:
        return empty
    m = upper_mask(det, height_crop) if upper else _crop_mask(det)
    if m is None:
        return empty
    x0, y0, x1, y1 = det.bbox
    crop = color[y0:y1, x0:x1]
    if crop.shape[:2] != m.shape:
        return empty
    return np.ascontiguousarray(crop[m], dtype=np.uint8)


def bgr_to_hsv_pixels(bgr: np.ndarray) -> np.ndarray:
    """Convert Nx3 uint8 BGR pixels to Nx3 uint8 HSV (OpenCV scale: H 0-179)."""
    bgr = np.asarray(bgr, dtype=np.uint8).reshape(-1, 1, 3)
    if bgr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def hue_histogram(hsv: np.ndarray, bins: int = HUE_BINS) -> np.ndarray:
    """
    Saturation-weighted circular hue histogram that sums to 1.

    hsv: Nx3 OpenCV-scale pixels. Each pixel adds S/255 split linearly between
    the two nearest bin centers (hue wraps at 180). All zeros when no pixel
    has saturation.
    """
    hsv = np.asarray(hsv).reshape(-1, 3)
    hist = np.zeros(bins, dtype=np.float64)
    if hsv.shape[0] == 0:
        return hist
    h = hsv[:, 0].astype(np.float64)
    w = hsv[:, 1].astype(np.float64) / 255.0
    pos = h / (180.0 / bins) - 0.5
    i0 = np.floor(pos)
    f = pos - i0
    i0 = i0.astype(np.int64) % bins
    i1 = (i0 + 1) % bins
    hist += np.bincount(i0, weights=w * (1.0 - f), minlength=bins)
    hist += np.bincount(i1, weights=w * f, minlength=bins)
    total = float(hist.sum())
    if total > 1e-9:
        hist /= total
    return hist


def footprint_shape(points_xy: np.ndarray, length: float, width: float,
                    yaw: float = 0.0, cell: float = GRID_CELL_M) -> Tuple[float, float]:
    """
    Return (fill ratio, circularity) of a base-XY footprint.

    The points (Nx2, m) are rotated by -yaw and rasterized on a cell x cell
    grid. fill = occupied area / (length * width); circularity = 4 pi A / P^2
    of the largest outer contour of the (closed) raster. NaN when undefined.
    """
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 3 or not (length > 0 and width > 0):
        return float('nan'), float('nan')
    c, s = math.cos(-yaw), math.sin(-yaw)
    rel = xy - xy.mean(axis=0)
    loc = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=1)
    ij = np.floor((loc - loc.min(axis=0)) / cell).astype(np.int64)
    ij = np.minimum(ij, _MAX_GRID - 1)
    nx, ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    grid = np.zeros((ny + 2, nx + 2), dtype=np.uint8)
    grid[ij[:, 1] + 1, ij[:, 0] + 1] = 1
    fill = float(np.count_nonzero(grid)) * cell * cell / (length * width)
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    res = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[0] if len(res) == 2 else res[1]
    if not contours:
        return fill, float('nan')
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    per = float(cv2.arcLength(cnt, True))
    circ = 4.0 * math.pi * area / (per * per) if per > 0 else float('nan')
    return fill, float(min(circ, 1.0))


def _num(v: Optional[float]) -> float:
    return float(v) if v is not None and math.isfinite(float(v)) else float('nan')


def feature_vector(frame: Frame, detection: Detection,
                   height_crop: Optional[np.ndarray] = None,
                   points_xy: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the classifier feature vector of one detection.

    height_crop / points_xy override det.extra['_height_crop'] /
    det.extra['_points_xy'] (see the module docstring). Returns (float32
    vector, FEATURE_NAMES). Entries that cannot be computed (no color pixels,
    missing geometry) are NaN; classifiers must treat NaN as "cannot classify".
    Color: 12-bin saturation-weighted hue histogram, mean / std of S and V
    (0-1), Lab a / b means (-1..1). Shape: length, width, height (m),
    length / width, footprint fill ratio, footprint circularity.
    """
    det = detection
    out = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)
    bgr = instance_pixels(frame, det, upper=True, height_crop=height_crop)
    if bgr.shape[0] > 0:
        hsv = bgr_to_hsv_pixels(bgr).astype(np.float64)
        out[:HUE_BINS] = hue_histogram(hsv)
        s = hsv[:, 1] / 255.0
        v = hsv[:, 2] / 255.0
        lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        lab = lab.astype(np.float64)
        out[HUE_BINS:HUE_BINS + 6] = [s.mean(), s.std(), v.mean(), v.std(),
                                      (lab[:, 1].mean() - 128.0) / 127.0,
                                      (lab[:, 2].mean() - 128.0) / 127.0]
    length, width, height = _num(det.length), _num(det.width), _num(det.height)
    g = HUE_BINS + 6
    out[g:g + 3] = [length, width, height]
    if math.isfinite(length) and math.isfinite(width) and width > 0:
        out[g + 3] = length / width
        pxy = points_xy if points_xy is not None else det.extra.get('_points_xy')
        if pxy is not None:
            yaw = det.yaw if det.yaw is not None else 0.0
            out[g + 4], out[g + 5] = footprint_shape(pxy, length, width, yaw)
    return out.astype(np.float32), list(FEATURE_NAMES)

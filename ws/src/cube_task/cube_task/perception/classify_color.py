"""
HSV range color classifier for colored blocks (L1).

Ranges use the OpenCV HSV scale: H 0-180 (degrees / 2), S and V 0-255. A
class may have several ranges, which are OR'ed. A range whose hmin > hmax
wraps around 180 (for example red [170 .. 8]).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .features import bgr_to_hsv_pixels, instance_pixels
from ..core.interfaces import InstanceClassifier
from ..core.types import Detection, Frame

LOG = logging.getLogger('cube_task.perception.classify_color')

Ranges = List[List[List[int]]]


def _check_ranges(label: str, ranges: Sequence) -> np.ndarray:
    arr = np.asarray(ranges, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (2, 3):
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (2, 3) or arr.shape[0] < 1:
        raise ValueError('color class %r: ranges must be a list of [[h,s,v],[h,s,v]]' % label)
    if not np.all(np.isfinite(arr)):
        raise ValueError('color class %r: ranges must be finite' % label)
    if np.any(arr[:, :, 0] < 0) or np.any(arr[:, :, 0] > 180):
        raise ValueError('color class %r: hue must be in [0, 180]' % label)
    if np.any(arr[:, :, 1:] < 0) or np.any(arr[:, :, 1:] > 255):
        raise ValueError('color class %r: saturation / value must be in [0, 255]' % label)
    if np.any(arr[:, 0, 1:] > arr[:, 1, 1:]):
        raise ValueError('color class %r: S / V minimum above maximum' % label)
    return arr


def in_ranges(hsv: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Return an N bool array: which Nx3 HSV pixels fall inside any of the Kx2x3 ranges."""
    hsv = np.asarray(hsv).reshape(-1, 3).astype(np.int32)
    out = np.zeros(hsv.shape[0], dtype=bool)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    for lo, hi in np.asarray(ranges):
        sv = (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])
        if lo[0] <= hi[0]:
            hue = (h >= lo[0]) & (h <= hi[0])
        else:
            hue = (h >= lo[0]) | (h <= hi[0])
        out |= hue & sv
    return out


class ColorClassifier(InstanceClassifier):
    """
    Label instances by the fraction of their upper pixels inside HSV ranges.

    classes: label -> list of [[hmin, smin, vmin], [hmax, smax, vmax]].
    min_fraction: the best class needs at least this fraction of the
    instance's upper pixels, else the label is 'unknown'. allowed_labels:
    restrict the classes to these labels (each must exist in classes).
    Score = winning fraction, score_kind = 'color'. Per-class fractions are
    stored in det.extra['color_fractions'].
    """

    def __init__(self, classes: Mapping[str, Sequence], min_fraction: float = 0.35,
                 allowed_labels: Optional[Iterable[str]] = None):
        if not classes:
            raise ValueError('ColorClassifier needs at least one color class')
        if not (0.0 < float(min_fraction) <= 1.0):
            raise ValueError('min_fraction must be in (0, 1]')
        ranges: Dict[str, np.ndarray] = {}
        for label, rr in classes.items():
            label = str(label)
            if not label or label == 'unknown':
                raise ValueError('invalid color class name %r' % label)
            ranges[label] = _check_ranges(label, rr)
        if allowed_labels is not None:
            allowed = [str(a) for a in allowed_labels]
            missing = [a for a in allowed if a not in ranges]
            if missing:
                raise ValueError('labels %s have no color ranges (perception.classifier.'
                                 'color.classes); run color_calib' % missing)
            ranges = {k: ranges[k] for k in allowed}
        self.ranges = ranges
        self.min_fraction = float(min_fraction)
        self.labels = list(ranges.keys())

    def fractions(self, hsv: np.ndarray) -> Dict[str, float]:
        """Return label -> fraction of the Nx3 HSV pixels inside that label's ranges."""
        n = max(int(np.asarray(hsv).reshape(-1, 3).shape[0]), 1)
        return {k: float(np.count_nonzero(in_ranges(hsv, r))) / n
                for k, r in self.ranges.items()}

    def pixel_classes(self, hsv: np.ndarray) -> np.ndarray:
        """
        Return the class of each Nx3 HSV pixel as an int array.

        The value is the index into self.labels of the only class whose
        ranges contain the pixel; -1 when no class or several classes match.
        """
        hsv = np.asarray(hsv).reshape(-1, 3)
        out = np.full(hsv.shape[0], -1, dtype=np.int32)
        hits = np.zeros(hsv.shape[0], dtype=np.int32)
        for i, label in enumerate(self.labels):
            m = in_ranges(hsv, self.ranges[label])
            out[m] = i
            hits += m
        out[hits != 1] = -1
        return out

    def classify(self, frame: Frame, detections: List[Detection]) -> None:
        """Fill label / score / score_kind of each detection in place."""
        for det in detections:
            bgr = instance_pixels(frame, det, upper=True)
            det.score_kind = 'color'
            if bgr.shape[0] == 0:
                det.label, det.score = 'unknown', 0.0
                det.extra['classify_reason'] = 'empty_mask'
                continue
            fr = self.fractions(bgr_to_hsv_pixels(bgr))
            det.extra['color_fractions'] = {k: round(v, 3) for k, v in fr.items()}
            best = max(fr, key=lambda k: (fr[k], k))
            det.score = fr[best]
            if fr[best] >= self.min_fraction:
                det.label = best
            else:
                det.label = 'unknown'
                det.extra['classify_reason'] = 'below_min_fraction'


def _circular_mean_hue(h: np.ndarray) -> float:
    ang = h.astype(np.float64) * (2.0 * math.pi / 180.0)
    m = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang))))
    return (m * 180.0 / (2.0 * math.pi)) % 180.0


def hsv_ranges_from_samples(pixels_by_label: Mapping[str, np.ndarray], lo_pct: float = 2.0,
                            hi_pct: float = 98.0, hue_margin: float = 4.0,
                            sv_margin: float = 25.0, min_saturation: int = 40,
                            min_chromatic_fraction: float = 0.3) -> Dict[str, Ranges]:
    """
    Derive HSV ranges for ColorClassifier from sample pixels.

    pixels_by_label: label -> Nx3 uint8 HSV pixels (OpenCV scale), for example
    from features.instance_pixels + features.bgr_to_hsv_pixels. Hue limits are
    robust percentiles of the hue around its circular mean (only pixels with
    S >= min_saturation), widened by hue_margin; a range crossing 0 / 180 is
    split into two ranges. S / V limits are the lo_pct / hi_pct percentiles
    widened by sv_margin. A class with fewer than min_chromatic_fraction
    saturated pixels (white, gray, black) gets the full hue range. Returns
    label -> [[[h0, s0, v0], [h1, s1, v1]], ...] with Python ints (YAML-ready).
    """
    out: Dict[str, Ranges] = {}
    for label, px in pixels_by_label.items():
        hsv = np.asarray(px).reshape(-1, 3)
        if hsv.shape[0] < 10:
            raise ValueError('label %r: need at least 10 sample pixels, got %d'
                             % (label, hsv.shape[0]))
        h = hsv[:, 0].astype(np.float64)
        s = hsv[:, 1].astype(np.float64)
        v = hsv[:, 2].astype(np.float64)
        s_lo, s_hi = np.percentile(s, [lo_pct, hi_pct])
        v_lo, v_hi = np.percentile(v, [lo_pct, hi_pct])
        s0 = int(max(0, math.floor(s_lo - sv_margin)))
        s1 = int(min(255, math.ceil(s_hi + sv_margin)))
        v0 = int(max(0, math.floor(v_lo - sv_margin)))
        v1 = int(min(255, math.ceil(v_hi + sv_margin)))
        chrom = s >= min_saturation
        if np.count_nonzero(chrom) < max(10, min_chromatic_fraction * hsv.shape[0]):
            out[str(label)] = [[[0, s0, v0], [180, s1, v1]]]
            continue
        hc = h[chrom]
        mu = _circular_mean_hue(hc)
        rel = (hc - mu + 90.0) % 180.0
        r_lo, r_hi = np.percentile(rel, [lo_pct, hi_pct])
        lo = r_lo - hue_margin + mu - 90.0
        hi = r_hi + hue_margin + mu - 90.0
        if hi - lo >= 180.0:
            out[str(label)] = [[[0, s0, v0], [180, s1, v1]]]
            continue
        lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
        if lo_i < 0:
            ranges = [[[lo_i + 180, s0, v0], [180, s1, v1]], [[0, s0, v0], [hi_i, s1, v1]]]
        elif hi_i > 180:
            ranges = [[[lo_i, s0, v0], [180, s1, v1]], [[0, s0, v0], [hi_i - 180, s1, v1]]]
        else:
            ranges = [[[lo_i, s0, v0], [hi_i, s1, v1]]]
        out[str(label)] = ranges
    return out

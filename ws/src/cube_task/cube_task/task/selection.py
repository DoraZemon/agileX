"""
Choose the next object to pick from the current candidates.

Positions are base-frame XY in meters. Only detections with a finite
``center_base`` are considered.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Detection

STRATEGIES = ('nearest', 'score', 'isolated')
#: clearances above this (m) are "free enough" and do not change the ranking.
CLEARANCE_CAP = 0.05
#: half width (m) of the band swept by a finger along the closing axis.
FINGER_HALF_BAND = 0.015


def _xy(det: Detection) -> Optional[np.ndarray]:
    c = det.center_base
    if c is None or len(c) < 2:
        return None
    xy = np.asarray(c[:2], dtype=float)
    return xy if np.all(np.isfinite(xy)) else None


def _radius(det: Detection) -> float:
    vals = [v for v in (det.length, det.width) if v is not None and math.isfinite(float(v))]
    return max(vals) / 2.0 if vals else 0.0


def is_blacklisted(det: Detection, positions: Iterable[Sequence[float]],
                   radius: float) -> bool:
    """Return True when det lies within radius (m) of any blacklisted XY position."""
    xy = _xy(det)
    if xy is None:
        return True
    for p in positions or ():
        if math.hypot(xy[0] - float(p[0]), xy[1] - float(p[1])) <= radius:
            return True
    return False


def clearances(det: Detection, neighbors: Sequence[Detection]) -> Tuple[float, float]:
    """
    Return (closing-direction clearance, overall clearance) in meters.

    Overall: smallest gap between bounding circles of det and any neighbor.
    Closing direction: smallest free distance beyond det's sides along its minor
    axis (yaw + pi/2) to a neighbor whose circle crosses the finger band.
    Both are math.inf without neighbors.
    """
    xy = _xy(det)
    if xy is None:
        return -math.inf, -math.inf
    r0 = _radius(det)
    half_w = (float(det.width) / 2.0 if det.width is not None and math.isfinite(float(det.width))
              else r0)
    overall = math.inf
    closing = math.inf
    u = v = None
    if det.yaw is not None and math.isfinite(float(det.yaw)):
        a = float(det.yaw) + math.pi / 2.0
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
    for n in neighbors:
        if n is det:
            continue
        nxy = _xy(n)
        if nxy is None:
            continue
        d = nxy - xy
        dist = float(np.hypot(d[0], d[1]))
        if dist < 1e-6:
            continue        # the same object seen twice
        rn = _radius(n)
        overall = min(overall, dist - r0 - rn)
        if u is not None and abs(float(d @ v)) - rn < FINGER_HALF_BAND:
            closing = min(closing, abs(float(d @ u)) - rn - half_w)
    return closing, overall


def select_target(candidates: Sequence[Detection], strategy: str = 'nearest',
                  order: Optional[Sequence[str]] = None,
                  arm_xy: Sequence[float] = (0.0, 0.0),
                  already_failed: Optional[Iterable[Sequence[float]]] = None,
                  blacklist_radius: float = 0.03,
                  neighbors: Optional[Sequence[Detection]] = None) -> Optional[Detection]:
    """
    Return the detection to pick next, or None.

    strategy: 'nearest' (to arm_xy, the robot base), 'score' (classifier score,
    ties by distance) or 'isolated' (largest free space along the closing
    direction, then overall clearance, then distance; clearances are capped at
    5 cm). order: optional label list; the first label that has a candidate wins.
    already_failed: blacklisted base XY positions; candidates within
    blacklist_radius of one are skipped. neighbors: all detections used for
    clearances (default: candidates).
    """
    if strategy not in STRATEGIES:
        raise ValueError('unknown selection strategy %r (use %s)' % (strategy, STRATEGIES))
    failed = [tuple(p) for p in (already_failed or ())]
    pool: List[Detection] = [d for d in candidates
                             if _xy(d) is not None and not is_blacklisted(d, failed,
                                                                          blacklist_radius)]
    if order:
        for label in order:
            sub = [d for d in pool if d.label == label]
            if sub:
                pool = sub
                break
        else:
            return None
    if not pool:
        return None
    base = np.asarray(arm_xy, dtype=float)[:2]
    others = list(neighbors) if neighbors is not None else list(candidates)

    def dist(d: Detection) -> float:
        xy = _xy(d)
        return float(np.hypot(*(xy - base)))

    def tie(d: Detection) -> Tuple[float, float]:
        xy = _xy(d)
        return (round(float(xy[0]), 6), round(float(xy[1]), 6))

    if strategy == 'nearest':
        return min(pool, key=lambda d: (dist(d), tie(d)))
    if strategy == 'score':
        return min(pool, key=lambda d: (-float(d.score), dist(d), tie(d)))

    def iso_key(d: Detection):
        c, o = clearances(d, others)
        return (-min(c, CLEARANCE_CAP), -min(o, CLEARANCE_CAP), dist(d), tie(d))
    return min(pool, key=iso_key)

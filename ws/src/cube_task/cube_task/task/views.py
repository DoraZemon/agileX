"""
Several observe poses for the wrist camera (eye_in_hand).

One wrist view does not cover an L2 pick area with six objects, so the runner
can scan a list of taught observe poses (robot.poses.observe_views, joint lists
in scan order; empty = robot.poses.observe_joints only). Each view gives the
valid detections of its latest capture in the base frame; merge_view_detections
joins them into one list (an object seen from two views once, the more complete
view winning) and view_coverage tells how much of the pick polygon at least one
view sees with room for an object around each point.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.types import Detection

#: default distance (m) below which two views' detections are one object
MERGE_DIST = 0.04
#: grid step (m) of the pick-polygon points tested by view_coverage
COVERAGE_STEP = 0.01
#: an object around a covered point must fit in one image: half size (m) of that box
OBJECT_HALF = 0.05
#: image border (px) a covered point's box must stay inside
IMAGE_MARGIN_PX = 6


def observe_views(cfg) -> List[List[float]]:
    """
    Return the observe poses to scan (joint lists), [] when none is taught.

    robot.poses.observe_views when it is a non-empty list, else
    [robot.poses.observe_joints] when that is set. Raises ValueError for a
    view that is not 6 finite numbers.
    """
    raw = cfg.get('robot.poses.observe_views', None) or []
    if not raw:
        q = cfg.get('robot.poses.observe_joints', None)
        raw = [q] if q is not None else []
    out = []
    for i, q in enumerate(raw):
        try:
            vals = [float(v) for v in q]
        except (TypeError, ValueError):
            raise ValueError('robot.poses.observe_views[%d] must be 6 numbers, got %r' % (i, q))
        if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
            raise ValueError('robot.poses.observe_views[%d] must be 6 finite numbers, got %r'
                             % (i, q))
        out.append(vals)
    return out


def _xy(d: Detection) -> Optional[Tuple[float, float]]:
    c = d.center_base
    if c is None:
        return None
    x, y = float(c[0]), float(c[1])
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def view_of(d: Detection) -> Optional[int]:
    """Return the observe view a merged detection came from (None if not tagged)."""
    v = (d.extra or {}).get('view')
    return None if v is None else int(v)


def tag_view(dets: Iterable[Detection], view: int) -> List[Detection]:
    """Record the observe view in each detection's extra['view']; return them as a list."""
    out = []
    for d in dets:
        if d.extra is None:
            d.extra = {}
        d.extra['view'] = int(view)
        out.append(d)
    return out


def merge_view_detections(scene: Dict[int, Sequence[Detection]], stale: Set[int] = frozenset(),
                          dist: float = MERGE_DIST) -> List[Detection]:
    """
    Return one detection per object over the views' latest detections.

    scene: view -> its valid detections (base frame). Detections of different
    views closer than dist (m, base XY, any label) are one object; the kept
    one is from a view not in stale (captured after the last arm action) if
    possible, then the one with more depth points (the more complete view).
    Detections without a base position are dropped.
    """
    kept: List[Tuple[Tuple[float, float], Detection]] = []

    def rank(d: Detection) -> Tuple[int, int]:
        return (0 if view_of(d) in stale else 1, int(d.n_points))

    for v in sorted(scene):
        for d in scene[v]:
            p = _xy(d)
            if p is None:
                continue
            hit = None
            for i, (q, _o) in enumerate(kept):
                if math.hypot(p[0] - q[0], p[1] - q[1]) <= dist:
                    hit = i
                    break
            if hit is None:
                kept.append((p, d))
            elif rank(d) > rank(kept[hit][1]):
                kept[hit] = (p, d)
    return [d for _p, d in kept]


def drop_near(scene: Dict[int, List[Detection]], xy: Sequence[float], dist: float) -> None:
    """Remove every view's detections within dist (m) of base xy (a picked object)."""
    for v in list(scene):
        scene[v] = [d for d in scene[v] if _xy(d) is None
                    or math.hypot(_xy(d)[0] - float(xy[0]), _xy(d)[1] - float(xy[1])) > dist]


def nearest_view(views: Sequence[Sequence[float]], todo: Iterable[int],
                 q_now: Optional[Sequence[float]]) -> int:
    """Return the view in todo closest to the joints q_now (max joint step), else the lowest."""
    todo = sorted(set(int(v) for v in todo))
    if not todo:
        raise ValueError('no view to choose from')
    if q_now is None:
        return todo[0]
    q = np.asarray(q_now, dtype=float).reshape(-1)[:6]
    if q.shape[0] != 6 or not np.all(np.isfinite(q)):
        return todo[0]
    return min(todo, key=lambda v: (float(np.max(np.abs(np.asarray(views[v]) - q))), v))


def polygon_points(poly: Sequence[Sequence[float]], step: float = COVERAGE_STEP) -> np.ndarray:
    """Return Nx2 grid points (step m) inside a convex or simple polygon in base XY."""
    P = np.asarray(poly, dtype=float).reshape(-1, 2)
    if P.shape[0] < 3:
        return np.zeros((0, 2))
    lo, hi = P.min(axis=0), P.max(axis=0)
    xs = np.arange(lo[0] + step / 2.0, hi[0], step)
    ys = np.arange(lo[1] + step / 2.0, hi[1], step)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 2))
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    j = P.shape[0] - 1
    for i in range(P.shape[0]):                 # even-odd ray casting
        xi, yi = P[i]
        xj, yj = P[j]
        cross = (yi > pts[:, 1]) != (yj > pts[:, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (xj - xi) * (pts[:, 1] - yi) / (yj - yi) + xi
        inside ^= cross & (pts[:, 0] < xint)
        j = i
    return pts[inside]


def _in_image(pts_base: np.ndarray, T_base_camera: np.ndarray, intr: Any,
              margin: float, bottom: float) -> np.ndarray:
    """Return a bool per base point: projects inside the image (margin px, bottom rows cut)."""
    T_cb = np.linalg.inv(np.asarray(T_base_camera, dtype=float))
    pc = pts_base @ T_cb[:3, :3].T + T_cb[:3, 3]
    z = pc[:, 2]
    ok = z > 0.05
    zs = np.where(ok, z, 1.0)
    u = float(intr.fx) * pc[:, 0] / zs + float(intr.cx)
    v = float(intr.fy) * pc[:, 1] / zs + float(intr.cy)
    w, h = float(intr.width), float(intr.height)
    return ok & (u >= margin) & (u <= w - 1 - margin) & (v >= margin) & (v <= h - 1 - bottom)


def view_coverage(poly: Sequence[Sequence[float]], table_z: float,
                  cameras: Sequence[np.ndarray], intr: Any, half: float = OBJECT_HALF,
                  margin_px: float = IMAGE_MARGIN_PX, bottom_px: float = 0.0,
                  step: float = COVERAGE_STEP) -> Tuple[float, List[int], np.ndarray]:
    """
    Return (covered fraction, points per view, uncovered points Nx2) of a pick polygon.

    A grid point counts as covered by a view when the square of half size half
    (m) around it on the table at table_z projects inside that view's image
    (margin_px, bottom_px rows cut off for the fingers): an object there is
    whole in one image. cameras: T_base_camera of each view.
    """
    pts = polygon_points(poly, step)
    if len(pts) == 0 or not cameras:
        return 0.0, [0] * len(cameras), pts
    covered = np.zeros(len(pts), dtype=bool)
    per_view = []
    offsets = [(-half, -half), (-half, half), (half, -half), (half, half)]
    for T in cameras:
        ok = np.ones(len(pts), dtype=bool)
        for dx, dy in offsets:
            p3 = np.column_stack([pts[:, 0] + dx, pts[:, 1] + dy, np.full(len(pts), table_z)])
            ok &= _in_image(p3, T, intr, margin_px, bottom_px)
        per_view.append(int(ok.sum()))
        covered |= ok
    return float(covered.mean()), per_view, pts[~covered]


def coverage_report(cfg, views: Sequence[Sequence[float]], T_flange_camera: np.ndarray,
                    intr: Any, kin=None, bottom_px: float = 0.0) -> Tuple[Optional[float], str]:
    """
    Return (covered fraction or None, one-line message) of the taught views on the pick area.

    Uses robot kinematics (kin, default PiperKinematics) for each view's flange
    pose, the pick polygon and perception.table.expected_z (0 when unset).
    """
    poly = cfg.get('perception.roi.pick_polygon', None) or []
    if len(poly) < 3:
        return None, 'view coverage not checked: no pick polygon'
    if not views:
        return None, 'view coverage not checked: no observe pose taught'
    if kin is None:
        from ..robot.kinematics import PiperKinematics
        kin = PiperKinematics()
    tz = cfg.get('perception.table.expected_z', None)
    cams = [kin.fk(q) @ np.asarray(T_flange_camera, dtype=float) for q in views]
    frac, per_view, missing = view_coverage(poly, 0.0 if tz is None else float(tz), cams, intr,
                                            bottom_px=bottom_px)
    msg = 'observe views cover %.0f%% of the pick area (an object fits whole in one image); ' \
          'per view %s grid points' % (100.0 * frac, per_view)
    if len(missing):
        lo, hi = missing.min(axis=0), missing.max(axis=0)
        msg += '; uncovered x %.2f..%.2f y %.2f..%.2f m' % (lo[0], hi[0], lo[1], hi[1])
    return frac, msg

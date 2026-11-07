"""
Table plane estimation in the robot base frame.

RANSAC on a random subset of base-frame points restricted to the pick
polygon grown by ``roi_margin`` (default 5 cm) with place zones removed,
followed by a least-squares refinement on the inliers. Units: meters.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Optional

import numpy as np

from .geometry import parse_polygon, points_in_polygon, points_in_zones
from ..core.types import TablePlane

LOG = logging.getLogger('cube_task.perception.table')


@dataclass
class TableParams:
    """Validated table-fit parameters (see config perception.table.*)."""

    ransac_iters: int = 300
    inlier_thresh: float = 0.005        # m, |signed distance| for an inlier
    min_inlier_ratio: float = 0.3       # inliers / points inside the restricted region
    max_tilt_deg: float = 20.0          # normal vs base +z
    expected_z: Optional[float] = None  # m, plane z at the reference point; None = no check
    expected_z_tol: float = 0.05        # m
    roi_margin: float = 0.05            # m, pick polygon grown by this for the fit
    zone_margin: float = 0.02           # m, zones grown by this are excluded from the fit
    ransac_points: int = 4000           # random subset size for hypothesis scoring
    min_points: int = 200               # fewer restricted points -> no fit

    @staticmethod
    def from_cfg(cfg: Any) -> 'TableParams':
        """Read perception.table.* (and roi.zone_exclude_margin); raise ValueError if bad."""
        p = TableParams(
            ransac_iters=int(cfg.get('perception.table.ransac_iters', 300)),
            inlier_thresh=float(cfg.get('perception.table.inlier_thresh', 0.005)),
            min_inlier_ratio=float(cfg.get('perception.table.min_inlier_ratio', 0.3)),
            max_tilt_deg=float(cfg.get('perception.table.max_tilt_deg', 20.0)),
            expected_z=cfg.get('perception.table.expected_z', None),
            expected_z_tol=float(cfg.get('perception.table.expected_z_tol', 0.05)),
            roi_margin=float(cfg.get('perception.table.roi_margin', 0.05)),
            zone_margin=float(cfg.get('perception.roi.zone_exclude_margin', 0.02)),
            ransac_points=int(cfg.get('perception.table.ransac_points', 4000)),
            min_points=int(cfg.get('perception.table.min_points', 200)),
        )
        if p.expected_z is not None:
            p.expected_z = float(p.expected_z)
        p.validate()
        return p

    def validate(self) -> None:
        """Raise ValueError when a parameter is out of its meaningful range."""
        if self.ransac_iters < 1:
            raise ValueError('perception.table.ransac_iters must be >= 1')
        if not (0.0 < self.inlier_thresh < 0.1):
            raise ValueError('perception.table.inlier_thresh must be in (0, 0.1) m')
        if not (0.0 <= self.min_inlier_ratio <= 1.0):
            raise ValueError('perception.table.min_inlier_ratio must be in [0, 1]')
        if not (0.0 < self.max_tilt_deg <= 90.0):
            raise ValueError('perception.table.max_tilt_deg must be in (0, 90]')
        if self.expected_z is not None and not math.isfinite(self.expected_z):
            raise ValueError('perception.table.expected_z must be finite or null')
        if self.expected_z_tol <= 0.0:
            raise ValueError('perception.table.expected_z_tol must be > 0')
        if self.roi_margin < 0.0 or self.zone_margin < 0.0:
            raise ValueError('table roi / zone margins must be >= 0')
        if self.ransac_points < 3 or self.min_points < 3:
            raise ValueError('perception.table.ransac_points / min_points must be >= 3')


@dataclass
class TableFit:
    """
    Detailed result of fit_table_plane_detailed.

    plane is None when the fit failed or was rejected; reason then names the
    check ('too_few_points', 'no_hypothesis', 'tilt', 'low_inlier_ratio',
    'unexpected_z') and candidate holds the rejected plane if there was one.
    """

    plane: Optional[TablePlane]
    reason: str = ''
    candidate: Optional[TablePlane] = None
    n_points: int = 0                   # points inside the restricted region
    n_inliers: int = 0
    tilt_deg: float = float('nan')
    z_ref: float = float('nan')         # plane z at the reference xy (base frame, m)


def _plane_from_points(pts: np.ndarray) -> Optional[tuple]:
    """Least-squares plane through Nx3 points; normal oriented to +z."""
    if pts.shape[0] < 3:
        return None
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q
    w, v = np.linalg.eigh(cov)
    if w[1] <= 1e-12:
        return None
    n = v[:, 0]
    if n[2] < 0:
        n = -n
    n = n / np.linalg.norm(n)
    return n, -float(n @ c)


def _restrict(points: np.ndarray, polygon: Optional[np.ndarray], zones: Any,
              params: TableParams) -> np.ndarray:
    keep = np.all(np.isfinite(points), axis=1)
    pts = points[keep]
    if polygon is not None and pts.shape[0]:
        pts = pts[points_in_polygon(pts[:, :2], polygon, margin=params.roi_margin)]
    if zones and pts.shape[0]:
        pts = pts[~points_in_zones(pts[:, :2], zones, margin=params.zone_margin)]
    return pts


def fit_table_plane_detailed(points_base: np.ndarray, cfg: Any,
                             rng: Optional[np.random.Generator] = None,
                             pick_polygon: Optional[np.ndarray] = None,
                             zones: Any = None) -> TableFit:
    """
    Fit the table plane and report why a fit was rejected.

    points_base: Nx3 base-frame points (meters); NaN rows are ignored.
    cfg: Config (perception.table.*) or a TableParams.
    pick_polygon: Mx2 base XY; None reads perception.roi.pick_polygon from cfg
    (when cfg is a Config). zones: dict / iterable of core Zone excluded from
    the fit. rng: numpy Generator for the RANSAC sampling (default seed 0).
    """
    params = cfg if isinstance(cfg, TableParams) else TableParams.from_cfg(cfg)
    if pick_polygon is None and not isinstance(cfg, TableParams):
        pick_polygon = parse_polygon(cfg.get('perception.roi.pick_polygon', None))
    rng = rng if rng is not None else np.random.default_rng(0)
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    pts = _restrict(pts, pick_polygon, zones, params)
    n = int(pts.shape[0])
    if n < params.min_points:
        return TableFit(None, 'too_few_points', n_points=n)

    if n <= params.ransac_points:
        sub = pts
    else:
        sub = pts[rng.choice(n, params.ransac_points, replace=False)]
    m = sub.shape[0]
    idx = rng.integers(0, m, size=(params.ransac_iters, 3))
    p0, p1, p2 = sub[idx[:, 0]], sub[idx[:, 1]], sub[idx[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(nrm, axis=1)
    ok = norm > 1e-9
    nrm[ok] /= norm[ok, None]
    nrm[nrm[:, 2] < 0] *= -1.0
    if not np.any(ok):
        return TableFit(None, 'no_hypothesis', n_points=n)
    cos_tilt = math.cos(math.radians(params.max_tilt_deg))
    ok &= nrm[:, 2] >= cos_tilt
    if not np.any(ok):
        return TableFit(None, 'tilt', n_points=n)
    nrm = nrm[ok]
    d = -np.sum(nrm * p0[ok], axis=1)
    dist = sub.astype(np.float32) @ nrm.T.astype(np.float32) + d.astype(np.float32)
    thr = params.inlier_thresh
    inl = np.count_nonzero(np.abs(dist) < thr, axis=0)
    # Points far below a candidate plane are impossible for the real table
    # (it is the lowest surface), so penalize planes fitted to object tops.
    below = np.count_nonzero(dist < -max(3.0 * thr, 0.01), axis=0)
    best = int(np.argmax(inl - below))
    n_best, d_best = nrm[best], float(d[best])

    inliers = None
    for _ in range(3):
        sd = pts @ n_best + d_best
        inliers = np.abs(sd) < thr
        fit = _plane_from_points(pts[inliers])
        if fit is None:
            return TableFit(None, 'no_hypothesis', n_points=n)
        n_best, d_best = fit
    sd = pts @ n_best + d_best
    inliers = np.abs(sd) < thr
    n_in = int(np.count_nonzero(inliers))
    rms = float(np.sqrt(np.mean(sd[inliers] ** 2))) if n_in else float('nan')
    ratio = n_in / float(n)
    plane = TablePlane(normal=n_best.astype(np.float64), d=d_best, frame='base',
                       inlier_ratio=ratio, rms=rms)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n_best[2])))))
    if pick_polygon is not None:
        ref = np.asarray(pick_polygon, dtype=float).mean(axis=0)
    else:
        ref = pts[inliers, :2].mean(axis=0) if n_in else pts[:, :2].mean(axis=0)
    z_ref = plane.z_at(float(ref[0]), float(ref[1]))
    res = TableFit(None, '', candidate=plane, n_points=n, n_inliers=n_in,
                   tilt_deg=tilt, z_ref=z_ref)
    if tilt > params.max_tilt_deg:
        res.reason = 'tilt'
    elif ratio < params.min_inlier_ratio:
        res.reason = 'low_inlier_ratio'
    elif (params.expected_z is not None
          and abs(z_ref - params.expected_z) > params.expected_z_tol):
        res.reason = 'unexpected_z'
    else:
        res.plane = plane
    if res.reason:
        LOG.debug('table fit rejected: %s (tilt %.1f deg, ratio %.2f, z %.4f)',
                  res.reason, tilt, ratio, z_ref)
    return res


def fit_table_plane(points_base: np.ndarray, cfg: Any,
                    rng: Optional[np.random.Generator] = None,
                    pick_polygon: Optional[np.ndarray] = None,
                    zones: Any = None) -> Optional[TablePlane]:
    """
    Fit the table plane (base frame, normal +z) or return None.

    Same arguments as fit_table_plane_detailed. Rejected when the normal is
    tilted more than max_tilt_deg from base z, the inlier ratio is below
    min_inlier_ratio, or expected_z is set and the plane z at the pick-polygon
    centroid differs from it by more than expected_z_tol.
    """
    return fit_table_plane_detailed(points_base, cfg, rng, pick_polygon, zones).plane

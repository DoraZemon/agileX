"""
Class-agnostic segmentation of objects standing on the table, plus geometry.

Every valid pixel gets its height above the fitted table plane. Pixels with
min_height < h < max_height inside the pick polygon (and outside the place
zones) form the object mask; connected components, optionally split where
objects touch, become instances. Each instance gets base-frame geometry as
specified in docs/CONTRACTS.md "Detection geometry". Units: meters, radians,
pixels. Instances that fail a check are kept with valid=False and a
reject_reason so logs show why they were not picked.

The pipeline can split an instance further after segmentation (touching
blocks of different colors, or of the same color): split_by_classes and
find_block_cuts compute the parts, split_instance recomputes the geometry of
each part with the same code as a normal instance (Segmentation.ctx).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import parse_polygon, points_in_polygon, points_in_zones, project_points
from ..core.types import CameraIntrinsics, Detection, TablePlane

LOG = logging.getLogger('cube_task.perception.segment')

# Most severe first; Detection.reject_reason is the first one that applies.
REASON_ORDER = ('too_tall', 'touches_image_border', 'touches_roi_border', 'touches_zone',
                'few_points', 'occluded', 'too_small', 'too_large', 'merged')
MERGED_MODES = ('split', 'reject', 'off')   # perception.segmentation.merged_blocks
CAP_DEPTH_FRACTION = 0.05   # 'cap' points: z >= top_z - max(2 mm, this * object height)
CAP_ABOVE_MIN_M = 0.003     # ... and z <= top_z + max(this, the band depth,
CAP_ABOVE_RMS_K = 3.0       # this * table rms) (see cap_band)
#: extra['_points_base'] (instance base-frame points without depth-edge pixels) keeps at
#: most this many points (uniform stride).
MAX_INSTANCE_POINTS = 4000

_K3 = np.ones((3, 3), np.uint8)
_BIG = np.float32(1e6)


@dataclass
class SegmentParams:
    """Validated segmentation parameters (config perception.segmentation.* and roi.*)."""

    min_height: float = 0.008           # m above the table
    max_height: float = 0.25            # m above the table
    min_area_m2: float = 0.00012        # footprint rectangle area limits
    max_area_m2: float = 0.040
    min_points: int = 40                # valid above-table points per instance
    morph_open_px: int = 3
    morph_close_px: int = 5
    split_touching: bool = True
    split_min_peak_dist_m: float = 0.025
    split_neck_ratio: float = 0.75      # merge two parts when neck >= ratio * smaller radius
    split_cell_m: float = 0.002         # top-down grid cell for splitting
    split_height_jump_m: float = 0.008  # height step between grid cells that separates objects
    split_jump_noise_k: float = 6.0     # step at least k * sqrt(2) * table rms (0 = off)
    top_percentile: float = 90.0
    edge_erode_px: int = 2              # mask border ignored for geometry
    edge_jump_abs_m: float = 0.006      # depth step to a neighbor counted as an edge:
    edge_jump_rel: float = 0.01         # abs + rel * depth
    despike: bool = True                # drop isolated pixels much nearer than their 3x3 median
    extent_trim_pct: float = 0.5        # robust extents: percentile trimmed on each side
    max_occlusion_ratio: float = 0.2    # fraction of the outline hidden by nearer surfaces
    min_blob_px: int = 15               # smaller components are dropped (counted)
    max_instances: int = 40
    exclude_zones: bool = True
    zone_exclude_margin: float = 0.02
    # Splitting of touching L1 blocks (used by the pipeline with a ColorClassifier).
    color_split: bool = True            # split instances that hold several block colors
    color_split_min_fraction: float = 0.25  # each color needs this share of the upper pixels
    merged_blocks: str = 'split'        # same-color merges: split | reject | off
    block_size: float = 0.035           # L1 cube edge (m), perception.l1_block_size

    @staticmethod
    def from_cfg(cfg: Any) -> 'SegmentParams':
        """Read the config; raise ValueError for values outside their valid range."""
        s = 'perception.segmentation.'
        p = SegmentParams(
            min_height=float(cfg.get(s + 'min_height', 0.008)),
            max_height=float(cfg.get(s + 'max_height', 0.25)),
            min_area_m2=float(cfg.get(s + 'min_area_m2', 0.00012)),
            max_area_m2=float(cfg.get(s + 'max_area_m2', 0.040)),
            min_points=int(cfg.get(s + 'min_points', 40)),
            morph_open_px=int(cfg.get(s + 'morph_open_px', 3)),
            morph_close_px=int(cfg.get(s + 'morph_close_px', 5)),
            split_touching=bool(cfg.get(s + 'split_touching', True)),
            split_min_peak_dist_m=float(cfg.get(s + 'split_min_peak_dist_m', 0.025)),
            split_neck_ratio=float(cfg.get(s + 'split_neck_ratio', 0.75)),
            split_cell_m=float(cfg.get(s + 'split_cell_m', 0.002)),
            split_height_jump_m=float(cfg.get(s + 'split_height_jump_m', 0.008)),
            split_jump_noise_k=float(cfg.get(s + 'split_jump_noise_k', 6.0)),
            top_percentile=float(cfg.get(s + 'top_percentile', 90)),
            edge_erode_px=int(cfg.get(s + 'edge_erode_px', 2)),
            edge_jump_abs_m=float(cfg.get(s + 'edge_jump_abs_m', 0.006)),
            edge_jump_rel=float(cfg.get(s + 'edge_jump_rel', 0.01)),
            despike=bool(cfg.get(s + 'despike', True)),
            extent_trim_pct=float(cfg.get(s + 'extent_trim_pct', 0.5)),
            max_occlusion_ratio=float(cfg.get(s + 'max_occlusion_ratio', 0.2)),
            min_blob_px=int(cfg.get(s + 'min_blob_px', 15)),
            max_instances=int(cfg.get(s + 'max_instances', 40)),
            exclude_zones=bool(cfg.get('perception.roi.exclude_zones', True)),
            zone_exclude_margin=float(cfg.get('perception.roi.zone_exclude_margin', 0.02)),
            color_split=bool(cfg.get(s + 'color_split', True)),
            color_split_min_fraction=float(cfg.get(s + 'color_split_min_fraction', 0.25)),
            merged_blocks=str(cfg.get(s + 'merged_blocks', 'split')).strip().lower(),
            block_size=float(cfg.get('perception.l1_block_size', 0.035)),
        )
        p.validate()
        return p

    def validate(self) -> None:
        """Raise ValueError when a parameter is out of its meaningful range."""
        if not (0.0 < self.min_height < self.max_height):
            raise ValueError('segmentation: need 0 < min_height < max_height')
        if not (0.0 <= self.min_area_m2 < self.max_area_m2):
            raise ValueError('segmentation: need 0 <= min_area_m2 < max_area_m2')
        if self.min_points < 3:
            raise ValueError('segmentation.min_points must be >= 3')
        if self.morph_open_px < 0 or self.morph_close_px < 0 or self.edge_erode_px < 0:
            raise ValueError('segmentation morphology sizes must be >= 0')
        if self.split_min_peak_dist_m <= 0.0:
            raise ValueError('segmentation.split_min_peak_dist_m must be > 0')
        if not (0.0 < self.split_neck_ratio <= 1.0):
            raise ValueError('segmentation.split_neck_ratio must be in (0, 1]')
        if not (0.0005 <= self.split_cell_m <= 0.01):
            raise ValueError('segmentation.split_cell_m must be in [0.0005, 0.01] m')
        if self.split_height_jump_m <= 0.0:
            raise ValueError('segmentation.split_height_jump_m must be > 0')
        if not (0.0 <= self.split_jump_noise_k <= 20.0):
            raise ValueError('segmentation.split_jump_noise_k must be in [0, 20]')
        if not (50.0 <= self.top_percentile <= 100.0):
            raise ValueError('segmentation.top_percentile must be in [50, 100]')
        if self.edge_jump_abs_m <= 0.0 or self.edge_jump_rel < 0.0:
            raise ValueError('segmentation edge jump thresholds must be positive')
        if not (0.0 <= self.extent_trim_pct < 10.0):
            raise ValueError('segmentation.extent_trim_pct must be in [0, 10)')
        if not (0.0 < self.max_occlusion_ratio <= 1.0):
            raise ValueError('segmentation.max_occlusion_ratio must be in (0, 1]')
        if self.min_blob_px < 1 or self.max_instances < 1:
            raise ValueError('segmentation min_blob_px / max_instances must be >= 1')
        if self.zone_exclude_margin < 0.0:
            raise ValueError('roi.zone_exclude_margin must be >= 0')
        if not (0.05 <= self.color_split_min_fraction <= 0.5):
            raise ValueError('segmentation.color_split_min_fraction must be in [0.05, 0.5]')
        if self.merged_blocks not in MERGED_MODES:
            raise ValueError('segmentation.merged_blocks must be one of %s' % (MERGED_MODES,))
        if not (0.01 <= self.block_size <= 0.15):
            raise ValueError('perception.l1_block_size must be in [0.01, 0.15] m')


@dataclass
class Segmentation:
    """Output of segment_objects."""

    detections: List[Detection]
    height: np.ndarray                  # HxW float32 height above the table (m), NaN invalid
    object_mask: np.ndarray             # HxW bool object pixels after morphology
    warnings: List[str] = field(default_factory=list)
    n_dropped: int = 0                  # components smaller than min_blob_px
    ctx: Optional['_Ctx'] = None        # frame arrays for split_instance
    split_jump_m: float = 0.0           # barrier height step used by split_component


def height_above_plane(points_base: np.ndarray, plane: TablePlane) -> np.ndarray:
    """Return the HxW float32 signed height (m) of base points above the plane, NaN invalid."""
    n = np.asarray(plane.normal, dtype=np.float32)
    h = points_base @ n
    h += np.float32(plane.d)
    return h.astype(np.float32, copy=False)


def depth_edges(z: np.ndarray, valid: np.ndarray, abs_thresh: float,
                rel_thresh: float) -> np.ndarray:
    """
    Mark valid pixels next to a depth discontinuity.

    A pixel is an edge pixel when some valid 8-neighbor differs from its depth
    by more than abs_thresh + rel_thresh * depth (meters). Mixed / flying
    pixels and the outermost pixels of objects against the background are
    edge pixels. Returns an HxW bool array.
    """
    zf = np.where(valid, z, 0.0).astype(np.float32)
    zmax = cv2.dilate(zf, _K3)
    zmin = cv2.erode(np.where(valid, z, _BIG).astype(np.float32), _K3)
    thr = np.float32(abs_thresh) + np.float32(rel_thresh) * zf
    return valid & (((zmax - zf) > thr) | ((zf - zmin) > thr))


def depth_spikes(z: np.ndarray, valid: np.ndarray, abs_thresh: float,
                 rel_thresh: float) -> np.ndarray:
    """
    Mark isolated valid pixels much nearer to the camera than their neighborhood.

    A pixel is a spike when its depth is more than abs_thresh + rel_thresh *
    depth (meters) below the median of its 3x3 neighborhood (itself
    included). Up to 4 outliers in a 3x3 window are found. Along a straight
    border of a nearer surface 6 of the 9 window pixels are on the surface,
    so the border is kept; a convex corner pixel and lines 1-2 pixels wide
    are marked. Such outliers (structured-light mismatches on glossy tops,
    persistent over averaged frames) lie above the object top and are
    displaced toward the camera in XY. Returns an HxW bool array.
    """
    zf = np.where(valid, z, 0.0).astype(np.float32)
    med = cv2.medianBlur(zf, 3)
    thr = np.float32(abs_thresh) + np.float32(rel_thresh) * med
    return valid & (zf < med - thr)


def _morph(mask: np.ndarray, op: int, size: int) -> np.ndarray:
    if size < 2:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(bool)


def wrap_half_pi(a: float) -> float:
    """Wrap an axis angle (rad) to [-pi/2, pi/2)."""
    return float((a + math.pi / 2.0) % math.pi - math.pi / 2.0)


def rect_geometry(xy: np.ndarray, xy_ext: Optional[np.ndarray] = None,
                  max_extend: float = 0.0,
                  trim_pct: float = 0.5) -> Tuple[np.ndarray, float, float, float]:
    """
    Fit a robust minimum-area rectangle to base-XY points (meters).

    The orientation comes from cv2.minAreaRect of xy. Extents along the
    rectangle axes are the trim_pct / (100 - trim_pct) percentiles of xy.
    When xy_ext (a superset without the eroded border) is given, each side is
    pushed out to the matching percentile of xy_ext, but by at most
    max_extend meters; this restores the border removed by erosion without
    letting flying pixels inflate the footprint. Returns (center_xy (2,),
    yaw of the major axis in [-pi/2, pi/2), length, width) with length >= width.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 1:
        raise ValueError('rect_geometry needs at least one point')
    c0 = xy.mean(axis=0)
    q = ((xy - c0) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(q)
    box = cv2.boxPoints(rect).astype(np.float64)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    a = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    na = float(np.hypot(*a))
    u = np.array([1.0, 0.0]) if na < 1e-9 else a / na
    v = np.array([-u[1], u[0]])
    c = c0 + np.asarray(rect[0], dtype=np.float64) / 1000.0
    rel = xy - c
    pu, pv = rel @ u, rel @ v
    lo_u, hi_u = np.percentile(pu, [trim_pct, 100.0 - trim_pct])
    lo_v, hi_v = np.percentile(pv, [trim_pct, 100.0 - trim_pct])
    if xy_ext is not None and max_extend > 0.0:
        xe = np.asarray(xy_ext, dtype=np.float64).reshape(-1, 2)
        if xe.shape[0] > 0:
            rel_e = xe - c
            elo_u, ehi_u = np.percentile(rel_e @ u, [trim_pct, 100.0 - trim_pct])
            elo_v, ehi_v = np.percentile(rel_e @ v, [trim_pct, 100.0 - trim_pct])
            lo_u = min(lo_u, max(elo_u, lo_u - max_extend))
            hi_u = max(hi_u, min(ehi_u, hi_u + max_extend))
            lo_v = min(lo_v, max(elo_v, lo_v - max_extend))
            hi_v = max(hi_v, min(ehi_v, hi_v + max_extend))
    len_u, len_v = float(hi_u - lo_u), float(hi_v - lo_v)
    center = c + u * (lo_u + hi_u) / 2.0 + v * (lo_v + hi_v) / 2.0
    if len_u >= len_v:
        return center, wrap_half_pi(math.atan2(u[1], u[0])), len_u, len_v
    return center, wrap_half_pi(math.atan2(v[1], v[0])), len_v, len_u


def completed_extents(xy: np.ndarray, center: np.ndarray, yaw: float,
                      trim_pct: float = 0.5) -> Tuple[float, float]:
    """
    Footprint extents along the yaw axis and across it, completed by symmetry.

    xy: Nx2 base XY (m) of the visible points; center: the cap center
    (directly above the center of a round or box-like object). Along each
    axis the extent is twice the larger trimmed distance from center, so a
    side hidden from an oblique camera (the far half of a fruit) is mirrored
    from the visible one. Returns (length_est along yaw, width_est across).
    In simulation this brings the width of round fruit from 11-17 mm too
    small to within 1-5 mm and leaves boxes unchanged.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64).ravel()[:2]
    out = []
    for ang in (yaw, yaw + math.pi / 2.0):
        q = (xy - c) @ np.array([math.cos(ang), math.sin(ang)])
        lo, hi = np.percentile(q, [trim_pct, 100.0 - trim_pct])
        out.append(2.0 * max(-float(lo), float(hi), 0.0))
    return out[0], out[1]


# --------------------------------------------------------------------------- splitting


def _find_peaks(dt: np.ndarray, r_min: float, min_dist_px: float) -> List[Tuple[float, int, int]]:
    """Distance-transform maxima (radius, x, y) at least min_dist_px apart."""
    k = max(3, int(round(min_dist_px / 2.0)) | 1)
    dil = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pm = (dt >= dil - 1e-4) & (dt >= r_min)
    n, lab = cv2.connectedComponents(pm.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    yy, xx = np.nonzero(lab)
    ids = lab[yy, xx]
    cnt = np.bincount(ids).astype(np.float64)
    cx = np.bincount(ids, xx)[ids] / cnt[ids]
    cy = np.bincount(ids, yy)[ids] / cnt[ids]
    dd = (xx - cx) ** 2 + (yy - cy) ** 2
    order = np.lexsort((dd, ids))
    sid = ids[order]
    first = order[np.r_[True, sid[1:] != sid[:-1]]]
    cands = sorted(((float(dt[yy[i], xx[i]]), int(xx[i]), int(yy[i])) for i in first),
                   key=lambda t: (-t[0], t[2], t[1]))
    kept: List[Tuple[float, int, int]] = []
    for r, x, y in cands:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_dist_px ** 2 for _, kx, ky in kept):
            kept.append((r, x, y))
    return kept


def _pair_stats(labels: np.ndarray, dt: np.ndarray, edges: np.ndarray) -> list:
    """Return [(i, j, saddle_dt, boundary_len, edge_fraction)] for adjacent labels i < j."""
    parts = []
    for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                   ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        a, b = labels[sa].ravel(), labels[sb].ravel()
        sel = (a != b) & (a > 0) & (b > 0)
        if np.any(sel):
            s = np.minimum(dt[sa], dt[sb]).ravel()[sel]
            e = (edges[sa] | edges[sb]).ravel()[sel]
            parts.append((a[sel], b[sel], s, e))
    if not parts:
        return []
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    e = np.concatenate([p[3] for p in parts]).astype(np.float64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    base = int(labels.max()) + 1
    uniq, inv = np.unique(lo.astype(np.int64) * base + hi, return_inverse=True)
    saddle = np.zeros(uniq.size)
    np.maximum.at(saddle, inv, s)
    length = np.bincount(inv).astype(np.float64)
    efrac = np.bincount(inv, e) / length
    return [(int(k // base), int(k % base), float(saddle[i]), float(length[i]),
             float(efrac[i])) for i, k in enumerate(uniq)]


def _relabel(labels: np.ndarray, parent: List[int]) -> np.ndarray:
    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i
    lut = np.array([root(i) for i in range(len(parent))], dtype=np.int32)
    out = lut[labels]
    ids = np.unique(out[out > 0])
    remap = np.zeros(len(parent), dtype=np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return remap[out]


def _fix_connectivity(labels: np.ndarray) -> np.ndarray:
    """Give pieces of a label that are cut off from its main piece to a neighbor."""
    out = labels.copy()
    for lab in range(1, int(labels.max()) + 1):
        reg = (labels == lab).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(reg, connectivity=4)
        if n <= 2:
            continue
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for j in range(1, n):
            if j == main:
                continue
            piece = cc == j
            ring = cv2.dilate(piece.astype(np.uint8), _K3).astype(bool) & ~piece
            nb = out[ring]
            nb = nb[(nb > 0) & (nb != lab)]
            if nb.size:
                out[piece] = int(np.bincount(nb).argmax())
    return out


def _grow_markers(peaks: List[Tuple[float, int, int]], core: np.ndarray,
                  comp: np.ndarray) -> np.ndarray:
    """
    Grow one region per marker, first inside core (depth edges are barriers).

    Marker i starts growing (r_max - r_i) steps late, so two touching shapes
    meet near their tangent line. Pixels that the core growth cannot reach
    (edge pixels, isolated bits) are then filled by unconstrained growth
    inside comp. Ties go to the higher label. Returns int32 labels.
    """
    lab = np.zeros(comp.shape, dtype=np.uint16)
    radii = np.array([0.0] + [p[0] for p in peaks])
    delays = np.round(radii.max() - radii).astype(np.int64)
    delays[0] = 0
    for i, (_, x, y) in enumerate(peaks):
        lab[y, x] = i + 1
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    free = core & (lab == 0)
    t, t_max, d_max = 0, comp.shape[0] + comp.shape[1] + int(delays.max()), int(delays.max())
    while t < t_max:
        active = np.where(delays[lab] <= t, lab, 0).astype(np.uint16)
        grown = cv2.dilate(active, cross if t % 2 else _K3)
        new = free & (grown > 0)
        if not np.any(new):
            if t >= d_max:
                break
        else:
            lab[new] = grown[new]
            free &= ~new
        t += 1
    free = comp & (lab == 0)
    while np.any(free):
        grown = cv2.dilate(lab, _K3)
        new = free & (grown > 0)
        if not np.any(new):
            break
        lab[new] = grown[new]
        free &= ~new
    lab[~comp] = 0
    return lab.astype(np.int32)


SPLIT_EDGE_FRACTION = 0.3   # boundary share on height barriers that always keeps a split
SPLIT_JUMP_MAX_M = 0.025    # noise-scaled barrier step is capped here
_MAX_GRID_CELLS = 600       # per side of the top-down split grid


def split_jump(params: 'SegmentParams', plane_rms: Optional[float] = None) -> float:
    """
    Return the height step (m) between split grid cells that counts as a barrier.

    The step is split_height_jump_m, raised to split_jump_noise_k * sqrt(2)
    * plane_rms (the table fit residual of the same frame; the difference of
    two noisy cells has sqrt(2) times their noise) but not above
    SPLIT_JUMP_MAX_M. Depth noise grows with distance, so a fixed step cuts
    long objects into pieces on a far or noisy camera. k is large (6)
    because cell maxima on curved surfaces add slope to the noise: in the
    simulation with the camera 0.35 m farther (table rms 2 mm) k = 4 left 11
    extra fragments on 96 L2 objects (47 with the fixed 8 mm), k = 6 left 3
    (2 of them grape bunches cut at a neck, which the step does not change). A
    missing or non-finite rms, or k = 0, leaves split_height_jump_m.
    """
    jump = float(params.split_height_jump_m)
    k = float(params.split_jump_noise_k)
    try:
        rms = float(plane_rms) if plane_rms is not None else float('nan')
    except (TypeError, ValueError):
        rms = float('nan')
    if k > 0.0 and math.isfinite(rms) and rms > 0.0:
        jump = max(jump, min(k * math.sqrt(2.0) * rms, SPLIT_JUMP_MAX_M))
    return jump


def _height_grid(xy: np.ndarray, h: np.ndarray, cell: float) -> Tuple[np.ndarray, ...]:
    """Rasterize base-XY points with heights into a top-down max-height grid."""
    lo = xy.min(axis=0)
    span = float(np.max(xy.max(axis=0) - lo))
    cell = max(cell, span / (_MAX_GRID_CELLS - 4))
    ij = np.floor((xy - lo) / cell).astype(np.int64) + 2
    nx, ny = int(ij[:, 0].max()) + 3, int(ij[:, 1].max()) + 3
    hmax = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(hmax, (ij[:, 1], ij[:, 0]), h.astype(np.float64))
    return hmax, ij, cell


def _barriers(hmax: np.ndarray, jump: float) -> np.ndarray:
    """Cells whose max height differs from an occupied 4-neighbor by more than jump."""
    hm = np.where(np.isfinite(hmax), hmax, np.nan)
    out = np.zeros(hm.shape, dtype=bool)
    with np.errstate(invalid='ignore'):
        dx = np.abs(hm[:, 1:] - hm[:, :-1]) > jump
        dy = np.abs(hm[1:, :] - hm[:-1, :]) > jump
    out[:, 1:] |= dx
    out[:, :-1] |= dx
    out[1:, :] |= dy
    out[:-1, :] |= dy
    return out


def _split_grid(occ: np.ndarray, barrier: np.ndarray, min_peak_dist: float, neck_ratio: float,
                min_region: int) -> np.ndarray:
    """Split an occupancy grid into touching objects (see split_component)."""
    labels = occ.astype(np.int32)
    if min_peak_dist < 4.0 or not np.any(occ):
        return labels
    core = occ & ~barrier
    dt_core = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    r_min = max(2.0, 0.25 * float(min_peak_dist))
    if float(dt_core.max()) < r_min:
        return labels
    peaks = _find_peaks(dt_core, r_min, float(min_peak_dist))
    if len(peaks) < 2:
        return labels
    labels = _fix_connectivity(_grow_markers(peaks, core, occ))
    dt_full = cv2.distanceTransform(occ.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = np.zeros(len(peaks) + 1)
    for i, (_, x, y) in enumerate(peaks):
        radius[i + 1] = dt_full[y, x]
    parent = list(range(len(peaks) + 1))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j, saddle, _length, efrac in sorted(_pair_stats(labels, dt_full, barrier),
                                               key=lambda t: -t[2]):
        if efrac >= SPLIT_EDGE_FRACTION:
            continue
        if saddle >= neck_ratio * min(radius[i], radius[j]):
            ri, rj = root(i), root(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    labels = _relabel(labels, parent)

    for _ in range(int(labels.max())):
        sizes = np.bincount(labels.ravel())
        small = [k for k in range(1, sizes.size) if 0 < sizes[k] < min_region]
        if not small or sizes.size <= 2:
            break
        k = min(small, key=lambda q: (sizes[q], q))
        best, best_len = 0, 0.0
        for i, j, _s, length, _e in _pair_stats(labels, dt_full, barrier):
            if k in (i, j) and length > best_len:
                best, best_len = (j if i == k else i), length
        if best == 0:
            break
        par = list(range(sizes.size))
        par[k] = best
        labels = _relabel(labels, par)
    return labels


def _fill_from_labels(labels: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Give unlabeled comp pixels the label of the nearest labeled pixel (8-growth)."""
    lab = labels.astype(np.uint16)
    free = comp & (lab == 0)
    while np.any(free):
        grown = cv2.dilate(lab, _K3)
        new = free & (grown > 0)
        if not np.any(new):
            break
        lab[new] = grown[new]
        free &= ~new
    lab[~comp] = 0
    return lab.astype(np.int32)


def split_component(comp: np.ndarray, points: np.ndarray, hmap: np.ndarray,
                    use: np.ndarray, params: 'SegmentParams',
                    jump: Optional[float] = None) -> np.ndarray:
    """
    Split one image component into touching objects using a top-down view.

    comp: bool crop of the component; points: crop HxWx3 base points (m);
    hmap: crop heights above the table (m); use: bool crop of the reliable
    above-table pixels (no depth-edge / flying pixels) that are rasterized.
    The base-XY footprint is rasterized on a split_cell_m grid keeping the
    maximum height per cell, which makes the result independent of the camera
    view. Cells whose height differs from a neighbor by more than jump
    (default split_height_jump_m; segment_objects passes split_jump()) are
    barriers. Markers are distance-transform maxima
    of the footprint without barriers, at least split_min_peak_dist_m apart
    with an inscribed radius of at least a quarter of that; regions grow from
    them (barriers block, larger shapes start first), so cuts land on
    tangent lines of touching shapes. Two regions merge again unless at
    least SPLIT_EDGE_FRACTION of their shared boundary is a barrier or their
    neck is below split_neck_ratio * the smaller inscribed radius, so long
    single objects stay whole. Regions under min_area_m2 join their longest
    neighbor. Image pixels take the label of their cell; the rest (edges,
    holes) take the nearest labeled pixel. Returns int32 labels (0 = none).
    """
    p = params
    labels = comp.astype(np.int32)
    use = use & comp
    if np.count_nonzero(use) < max(p.min_points, 10):
        return labels
    xy = points[..., :2][use].astype(np.float64)
    hmax, ij, cell = _height_grid(xy, hmap[use], p.split_cell_m)
    occ = np.isfinite(hmax)
    occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, _K3).astype(bool) | occ
    barrier = _barriers(hmax, p.split_height_jump_m if jump is None else float(jump))
    grid = _split_grid(occ, barrier, p.split_min_peak_dist_m / cell, p.split_neck_ratio,
                       max(4, int(p.min_area_m2 / (cell * cell))))
    if int(grid.max()) < 2:
        return labels
    out = np.zeros(comp.shape, dtype=np.int32)
    out[use] = grid[ij[:, 1], ij[:, 0]]
    out = _fill_from_labels(out, comp)
    return _fix_connectivity(out)


# --------------------------------------------------------------------------- per instance


def cap_band(top_z: float, height: float, plane_rms: Optional[float] = None
             ) -> Tuple[float, float]:
    """
    Return the (low, high) base z (m) of an instance's 'cap' (topmost) points.

    low = top_z - max(2 mm, CAP_DEPTH_FRACTION * height). high = top_z +
    max(CAP_ABOVE_MIN_M, the band depth, CAP_ABOVE_RMS_K * plane_rms): top_z
    is the top_percentile height, so real top points lie only noise (or, on
    a round top, about the band depth) above it, while depth outliers nearer
    the camera lie higher and are displaced toward the camera in XY.
    """
    band = max(0.002, CAP_DEPTH_FRACTION * float(height)) if math.isfinite(height) else 0.002
    try:
        rms = float(plane_rms) if plane_rms is not None else 0.0
    except (TypeError, ValueError):
        rms = 0.0
    rms = rms if math.isfinite(rms) and rms > 0.0 else 0.0
    return top_z - band, top_z + max(CAP_ABOVE_MIN_M, band, CAP_ABOVE_RMS_K * rms)


def cap_points(pts: np.ndarray, top_z: float, height: float,
               plane_rms: Optional[float] = None) -> np.ndarray:
    """Return the rows of pts (Nx3 base points, m) whose z lies in cap_band()."""
    lo, hi = cap_band(top_z, height, plane_rms)
    zz = pts[:, 2]
    return pts[(zz >= lo) & (zz <= hi)]


def occlusion_ratio(region: np.ndarray, z: np.ndarray, valid: np.ndarray, rho: np.ndarray,
                    abs_thresh: float, rel_thresh: float, rho_margin: float = 0.002) -> float:
    """
    Fraction of the outline of region next to surfaces that may hide part of it.

    region: bool crop; z: depth crop (m); valid: bool crop; rho: crop of the
    horizontal (base XY) distance of each pixel's point from the camera (m).
    A ring pixel occludes when it is nearer to the camera than every adjacent
    region pixel by more than abs_thresh + rel_thresh * depth and also
    horizontally closer to the camera by more than rho_margin; a taller
    neighbor beyond the region (seen from above) is therefore not counted.
    """
    ring = cv2.dilate(region.astype(np.uint8), _K3).astype(bool) & ~region
    n_ring = int(np.count_nonzero(ring))
    if n_ring == 0:
        return 0.0
    inner = region & valid
    adj = cv2.erode(np.where(inner, z, _BIG).astype(np.float32), _K3)
    adj_rho = cv2.erode(np.where(inner, rho, _BIG).astype(np.float32), _K3)
    thr = np.float32(abs_thresh) + np.float32(rel_thresh) * adj
    zr = np.where(valid, z, _BIG).astype(np.float32)
    rr = np.where(valid, rho, _BIG).astype(np.float32)
    occ = (ring & valid & (adj < _BIG / 2) & (zr < adj - thr)
           & (rr < adj_rho - np.float32(rho_margin)))
    return float(np.count_nonzero(occ)) / n_ring


@dataclass
class _Ctx:
    """Full-image arrays shared by all instances of one frame."""

    points: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    hmap: np.ndarray
    cand: np.ndarray
    outside: np.ndarray
    in_zone: np.ndarray
    tall: np.ndarray
    edges: np.ndarray
    rho: np.ndarray
    plane: TablePlane
    intrinsics: CameraIntrinsics
    T_base_camera: np.ndarray
    params: SegmentParams


def _crop(a: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def subsample_points(pts: np.ndarray, max_points: int = MAX_INSTANCE_POINTS) -> np.ndarray:
    """Return at most max_points rows of pts (uniform stride) as Nx3 float32."""
    pts = np.asarray(pts).reshape(-1, 3)
    step = max(1, int(math.ceil(pts.shape[0] / float(max(1, int(max_points))))))
    return np.ascontiguousarray(pts[::step], dtype=np.float32)


def _make_detection(ctx: _Ctx, region: np.ndarray, box: Tuple[int, int, int, int],
                    comp_id: int) -> Optional[Detection]:
    p = ctx.params
    H, W = ctx.valid.shape
    x0, y0, _x1, _y1 = box
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return None
    bbox = (int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1),
            int(ys.max() + y0 + 1))
    full = np.zeros((H, W), dtype=bool)
    full[y0:y0 + region.shape[0], x0:x0 + region.shape[1]] = region

    cand = _crop(ctx.cand, box)
    pts = _crop(ctx.points, box)
    z = _crop(ctx.z, box)
    valid = _crop(ctx.valid, box)
    edges = _crop(ctx.edges, box)
    inst = region & cand
    n_points = int(np.count_nonzero(inst))
    reasons: List[str] = []
    if n_points < p.min_points:
        reasons.append('few_points')
    if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= W or bbox[3] >= H:
        reasons.append('touches_image_border')
    near = cv2.dilate(region.astype(np.uint8), _K3, iterations=2).astype(bool)
    touch_min = max(8, int(0.02 * max(n_points, 1)))
    if np.count_nonzero(near & _crop(ctx.outside, box)) >= touch_min:
        reasons.append('touches_roi_border')
    if np.count_nonzero(near & _crop(ctx.in_zone, box)) >= touch_min:
        reasons.append('touches_zone')
    if np.count_nonzero(near & _crop(ctx.tall, box)) >= touch_min:
        reasons.append('too_tall')
    occ = occlusion_ratio(region, z, valid, _crop(ctx.rho, box), p.edge_jump_abs_m,
                          p.edge_jump_rel)
    if occ > p.max_occlusion_ratio:
        reasons.append('occluded')

    extra = {'component': int(comp_id), 'occlusion_ratio': round(occ, 3)}
    det = Detection(label='unknown', score=0.0, score_kind='none', bbox=bbox,
                    center_px=(float(xs.mean() + x0), float(ys.mean() + y0)),
                    mask=full, n_points=n_points, extra=extra)
    hm = _crop(ctx.hmap, box)
    bx0, by0, bx1, by1 = bbox
    sub = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
    extra['_height_crop'] = np.where(region[sub] & valid[sub], hm[sub],
                                     np.nan).astype(np.float32)

    if p.edge_erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * p.edge_erode_px + 1, 2 * p.edge_erode_px + 1))
        eroded = cv2.erode(region.astype(np.uint8), k).astype(bool)
    else:
        eroded = region
    # Depth-edge pixels (flying pixels, and isolated depth outliers nearer the
    # camera, which also lie above the object top) never enter the geometry.
    clean = inst & ~edges
    pts_all = pts[inst].astype(np.float64)
    pts_geom = pts[clean & eroded].astype(np.float64)
    pts_ext = pts[clean].astype(np.float64)
    geom_eroded = p.edge_erode_px > 0
    need = max(5, p.min_points // 4)
    if pts_geom.shape[0] < need:
        pts_geom, geom_eroded = (pts_ext if pts_ext.shape[0] >= need else pts_all), False
    extra['geom_eroded'] = geom_eroded
    extra['n_geom_points'] = int(pts_geom.shape[0])
    if pts_geom.shape[0] < 3:
        if 'few_points' not in reasons:
            reasons.append('few_points')
        _finish(det, reasons)
        return det

    zc = z[inst]
    gsd = float(np.median(zc)) / float(ctx.intrinsics.fx)
    cam = np.asarray(ctx.T_base_camera, dtype=np.float64)[:3, 3]
    ray = pts_geom.mean(axis=0) - cam
    sin_elev = abs(float(ray[2])) / max(float(np.linalg.norm(ray)), 1e-9)
    max_ext = (p.edge_erode_px + 1) * gsd / max(sin_elev, 0.3) if geom_eroded else 0.0
    ext_xy = pts_ext[:, :2] if pts_ext.shape[0] >= 3 else None
    cxy, yaw, length, width = rect_geometry(pts_geom[:, :2], ext_xy, max_ext,
                                            p.extent_trim_pct)
    top_z = float(np.percentile(pts_geom[:, 2], p.top_percentile))
    table_z = ctx.plane.z_at(float(cxy[0]), float(cxy[1]))
    det.center_base = np.array([cxy[0], cxy[1], top_z], dtype=np.float64)
    det.top_z = top_z
    det.height = top_z - table_z
    det.yaw = yaw
    det.length = length
    det.width = width
    uv = project_points(det.center_base, ctx.T_base_camera, ctx.intrinsics)[0]
    if np.all(np.isfinite(uv)):
        det.center_px = (float(uv[0]), float(uv[1]))
    area = length * width
    extra['table_z'] = table_z
    extra['area_m2'] = area
    extra['max_extend_m'] = max_ext
    fp = pts_ext if pts_ext.shape[0] >= 3 else pts_geom
    extra['_points_xy'] = fp[:, :2].astype(np.float32)
    extra['_points_base'] = subsample_points(fp)
    # Center of the topmost points: on round objects seen obliquely it is a
    # better grasp point than the footprint center, whose far side is hidden.
    # All band points are used: near an oblique silhouette the top of a round
    # object lies within the eroded border, and the far top edge of a block
    # is a depth-edge pixel (dropping it moves the cap 1-2 mm toward the
    # camera). The band is bounded from above (cap_band), which drops depth
    # outliers nearer the camera; flying pixels are lower.
    cap = cap_points(pts_all, top_z, float(det.height), ctx.plane.rms)
    extra['length_est'], extra['width_est'] = length, width
    if cap.shape[0] >= 10:
        cc, _, _, _ = rect_geometry(cap[:, :2], None, 0.0, p.extent_trim_pct)
        extra['cap_center_xy'] = [float(cc[0]), float(cc[1])]
        l_est, w_est = completed_extents(fp[:, :2], cc, yaw, p.extent_trim_pct)
        extra['length_est'], extra['width_est'] = max(length, l_est), max(width, w_est)
    if area < p.min_area_m2:
        reasons.append('too_small')
    if area > p.max_area_m2:
        reasons.append('too_large')
    _finish(det, reasons)
    return det


def _finish(det: Detection, reasons: List[str]) -> None:
    ordered = [r for r in REASON_ORDER if r in reasons]
    det.extra['reject_reasons'] = ordered
    det.valid = not ordered
    det.reject_reason = ordered[0] if ordered else ''


def mark_invalid(det: Detection, reason: str) -> None:
    """Add a reject reason (see REASON_ORDER) to det and update valid / reject_reason."""
    reasons = list(det.extra.get('reject_reasons', []))
    if reason not in reasons:
        reasons.append(reason)
    _finish(det, reasons)


def _pad(params: SegmentParams) -> int:
    """Crop padding (px) around an instance, enough for the border and occlusion tests."""
    return max(4, params.edge_erode_px + 3)


# --------------------------------------------------------------------------- re-splitting


def split_instance(ctx: _Ctx, det: Detection, labels: np.ndarray, how: str) -> List[Detection]:
    """
    Replace det by one detection per label of labels (its bbox crop, 0 = none).

    Each part gets the full geometry and every reject check of a normal
    instance (same code path), extra['split'] = how and extra['split_parts'].
    Returns [det] unchanged when fewer than two non-empty parts result.
    """
    x0, y0, x1, y1 = det.bbox
    labels = np.asarray(labels)
    if labels.shape != (y1 - y0, x1 - x0):
        raise ValueError('split_instance: labels must have the bbox crop shape')
    H, W = ctx.valid.shape
    pad = _pad(ctx.params)
    comp_id = int(det.extra.get('component', -1))
    parts: List[Detection] = []
    for k in range(1, int(labels.max()) + 1 if labels.size else 1):
        region = labels == k
        if not np.any(region):
            continue
        ys, xs = np.nonzero(region)
        box = (max(0, x0 + int(xs.min()) - pad), max(0, y0 + int(ys.min()) - pad),
               min(W, x0 + int(xs.max()) + 1 + pad), min(H, y0 + int(ys.max()) + 1 + pad))
        crop = np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
        crop[y0 - box[1] + ys, x0 - box[0] + xs] = True
        part = _make_detection(ctx, crop, box, comp_id)
        if part is not None:
            parts.append(part)
    if len(parts) < 2:
        return [det]
    for part in parts:
        part.extra['split'] = how
        part.extra['split_parts'] = len(parts)
    return parts


def split_by_classes(mask: np.ndarray, classes: np.ndarray, consider: Optional[np.ndarray] = None,
                     min_fraction: float = 0.25, min_seed_px: int = 15
                     ) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    Split one instance into parts of uniform pixel class (for example block colors).

    mask: HxW bool crop of the instance; classes: HxW int class index per
    pixel (-1 = none); consider: bool crop of the pixels whose classes count
    (default mask, the pipeline passes the upper pixels). When at least two
    classes each cover min_fraction of the considered pixels, every connected
    component (8-connected, after a 3x3 opening) of such a class with at least
    min_seed_px pixels seeds a part. All other mask pixels (other or no class,
    lower pixels, small components) join the nearest seed by growth inside
    the mask. Returns (int32 labels 0..n, class index of each part) or None.
    """
    mask = np.asarray(mask, dtype=bool)
    cons = mask if consider is None else (np.asarray(consider, dtype=bool) & mask)
    n = int(np.count_nonzero(cons))
    if n == 0:
        return None
    cls = np.where(cons, classes, -1)
    ids, counts = np.unique(cls[cls >= 0], return_counts=True)
    major = [int(k) for k, c in zip(ids, counts) if c >= min_fraction * n]
    if len(major) < 2:
        return None
    labels = np.zeros(mask.shape, dtype=np.int32)
    part_class: List[int] = []
    for k in major:
        mk = cv2.morphologyEx((cls == k).astype(np.uint8), cv2.MORPH_OPEN, _K3)
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        for j in range(1, n_cc):
            if stats[j, cv2.CC_STAT_AREA] >= min_seed_px:
                part_class.append(k)
                labels[cc == j] = len(part_class)
    if len(part_class) < 2:
        return None
    return _fill_from_labels(labels, mask), part_class


# Same-color merges of L1 cubes (find_block_cuts). Sizes are relative to the block size.
BLOCK_SIDE_RANGE = (0.75, 1.4)  # accepted part side lengths
BLOCK_SQUARE_TOL = 0.2          # accepted |side_u - side_v| / max side of every part
BLOCK_MAX_PARTS = 4
BLOCK_BIN_M = 0.002             # cut position grid along the major axis
BLOCK_SNAP_M = 0.010            # find_block_cuts: search a height step this far around a cut
BLOCK_STEP_M = 0.003            # top-height step (m) that marks the boundary of two blocks


def find_block_cuts(points_xy: np.ndarray, yaw: float, block_size: float,
                    trim_pct: float = 0.5, z: Optional[np.ndarray] = None
                    ) -> Optional[Tuple[List[float], float]]:
    """
    Find where to cut a footprint into touching square blocks along its major axis.

    points_xy: Nx2 base XY (m) of the instance's reliable points; yaw: major
    axis angle (rad). The points are clipped to their trimmed extents and
    binned every BLOCK_BIN_M along the axis. For 2..BLOCK_MAX_PARTS parts,
    dynamic programming chooses the cuts that minimize the worst part's
    squareness error |side_u - side_v| / max side, where side_v is the
    part's extent across the axis; parts with a side outside BLOCK_SIDE_RANGE
    x block_size are not allowed. Works for blocks of different sizes and
    with a lateral offset, because each block keeps its own cross extent.
    z: optional (N,) base z of the points. A taller block hides a strip of a
    lower one behind it, which biases the squareness cut toward the taller
    block, so each cut moves to the largest step of the per-bin top height
    within BLOCK_SNAP_M when that step is at least BLOCK_STEP_M. Returns (cut
    positions along the axis relative to the projection origin, worst
    error) when the best split has error <= BLOCK_SQUARE_TOL, else None.
    """
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    zz = None if z is None else np.asarray(z, dtype=np.float64).ravel()
    fin = np.all(np.isfinite(xy), axis=1)
    if zz is not None:
        if zz.shape[0] != xy.shape[0]:
            raise ValueError('find_block_cuts: z must have one value per point')
        fin &= np.isfinite(zz)
        zz = zz[fin]
    xy = xy[fin]
    if xy.shape[0] < 20 or not (block_size > 0.0):
        return None
    ax_u = np.array([math.cos(yaw), math.sin(yaw)])
    ax_v = np.array([-ax_u[1], ax_u[0]])
    u, v = xy @ ax_u, xy @ ax_v
    lo_u, hi_u = np.percentile(u, [trim_pct, 100.0 - trim_pct])
    lo_v, hi_v = np.percentile(v, [trim_pct, 100.0 - trim_pct])
    keep = (u >= lo_u) & (u <= hi_u) & (v >= lo_v) & (v <= hi_v)
    u, v = u[keep], v[keep]
    n_bins = int(math.ceil((hi_u - lo_u) / BLOCK_BIN_M))
    if n_bins < 4 or n_bins > 400:
        return None
    edges = lo_u + BLOCK_BIN_M * np.arange(n_bins + 1)
    edges[-1] = hi_u
    b = np.clip(((u - lo_u) / BLOCK_BIN_M).astype(np.int64), 0, n_bins - 1)
    bmin = np.full(n_bins, np.inf)
    bmax = np.full(n_bins, -np.inf)
    np.minimum.at(bmin, b, v)
    np.maximum.at(bmax, b, v)
    smin, smax = BLOCK_SIDE_RANGE[0] * block_size, BLOCK_SIDE_RANGE[1] * block_size
    # cost[i, j]: squareness error of the part covering bins i .. j - 1.
    cost = np.full((n_bins + 1, n_bins + 1), np.inf)
    for i in range(n_bins):
        with np.errstate(invalid='ignore'):
            ext_v = np.maximum.accumulate(bmax[i:]) - np.minimum.accumulate(bmin[i:])
        ext_u = edges[i + 1:] - edges[i]
        big = np.maximum(ext_u, ext_v)
        with np.errstate(invalid='ignore', divide='ignore'):
            c = np.abs(ext_u - ext_v) / big
        ok = (np.isfinite(ext_v) & (ext_u >= smin) & (ext_u <= smax) & (ext_v >= smin)
              & (ext_v <= smax))
        cost[i, i + 1:] = np.where(ok, c, np.inf)
    best = cost[0].copy()                     # best[j]: worst error of parts covering 0 .. j
    back: List[np.ndarray] = []
    result: Optional[Tuple[List[int], float]] = None
    for _n_parts in range(2, BLOCK_MAX_PARTS + 1):
        cand = np.maximum(best[:, None], cost)   # [i, j]: previous parts end at i
        arg = np.argmin(cand, axis=0)
        best = cand[arg, np.arange(n_bins + 1)]
        back.append(arg)
        if not math.isfinite(float(best[n_bins])):
            continue
        err = float(best[n_bins])
        if result is None or err < result[1] - 1e-9:
            cut_bins, j = [], n_bins
            for a in reversed(back):
                j = int(a[j])
                cut_bins.append(j)
            result = (sorted(cut_bins), err)
    if result is None or result[1] > BLOCK_SQUARE_TOL:
        return None
    cut_bins = result[0]
    if zz is not None:
        top = np.full(n_bins, -np.inf)
        np.maximum.at(top, b, zz[keep])
        cut_bins = [_snap_to_step(top, k) for k in cut_bins]
    return [float(edges[k]) for k in cut_bins], result[1]


def _snap_to_step(top: np.ndarray, k: int) -> int:
    """
    Move bin boundary k to a top-height step of the blocks within BLOCK_SNAP_M.

    top: per-bin maximum z (-inf for empty bins). A two-level step function
    is fitted (least squares) to the non-empty bins within BLOCK_SNAP_M of k
    plus 3 more on each side, with the change point within BLOCK_SNAP_M of
    k and at least 3 bins on each side of it. The cut moves
    to the first filled bin of the taller side (a strip of the lower block
    hidden behind the taller one has no points) when the levels differ by at
    least BLOCK_STEP_M and 4 times the residual noise; else k is returned.
    """
    reach = int(round(BLOCK_SNAP_M / BLOCK_BIN_M))
    filled = np.nonzero(np.isfinite(top))[0]
    i0 = max(0, int(np.searchsorted(filled, k - reach)) - 3)
    i1 = int(np.searchsorted(filled, k + reach, side='right')) + 3
    idx = filled[i0:i1]
    if idx.size < 6:
        return k
    vals = top[idx]
    best = None
    for p in range(3, idx.size - 2):
        if not (k - reach <= int(idx[p]) and int(idx[p - 1]) + 1 <= k + reach):
            continue
        left, right = vals[:p], vals[p:]
        sse = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        if best is None or sse < best[0]:
            best = (sse, p, float(right.mean() - left.mean()))
    if best is None:
        return k
    sse, p, step = best
    noise = math.sqrt(sse / max(idx.size - 2, 1))
    if abs(step) < max(BLOCK_STEP_M, 4.0 * noise):
        return k
    return int(idx[p]) if step > 0 else int(idx[p - 1]) + 1


def labels_from_cuts(ctx: _Ctx, det: Detection, yaw: float,
                     cuts: Sequence[float]) -> np.ndarray:
    """
    Label det's bbox crop by the base-XY position of each pixel along the yaw axis.

    Part k + 1 holds the pixels whose point projects between cuts[k - 1] and
    cuts[k] (positions as returned by find_block_cuts); mask pixels without
    a valid point join the nearest labeled pixel. Returns int32 labels.
    """
    x0, y0, x1, y1 = det.bbox
    mask = np.asarray(det.mask[y0:y1, x0:x1], dtype=bool)
    pts = ctx.points[y0:y1, x0:x1]
    ax = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    labels = np.zeros(mask.shape, dtype=np.int32)
    ok = mask & np.isfinite(pts[..., 0])
    u = pts[..., :2][ok].astype(np.float64) @ ax
    labels[ok] = 1 + np.searchsorted(np.asarray(cuts, dtype=np.float64), u)
    return _fill_from_labels(labels, mask)


def segment_objects(points_base: np.ndarray, depth: np.ndarray, plane: TablePlane,
                    intrinsics: CameraIntrinsics, T_base_camera: np.ndarray,
                    params: SegmentParams, pick_polygon: Optional[np.ndarray] = None,
                    zones: Any = None) -> Segmentation:
    """
    Find object instances standing on the table.

    points_base: HxWx3 float32 base-frame points (m), NaN where depth is invalid.
    depth: HxW float depth (m) of the same frame. plane: table plane in the
    base frame. pick_polygon: Mx2 base XY (m) or None for no restriction.
    zones: dict / iterable of core Zone; with params.exclude_zones their area
    grown by zone_exclude_margin is ignored. Returns detections (label
    'unknown', geometry filled, invalid ones carry reject_reason), the height
    map and the object mask.
    """
    if not isinstance(params, SegmentParams):
        params = SegmentParams.from_cfg(params)
    p = params
    H, W = points_base.shape[:2]
    valid = np.isfinite(points_base[..., 2])
    z = np.where(valid, depth, np.nan).astype(np.float32)
    if p.despike:
        spikes = depth_spikes(z, valid, p.edge_jump_abs_m, p.edge_jump_rel)
        valid &= ~spikes
        z[spikes] = np.nan
    hmap = height_above_plane(points_base, plane)
    with np.errstate(invalid='ignore'):
        band = valid & (hmap > p.min_height) & (hmap < p.max_height)
        tall = valid & (hmap >= p.max_height)
    poly = parse_polygon(pick_polygon) if pick_polygon is not None else None
    ys, xs = np.nonzero(band | tall)
    xy = points_base[ys, xs, :2]
    in_poly = (points_in_polygon(xy, poly) if poly is not None
               else np.ones(ys.size, dtype=bool))
    if p.exclude_zones and zones:
        in_zone_pts = points_in_zones(xy, zones, margin=p.zone_exclude_margin)
    else:
        in_zone_pts = np.zeros(ys.size, dtype=bool)
    keep = in_poly & ~in_zone_pts
    is_band = band[ys, xs]
    cand = np.zeros((H, W), dtype=bool)
    cand[ys[keep & is_band], xs[keep & is_band]] = True
    tall_roi = np.zeros((H, W), dtype=bool)
    tall_roi[ys[keep & ~is_band], xs[keep & ~is_band]] = True
    out_b = ~in_poly & is_band
    outside = np.zeros((H, W), dtype=bool)
    outside[ys[out_b], xs[out_b]] = True
    zone_b = in_poly & in_zone_pts & is_band
    in_zone = np.zeros((H, W), dtype=bool)
    in_zone[ys[zone_b], xs[zone_b]] = True
    outside = _morph(outside, cv2.MORPH_OPEN, p.morph_open_px)
    in_zone = _morph(in_zone, cv2.MORPH_OPEN, p.morph_open_px)
    tall = _morph(tall, cv2.MORPH_OPEN, p.morph_open_px)
    tall_roi = _morph(tall_roi, cv2.MORPH_OPEN, p.morph_open_px)

    # Tall pixels inside the ROI join the components so a too-tall object (or
    # the arm) is reported as an invalid detection instead of vanishing.
    mask = _morph(cand, cv2.MORPH_OPEN, p.morph_open_px) | tall_roi
    mask = _morph(mask, cv2.MORPH_CLOSE, p.morph_close_px)
    mask &= ~(outside | in_zone)
    edges = depth_edges(z, valid, p.edge_jump_abs_m, p.edge_jump_rel)
    T = np.asarray(T_base_camera, dtype=np.float64)
    rho = np.hypot(points_base[..., 0] - np.float32(T[0, 3]),
                   points_base[..., 1] - np.float32(T[1, 3])).astype(np.float32)
    ctx = _Ctx(points_base, z, valid, hmap, cand, outside, in_zone, tall, edges, rho, plane,
               intrinsics, T, p)

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                              connectivity=8)
    order = sorted(range(1, n_cc), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
    warnings: List[str] = []
    dets: List[Detection] = []
    n_dropped = 0
    n_done = 0
    pad = _pad(p)
    jump = split_jump(p, getattr(plane, 'rms', None))
    if jump > p.split_height_jump_m:
        LOG.debug('split barrier step %.1f mm (table rms %.2f mm)', jump * 1000.0,
                  float(plane.rms) * 1000.0)
    for i in order:
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < p.min_blob_px:
            n_dropped += 1
            continue
        if n_done >= p.max_instances:
            warnings.append('too_many_blobs')
            break
        n_done += 1
        box = (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad))
        comp = _crop(labels, box) == i
        regions = comp.astype(np.int32)
        if p.split_touching:
            use = comp & _crop(cand, box) & ~_crop(edges, box)
            regions = split_component(comp, _crop(points_base, box), _crop(hmap, box), use, p,
                                      jump=jump)
        for r in range(1, int(regions.max()) + 1):
            det = _make_detection(ctx, regions == r, box, i)
            if det is not None:
                dets.append(det)
    if n_dropped:
        LOG.debug('dropped %d blobs smaller than %d px', n_dropped, p.min_blob_px)
    return Segmentation(dets, hmap, mask, warnings, n_dropped, ctx, jump)

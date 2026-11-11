"""
L1 detection: 50 mm red / green / yellow cubes and the 200 mm target cards.

The cubes and the cards share the same three colors, so color alone cannot
tell them apart. Height above the table does: a cube top is ~50 mm up, a card
is flat. Pipeline per frame:

1. Depth -> points in the base frame (T_base_camera from the camera config).
2. Fit the table plane (RANSAC, normal near base z), or use a given table z.
3. Per color: HSV mask AND height band -> cube candidates; HSV mask AND flat
   -> card candidates.
4. Per connected component: rectangle of the top-face points in base XY gives
   the center, yaw and edge length; touching same-color cubes are split by
   k-means when the footprint is too large for one cube.

Without depth, each color blob is intersected with the plane at
``table_z + cube_size`` (cubes) or ``table_z`` (cards) and classified by its
metric size; this needs ``table_z`` in the config.

All coordinates are in the robot base frame, meters / radians.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import Intrinsics
from .transforms import wrap_angle

LOG = logging.getLogger('piper_vision.l1')

# OpenCV HSV scale: H 0-180, S / V 0-255. A range with hmin > hmax wraps (red).
DEFAULT_COLORS: Dict[str, List[List[List[int]]]] = {
    'red': [[[0, 90, 60], [8, 255, 255]], [[170, 90, 60], [180, 255, 255]]],
    'yellow': [[[18, 90, 80], [34, 255, 255]]],
    'green': [[[35, 60, 40], [85, 255, 255]]],
}


@dataclass
class L1Params:
    """Tunable values, loaded from config/l1.yaml (see that file for meanings)."""

    colors: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    cube_size: float = 0.050
    size_tol: float = 0.015
    min_height: float = 0.020
    max_height: float = 0.090
    top_band: float = 0.012
    card_size: float = 0.200
    card_size_tol: float = 0.050
    card_max_height: float = 0.012
    min_pixels: int = 80
    min_color_fraction: float = 0.5
    table_z: Optional[float] = None
    fit_table: bool = True
    table_max_tilt_deg: float = 15.0
    table_inlier_m: float = 0.006
    depth_min: float = 0.15
    depth_max: float = 2.5
    roi_polygon: Optional[List[List[float]]] = None   # base XY; filters object centers
    morph_kernel: int = 5
    seed: int = 0

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> 'L1Params':
        """Build params from a dict; unknown keys raise so typos are caught."""
        d = dict(d or {})
        p = L1Params()
        unknown = sorted(set(d) - set(p.__dict__))
        if unknown:
            raise ValueError('unknown L1 parameter(s): %s' % ', '.join(unknown))
        for k, v in d.items():
            setattr(p, k, v)
        p.colors = {str(k): _check_ranges(k, v) for k, v in p.colors.items()}
        if p.table_z is not None:
            p.table_z = float(p.table_z)
        return p


@dataclass
class CubeDetection:
    """One cube. center_base is the center of the top face."""

    label: str
    center_base: np.ndarray
    yaw: float                      # edge direction in base XY, in [-pi/4, pi/4)
    edge: float                     # measured top-face edge (m)
    height: float                   # top above the table (m)
    score: float
    center_px: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    n_points: int = 0
    mask: Optional[np.ndarray] = None
    source: str = 'depth'           # 'depth' | 'color'
    warnings: List[str] = field(default_factory=list)

    @property
    def grasp_center(self) -> np.ndarray:
        """Return the cube center (half an edge below the top face)."""
        return self.center_base - np.array([0.0, 0.0, self.height / 2.0])

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {'label': self.label, 'center_base': [round(float(v), 4) for v in self.center_base],
                'yaw_deg': round(math.degrees(self.yaw), 1), 'edge': round(self.edge, 4),
                'height': round(self.height, 4), 'score': round(self.score, 3),
                'center_px': [round(float(v), 1) for v in self.center_px],
                'n_points': self.n_points, 'source': self.source, 'warnings': self.warnings}


@dataclass
class ZoneDetection:
    """One target card, lying flat on the table."""

    label: str
    center_base: np.ndarray
    yaw: float
    size: Tuple[float, float]
    score: float
    center_px: Tuple[float, float]

    @property
    def name(self) -> str:
        """Return the zone name used in cube_task configs ('zone_<color>')."""
        return 'zone_%s' % self.label

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {'name': self.name, 'center_base': [round(float(v), 4) for v in self.center_base],
                'yaw_deg': round(math.degrees(self.yaw), 1),
                'size': [round(float(v), 3) for v in self.size], 'score': round(self.score, 3)}


@dataclass
class L1Result:
    """Everything found in one frame."""

    cubes: List[CubeDetection]
    zones: List[ZoneDetection]
    table_z: Optional[float]
    table_normal: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)

    def by_label(self, label: str) -> List[CubeDetection]:
        """Return the cubes of one color, best score first."""
        return sorted([c for c in self.cubes if c.label == label], key=lambda c: -c.score)

    def zone(self, label: str) -> Optional[ZoneDetection]:
        """Return the card of one color, or None."""
        found = [z for z in self.zones if z.label == label]
        return max(found, key=lambda z: z.score) if found else None


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _check_ranges(label: str, ranges: Any) -> List[List[List[int]]]:
    arr = np.asarray(ranges, dtype=float)
    if arr.shape == (2, 3):
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (2, 3):
        raise ValueError('color %r: ranges must be a list of [[h,s,v],[h,s,v]]' % label)
    if np.any(arr[:, :, 0] < 0) or np.any(arr[:, :, 0] > 180):
        raise ValueError('color %r: hue must be in [0, 180]' % label)
    if np.any(arr[:, :, 1:] < 0) or np.any(arr[:, :, 1:] > 255):
        raise ValueError('color %r: S / V must be in [0, 255]' % label)
    return arr.astype(int).tolist()


def color_mask(hsv: np.ndarray, ranges: Sequence) -> np.ndarray:
    """Return an HxW bool mask of pixels inside any HSV range (hue may wrap)."""
    out = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in ranges:
        sv = cv2.inRange(hsv, (0, int(lo[1]), int(lo[2])), (180, int(hi[1]), int(hi[2]))) > 0
        h = hsv[:, :, 0]
        if lo[0] <= hi[0]:
            hue = (h >= lo[0]) & (h <= hi[0])
        else:
            hue = (h >= lo[0]) | (h <= hi[0])
        out |= hue & sv
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def depth_to_base(depth: np.ndarray, intr: Intrinsics, T_base_camera: np.ndarray,
                  zmin: float, zmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (HxWx3 base points, HxW valid mask) from an aligned depth image (m)."""
    h, w = depth.shape
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > zmin) & (d < zmax)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where(valid, d, 0.0)
    pc = np.stack(((u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z), axis=-1)
    T = np.asarray(T_base_camera, dtype=float)
    pb = pc @ T[:3, :3].T + T[:3, 3]
    return pb, valid


def fit_table_plane(pts: np.ndarray, max_tilt_deg: float, inlier_m: float,
                    rng: np.random.Generator, iters: int = 200,
                    max_points: int = 6000) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    RANSAC a plane n.p + d = 0 with n near base +z. Returns (n, d, inlier ratio).

    The plane with the most inliers wins; the table covers far more of the
    view than the top faces of a few cubes.
    """
    if len(pts) < 50:
        return None
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    best = None
    best_n = 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        d = -float(n @ a)
        inl = np.abs(pts @ n + d) < inlier_m
        k = int(inl.sum())
        if k > best_n:
            best, best_n = (n, d, inl), k
    if best is None:
        return None
    n, d, inl = best
    # Least-squares refine on the inliers.
    P = pts[inl]
    centroid = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - centroid)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    d = -float(n @ centroid)
    ratio = float(np.mean(np.abs(pts @ n + d) < inlier_m))
    return n, d, ratio


def _rect_xy(xy: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Return (center, angle of the first side in rad, (side_a, side_b)) of the min-area rect."""
    (cx, cy), (wa, wb), ang = cv2.minAreaRect(xy.astype(np.float32))
    return np.array([cx, cy]), math.radians(ang), (float(wa), float(wb))


def _ray_plane(intr: Intrinsics, T_base_camera: np.ndarray, uv: np.ndarray,
               z_plane: float) -> np.ndarray:
    """Intersect pixel rays with the horizontal plane z = z_plane; Nx3 base points."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    rays_c = np.column_stack(((uv[:, 0] - intr.cx) / intr.fx,
                              (uv[:, 1] - intr.cy) / intr.fy, np.ones(len(uv))))
    T = np.asarray(T_base_camera, dtype=float)
    rays_b = rays_c @ T[:3, :3].T
    o = T[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (z_plane - o[2]) / rays_b[:, 2]
    s = np.where(s > 0, s, np.nan)
    return o + rays_b * s[:, None]


def _in_polygon(xy: np.ndarray, poly: Optional[Sequence]) -> np.ndarray:
    if not poly:
        return np.ones(len(xy), dtype=bool)
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
                     for x, y in xy], dtype=bool)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class L1Detector:
    """Find L1 cubes and target cards in one RGB-D frame."""

    def __init__(self, params: Optional[L1Params] = None):
        self.p = params or L1Params()
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(self.p.morph_kernel), int(self.p.morph_kernel)))

    # -- public ------------------------------------------------------------

    def detect(self, color: np.ndarray, depth: Optional[np.ndarray], intr: Intrinsics,
               T_base_camera: np.ndarray) -> L1Result:
        """
        Detect cubes and cards. Never raises on bad image content.

        color: HxWx3 BGR uint8. depth: HxW float32 meters aligned to color, or None.
        """
        if color is None or color.ndim != 3:
            return L1Result([], [], None, warnings=['no color image'])
        if (intr.width, intr.height) != (color.shape[1], color.shape[0]):
            intr = intr.scaled(color.shape[1], color.shape[0])
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        masks = {lab: self._clean(color_mask(hsv, rng)) for lab, rng in self.p.colors.items()}
        if depth is not None and depth.shape != color.shape[:2]:
            return L1Result([], [], None, warnings=[
                'depth %s does not match color %s; align depth to color first'
                % (depth.shape, color.shape[:2])])
        if depth is not None and np.count_nonzero(np.isfinite(depth) & (depth > 0)) > 1000:
            return self._detect_depth(masks, depth, intr, T_base_camera)
        res = self._detect_color_only(masks, intr, T_base_camera)
        res.warnings.insert(0, 'no usable depth; positions from the table plane only')
        return res

    # -- depth path ------------------------------------------------------------

    def _detect_depth(self, masks: Dict[str, np.ndarray], depth: np.ndarray,
                      intr: Intrinsics, T_bc: np.ndarray) -> L1Result:
        p = self.p
        warnings: List[str] = []
        pb, valid = depth_to_base(depth, intr, T_bc, p.depth_min, p.depth_max)
        n, d = np.array([0.0, 0.0, 1.0]), None
        if p.fit_table:
            rng = np.random.default_rng(p.seed)
            fit = fit_table_plane(pb[valid], p.table_max_tilt_deg, p.table_inlier_m, rng)
            if fit is not None and fit[2] > 0.2:
                n, d = fit[0], fit[1]
                if p.table_z is not None and abs(-d / n[2] - p.table_z) > 0.02:
                    warnings.append('fitted table z %.3f differs from configured %.3f'
                                    % (-d / n[2], p.table_z))
            else:
                warnings.append('table plane fit failed')
        if d is None:
            if p.table_z is None:
                return L1Result([], [], None, warnings=warnings + [
                    'no table plane: set table_z in l1.yaml or check the extrinsics'])
            d = -float(p.table_z)
        height = np.where(valid, pb @ n + d, np.nan)
        table_z = float(-d / n[2])

        cubes: List[CubeDetection] = []
        zones: List[ZoneDetection] = []
        with np.errstate(invalid='ignore'):
            raised = (height > p.min_height) & (height < p.max_height)
            flat = np.abs(height) < p.card_max_height
        for label, cmask in masks.items():
            cube_mask = self._clean(cmask & raised)
            for comp in self._components(cube_mask):
                cubes.extend(self._cube_from_points(label, comp, pb, height, cmask))
            for comp in self._components(self._clean(cmask & flat)):
                z = self._zone_from_xy(label, comp, pb[comp][:, :2], table_z)
                if z is not None:
                    zones.append(z)
        cubes, zones = self._in_roi(cubes, zones)
        return L1Result(cubes, zones, table_z, n, warnings)

    def _cube_from_points(self, label: str, comp: np.ndarray, pb: np.ndarray,
                          height: np.ndarray, cmask: np.ndarray) -> List[CubeDetection]:
        p = self.p
        hs = height[comp]
        top_h = float(np.percentile(hs, 90))
        top = comp & (height > top_h - p.top_band)
        pts = pb[top]
        if len(pts) < max(10, p.min_pixels // 4):
            return []
        center, _, sides = _rect_xy(pts[:, :2])
        n_cubes = 1
        area = sides[0] * sides[1]
        if max(sides) > 1.6 * p.cube_size:
            n_cubes = max(1, int(round(area / (p.cube_size ** 2))))
        groups = [top] if n_cubes == 1 else self._split(top, pts[:, :2], n_cubes)
        out = []
        for g in groups:
            gp = pb[g]
            if len(gp) < 10:
                continue
            c, ang, s = _rect_xy(gp[:, :2])
            edge = float(np.median(s)) if n_cubes == 1 else float(max(s))
            h_top = float(np.percentile(height[g], 90))
            warn = []
            score = 1.0
            if abs(h_top - p.cube_size) > p.size_tol:
                warn.append('height %.3f m, expected %.3f' % (h_top, p.cube_size))
                score *= 0.5
            if abs(edge - p.cube_size) > p.size_tol:
                warn.append('top edge %.3f m, expected %.3f' % (edge, p.cube_size))
                score *= 0.6
            if h_top < p.cube_size * 0.5 or edge > 2.0 * p.cube_size:
                continue
            frac = float(cmask[g].mean())
            score *= min(1.0, frac / max(p.min_color_fraction, 1e-6))
            top_z = float(np.percentile(gp[:, 2], 90))
            out.append(self._make_cube(label, np.array([c[0], c[1], top_z]), ang, edge,
                                       h_top, score, g, len(gp), 'depth', warn))
        return out

    def _split(self, mask: np.ndarray, xy: np.ndarray, k: int) -> List[np.ndarray]:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        cv2.setRNGSeed(self.p.seed)
        _, lab, _ = cv2.kmeans(xy.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        lab = lab.reshape(-1)
        idx = np.flatnonzero(mask)
        groups = []
        for i in range(k):
            g = np.zeros_like(mask)
            g.reshape(-1)[idx[lab == i]] = True
            groups.append(g)
        return groups

    # -- color-only path -------------------------------------------------------

    def _detect_color_only(self, masks: Dict[str, np.ndarray], intr: Intrinsics,
                           T_bc: np.ndarray) -> L1Result:
        p = self.p
        if p.table_z is None:
            return L1Result([], [], None, warnings=['table_z is required without depth'])
        cubes, zones = [], []
        for label, cmask in masks.items():
            for comp in self._components(cmask):
                v, u = np.nonzero(comp)
                uv = np.column_stack((u, v)).astype(float)
                card_xy = _ray_plane(intr, T_bc, uv, p.table_z)[:, :2]
                z = self._zone_from_xy(label, comp, card_xy, p.table_z)
                if z is not None:
                    zones.append(z)
                    continue
                # A cube shows its top face and some sides; the sides make the
                # blob look larger, so judge by the top-plane footprint loosely.
                top_z = p.table_z + p.cube_size
                pts = _ray_plane(intr, T_bc, uv, top_z)
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if len(pts) < 10:
                    continue
                c, ang, s = _rect_xy(pts[:, :2])
                edge = float(min(s))
                if not 0.6 * p.cube_size < edge < 1.8 * p.cube_size:
                    continue
                cubes.append(self._make_cube(label, np.array([c[0], c[1], top_z]), ang, edge,
                                             p.cube_size, 0.6, comp, len(pts), 'color',
                                             ['position from color only; side faces bias it '
                                              'away from the camera']))
        cubes, zones = self._in_roi(cubes, zones)
        return L1Result(cubes, zones, p.table_z)

    def _in_roi(self, cubes: List[CubeDetection],
                zones: List[ZoneDetection]) -> Tuple[List[CubeDetection], List[ZoneDetection]]:
        """Keep only objects whose center lies in roi_polygon (cubes and cards alike)."""
        poly = self.p.roi_polygon
        if not poly:
            return cubes, zones
        return ([c for c in cubes if _in_polygon(c.center_base[None, :2], poly)[0]],
                [z for z in zones if _in_polygon(z.center_base[None, :2], poly)[0]])

    # -- shared ------------------------------------------------------------------

    def _zone_from_xy(self, label: str, comp: np.ndarray, xy: np.ndarray,
                      table_z: float) -> Optional[ZoneDetection]:
        p = self.p
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) < p.min_pixels:
            return None
        c, ang, s = _rect_xy(xy)
        if not all(abs(v - p.card_size) <= p.card_size_tol for v in s):
            return None
        v, u = np.nonzero(comp)
        score = 1.0 - max(abs(v_ - p.card_size) for v_ in s) / max(p.card_size_tol, 1e-6) * 0.5
        return ZoneDetection(label, np.array([c[0], c[1], table_z]),
                             wrap_angle(ang, math.pi / 2), (s[0], s[1]), float(score),
                             (float(u.mean()), float(v.mean())))

    def _make_cube(self, label: str, center: np.ndarray, ang: float, edge: float,
                   height: float, score: float, mask: np.ndarray, n: int, source: str,
                   warn: List[str]) -> CubeDetection:
        v, u = np.nonzero(mask)
        return CubeDetection(
            label=label, center_base=center, yaw=wrap_angle(ang, math.pi / 2), edge=edge,
            height=height, score=float(max(0.0, min(1.0, score))),
            center_px=(float(u.mean()), float(v.mean())),
            bbox=(int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1),
            n_points=int(n), mask=mask, source=source, warnings=warn)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
        return m > 0

    def _components(self, mask: np.ndarray) -> List[np.ndarray]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.p.min_pixels]


def draw_result(color: np.ndarray, res: L1Result) -> np.ndarray:
    """Return a BGR overlay with cube boxes, labels and card centers."""
    out = color.copy()
    bgr = {'red': (0, 0, 255), 'green': (0, 200, 0), 'yellow': (0, 220, 255)}
    for z in res.zones:
        cv2.drawMarker(out, (int(z.center_px[0]), int(z.center_px[1])),
                       bgr.get(z.label, (255, 255, 255)), cv2.MARKER_SQUARE, 30, 2)
        cv2.putText(out, z.name, (int(z.center_px[0]) + 18, int(z.center_px[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, c in enumerate(res.cubes):
        x0, y0, x1, y1 = c.bbox
        col = bgr.get(c.label, (255, 255, 255))
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        txt = '%d %s %.2f (%.3f, %.3f)' % (i, c.label, c.score, c.center_base[0], c.center_base[1])
        cv2.putText(out, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return out

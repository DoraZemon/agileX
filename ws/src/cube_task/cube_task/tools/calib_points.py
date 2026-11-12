"""
Fast camera extrinsics from point pairs: touch a target with the TCP, no board.

At N spread-out spots on the table the camera measures a target in the camera
frame, then the operator drags the arm (drag-teach mode) so the fingertip
center (TCP) touches the measured point and presses Enter. A rigid fit of the
pairs (core.transforms.fit_rigid) gives the wrist camera extrinsics (the
competition's only camera, eye_in_hand): each measurement stores the flange pose
it was taken at and inv(T_base_flange_i) @ p_base = T_flange_camera @ p_cam, so
the arm does not have to come back to exactly the same view pose between points.

Targets (--target):
  block  one block 8-80 mm high (e.g. an L1 cube) alone on the table: the table
         plane is fitted in the camera frame and the top face is found on the
         block standing on it. Touch the CENTER OF THE TOP FACE.
  aruco  a printed ArUco marker lying flat (--aruco-dict; --aruco-size = side of
         the black square in m), e.g. taped on a block. Touch its CENTER.

Per point: Enter measures (--frames frames, averaged, rejected when they
disagree by more than --max-spread mm), then touch and Enter. From the 4th
point on, a touch far from where the previous points predict it is reported
at once. After --points points the tool solves, prints per-point residuals and
the RMS, offers to drop outliers (> 3x the median residual and > 3 mm), prints
sanity checks (view angle, camera height, table tilt in the base frame, change
from the previous extrinsics) and writes camera.extrinsics_file after a
timestamped backup (sim: <logging.dir>/sim_site). The pairs are saved after
every point; --resume continues a session.

The tool never moves the arm, it only reads poses (drag-teach mode is fine).
Touches use robot.tcp_offset, which must be the fingertip center.

Placing (headless: type + Enter; window: keys in the preview window):
    Enter measure, s solve now, u undo the last point, q quit
After solving: w write the file, a add a point, d N drop point N, u undo, q quit.

Example commands
    calib_points
    calib_points --target aruco --aruco-size 0.05 --points 8 --headless
    calib_points --mode sim --mount eye_in_hand --headless
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import CameraIntrinsics, Frame, TablePlane
from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_onsite_args, compose_view, connect_arm_readonly, coverage_check,
                     extrinsics_out_path, flange_to_tcp, fmt_mm, format_table, grab_frames,
                     load_extrinsics_or_none, logging_dir, make_ui, operator_action,
                     read_pose_averaged, sim_drag_joints, sim_drag_tcp, teach_hint,
                     teach_mode_reminder, text_panel, use_hint, wall_stamp,
                     write_extrinsics_site, write_report)

_LOG = logging.getLogger('cube_task.tools.calib_points')

TOOL = 'calib_points'
TARGETS = ('block', 'aruco')
MIN_SOLVE_POINTS = 4            # never solve with fewer pairs
RECOMMENDED_POINTS = 5
MIN_LINE_OFFSET_M = 0.015       # points must lie this far (RMS) from their best-fit line
OUTLIER_FACTOR = 3.0            # flag a residual > OUTLIER_FACTOR x the median ...
OUTLIER_MIN_M = 0.003           # ... and > OUTLIER_MIN_M
PREDICT_WARN_M = 0.010          # touch this far from where the previous points predict it
RMS_WARN_M = 0.003
TABLE_TILT_WARN_DEG = 3.0
STILL_M = 0.001                 # arm pose change allowed while measuring / touching
STILL_DEG = 0.3
MIN_TOUCH_MOVE_M = 0.02         # the TCP must move this far between measuring and touching
MAX_TRIES = 10

HEADLESS_PLACE = '[Enter = measure, s = solve now, u = undo last point, q = quit]'
WINDOW_PLACE = 'keys in the preview window: Enter/space measure, s solve now, u undo, q quit'
WINDOW_KEYS = {'\n': 'measure', ' ': 'measure', 'm': 'measure', 's': 'solve', 'u': 'undo',
               'q': 'quit', 'esc': 'quit'}
MENU = ('w = write %s, a = add a point, d N = drop point N, u = undo the last point, '
        'q = quit without writing')
COMMANDS = {'m': 'measure', 's': 'solve', 'u': 'undo', 'q': 'quit', 'w': 'write', 'a': 'add',
            'd': 'drop', 'r': 'retouch', 'x': 'discard', 'k': 'keep'}
RESIDUAL_HEADERS = ('#', 'camera point mm', 'TCP (base) mm', 'residual mm', 'spread mm', '')

# Simulated operator (--mode sim).
SIM_GRID = ((0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.5, 0.1),
            (0.5, 0.9), (0.1, 0.5), (0.9, 0.5), (0.3, 0.3), (0.7, 0.7), (0.3, 0.7),
            (0.7, 0.3), (0.3, 0.5), (0.7, 0.5))
SIM_BLOCK_M = 0.035
SIM_PLATE_M = 0.004             # thickness of the simulated marker plate
SIM_VIEW_OFFSETS = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.04, 0.0, 0.0, 0.0, 0.0, 0.06),
                    (-0.04, 0.0, 0.0, 0.0, 0.0, -0.06), (0.02, 0.0, 0.0, 0.0, 0.0, -0.03),
                    (-0.02, 0.0, 0.0, 0.0, 0.0, 0.03))
SIM_FINGER_ROWS = 45            # bottom rows of the eye-in-hand image showing the fingers
SIM_BAD_TOUCH_M = 0.015


# ------------------------------------------------------------------ data
@dataclasses.dataclass
class BlockParams:
    """Settings of measure_target_block (meters unless noted)."""

    min_height: float = 0.008       # block pixels stand at least this high above the table
    max_height: float = 0.080       # ... and at most this high
    top_band: float = 0.004         # top face: within this of the top-percentile height
    top_percentile: float = 90.0
    max_extent: float = 0.10        # wider components (bins, the robot) are not the block
    min_pixels: int = 30            # smaller components are noise
    min_top_pixels: int = 12
    depth_median_px: int = 3        # edge-preserving median filter of the depth (0 = off)
    plane_thresh: float = 0.005     # RANSAC inlier distance of the table plane
    plane_iters: int = 200
    plane_points: int = 5000        # RANSAC subsample size
    zmin: float = 0.15
    zmax: float = 3.0
    seed: int = 0

    @staticmethod
    def from_cfg(cfg) -> 'BlockParams':
        """Return defaults with the plane threshold and depth range from perception.*."""
        return BlockParams(plane_thresh=float(cfg.get('perception.table.inlier_thresh', 0.005)),
                           zmin=float(cfg.get('perception.depth.min_z', 0.15)),
                           zmax=float(cfg.get('perception.depth.max_z', 3.0)))


@dataclasses.dataclass
class Measurement:
    """
    Target position in the camera frame, from one frame or averaged.

    ok False: reason says why. p_cam (3,) m; pixel: image position of p_cam;
    method 'block' | 'aruco_depth' | 'aruco_pnp'; plane: table plane in the
    camera frame (None if not found); height: target top above the table (m);
    size: (long, short) side of the top face / marker (m); spread: largest
    distance of one frame's point from the mean (m); mask / outline: pixels and
    pixel polygon of the top face / marker (previews).
    """

    ok: bool
    reason: str = ''
    p_cam: Optional[np.ndarray] = None
    pixel: Optional[Tuple[float, float]] = None
    method: str = ''
    plane: Optional[TablePlane] = None
    height: Optional[float] = None
    size: Optional[Tuple[float, float]] = None
    n_points: int = 0
    n_frames: int = 1
    spread: float = 0.0
    marker_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    outline: Optional[np.ndarray] = None


def _as_list(a: Any) -> Optional[list]:
    """Return a nested float list of a, or None."""
    return None if a is None else np.asarray(a, dtype=float).tolist()


def _copy_or_none(a: Any) -> Optional[np.ndarray]:
    """Return a float copy of a, or None."""
    return None if a is None else np.array(a, dtype=float)


def _as_array(d: Dict[str, Any], key: str, shape: Tuple[int, ...],
              required: bool = False) -> Optional[np.ndarray]:
    """Return d[key] as a finite float array of this shape (None when absent and optional)."""
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError('point %s: %s is missing' % (d.get('id', '?'), key))
        return None
    a = np.asarray(v, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError('point %s: %s must be %s finite numbers'
                         % (d.get('id', '?'), key, 'x'.join(str(s) for s in shape)))
    return a


@dataclasses.dataclass
class PointPair:
    """
    One calibration point (meters, 4x4 poses).

    p_cam: target in the camera frame. p_base: TCP position at the touch.
    T_base_flange_cam: flange pose while the camera measured (eye_in_hand
    needs it). T_base_tcp / T_base_flange: arm pose at the touch.
    plane_normal / plane_d: table plane in the camera frame (sanity checks).
    """

    id: int
    p_cam: np.ndarray
    p_base: np.ndarray
    T_base_flange_cam: Optional[np.ndarray] = None
    T_base_tcp: Optional[np.ndarray] = None
    T_base_flange: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    plane_d: Optional[float] = None
    spread: float = 0.0
    method: str = ''

    def plane(self) -> Optional[TablePlane]:
        """Return the stored camera-frame table plane, or None."""
        if self.plane_normal is None or self.plane_d is None:
            return None
        return TablePlane(np.asarray(self.plane_normal, dtype=float), float(self.plane_d),
                          'camera')

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict (inverse of from_dict)."""
        return {'id': int(self.id), 'p_cam': _as_list(self.p_cam),
                'p_base': _as_list(self.p_base),
                'T_base_flange_cam': _as_list(self.T_base_flange_cam),
                'T_base_tcp': _as_list(self.T_base_tcp),
                'T_base_flange': _as_list(self.T_base_flange),
                'plane_normal': _as_list(self.plane_normal),
                'plane_d': None if self.plane_d is None else float(self.plane_d),
                'spread': float(self.spread), 'method': str(self.method)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'PointPair':
        """Build a pair from to_dict() output; raise ValueError on bad data."""
        if not isinstance(d, dict):
            raise ValueError('a point must be a mapping, got %s' % type(d).__name__)
        plane_d = d.get('plane_d')
        return PointPair(int(d.get('id', 0)), _as_array(d, 'p_cam', (3,), True),
                         _as_array(d, 'p_base', (3,), True),
                         _as_array(d, 'T_base_flange_cam', (4, 4)),
                         _as_array(d, 'T_base_tcp', (4, 4)),
                         _as_array(d, 'T_base_flange', (4, 4)),
                         _as_array(d, 'plane_normal', (3,)),
                         None if plane_d is None else float(plane_d),
                         float(d.get('spread', 0.0) or 0.0), str(d.get('method', '') or ''))


# ------------------------------------------------------------------ geometry
def _unit(v: Any) -> np.ndarray:
    """Return v / |v| (ValueError for a zero vector)."""
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if not n > 1e-12:
        raise ValueError('zero-length vector')
    return a / n


def plane_basis(normal: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit vectors (e1, e2) spanning the plane with this normal (e1 x e2 = n)."""
    n = _unit(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(a - (a @ n) * n)
    return e1, np.cross(n, e1)


def _lsq_plane(P: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Return the least-squares plane (n, d) of Nx3 points, oriented so that d >= 0."""
    if P.shape[0] < 3:
        return None
    c = P.mean(axis=0)
    q = P - c
    w, v = np.linalg.eigh(q.T @ q)
    if w[1] <= 1e-12:
        return None
    n = v[:, 0] / np.linalg.norm(v[:, 0])
    d = -float(n @ c)
    return (-n, -d) if d < 0 else (n, d)


def fit_plane_ransac(points: Any, thresh: float = 0.005, iters: int = 200,
                     max_points: int = 5000, rng: Optional[np.random.Generator] = None,
                     min_points: int = 200) -> Optional[TablePlane]:
    """
    Fit the dominant plane to camera-frame points (RANSAC, then least squares).

    points: Nx3 (m), NaN rows ignored. The normal points toward the camera
    origin (d > 0 is the camera's distance from the plane), so
    plane.signed_height(p) is the height above the table. Returns None with
    fewer than min_points finite points or without a plane hypothesis.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    n_all = int(P.shape[0])
    if n_all < max(3, int(min_points)):
        return None
    rng = rng if rng is not None else np.random.default_rng(0)
    sub = P if n_all <= max_points else P[rng.choice(n_all, int(max_points), replace=False)]
    idx = rng.integers(0, sub.shape[0], size=(max(1, int(iters)), 3))
    a, b, c = sub[idx[:, 0]], sub[idx[:, 1]], sub[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    length = np.linalg.norm(nrm, axis=1)
    ok = length > 1e-12
    if not np.any(ok):
        return None
    nrm = nrm[ok] / length[ok, None]
    d = -np.sum(nrm * a[ok], axis=1)
    count = np.count_nonzero(np.abs(sub @ nrm.T + d) < thresh, axis=0)
    best = int(np.argmax(count))
    n, dd = nrm[best], float(d[best])
    for _ in range(3):
        fit = _lsq_plane(P[np.abs(P @ n + dd) < thresh])
        if fit is None:
            return None
        n, dd = fit
    sd = P @ n + dd
    inl = np.abs(sd) < thresh
    n_in = int(np.count_nonzero(inl))
    rms = float(np.sqrt(np.mean(sd[inl] ** 2))) if n_in else float('nan')
    return TablePlane(normal=n, d=dd, frame='camera', inlier_ratio=n_in / float(n_all), rms=rms)


def project_cam(points: Any, intr: CameraIntrinsics) -> np.ndarray:
    """Return Nx2 pixels of camera-frame points (Nx3 or (3,), m); NaN behind the camera."""
    from ..perception.geometry import project_points
    return project_points(np.asarray(points, dtype=float).reshape(-1, 3), np.eye(4), intr)


def pixel_ray(pixel: Sequence[float], intr: CameraIntrinsics) -> np.ndarray:
    """Return the camera-frame ray (x, y, 1) through pixel (u, v), undistorted."""
    from ..perception.geometry import has_distortion
    u, v = float(pixel[0]), float(pixel[1])
    if has_distortion(intr):
        import cv2
        p = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), intr.K(),
                                np.asarray(intr.dist, dtype=np.float64).ravel())
        return np.array([float(p[0, 0, 0]), float(p[0, 0, 1]), 1.0])
    return np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])


def clean_depth(depth: Any, median_px: int = 3) -> np.ndarray:
    """
    Return float32 depth (m) with invalid pixels set to 0.

    median_px >= 3: 3x3 (5 and up: 5x5) median filter. It keeps depth edges
    and removes the noise that pushes the extreme top-face points outward (in
    the simulation it cut the error of the far eye-to-hand camera from 0.8 to
    0.35 mm).
    """
    d = np.asarray(depth, dtype=np.float32)
    with np.errstate(invalid='ignore'):
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
    if int(median_px) >= 3:
        import cv2
        d = cv2.medianBlur(np.ascontiguousarray(d), 3 if int(median_px) < 5 else 5)
    return d


# ------------------------------------------------------------------ block target
def _pick_block(cand: np.ndarray, pts: np.ndarray, plane: TablePlane,
                p: BlockParams) -> Tuple[Optional[np.ndarray], str]:
    """Return (mask of the largest block-like component, '') or (None, why)."""
    import cv2
    H, W = cand.shape
    _n, lab, stats, _c = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    E = np.column_stack(plane_basis(plane.normal))
    border = wide = 0
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        x, y, w, h, area = (int(v) for v in stats[i, :5])
        if area < p.min_pixels:
            break
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            border += 1             # eye-in-hand fingers, objects cut by the image edge
            continue
        comp = lab == i
        uv = pts[comp].astype(np.float64) @ E * 1000.0
        _ctr, sides, _ang = cv2.minAreaRect(uv.astype(np.float32).reshape(-1, 1, 2))
        if max(sides) > p.max_extent * 1000.0:
            wide += 1               # bins, the robot base, an arm low over the table
            continue
        return comp, ''
    why = [t for t in ('%d at the image border' % border if border else '',
                       '%d wider than %.0f mm' % (wide, p.max_extent * 1e3) if wide else '') if t]
    return None, ('no block found %.0f-%.0f mm above the table%s'
                  % (p.min_height * 1e3, p.max_height * 1e3,
                     ' (ignored: %s)' % ', '.join(why) if why else ''))


def measure_target_block(frame: Frame, params: Optional[BlockParams] = None,
                         plane: Optional[TablePlane] = None) -> Measurement:
    """
    Measure the top-face center of the block on the table, in the camera frame.

    The table is the dominant plane of the back-projected depth (RANSAC; pass
    plane to skip the fit). The block is the largest connected component of
    pixels min_height..max_height above it that does not touch the image
    border (eye-in-hand fingers) and is at most max_extent wide. Its top face
    is the pixels within top_band of the top_percentile height, cleaned by a
    3x3 opening. The center is the cv2.minAreaRect center of the top-face
    points in plane coordinates (no erosion: it biases the center toward the
    camera), at the median top height above the plane.
    """
    import cv2
    from ..perception.geometry import backproject
    p = params or BlockParams()
    if frame.depth is None:
        return Measurement(False, 'the frame has no depth image')
    intr = frame.intrinsics
    pts = backproject(clean_depth(frame.depth, p.depth_median_px), intr, p.zmin, p.zmax)
    valid = np.isfinite(pts[..., 2])
    if plane is None:
        plane = fit_plane_ransac(pts[valid], p.plane_thresh, p.plane_iters, p.plane_points,
                                 np.random.default_rng(p.seed))
    if plane is None:
        return Measurement(False, 'no table plane in the depth image (too few valid pixels)')
    n, d = np.asarray(plane.normal, dtype=float), float(plane.d)
    h = np.full(valid.shape, np.nan)
    h[valid] = pts[valid].astype(np.float64) @ n + d
    with np.errstate(invalid='ignore'):
        cand = valid & (h >= p.min_height) & (h <= p.max_height)
    comp, why = _pick_block(cand, pts, plane, p)
    if comp is None:
        return Measurement(False, why, plane=plane)
    top_h = float(np.percentile(h[comp], p.top_percentile))
    with np.errstate(invalid='ignore'):
        top = comp & (np.abs(h - top_h) <= p.top_band)
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n_top = int(np.count_nonzero(top))
    if n_top < p.min_top_pixels:
        return Measurement(False, 'top face too small (%d pixels): is the block seen from the '
                           'side or too far away?' % n_top, plane=plane)
    E = np.column_stack(plane_basis(n))
    origin = -d * n
    uv = (pts[top].astype(np.float64) - origin) @ E * 1000.0
    (ru, rv), (sa, sb), ang = cv2.minAreaRect(uv.astype(np.float32).reshape(-1, 1, 2))
    height = float(np.median(h[top]))
    center = origin + E @ (np.array([ru, rv]) / 1000.0) + height * n
    box = cv2.boxPoints(((ru, rv), (sa, sb), ang)).astype(np.float64) / 1000.0
    outline = project_cam(origin + box @ E.T + height * n, intr)
    pix = project_cam(center, intr)[0]
    return Measurement(True, p_cam=center, pixel=(float(pix[0]), float(pix[1])), method='block',
                       plane=plane, height=height,
                       size=(max(sa, sb) / 1000.0, min(sa, sb) / 1000.0), n_points=n_top,
                       mask=top, outline=outline)


# ------------------------------------------------------------------ ArUco target
def aruco_dictionary(name: str):
    """Return the predefined cv2.aruco dictionary called name (ValueError if unknown)."""
    import cv2
    aruco = getattr(cv2, 'aruco', None)
    if aruco is None:
        raise ValueError('this OpenCV build has no aruco module')
    code = getattr(aruco, str(name), None)
    if not str(name).startswith('DICT_') or not isinstance(code, int):
        raise ValueError('unknown ArUco dictionary %r (e.g. DICT_4X4_50, DICT_5X5_100)' % name)
    if hasattr(aruco, 'getPredefinedDictionary'):
        return aruco.getPredefinedDictionary(code)
    return aruco.Dictionary_get(code)


def detect_markers(gray: np.ndarray, dictionary) -> Tuple[List[np.ndarray], List[int]]:
    """Return (4x2 corner arrays in ArUco order TL, TR, BR, BL, ids), sub-pixel refined."""
    import cv2
    aruco = cv2.aruco
    if hasattr(aruco, 'ArucoDetector'):            # OpenCV >= 4.7
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _rej = aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _rej = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(corners) == 0:
        return [], []
    return ([np.asarray(c, dtype=float).reshape(4, 2) for c in corners],
            [int(i) for i in np.asarray(ids).ravel()])


def marker_image(dictionary, marker_id: int, side_px: int) -> np.ndarray:
    """Return the side_px x side_px uint8 image of a marker (black border included)."""
    import cv2
    aruco = cv2.aruco
    if hasattr(aruco, 'generateImageMarker'):
        return aruco.generateImageMarker(dictionary, int(marker_id), int(side_px))
    return aruco.drawMarker(dictionary, int(marker_id), int(side_px))


def quad_center(corners: Any) -> np.ndarray:
    """
    Return the intersection of the diagonals of a 4x2 pixel quad.

    It is the exact image of the center of a square seen in perspective (the
    corner mean is not). A degenerate quad gives the corner mean.
    """
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    d1, d2 = c[2] - c[0], c[3] - c[1]
    A = np.column_stack([d1, -d2])
    if abs(float(np.linalg.det(A))) < 1e-9:
        return c.mean(axis=0)
    s = float(np.linalg.solve(A, c[1] - c[0])[0])
    return c[0] + s * d1


def _robust_plane(P: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Return a least-squares plane of Nx3 points refitted without > 3 sigma (>= 2 mm) points."""
    fit = _lsq_plane(P)
    if fit is None:
        return None
    r = np.abs(P @ fit[0] + fit[1])
    keep = r <= max(3.0 * float(np.sqrt(np.mean(r ** 2))), 0.002)
    return (_lsq_plane(P[keep]) or fit) if np.count_nonzero(keep) >= 3 else fit


def measure_target_aruco(frame: Frame, dictionary, marker_size: float = 0.04,
                         marker_id: Optional[int] = None, zmin: float = 0.15,
                         zmax: float = 3.0) -> Measurement:
    """
    Measure the center of an ArUco marker in the camera frame.

    The center pixel is the intersection of the marker diagonals; its ray
    meets the plane fitted to the depth inside the marker (median depth when
    that fit fails). With valid depth on less than 30 % of the marker the
    position comes from cv2.solvePnP with marker_size (side of the black
    square, m), which is less accurate in depth. marker_id None = the largest
    marker in the image.
    """
    import cv2
    from ..perception.geometry import backproject
    color = np.asarray(frame.color)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    corners, ids = detect_markers(gray, dictionary)
    if not ids:
        return Measurement(False, 'no ArUco marker found (dictionary? light? marker too small '
                           'in the image?)')
    found = [(c, i) for c, i in zip(corners, ids) if marker_id is None or i == int(marker_id)]
    if not found:
        return Measurement(False, 'marker id %d not found (seen: %s)'
                           % (int(marker_id), ', '.join(str(i) for i in ids)))
    quad, mid = max(found, key=lambda ci: float(cv2.arcLength(ci[0].astype(np.float32), True)))
    intr = frame.intrinsics
    center_px = quad_center(quad)
    ray = pixel_ray(center_px, intr)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    inside = mask > 0
    p_cam: Optional[np.ndarray] = None
    method, table, n_pts = '', None, 0
    if frame.depth is not None:
        pts = backproject(clean_depth(frame.depth, 0), intr, zmin, zmax)
        valid = np.isfinite(pts[..., 2])
        P = pts[inside & valid].astype(np.float64)
        n_pts = int(P.shape[0])
        if n_pts >= max(10.0, 0.3 * np.count_nonzero(inside)):
            fit = _robust_plane(P)
            if fit is not None and abs(float(fit[0] @ ray)) > 1e-6:
                p_cam = ray * (-fit[1] / float(fit[0] @ ray))
            else:
                p_cam = ray * float(np.median(P[:, 2]))
            method = 'aruco_depth'
        table = fit_plane_ransac(pts[valid & ~inside])
    if p_cam is None:
        s = 0.5 * float(marker_size)
        obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
        ok, _rvec, tvec = cv2.solvePnP(obj, quad.reshape(4, 1, 2), intr.K(),
                                       np.asarray(intr.dist, dtype=np.float64).ravel(),
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return Measurement(False, 'marker found but solvePnP failed (no valid depth)',
                               marker_id=mid)
        p_cam, method = np.asarray(tvec, dtype=float).ravel(), 'aruco_pnp'
    height = None if table is None else float(table.signed_height(p_cam[None])[0])
    return Measurement(True, p_cam=p_cam, pixel=(float(center_px[0]), float(center_px[1])),
                       method=method, plane=table, height=height,
                       size=(float(marker_size), float(marker_size)), n_points=n_pts,
                       marker_id=mid, mask=inside, outline=quad)


# ------------------------------------------------------------------ averaging
def average_measurements(ms: Sequence[Measurement], max_spread: float = 0.002,
                         min_ok: Optional[int] = None) -> Measurement:
    """
    Average per-frame measurements of a still target.

    Needs min_ok good frames (default: more than half of ms). spread = largest
    distance of a frame's point from the mean; above max_spread (m) the result
    is not ok. The result keeps the last good frame's plane, mask and outline.
    """
    total = len(ms)
    good = [m for m in ms if m.ok and m.p_cam is not None]
    need = total // 2 + 1 if min_ok is None else int(min_ok)
    if not good or len(good) < need:
        reasons = [m.reason for m in ms if not m.ok and m.reason]
        why = max(reasons, key=reasons.count) if reasons else 'no frames'
        return Measurement(False, 'target found in %d of %d frames: %s' % (len(good), total, why),
                           n_frames=len(good))
    P = np.array([m.p_cam for m in good], dtype=float)
    mean = P.mean(axis=0)
    spread = float(np.max(np.linalg.norm(P - mean, axis=1)))
    pix = [m.pixel for m in good if m.pixel is not None]
    hs = [m.height for m in good if m.height is not None]
    out = dataclasses.replace(good[-1], p_cam=mean, spread=spread, n_frames=len(good),
                              pixel=tuple(np.mean(pix, axis=0).tolist()) if pix else None,
                              height=float(np.mean(hs)) if hs else None)
    if spread > max_spread:
        out.ok = False
        out.reason = ('the %d frames disagree by %.1f mm (limit %.1f mm): keep the target and '
                      'the arm still, check the light'
                      % (len(good), spread * 1e3, max_spread * 1e3))
    return out


# ------------------------------------------------------------------ solving
def pair_arrays(pairs: Sequence[PointPair], mount: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (src, dst) Nx3 for the rigid fit dst = T @ src.

    src = p_cam, dst = inv(T_base_flange_cam) @ p_base (T = T_flange_camera).
    """
    from ..core.transforms import inv_T, transform_points
    src = np.array([np.asarray(p.p_cam, dtype=float) for p in pairs]).reshape(-1, 3)
    if mount != 'eye_in_hand':
        raise ValueError('mount must be eye_in_hand (the wrist camera), got %r' % (mount,))
    rows = []
    for p in pairs:
        if p.T_base_flange_cam is None:
            raise ValueError('point #%d has no flange pose of its measurement (eye_in_hand '
                             'needs it)' % p.id)
        rows.append(transform_points(inv_T(p.T_base_flange_cam),
                                     np.asarray(p.p_base, dtype=float)))
    return src, np.array(rows).reshape(-1, 3)


def line_offset(points: Any) -> float:
    """Return the RMS distance (m) of Nx3 points from their best-fit line (0 for < 2 points)."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if P.shape[0] < 2:
        return 0.0
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return float(np.sqrt(np.sum(s[1:] ** 2) / P.shape[0]))


def check_point_geometry(src: Any, dst: Any, min_points: int = MIN_SOLVE_POINTS,
                         min_offset: float = MIN_LINE_OFFSET_M) -> Tuple[bool, str]:
    """Return (True, '') when there are enough points and they are not (nearly) collinear."""
    n = int(np.asarray(src, dtype=float).reshape(-1, 3).shape[0])
    if n < min_points:
        return False, 'need at least %d points, have %d' % (min_points, n)
    for what, P in (('camera', src), ('touched', dst)):
        off = line_offset(P)
        if off < min_offset:
            return False, ('the %s points lie almost on one line (%.1f mm off the line, need '
                           '%.0f mm): add points to the side'
                           % (what, off * 1e3, min_offset * 1e3))
    return True, ''


def solve_extrinsics(pairs: Sequence[PointPair], mount: str, min_points: int = MIN_SOLVE_POINTS,
                     min_offset: float = MIN_LINE_OFFSET_M,
                     stamp: Optional[str] = None) -> Tuple[Any, np.ndarray]:
    """
    Fit the extrinsics to the point pairs (Kabsch, core.transforms.fit_rigid).

    Returns (core.extrinsics.Extrinsics with source 'calib_points' and the
    RMS, per-point residuals in m): T_flange_camera (see pair_arrays). Raises
    ValueError with fewer than min_points pairs, (nearly) collinear points or a
    missing flange pose.
    """
    from ..core.extrinsics import Extrinsics, normalize_mount
    from ..core.transforms import fit_rigid, transform_points
    mount = normalize_mount(mount)
    src, dst = pair_arrays(pairs, mount)
    ok, why = check_point_geometry(src, dst, min_points, min_offset)
    if not ok:
        raise ValueError(why)
    T, rms = fit_rigid(src, dst)
    res = np.linalg.norm(dst - transform_points(T, src), axis=1)
    return Extrinsics(T, mount, source=TOOL, rms=rms, stamp=stamp), res


def find_outliers(residuals: Any, factor: float = OUTLIER_FACTOR,
                  min_residual: float = OUTLIER_MIN_M) -> List[int]:
    """Return the indices of residuals > factor x median and > min_residual, worst first."""
    r = np.asarray(residuals, dtype=float).ravel()
    if r.size < 3:
        return []
    med = float(np.median(r))
    idx = [i for i in range(r.size) if r[i] > factor * med and r[i] > min_residual]
    return sorted(idx, key=lambda i: -r[i])


def prediction_error(pairs: Sequence[PointPair], new: PointPair, mount: str) -> Optional[float]:
    """
    Return how far (m) a new touch is from where the previous pairs put the target.

    None until 3 previous pairs exist that are not collinear.
    """
    from ..core.transforms import transform_points
    if len(pairs) < 3:
        return None
    try:
        ext, _res = solve_extrinsics(pairs, mount, min_points=3)
        src, dst = pair_arrays([new], mount)
    except ValueError:
        return None
    return float(np.linalg.norm(dst[0] - transform_points(ext.T, src[0])))


# ------------------------------------------------------------------ sanity checks
def view_geometry(plane: TablePlane) -> Tuple[float, float]:
    """Return (angle deg between the optical axis and the table normal, camera height m)."""
    n = _unit(plane.normal)
    return math.degrees(math.acos(min(1.0, abs(float(n[2]))))), float(plane.d)


def table_in_base(T_base_camera: Any, plane: TablePlane) -> Tuple[np.ndarray, float]:
    """Return (unit normal, d) of a camera-frame plane expressed in the base frame."""
    T = np.asarray(T_base_camera, dtype=float)
    n_b = T[:3, :3] @ _unit(plane.normal)
    return n_b, float(plane.d) - float(n_b @ T[:3, 3])


def _deg_between(a: Any, b: Any) -> float:
    """Return the angle (deg) between two vectors."""
    c = float(_unit(a) @ _unit(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _table_rows(ext: Any, pairs: Sequence[PointPair]) -> List[Tuple[float, float, Any, float]]:
    """Return (view angle deg, camera height m, base normal, base d) per pair with a plane."""
    rows = []
    for p in pairs:
        pl = p.plane()
        if pl is None:
            continue
        if p.T_base_flange_cam is None:
            continue
        T_bc = np.asarray(p.T_base_flange_cam, dtype=float) @ ext.T
        ang, height = view_geometry(pl)
        n_b, d_b = table_in_base(T_bc, pl)
        rows.append((ang, height, n_b, d_b))
    return rows


def sanity_report(ext: Any, pairs: Sequence[PointPair],
                  polygon: Optional[Sequence[Sequence[float]]] = None,
                  expected_z: Optional[float] = None,
                  expected_tol: float = 0.05) -> Tuple[List[str], List[str]]:
    """
    Return (lines, warnings) of sanity checks of solved extrinsics.

    From the depth alone: angle between the optical axis and the table normal
    and the camera height above the table. Through the extrinsics: the camera
    position, the table tilt against base z (the robot stands on the table, so
    more than TABLE_TILT_WARN_DEG is suspicious) and the table z at the
    pick-area center (compared with expected_z when given).
    """
    lines: List[str] = []
    warns: List[str] = []
    lines.append('camera in the flange frame: %s mm, optical axis %.1f deg from flange z'
                 % (fmt_mm(ext.T[:3, 3]), _deg_between(ext.T[:3, 2], (0.0, 0.0, 1.0))))
    rows = _table_rows(ext, pairs)
    if not rows:
        lines.append('no table plane stored with the points: table checks skipped')
    else:
        n_b = _unit(np.mean([r[2] for r in rows], axis=0))
        d_b = float(np.mean([r[3] for r in rows]))
        tilt = _deg_between(n_b, (0.0, 0.0, 1.0))
        if polygon is not None and len(polygon) >= 3:
            ref, where = np.asarray(polygon, dtype=float).mean(axis=0), 'the pick-area center'
        else:
            ref = np.mean([np.asarray(p.p_base, dtype=float)[:2] for p in pairs], axis=0)
            where = 'the touched points'
        z = (-(n_b[0] * ref[0] + n_b[1] * ref[1] + d_b) / n_b[2]
             if abs(float(n_b[2])) > 1e-6 else float('nan'))
        lines.append('from depth only: optical axis %.1f deg from the table normal, camera '
                     '%.0f mm above the table' % (float(np.mean([r[0] for r in rows])),
                                                  float(np.mean([r[1] for r in rows])) * 1e3))
        lines.append('table in the base frame: tilt %.2f deg from base z, z = %.1f mm at %s'
                     % (tilt, z * 1e3, where))
        if tilt > TABLE_TILT_WARN_DEG:
            warns.append('the table is tilted %.1f deg in the base frame (a robot standing on '
                         'the table gives < %.0f deg): check the touches and robot.tcp_offset'
                         % (tilt, TABLE_TILT_WARN_DEG))
        if (expected_z is not None and math.isfinite(z)
                and abs(z - float(expected_z)) > float(expected_tol)):
            warns.append('table z %.1f mm differs from perception.table.expected_z %.1f mm by '
                         'more than %.0f mm' % (z * 1e3, float(expected_z) * 1e3,
                                                float(expected_tol) * 1e3))
    if ext.rms is not None and ext.rms > RMS_WARN_M:
        warns.append('RMS %.1f mm is high: drop or re-touch the worst points (d N, a), check that '
                     'the target did not move and robot.tcp_offset' % (ext.rms * 1e3))
    return lines, warns


def compare_extrinsics(new: Any, old: Any, T_flange_tcp: Optional[np.ndarray] = None
                       ) -> Tuple[Optional[float], Optional[float], str]:
    """Return (delta translation m, delta rotation deg, text) between two extrinsics."""
    from ..core.transforms import pose_error
    src = ' (%s)' % old.source if getattr(old, 'source', '') else ''
    if old.mount != new.mount:
        return None, None, ('previous extrinsics%s are for %s, not %s'
                            % (src, old.mount, new.mount))
    T_old = np.asarray(old.T, dtype=float)
    if old.parent_frame == 'tcp':
        if T_flange_tcp is None:
            return None, None, 'previous extrinsics%s are T_tcp_camera: cannot compare' % src
        T_old = np.asarray(T_flange_tcp, dtype=float) @ T_old
    dp, dr = pose_error(np.asarray(new.T, dtype=float), T_old)
    return dp, math.degrees(dr), ('change from the previous extrinsics%s: %.1f mm, %.2f deg'
                                  % (src, dp * 1e3, math.degrees(dr)))


# ------------------------------------------------------------------ text, files, previews
def measurement_line(m: Measurement) -> str:
    """Return a one-line summary of a measurement (mm)."""
    if not m.ok:
        return 'measurement FAILED: %s' % m.reason
    parts = ['camera point %s mm' % fmt_mm(m.p_cam)]
    if m.n_frames > 1:
        parts.append('%d frames, spread %.1f mm' % (m.n_frames, m.spread * 1e3))
    if m.method == 'block' and m.size is not None:
        parts.append('top face %.0f x %.0f mm' % (m.size[0] * 1e3, m.size[1] * 1e3))
    if m.marker_id is not None:
        parts.append('marker id %d' % m.marker_id)
    if m.height is not None:
        parts.append('%.1f mm above the table' % (m.height * 1e3))
    if m.method == 'aruco_pnp':
        parts.append('NO DEPTH: position from the marker corners only (less accurate)')
    return ', '.join(parts)


def residual_table(pairs: Sequence[PointPair], residuals: Any,
                   outliers: Sequence[int] = ()) -> str:
    """Return the per-point residual table and an RMS / median / max line (mm)."""
    r = np.asarray(residuals, dtype=float).ravel()
    flag = set(int(i) for i in outliers)
    rows = [[str(p.id), fmt_mm(p.p_cam), fmt_mm(p.p_base), '%.2f' % (e * 1e3),
             '%.1f' % (p.spread * 1e3), 'OUTLIER?' if i in flag else '']
            for i, (p, e) in enumerate(zip(pairs, r))]
    if not r.size:
        return format_table(RESIDUAL_HEADERS, rows)
    return (format_table(RESIDUAL_HEADERS, rows)
            + '\nRMS %.2f mm, median %.2f mm, max %.2f mm over %d points'
            % (float(np.sqrt(np.mean(r ** 2))) * 1e3, float(np.median(r)) * 1e3,
               float(r.max()) * 1e3, r.size))


def parse_command(text: Any) -> Tuple[str, Optional[int]]:
    """
    Parse an operator command line.

    '' -> ('enter', None). Otherwise the first letter picks the command (see
    COMMANDS) and an integer may follow ('d 3', 'd3', 'drop 3'). Anything
    else -> ('unknown', None).
    """
    t = str(text or '').strip().lower()
    if not t:
        return 'enter', None
    name = COMMANDS.get(t[0])
    if name is None:
        return 'unknown', None
    rest = t[1:].lstrip('abcdefghijklmnopqrstuvwxyz').strip()
    if not rest:
        return name, None
    try:
        return name, int(rest)
    except ValueError:
        return 'unknown', None


def pairs_to_json(pairs: Sequence[PointPair], mount: str, target: str,
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the session file content (tool, mount, target, pairs and extra keys)."""
    out: Dict[str, Any] = {'tool': TOOL, 'version': 1, 'mount': str(mount),
                           'target': str(target), 'pairs': [p.to_dict() for p in pairs]}
    out.update(extra or {})
    return out


def pairs_from_json(data: Any, mount: Optional[str] = None) -> List[PointPair]:
    """Return the pairs of a session file; ValueError for other files or another mount."""
    if (not isinstance(data, dict) or data.get('tool') != TOOL
            or not isinstance(data.get('pairs'), list)):
        raise ValueError('not a %s session file' % TOOL)
    if mount is not None and data.get('mount') != mount:
        raise ValueError('the session is for mount %s, camera.mount is %s'
                         % (data.get('mount'), mount))
    pairs = [PointPair.from_dict(d) for d in data['pairs']]
    ids = [p.id for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate point ids in the session file')
    if any(p.T_base_flange_cam is None for p in pairs):
        raise ValueError('session points need T_base_flange_cam (the wrist camera pose)')
    return pairs


def draw_measurement(color: np.ndarray, m: Measurement, label: str = '') -> np.ndarray:
    """Return a copy of color with the measured face / marker, its outline and center drawn."""
    import cv2
    img = np.array(color, dtype=np.uint8, copy=True)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if m.mask is not None and np.shape(m.mask) == img.shape[:2]:
        sel = np.asarray(m.mask, dtype=bool)
        img[sel] = (0.5 * img[sel] + np.array([0.0, 127.5, 0.0])).astype(np.uint8)
    if m.outline is not None and np.all(np.isfinite(m.outline)):
        pts = np.round(np.asarray(m.outline, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
    if m.pixel is not None and all(math.isfinite(v) for v in m.pixel):
        cv2.drawMarker(img, (int(round(m.pixel[0])), int(round(m.pixel[1]))), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
    if label:
        for col, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, thick,
                        cv2.LINE_AA)
    return img


def iso_now() -> str:
    """Return the local wall time as an ISO string (stamp of the calibration)."""
    return datetime.datetime.now().isoformat(timespec='seconds')


# ------------------------------------------------------------------ simulated operator
def sim_spots(polygon: Sequence[Sequence[float]],
              margin: float = 0.03) -> List[Tuple[float, float]]:
    """Return spread-out (x, y) spots (m) inside the pick polygon, corners first (sim)."""
    from ..sim.world import point_in_polygon
    poly = np.asarray(polygon, dtype=float)
    lo, hi = poly.min(axis=0) + margin, poly.max(axis=0) - margin
    out = []
    for f in SIM_GRID:
        x, y = (lo + (hi - lo) * np.asarray(f, dtype=float)).tolist()
        if point_in_polygon(x, y, poly):
            out.append((float(x), float(y)))
    return out


def sim_target(target: str, aruco_size: float = 0.04):
    """Return the simulated calibration target: a red cube or a white marker plate."""
    from ..sim.world import SimObject
    if target == 'aruco':
        side = 1.5 * float(aruco_size)          # white margin around the black square
        return SimObject('calib_marker', 'marker', 'box', (side, side, SIM_PLATE_M),
                         (235, 235, 235), np.eye(4))
    return SimObject('calib_block', 'red', 'box', (SIM_BLOCK_M,) * 3, (40, 40, 200), np.eye(4))


def sim_visible(obj, T_base_camera: Any, intr: CameraIntrinsics, margin_px: float = 10.0,
                bottom_px: float = 0.0) -> bool:
    """Return True when obj's bounding box projects inside the image (margins in pixels)."""
    from ..perception.geometry import project_points
    lo, hi = obj.aabb()
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    uv = project_points(corners, T_base_camera, intr)
    if not np.all(np.isfinite(uv)):
        return False
    return bool(uv[:, 0].min() >= margin_px and uv[:, 0].max() <= intr.width - 1 - margin_px
                and uv[:, 1].min() >= margin_px
                and uv[:, 1].max() <= intr.height - 1 - max(margin_px, bottom_px))


def sim_touch_pose(obj) -> np.ndarray:
    """Return the top-down TCP pose touching the center of obj's top face."""
    from ..core.transforms import neutral_closing_angle, top_down_T
    c = obj.center_xy()
    return top_down_T(float(c[0]), float(c[1]), obj.top_z(),
                      neutral_closing_angle(float(c[0]), float(c[1])))


def marker_corners_base(obj, size: float) -> np.ndarray:
    """Return the 4x3 base-frame corners (ArUco order TL, TR, BR, BL) of a marker on obj."""
    R = np.asarray(obj.T_base_obj, dtype=float)[:3, :3]
    c = obj.center_xy()
    ctr = np.array([c[0], c[1], obj.top_z() + 1e-4])
    ex, ey, h = R[:, 0], R[:, 1], 0.5 * float(size)
    return np.array([ctr - h * ex + h * ey, ctr + h * ex + h * ey, ctr + h * ex - h * ey,
                     ctr - h * ex - h * ey])


def paint_marker(color: np.ndarray, corners_px: Any, marker: np.ndarray) -> np.ndarray:
    """Return color with the gray marker image warped onto the pixel quad (TL, TR, BR, BL)."""
    import cv2
    s = float(marker.shape[0])
    src = np.float32([[-0.5, -0.5], [s - 0.5, -0.5], [s - 0.5, s - 0.5], [-0.5, s - 0.5]])
    H = cv2.getPerspectiveTransform(src, np.asarray(corners_px, dtype=np.float32).reshape(4, 2))
    size = (int(color.shape[1]), int(color.shape[0]))
    ink = cv2.warpPerspective(marker, H, size, flags=cv2.INTER_LINEAR).astype(np.float32)
    alpha = cv2.warpPerspective(np.full(marker.shape[:2], 255, np.uint8), H, size,
                                flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    out = (color.astype(np.float32) * (1.0 - alpha[..., None])
           + (0.9 * ink * alpha)[..., None])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class _SimOperator:
    """Simulated operator: clears the table, places the target, poses and touches with the arm."""

    def __init__(self, cal: '_Calib'):
        from ..sim.scenarios import DEFAULT_PICK_POLYGON
        cfg = cal.cfg
        self.cal, self.cfg, self.arm, self.world = cal, cfg, cal.s.arm, cal.s.world
        self.obj = sim_target(cal.args.target, cal.args.aruco_size)
        self.world.objects = [self.obj]
        self.world.held = None
        self.spots = sim_spots(cfg.get('perception.roi.pick_polygon', None)
                               or DEFAULT_PICK_POLYGON)
        self.next_spot = 0
        self.k = 0
        self.q_view = np.asarray(cfg.get('robot.poses.observe_joints', None) or [0.0] * 6,
                                 dtype=float)
        T_fc = cfg.get('sim.T_flange_camera', None)
        if T_fc is None:
            raise ValueError('sim calibration needs sim.T_flange_camera (the wrist camera)')
        self.T_fc = np.asarray(T_fc, dtype=float)
        self.intr = CameraIntrinsics.from_dict(cfg.get('sim.intrinsics'))
        self.bad_id = int(cal.args.sim_bad_touch or 0)
        self.bad_done = False
        self.marker = None
        if cal.dictionary is not None:
            self.marker = marker_image(cal.dictionary, cal.args.aruco_id or 0, 240)

    def truth(self) -> Optional[np.ndarray]:
        """Return the true camera pose the extrinsics should match (sim config)."""
        return self.T_fc

    def _view_joints(self) -> np.ndarray:
        """Return the arm pose for the next measurement (varied wrist view poses)."""
        q = self.q_view.copy()
        q2 = q + np.asarray(SIM_VIEW_OFFSETS[self.k % len(SIM_VIEW_OFFSETS)])
        if self.arm.kin.within_limits(q2):
            q = q2
        return q

    def place(self) -> bool:
        """Pose the arm, put the target on the next visible, reachable spot (False: none left)."""
        q = self._view_joints()
        sim_drag_joints(self.arm, q)
        T_bc = self.arm.kin.fk(q) @ self.T_fc
        bottom = SIM_FINGER_ROWS
        while self.next_spot < len(self.spots):
            x, y = self.spots[self.next_spot]
            self.next_spot += 1
            self.obj.place_upright(x, y, 0.35 * self.next_spot, self.world.table_z)
            if not sim_visible(self.obj, T_bc, self.intr, 10.0, bottom):
                continue
            if self.arm.kin.solve_tcp_with_tilt(sim_touch_pose(self.obj), self.cal.T_ft,
                                                max_tilt=math.radians(45.0), q_seed=q) is None:
                continue
            self.k += 1
            return True
        return False

    def where(self) -> str:
        """Return where the target is now (for the sim log)."""
        c = self.obj.center_xy()
        return 'target at %s mm, yaw %.0f deg' % (fmt_mm([c[0], c[1], self.obj.top_z()]),
                                                  math.degrees(self.obj.yaw()))

    def touch(self, point_id: int) -> bool:
        """Drag the TCP onto the target (the first touch of --sim-bad-touch is 15 mm off)."""
        T = sim_touch_pose(self.obj)
        if point_id == self.bad_id and not self.bad_done:
            self.bad_done = True
            T = T.copy()
            T[0, 3] += SIM_BAD_TOUCH_M
        return sim_drag_tcp(self.arm, self.cfg, T)

    def paint(self, frame: Frame) -> Frame:
        """Return frame with the ArUco marker drawn on the plate (target aruco)."""
        from ..perception.geometry import project_points
        T_bc = (frame.meta or {}).get('T_base_camera_true')
        if self.marker is None or T_bc is None:
            return frame
        px = project_points(marker_corners_base(self.obj, self.cal.args.aruco_size), T_bc,
                            frame.intrinsics)
        if not np.all(np.isfinite(px)):
            return frame
        return dataclasses.replace(frame, color=paint_marker(frame.color, px, self.marker))


# ------------------------------------------------------------------ main
class _Calib:
    """State of one calibration run (thin operator / camera / arm I/O around the helpers)."""

    def __init__(self, args, s, ui):
        from ..core.extrinsics import normalize_mount
        self.args, self.s, self.ui, self.cfg = args, s, ui, s.cfg
        self.mode = args.mode
        self.mount = normalize_mount(self.cfg.get('camera.mount', 'eye_in_hand'))
        self.block = BlockParams.from_cfg(self.cfg)
        self.dictionary = aruco_dictionary(args.aruco_dict) if args.target == 'aruco' else None
        self.max_spread = float(args.max_spread) / 1000.0
        self.T_ft = flange_to_tcp(self.cfg)
        self.polygon = self.cfg.get('perception.roi.pick_polygon', None)
        self.out_path = extrinsics_out_path(self.cfg, self.mode, getattr(args, 'site_dir', None))
        if self.out_path.suffix.lower() == '.npy':
            # .npy is load-only (save_extrinsics refuses it): write YAML next to it; write()
            # then prints the --set camera.extrinsics_file line for the new file.
            self.out_path = self.out_path.with_suffix('.yaml')
        self.run_dir = logging_dir(self.cfg) / TOOL / wall_stamp()
        self.session_path = self.run_dir / 'pairs.json'
        self.pairs: List[PointPair] = []
        self.next_id = 1
        self.target = int(args.points)
        self.failures = 0
        self.last_stamp: Optional[float] = None
        if args.resume:
            data = json.loads(Path(args.resume).expanduser().read_text(encoding='utf-8'))
            self.pairs = pairs_from_json(data, self.mount)
            self.next_id = max([p.id for p in self.pairs] + [0]) + 1
        self.sim = _SimOperator(self) if self.mode == 'sim' else None

    # -------------------------------------------------------------- operator text
    def intro(self) -> None:
        """Print what the run does and how to place the target."""
        a, ui = self.args, self.ui
        ui.say('%s: %s camera (camera.mount), target %s, %d points, %d frames per measurement'
               % (TOOL, self.mount, a.target, a.points, a.frames))
        ui.say('result -> %s' % self.out_path)
        ui.say('session -> %s' % self.session_path)
        if self.pairs:
            ui.say('resumed %d points from %s' % (len(self.pairs), a.resume))
        if a.points < RECOMMENDED_POINTS:
            ui.say('note: use at least %d spread-out points for a reliable fit'
                   % RECOMMENDED_POINTS)
        what = ('a printed ArUco marker (%s, %.0f mm) lying flat' % (a.aruco_dict,
                                                                     a.aruco_size * 1e3)
                if self.dictionary is not None else 'one block (8-80 mm high) alone on the table')
        ui.say('Target: %s. Spread the spots over the whole pick area (corners and center).'
               % what)
        ui.say('Before each measurement drag the arm so the camera sees the target (not at '
               'the image border) and let the arm rest still; the view may change between '
               'points.')
        if self.mode == 'real':
            ui.say('The tool never moves the arm, it only reads its pose.')
            teach_hint(ui, self.s.arm)
        if ui.headless:
            ui.say('preview image: %s' % ui.preview_path(TOOL))

    def _what(self) -> str:
        """Return the target name used in prompts."""
        return 'marker' if self.dictionary is not None else 'block'

    def place_step(self) -> str:
        """Let the operator place the target; return the command ('measure', 'quit', ...)."""
        k, ui = self.next_id, self.ui
        if self.sim is not None:
            text = 'point #%d: put the %s at a new spot, arm at a view pose' % (
                k, self._what())
            if operator_action(ui, 'sim', text, self.sim.place):
                ui.say('[sim operator] %s' % self.sim.where())
                return 'measure'
            ui.say('[sim operator] no free spot left in the pick area')
            return 'fail'
        text = ('point #%d (%d of %d): put the %s at a new spot, drag the arm so the camera '
                'sees it and let the arm rest still'
                % (k, len(self.pairs) + 1, self.target, self._what()))
        if ui.headless:
            line = ui.prompt('%s %s' % (text, HEADLESS_PLACE))
            return 'quit' if ui.eof else parse_command(line)[0]
        ui.say(text)
        ui.say(WINDOW_PLACE)
        while True:                 # live preview until a key is pressed in the window
            frames = grab_frames(self.s.camera, 1, float(self.cfg.get('camera.frame_timeout_s',
                                                                      2.0)), self.last_stamp)
            if frames:
                self.last_stamp = float(frames[0].stamp)
                shown, m = self.measure_frame(frames[0])
                self.show(shown, m, 'live: point #%d' % k, refresh=False)
            cmd = WINDOW_KEYS.get(ui.key(30) or '')
            if cmd:
                return cmd

    def wait_touch(self, text: str) -> bool:
        """Wait until the operator touched the target (sim: touch); False = cancel the point."""
        ui = self.ui
        if self.sim is not None:
            return operator_action(ui, 'sim', text, lambda: self.sim.touch(self.next_id))
        if ui.headless:
            return operator_action(ui, self.mode, text)
        ui.say('%s, then Enter or space in the preview window (q = cancel this point)' % text)
        while True:
            k = ui.key(50)
            if k in ('\n', ' '):
                return True
            if k in ('q', 'esc'):
                return False

    # -------------------------------------------------------------- camera / arm I/O
    def measure_frame(self, frame: Frame) -> Tuple[Frame, Measurement]:
        """Measure the target on one frame; return (frame as shown, measurement)."""
        if self.dictionary is None:
            return frame, measure_target_block(frame, self.block)
        if self.sim is not None:
            frame = self.sim.paint(frame)
        return frame, measure_target_aruco(frame, self.dictionary, self.args.aruco_size,
                                           self.args.aruco_id, self.block.zmin, self.block.zmax)

    def show(self, frame: Frame, m: Measurement, title: str, save_as: str = '',
             refresh: bool = True) -> None:
        """
        Show the measurement preview (headless: write it); save_as also keeps a copy.

        refresh=False leaves drawing the window to the caller's next ui.key().
        """
        img = compose_view(frame, draw_measurement(frame.color, m, title))
        img = text_panel(img, [title, measurement_line(m)])
        self.ui.show(TOOL, img)
        if refresh and not self.ui.headless:
            self.ui.key(1)
        if save_as:
            import cv2
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.run_dir / save_as), img)
            except (OSError, cv2.error) as e:
                _LOG.warning('could not save %s: %s', save_as, e)

    def measure(self) -> Tuple[Measurement, Optional[Any], Optional[Frame]]:
        """Measure the target on --frames new frames; return (measurement, arm pose, frame)."""
        s, cfg, n = self.s, self.cfg, int(self.args.frames)
        pose0, _dev = read_pose_averaged(s.arm, s.clock, n=4)
        if pose0 is None:
            return Measurement(False, 'no arm pose: the wrist camera needs the flange pose of '
                               'every measurement (CAN link?)'), None, None
        s.clock.sleep(float(cfg.get('camera.settle_s', 0.4)))
        now = s.clock.now()
        newer = now if self.last_stamp is None else max(self.last_stamp, now)
        frames = grab_frames(s.camera, n, float(cfg.get('camera.frame_timeout_s', 2.0)), newer)
        if not frames:
            fail = Measurement(False, 'no camera frame (is the camera driver running?)')
            return fail, pose0, None
        self.last_stamp = float(frames[-1].stamp)
        pose1, _dev = read_pose_averaged(s.arm, s.clock, n=4)
        results = [self.measure_frame(f) for f in frames]
        m = average_measurements([r[1] for r in results], self.max_spread, min_ok=n // 2 + 1)
        if m.ok:
            moved = self._moved(pose0, pose1)
            if moved:
                m.ok, m.reason = False, moved
        return m, pose0, results[-1][0]

    @staticmethod
    def _moved(pose0, pose1) -> str:
        """Return why the flange poses before / after a measurement differ, or ''."""
        from ..core.transforms import pose_error
        if pose1 is None:
            return 'no arm pose after the measurement (CAN link?)'
        dp, dr = pose_error(pose0.T_base_flange, pose1.T_base_flange)
        if dp > STILL_M or math.degrees(dr) > STILL_DEG:
            return ('the arm moved %.1f mm / %.2f deg while measuring: let it rest still and '
                    'measure again' % (dp * 1e3, math.degrees(dr)))
        return ''

    def touch(self, pose_m) -> Optional[Any]:
        """Wait for the touch and read the arm pose (onsite.ArmPose); None = cancelled."""
        spot = ('CENTER OF THE MARKER' if self.dictionary is not None
                else 'CENTER OF THE BLOCK TOP FACE')
        text = ('point #%d: drag the arm so the fingertip center (TCP) touches the %s (red '
                'cross in the preview) and hold it there' % (self.next_id, spot))
        for _ in range(MAX_TRIES):
            if not self.wait_touch(text):
                return None
            pose, dev = read_pose_averaged(self.s.arm, self.s.clock, n=8, dt=0.02)
            if pose is None:
                self.ui.say('no arm pose (CAN link? arm powered?): touch again')
                continue
            if dev > STILL_M:
                self.ui.say('the TCP moved %.1f mm while it was read: hold the arm still'
                            % (dev * 1e3))
                continue
            if pose_m is not None and float(np.linalg.norm(
                    pose.T_base_tcp[:3, 3] - pose_m.T_base_tcp[:3, 3])) < MIN_TOUCH_MOVE_M:
                self.ui.say('the TCP is still where it was during the measurement: touch the '
                            'target first')
                continue
            return pose
        return None

    def make_pair(self, m: Measurement, pose_m, pose) -> PointPair:
        """Return the pair of a measurement and the touch pose."""
        copy = _copy_or_none
        return PointPair(self.next_id, copy(m.p_cam), copy(pose.T_base_tcp[:3, 3]),
                         T_base_flange_cam=None if pose_m is None else copy(pose_m.T_base_flange),
                         T_base_tcp=copy(pose.T_base_tcp), T_base_flange=copy(pose.T_base_flange),
                         plane_normal=None if m.plane is None else copy(m.plane.normal),
                         plane_d=None if m.plane is None else float(m.plane.d),
                         spread=float(m.spread), method=m.method)

    def check_prediction(self, pair: PointPair) -> str:
        """Ask about a touch far from the previous points' prediction: keep / retouch / discard."""
        err = prediction_error(self.pairs, pair, self.mount)
        if err is None or err <= PREDICT_WARN_M:
            return 'keep'
        self.ui.say('WARNING: point #%d: the touch is %.1f mm from where the previous %d points '
                    'put the target. Wrong spot touched, or did the target move?'
                    % (pair.id, err * 1e3, len(self.pairs)))
        cmd = parse_command(self.ui.prompt('r = touch again, x = discard this point, '
                                           'Enter = keep it'))[0]
        return cmd if cmd in ('retouch', 'discard') else 'keep'

    def record(self, pair: PointPair) -> None:
        """Store a pair, print it with the coverage of the pick area and save the session."""
        self.pairs.append(pair)
        self.next_id = pair.id + 1
        self.failures = 0
        ui = self.ui
        ui.say('point #%d recorded: TCP %s mm (base), camera %s mm'
               % (pair.id, fmt_mm(pair.p_base), fmt_mm(pair.p_cam)))
        cov = coverage_check(np.array([p.p_base for p in self.pairs]), self.polygon,
                             min_points=MIN_SOLVE_POINTS, min_minor_rms=MIN_LINE_OFFSET_M)
        ui.say(cov['text'])
        note = ''
        if len(self.pairs) >= MIN_SOLVE_POINTS and not cov['ok']:
            note = ' (%s)' % cov['reason']
        ui.say('%d of %d points%s' % (len(self.pairs), self.target, note))
        self.save_session()

    def undo(self) -> None:
        """Remove the last recorded point."""
        if not self.pairs:
            self.ui.say('no point to undo')
            return
        p = self.pairs.pop()
        self.ui.say('removed point #%d' % p.id)
        self.save_session()

    def save_session(self, ext: Any = None) -> None:
        """Write the pairs (and the result) to the session file (pairs.json)."""
        from ..core.logger import atomic_write_text
        extra: Dict[str, Any] = {'saved': iso_now(), 'mode': self.mode,
                                 'config_sources': [str(v) for v in self.cfg.sources]}
        if ext is not None:
            extra.update(result=ext.to_dict(), output=str(self.out_path))
        try:
            atomic_write_text(self.session_path, json.dumps(
                pairs_to_json(self.pairs, self.mount, self.args.target, extra), indent=1) + '\n')
        except OSError as e:
            _LOG.warning('could not save the session %s: %s', self.session_path, e)

    # -------------------------------------------------------------- flow
    def collect_one(self) -> str:
        """Place, measure and touch one point; return 'ok', 'retry', 'solve', 'quit' or 'fail'."""
        ui = self.ui
        cmd = self.place_step()
        if cmd in ('quit', 'fail'):
            return cmd
        if cmd == 'solve':
            if len(self.pairs) >= MIN_SOLVE_POINTS:
                return 'solve'
            ui.say('need at least %d points to solve, have %d'
                   % (MIN_SOLVE_POINTS, len(self.pairs)))
            return 'retry'
        if cmd == 'undo':
            self.undo()
            return 'retry'
        if cmd not in ('enter', 'measure'):
            ui.say('unknown command: %s' % HEADLESS_PLACE)
            return 'retry'
        m, pose_m, frame = self.measure()
        title = 'point #%d' % self.next_id
        if frame is not None:
            self.show(frame, m, title, 'point_%02d.jpg' % self.next_id)
        ui.say('%s: %s' % (title, measurement_line(m)))
        if not m.ok:
            self.failures += 1
            return 'fail' if self.sim is not None and self.failures >= 3 else 'retry'
        for _ in range(MAX_TRIES):
            pose = self.touch(pose_m)
            if pose is None:
                if ui.eof:
                    return 'quit'
                ui.say('point #%d cancelled' % self.next_id)
                return 'retry'
            pair = self.make_pair(m, pose_m, pose)
            verdict = self.check_prediction(pair)
            if verdict == 'retouch':
                continue
            if verdict == 'discard':
                ui.say('point #%d discarded' % pair.id)
                return 'retry'
            self.record(pair)
            return 'ok'
        return 'retry'

    def solve(self) -> Optional[Tuple[Any, np.ndarray]]:
        """Solve, print the residuals, offer to drop flagged outliers; None when unsolvable."""
        ui, declined = self.ui, set()
        while True:
            try:
                ext, res = solve_extrinsics(self.pairs, self.mount, stamp=iso_now())
            except ValueError as e:
                ui.say('cannot solve: %s' % e)
                return None
            flagged = find_outliers(res)
            ui.say(residual_table(self.pairs, res, flagged))
            dropped = False
            for i in flagged:
                p = self.pairs[i]
                if p.id in declined:
                    continue
                if len(self.pairs) <= MIN_SOLVE_POINTS:
                    ui.say('point #%d looks like an outlier, but dropping it leaves fewer than %d '
                           'points: add points (a), then drop it (d %d)'
                           % (p.id, MIN_SOLVE_POINTS, p.id))
                    break
                if ui.confirm('point #%d: residual %.1f mm is more than %.0fx the median '
                              '(%.1f mm). Drop it and solve again?'
                              % (p.id, res[i] * 1e3, OUTLIER_FACTOR,
                                 float(np.median(res)) * 1e3), default=False):
                    self.pairs.pop(i)
                    ui.say('dropped point #%d' % p.id)
                    self.save_session()
                    dropped = True
                    break
                declined.add(p.id)
            if not dropped:
                return ext, res

    def previous(self) -> Optional[Any]:
        """Return the extrinsics in the output file, else the configured ones, or None."""
        from ..core.extrinsics import load_extrinsics
        if self.out_path.is_file():
            try:
                return load_extrinsics(self.out_path, default_mount=self.mount)
            except (ValueError, OSError) as e:
                self.ui.say('the previous file %s is unreadable: %s' % (self.out_path, e))
                return None
        return load_extrinsics_or_none(self.cfg)

    def report(self, ext: Any) -> None:
        """Print the result, sanity checks and the change from the previous extrinsics."""
        from ..core.transforms import pose_error
        ui = self.ui
        ui.say('result: %s' % ext.describe())
        lines, warns = sanity_report(ext, self.pairs, self.polygon,
                                     self.cfg.get('perception.table.expected_z', None),
                                     float(self.cfg.get('perception.table.expected_z_tol', 0.05)))
        for line in lines:
            ui.say('  ' + line)
        prev = self.previous()
        if prev is not None:
            ui.say('  ' + compare_extrinsics(ext, prev, self.T_ft)[2])
        truth = None if self.sim is None else self.sim.truth()
        if truth is not None:
            dp, dr = pose_error(ext.T, truth)
            ui.say('  [sim] error against the true camera pose: %.2f mm, %.3f deg'
                   % (dp * 1e3, math.degrees(dr)))
        for w in warns:
            ui.say('WARNING: ' + w)

    def menu(self) -> str:
        """Ask what to do with a solution; return 'write', 'add', 'resolve' or 'quit'."""
        ui = self.ui
        while True:
            cmd, arg = parse_command(ui.prompt(MENU % self.out_path))
            if ui.eof or cmd == 'quit':
                return 'quit'
            if cmd in ('write', 'add'):
                return cmd
            if cmd == 'undo':
                self.undo()
                return 'resolve'
            if cmd == 'drop':
                ids = [p.id for p in self.pairs]
                if arg not in ids:
                    ui.say('usage: d N with N one of %s' % ', '.join(str(i) for i in ids))
                    continue
                if len(self.pairs) <= MIN_SOLVE_POINTS:
                    ui.say('cannot drop: %d points must remain; add points first (a)'
                           % MIN_SOLVE_POINTS)
                    continue
                self.pairs.pop(ids.index(arg))
                ui.say('dropped point #%d' % arg)
                self.save_session()
                return 'resolve'
            ui.say('unknown command')

    def write(self, ext: Any) -> None:
        """Back up and write the extrinsics file, print the keys written and how to use it."""
        path, backup, keys = write_extrinsics_site(self.out_path, ext)
        self.ui.say(write_report(path, keys, backup))
        self.ui.say(use_hint(self.cfg, self.mode, path, 'camera.extrinsics_file'))
        self.save_session(ext)
        self.ui.say('session with all points: %s' % self.session_path)
        self.ui.say('check: view --task L1 draws the pick polygon and zones where they must lie '
                    'on the table')

    def quit_message(self) -> int:
        """Say that nothing was written (and how to resume); return exit code 1."""
        if self.pairs:
            self.save_session()
            self.ui.say('extrinsics NOT written. %d points are saved: continue with --resume %s'
                        % (len(self.pairs), self.session_path))
        else:
            self.ui.say('extrinsics NOT written')
        return 1

    def run(self) -> int:
        """Collect points, solve, check and write; return the exit code (0 = written)."""
        self.intro()
        self.target = max(int(self.args.points), len(self.pairs))
        while True:
            while len(self.pairs) < self.target:
                r = self.collect_one()
                if r in ('quit', 'fail'):
                    return self.quit_message()
                if r == 'solve':
                    break
            cov = coverage_check(np.array([p.p_base for p in self.pairs]).reshape(-1, 3),
                                 self.polygon, min_points=MIN_SOLVE_POINTS,
                                 min_minor_rms=MIN_LINE_OFFSET_M)
            if not cov['ok']:
                self.ui.say('cannot solve yet: %s' % cov['reason'])
                self.target = len(self.pairs) + 1
                continue
            solved = self.solve()
            if solved is None:
                self.target = len(self.pairs) + 1
                continue
            self.report(solved[0])
            act = self.menu()
            if act == 'write':
                self.write(solved[0])
                return 0
            if act == 'quit':
                return self.quit_message()
            self.target = len(self.pairs) + (1 if act == 'add' else 0)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser (common + on-site options)."""
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__.split('\n\n')[0],
                                epilog=__doc__.split('\n\n', 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_onsite_args(p, site=True)
    p.add_argument('--target', choices=TARGETS, default='block',
                   help='block: one block on the table, touch its top-face center; aruco: a '
                        'flat ArUco marker, touch its center')
    p.add_argument('--aruco-dict', default='DICT_4X4_50', help='ArUco dictionary (cv2.aruco)')
    p.add_argument('--aruco-size', type=float, default=0.04, metavar='M',
                   help='side of the black marker square (m), used when the depth is invalid')
    p.add_argument('--aruco-id', type=int, default=None,
                   help='use only this marker id (default: the largest marker in view)')
    p.add_argument('--points', type=int, default=6,
                   help='touch points to collect before solving (>= 4, 5 or more recommended)')
    p.add_argument('--frames', type=int, default=5, help='frames averaged per measurement')
    p.add_argument('--max-spread', type=float, default=2.0, metavar='MM',
                   help='reject a measurement whose frames disagree by more than this (mm)')
    p.add_argument('--resume', default=None, metavar='PAIRS_JSON',
                   help='continue a session from its pairs.json')
    p.add_argument('--sim-bad-touch', type=int, default=0, metavar='N',
                   help='sim: the first touch of point N is 15 mm off (rehearse the checks)')
    return p


def check_args(args: argparse.Namespace) -> str:
    """Return why the arguments are unusable, or ''."""
    if args.mode == 'replay':
        return 'needs live touches: use --mode real or --mode sim'
    if args.points < MIN_SOLVE_POINTS:
        return '--points must be >= %d' % MIN_SOLVE_POINTS
    if args.frames < 1:
        return '--frames must be >= 1'
    if not (args.max_spread > 0 and math.isfinite(args.max_spread)):
        return '--max-spread must be > 0 (mm)'
    if not (0.005 <= args.aruco_size <= 0.5):
        return '--aruco-size is the marker side in meters (e.g. 0.04)'
    if args.sim_bad_touch and args.mode != 'sim':
        return '--sim-bad-touch is for --mode sim only'
    return ''


def main(argv=None, ui=None) -> int:
    """Run the point-pair calibration; return the exit code (0 = extrinsics written)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    problem = check_args(args)
    if problem:
        parser.error(problem)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    ui = make_ui(args, cfg_world_info[0], TOOL, ui)
    try:
        with open_session(args, arm=True, camera=True, perception=False,
                          connect=args.mode != 'real', cfg_world_info=cfg_world_info) as s:
            if not connect_arm_readonly(s, ui, required=True, why='touch poses'):
                return 2
            try:
                cal = _Calib(args, s, ui)
            except (ValueError, OSError) as e:
                ui.say('%s: %s' % (TOOL, e))
                return 2
            code = cal.run()
            if args.mode == 'real':
                teach_mode_reminder(ui, s.arm)
            return code
    finally:
        ui.close()


if __name__ == '__main__':
    raise SystemExit(main())

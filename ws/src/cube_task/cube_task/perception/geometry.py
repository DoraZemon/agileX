"""
Pixel and point geometry used by the perception pipeline.

Frames: ``camera`` is the color optical frame (x right, y down, z forward) and
``base`` the robot base frame (z up). Points and depth are in meters, image
coordinates in pixels with the OpenCV convention (pixel (u, v) has its center
at integer u, v). Depth images are HxW float meters aligned to the color
image; 0, NaN and inf mean invalid. ``T_base_camera`` maps camera points into
the base frame: ``p_base = T_base_camera @ [p_cam, 1]``.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ..core.types import CameraIntrinsics, Zone

DEFAULT_ZMIN = 0.15
DEFAULT_ZMAX = 3.0

_GRID_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
_GRID_CACHE_MAX = 8
_GRID_LOCK = threading.Lock()


def has_distortion(intrinsics: CameraIntrinsics) -> bool:
    """Return True when the intrinsics carry non-zero distortion coefficients."""
    d = np.asarray(intrinsics.dist, dtype=float).ravel()
    return bool(d.size > 0 and np.any(np.abs(d) > 0.0))


def _dist_array(intrinsics: CameraIntrinsics) -> np.ndarray:
    d = np.asarray(intrinsics.dist, dtype=np.float64).ravel()
    if d.size not in (4, 5, 8, 12, 14):
        raise ValueError('distortion needs 4, 5, 8, 12 or 14 coefficients, got %d' % d.size)
    if not np.all(np.isfinite(d)):
        raise ValueError('distortion coefficients must be finite')
    return d


def check_intrinsics(intrinsics: CameraIntrinsics) -> None:
    """Raise ValueError when focal lengths, center or image size are unusable."""
    vals = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError('intrinsics must be finite')
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError('focal lengths must be positive')
    if int(intrinsics.width) <= 0 or int(intrinsics.height) <= 0:
        raise ValueError('image size must be positive')
    if has_distortion(intrinsics):
        _dist_array(intrinsics)


def _intrinsics_key(intrinsics: CameraIntrinsics) -> tuple:
    d = tuple(float(v) for v in np.asarray(intrinsics.dist, dtype=float).ravel())
    return (float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx),
            float(intrinsics.cy), int(intrinsics.width), int(intrinsics.height), d)


def normalized_grid(intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return undistorted normalized image coordinates (x/z, y/z) of every pixel.

    The two HxW float32 arrays are cached per intrinsics and read-only. With
    non-zero distortion the pixel grid is undistorted once with OpenCV's
    iterative undistortion, so ``(x, y, 1) * depth`` is the camera point seen
    by each pixel.
    """
    check_intrinsics(intrinsics)
    key = _intrinsics_key(intrinsics)
    with _GRID_LOCK:
        hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    w, h = int(intrinsics.width), int(intrinsics.height)
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    if has_distortion(intrinsics):
        uu, vv = np.meshgrid(u, v)
        pix = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        und = cv2.undistortPointsIter(pix, intrinsics.K(), _dist_array(intrinsics),
                                      None, None, crit)
        und = und.reshape(h, w, 2)
        xn = und[..., 0].astype(np.float32)
        yn = und[..., 1].astype(np.float32)
    else:
        xn = np.broadcast_to(((u - intrinsics.cx) / intrinsics.fx).astype(np.float32),
                             (h, w)).copy()
        yn = np.broadcast_to(((v - intrinsics.cy) / intrinsics.fy).astype(np.float32)[:, None],
                             (h, w)).copy()
    xn.setflags(write=False)
    yn.setflags(write=False)
    with _GRID_LOCK:
        if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
            _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
        _GRID_CACHE[key] = (xn, yn)
    return xn, yn


def check_depth(depth: np.ndarray, intrinsics: Optional[CameraIntrinsics] = None) -> None:
    """Raise ValueError unless depth is an HxW float image matching the intrinsics."""
    if depth is None:
        raise ValueError('depth is None')
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError('depth must be HxW, got shape %s' % (depth.shape,))
    if depth.dtype.kind != 'f':
        raise ValueError('depth must be float meters, got dtype %s' % depth.dtype)
    if intrinsics is not None:
        if depth.shape != (int(intrinsics.height), int(intrinsics.width)):
            raise ValueError('depth shape %s does not match intrinsics %dx%d'
                             % (depth.shape, intrinsics.width, intrinsics.height))


def valid_depth_mask(depth: np.ndarray, zmin: float = DEFAULT_ZMIN,
                     zmax: float = DEFAULT_ZMAX) -> np.ndarray:
    """
    Return the HxW bool mask of usable depth pixels.

    A pixel is valid when its depth (meters) is finite and zmin <= z <= zmax.
    """
    check_depth(depth)
    d = np.asarray(depth)
    with np.errstate(invalid='ignore'):
        return np.isfinite(d) & (d >= float(zmin)) & (d <= float(zmax))


def backproject(depth: np.ndarray, intrinsics: CameraIntrinsics,
                zmin: float = DEFAULT_ZMIN, zmax: float = DEFAULT_ZMAX) -> np.ndarray:
    """
    Back-project a depth image into camera-frame points.

    depth: HxW float meters (z along the optical axis), aligned to the image
    described by intrinsics. Returns HxWx3 float32 camera points (meters) with
    NaN in all three channels where the depth is invalid (see valid_depth_mask).
    """
    check_depth(depth, intrinsics)
    xn, yn = normalized_grid(intrinsics)
    valid = valid_depth_mask(depth, zmin, zmax)
    z = np.where(valid, depth, np.nan).astype(np.float32)
    pts = np.empty(z.shape + (3,), dtype=np.float32)
    np.multiply(xn, z, out=pts[..., 0])
    np.multiply(yn, z, out=pts[..., 1])
    pts[..., 2] = z
    return pts


def transform_cloud(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply the 4x4 transform T to points of shape (..., 3).

    Returns float32 points of the same shape. NaN points stay NaN. Use it as
    ``transform_cloud(points_cam, T_base_camera)`` to get base-frame points.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError('T must be a finite 4x4 matrix')
    p = np.asarray(points, dtype=np.float32)
    if p.shape[-1] != 3:
        raise ValueError('points must have shape (..., 3)')
    flat = p.reshape(-1, 3)
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    out = flat @ R.T
    out += t
    return out.reshape(p.shape)


def project_points(points_base: np.ndarray, T_base_camera: np.ndarray,
                   intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Project base-frame points (Nx3 or (3,), meters) into the image.

    Returns Nx2 float64 pixel coordinates (u, v). Uses cv2.projectPoints when
    the intrinsics have non-zero distortion. Points that are not finite or not
    in front of the camera (camera z <= 1 mm) get NaN.
    """
    check_intrinsics(intrinsics)
    p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    T_cam_base = np.linalg.inv(np.asarray(T_base_camera, dtype=np.float64))
    pc = p @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    out = np.full((p.shape[0], 2), np.nan)
    ok = np.all(np.isfinite(pc), axis=1) & (pc[:, 2] > 1e-3)
    if not np.any(ok):
        return out
    q = pc[ok]
    if has_distortion(intrinsics):
        uv, _ = cv2.projectPoints(q.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  intrinsics.K(), _dist_array(intrinsics))
        out[ok] = uv.reshape(-1, 2)
    else:
        out[ok, 0] = intrinsics.fx * q[:, 0] / q[:, 2] + intrinsics.cx
        out[ok, 1] = intrinsics.fy * q[:, 1] / q[:, 2] + intrinsics.cy
    return out


def parse_polygon(value: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """
    Validate a base-XY polygon from the config (list of [x, y], meters).

    Returns an Mx2 float64 array, or None when value is None or empty.
    Raises ValueError for fewer than three vertices, non-finite numbers or a
    polygon with (near) zero area.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError('polygon must be a list of at least 3 [x, y] points')
    if not np.all(np.isfinite(arr)):
        raise ValueError('polygon coordinates must be finite')
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 1e-6:
        raise ValueError('polygon area is zero')
    return arr


def _segment_distance(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    ap = xy - a
    if denom <= 0.0:
        return np.hypot(ap[:, 0], ap[:, 1])
    t = np.clip((ap @ ab) / denom, 0.0, 1.0)
    dx = ap[:, 0] - t * ab[0]
    dy = ap[:, 1] - t * ab[1]
    return np.hypot(dx, dy)


def points_in_polygon(xy: np.ndarray, polygon: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Test which base-XY points lie inside a polygon (vectorized even-odd rule).

    xy: Nx2 meters; polygon: Mx2 meters (any vertex order, simple polygon).
    margin > 0 also accepts points within margin of the boundary (polygon
    grown by margin); margin < 0 only accepts points at least -margin inside.
    Non-finite points are outside. Returns an N bool array.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=np.float64)
    n = xy.shape[0]
    finite = np.all(np.isfinite(xy), axis=1)
    inside = np.zeros(n, dtype=bool)
    if n == 0:
        return inside
    x, y = xy[:, 0], xy[:, 1]
    m = poly.shape[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(m):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % m]
            if y0 == y1:
                continue
            crosses = (y0 > y) != (y1 > y)
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < xint)
    if margin != 0.0:
        dmin = np.full(n, np.inf)
        for i in range(m):
            dmin = np.minimum(dmin, _segment_distance(xy, poly[i], poly[(i + 1) % m]))
        if margin > 0.0:
            inside |= dmin <= margin
        else:
            inside &= dmin >= -margin
    return inside & finite


def as_zone_list(zones: Union[None, Mapping[str, Zone], Iterable[Zone]]) -> list:
    """Return zones (dict name->Zone, iterable of Zone or None) as a list."""
    if zones is None:
        return []
    if isinstance(zones, Mapping):
        return list(zones.values())
    return list(zones)


def points_in_zones(xy: np.ndarray, zones: Union[None, Mapping[str, Zone], Iterable[Zone]],
                    margin: float = 0.0) -> np.ndarray:
    """
    Test which base-XY points (Nx2, meters) lie inside any zone grown by margin.

    Same test as Zone.contains_xy, vectorized. Returns an N bool array.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.zeros(xy.shape[0], dtype=bool)
    for z in as_zone_list(zones):
        c, s = np.cos(-z.yaw), np.sin(-z.yaw)
        dx = xy[:, 0] - float(z.center[0])
        dy = xy[:, 1] - float(z.center[1])
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        with np.errstate(invalid='ignore'):
            out |= ((np.abs(lx) <= z.size[0] / 2.0 + margin)
                    & (np.abs(ly) <= z.size[1] / 2.0 + margin))
    return out


def densify_polyline(xy: np.ndarray, step: float = 0.01, closed: bool = True) -> np.ndarray:
    """Insert points every ``step`` meters along a polyline (for projection)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return pts
    seq = np.vstack([pts, pts[:1]]) if closed else pts
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / max(step, 1e-4))))
        t = np.arange(n, dtype=np.float64)[:, None] / n
        out.append(a + t * (b - a))
    out.append(seq[-1:])
    return np.vstack(out)

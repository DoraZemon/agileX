"""
Simulated tabletop world: objects, place zones and a parallel gripper.

All poses are 4x4 matrices in the robot base frame (z up, meters, radians).
Object frames have their origin at the bottom center of the object with z up.
The gripper model matches core conventions: tool z is the approach direction
(points down for a top-down grasp) and the fingers close along tool y. The
TCP is the fingertip center, so the fingers extend from the TCP upwards
(along -tool z) by FINGER_SIZE[2].

The world is pure geometry, no dynamics: a grasped object follows the TCP
rigidly and a released object drops straight down onto a zone floor or the
table. SimArm calls try_grasp / update_tcp / release; SimCamera renders it.
"""
from __future__ import annotations

import copy
import logging
import math
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import inv_T, is_rotation, make_T, rot_z
from ..core.types import Zone

_log = logging.getLogger('cube_task.sim.world')

SHAPES = ('box', 'sphere', 'ellipsoid', 'cylinder', 'capsule_chain', 'sphere_cluster')

# Gripper geometry, in the TCP frame (m). Finger box: (tool x = pad width,
# tool y = thickness, tool z = length). The inner finger faces are at
# +-width / 2 along tool y; fingers span 0 .. FINGER_SIZE[2] above the TCP.
FINGER_SIZE = (0.020, 0.012, 0.050)
# Palm / gripper body box (tool x, tool y, tool z), spanning PALM_BOTTOM ..
# PALM_BOTTOM + PALM_SIZE[2] above the TCP (top = flange for a 0.14 m TCP).
PALM_SIZE = (0.045, 0.090, 0.090)
PALM_BOTTOM = 0.050
GRIPPER_COLOR_BGR = (45, 45, 50)
FINGER_COLOR_BGR = (70, 70, 75)

# Default grasp tolerances (see SimWorld.try_grasp).
GRASP_TOL_X = 0.008          # m, TCP offset from the footprint center along tool x
GRASP_TOL_Y_FRAC = 0.3       # fraction of the footprint half extent along tool y

WALL_THICKNESS = 0.004       # m, walls of binned zones, outside the zone interior
SAMPLE_SPACING = 0.0025      # m, surface sample spacing used for grasp geometry


def _finite_array(value: Any, shape: Tuple[int, ...], what: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != shape:
        raise ValueError('%s: expected shape %s, got %s' % (what, shape, a.shape))
    if not np.all(np.isfinite(a)):
        raise ValueError('%s: contains NaN or inf' % what)
    return a


def check_pose(T: Any, what: str = 'pose') -> np.ndarray:
    """Return T as a validated 4x4 float rigid transform (raise ValueError)."""
    a = _finite_array(T, (4, 4), what)
    if not is_rotation(a[:3, :3], tol=1e-3) or not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError('%s: not a rigid transform' % what)
    return a.copy()


def _check_color(c: Sequence[Any], what: str) -> Tuple[int, int, int]:
    vals = [int(round(float(v))) for v in c]
    if len(vals) != 3 or min(vals) < 0 or max(vals) > 255:
        raise ValueError('%s: color must be 3 values in 0..255, got %r' % (what, c))
    return vals[0], vals[1], vals[2]


def _positive(v: Any, what: str, max_value: float = 2.0) -> float:
    f = float(v)
    if not math.isfinite(f) or f <= 0.0 or f > max_value:
        raise ValueError('%s must be in (0, %g], got %r' % (what, max_value, v))
    return f


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n roughly uniform unit vectors (n x 3)."""
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])


def densify_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Return points along the polyline with consecutive spacing <= spacing."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return pts.copy()
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / spacing)))
        s = (np.arange(1, n + 1, dtype=float) / n)[:, None]
        out.append(a + s * (b - a))
    return np.vstack(out)


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd test of (x, y) against a polygon given as [[x, y], ...] (m)."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError('polygon needs >= 3 [x, y] vertices')
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


class SimObject:
    """
    Rigid object on the table.

    shape / size (m):
      'box' (sx, sy, sz); 'sphere' (r,); 'ellipsoid' (rx, ry, rz);
      'cylinder' (r, h) with a vertical axis; 'capsule_chain' (r, [(x, y, z), ...])
      rendered as spheres of radius r placed every <= r / 2 along the local
      polyline; 'sphere_cluster' (r, [(x, y, z), ...]) spheres at the given
      local centers (e.g. a grape bunch).
    T_base_obj: object pose in the base frame; origin at the bottom center
    for the primitive shapes. For point-based shapes the local points define
    the geometry; scenarios put the bottom at local z = 0.
    color_bgr: base color (0..255, OpenCV channel order).
    """

    def __init__(self, name: str, label: str, shape: str, size: Sequence[Any],
                 color_bgr: Sequence[Any], T_base_obj: Any):
        if not name:
            raise ValueError('SimObject needs a name')
        if shape not in SHAPES:
            raise ValueError('unknown shape %r (expected one of %s)' % (shape, SHAPES))
        self.name = str(name)
        self.label = str(label)
        self._shape = shape
        self._size = self._normalize_size(shape, size)
        self.color_bgr = _check_color(color_bgr, 'object %s' % name)
        self._T = check_pose(T_base_obj, 'object %s T_base_obj' % name)
        self.texture_seed = zlib.crc32(self.name.encode('utf-8')) & 0xFFFF
        self._centers = self._sphere_centers()
        self._local_samples: Optional[np.ndarray] = None

    # ----------------------------------------------------------- definition
    @staticmethod
    def _normalize_size(shape: str, size: Sequence[Any]) -> tuple:
        what = 'size of %s' % shape
        if shape in ('capsule_chain', 'sphere_cluster'):
            if len(size) != 2:
                raise ValueError('%s: expected (r, [points])' % what)
            r = _positive(size[0], what + ' radius', 0.5)
            pts = _finite_array(size[1], (len(size[1]), 3), what + ' points')
            if len(pts) < 1:
                raise ValueError('%s: needs at least one point' % what)
            return (r, tuple(tuple(float(v) for v in p) for p in pts))
        n = {'box': 3, 'sphere': 1, 'ellipsoid': 3, 'cylinder': 2}[shape]
        if len(size) != n:
            raise ValueError('%s: expected %d values, got %r' % (what, n, size))
        return tuple(_positive(v, what) for v in size)

    def _sphere_centers(self) -> Optional[np.ndarray]:
        if self._shape == 'capsule_chain':
            r = self._size[0]
            return densify_polyline(np.array(self._size[1]), r / 2.0)
        if self._shape == 'sphere_cluster':
            return np.array(self._size[1], dtype=float)
        return None

    @property
    def shape(self) -> str:
        """Return the shape name (read-only)."""
        return self._shape

    @property
    def size(self) -> tuple:
        """Return the normalized size tuple (read-only)."""
        return self._size

    @property
    def T_base_obj(self) -> np.ndarray:
        """Return the object pose in the base frame (a copy)."""
        return self._T.copy()

    @T_base_obj.setter
    def T_base_obj(self, T: Any) -> None:
        self._T = check_pose(T, 'object %s T_base_obj' % self.name)

    def primitives(self) -> List[Tuple[Any, ...]]:
        """
        Local-frame primitives for rendering.

        Returns a list of ('box', center(3), half(3)), ('ellipsoid', center(3),
        radii(3)), ('cylinder', r, h) (axis = local z, bottom at z = 0) or
        ('spheres', r, centers Nx3).
        """
        s = self._size
        if self._shape == 'box':
            return [('box', np.array([0.0, 0.0, s[2] / 2.0]), np.array(s) / 2.0)]
        if self._shape == 'sphere':
            return [('ellipsoid', np.array([0.0, 0.0, s[0]]), np.array([s[0]] * 3))]
        if self._shape == 'ellipsoid':
            return [('ellipsoid', np.array([0.0, 0.0, s[2]]), np.array(s))]
        if self._shape == 'cylinder':
            return [('cylinder', s[0], s[1])]
        return [('spheres', s[0], self._centers.copy())]

    def local_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds (lo, hi) in the object frame (m)."""
        s = self._size
        if self._shape == 'box':
            return (np.array([-s[0] / 2, -s[1] / 2, 0.0]),
                    np.array([s[0] / 2, s[1] / 2, s[2]]))
        if self._shape == 'sphere':
            return np.array([-s[0], -s[0], 0.0]), np.array([s[0], s[0], 2 * s[0]])
        if self._shape == 'ellipsoid':
            return np.array([-s[0], -s[1], 0.0]), np.array([s[0], s[1], 2 * s[2]])
        if self._shape == 'cylinder':
            return np.array([-s[0], -s[0], 0.0]), np.array([s[0], s[0], s[1]])
        r = s[0]
        return self._centers.min(axis=0) - r, self._centers.max(axis=0) + r

    # ------------------------------------------------------------- geometry
    def support(self, direction: Sequence[float]) -> float:
        """Return max over the object of p . u (base frame, m); u is normalized."""
        u = np.asarray(direction, dtype=float).reshape(3)
        nrm = float(np.linalg.norm(u))
        if not math.isfinite(nrm) or nrm < 1e-12:
            raise ValueError('support direction must be a non-zero finite vector')
        u = u / nrm
        R = self._T[:3, :3]
        o = self._T[:3, 3]
        ul = R.T @ u
        s = self._size
        if self._shape == 'box':
            c = o + R @ np.array([0.0, 0.0, s[2] / 2.0])
            return float(c @ u + np.sum(np.abs(ul) * np.array(s) / 2.0))
        if self._shape in ('sphere', 'ellipsoid'):
            radii = np.array([s[0]] * 3) if self._shape == 'sphere' else np.array(s)
            c = o + R @ np.array([0.0, 0.0, radii[2]])
            return float(c @ u + np.linalg.norm(radii * ul))
        if self._shape == 'cylinder':
            r, h = s
            c = o + R @ np.array([0.0, 0.0, h / 2.0])
            az = abs(float(ul[2]))
            return float(c @ u + h / 2.0 * az + r * math.sqrt(max(0.0, 1.0 - az * az)))
        return float(np.max(self._centers @ ul) + o @ u + s[0])

    def extent_along(self, axis: Sequence[float]) -> Tuple[float, float]:
        """
        Return (lo, hi) of the object's projection on a base-frame axis (m).

        axis: 3-vector, or a 2-vector meaning the horizontal axis (x, y, 0).
        """
        a = np.asarray(axis, dtype=float).ravel()
        if a.size == 2:
            a = np.array([a[0], a[1], 0.0])
        a = a / max(float(np.linalg.norm(a)), 1e-12)
        return -self.support(-a), self.support(a)

    def footprint_extent(self, axis_xy: Sequence[float]) -> float:
        """Return the width (m) of the object along a horizontal base axis."""
        lo, hi = self.extent_along(axis_xy[:2])
        return hi - lo

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Base-frame axis-aligned bounds (lo, hi), exact for every shape (m)."""
        R = self._T[:3, :3]
        o = self._T[:3, 3]
        s = self._size
        if self._shape == 'box':
            c = o + R[:, 2] * (s[2] / 2.0)
            e = np.abs(R) @ (np.array(s) / 2.0)
        elif self._shape in ('sphere', 'ellipsoid'):
            radii = np.array([s[0]] * 3) if self._shape == 'sphere' else np.array(s)
            c = o + R[:, 2] * radii[2]
            e = np.sqrt((R * R) @ (radii * radii))
        elif self._shape == 'cylinder':
            r, h = s
            c = o + R[:, 2] * (h / 2.0)
            az = np.abs(R[:, 2])
            e = h / 2.0 * az + r * np.sqrt(np.maximum(0.0, 1.0 - az * az))
        else:
            pts = self._centers @ R.T + o
            return pts.min(axis=0) - s[0], pts.max(axis=0) + s[0]
        return c - e, c + e

    def bounding_sphere(self) -> Tuple[np.ndarray, float]:
        """Return a conservative (center (3,), radius) in the base frame (m)."""
        lo, hi = self.aabb()
        return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo) / 2.0)

    def top_z(self) -> float:
        """Return the highest base z of the object (m)."""
        return self.support((0.0, 0.0, 1.0))

    def bottom_z(self) -> float:
        """Return the lowest base z of the object (m)."""
        return -self.support((0.0, 0.0, -1.0))

    def height(self) -> float:
        """Return the vertical extent of the object (m)."""
        return self.top_z() - self.bottom_z()

    def center_xy(self) -> np.ndarray:
        """Return the footprint (base AABB) center in base XY (m)."""
        lo, hi = self.aabb()
        return (lo[:2] + hi[:2]) / 2.0

    def yaw(self) -> float:
        """Return the base-frame yaw of the object x axis (rad)."""
        return float(math.atan2(self._T[1, 0], self._T[0, 0]))

    def rest_on(self, z_floor: float) -> None:
        """Translate vertically so the lowest point is at z_floor (m)."""
        T = self._T.copy()
        T[2, 3] += float(z_floor) - self.bottom_z()
        self._T = T

    def place_upright(self, x: float, y: float, yaw: float, z_floor: float) -> None:
        """Pose the object upright: yaw (rad), footprint center (x, y), resting on z_floor (m)."""
        vals = [float(v) for v in (x, y, yaw, z_floor)]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('place_upright: non-finite argument %r' % (vals,))
        self._T = make_T(rot_z(vals[2]), (vals[0], vals[1], 0.0))
        c = self.center_xy()
        self._T[0, 3] += vals[0] - c[0]
        self._T[1, 3] += vals[1] - c[1]
        self.rest_on(vals[3])

    def local_surface_points(self) -> np.ndarray:
        """Return surface samples in the object frame (N x 3, spacing ~2.5 mm)."""
        if self._local_samples is None:
            self._local_samples = self._make_samples()
        return self._local_samples

    def surface_points(self) -> np.ndarray:
        """Return surface samples in the base frame (N x 3, m)."""
        p = self.local_surface_points()
        return p @ self._T[:3, :3].T + self._T[:3, 3]

    def _make_samples(self) -> np.ndarray:
        s = self._size
        sp = SAMPLE_SPACING
        if self._shape == 'box':
            return _box_surface(np.array(s) / 2.0, sp) + np.array([0.0, 0.0, s[2] / 2.0])
        if self._shape in ('sphere', 'ellipsoid'):
            radii = np.array([s[0]] * 3) if self._shape == 'sphere' else np.array(s)
            n = int(min(20000, max(400, 4.0 * math.pi * float(np.max(radii)) ** 2 / sp ** 2 * 2)))
            return fibonacci_sphere(n) * radii + np.array([0.0, 0.0, radii[2]])
        if self._shape == 'cylinder':
            r, h = s
            na = max(48, int(math.ceil(2 * math.pi * r / sp)))
            nz = max(2, int(math.ceil(h / sp)) + 1)
            a = np.linspace(0.0, 2 * math.pi, na, endpoint=False)
            zz = np.linspace(0.0, h, nz)
            A, Z = np.meshgrid(a, zz)
            side = np.column_stack([r * np.cos(A).ravel(), r * np.sin(A).ravel(), Z.ravel()])
            g = np.arange(-r, r + 1e-9, sp)
            X, Y = np.meshgrid(g, g)
            keep = X ** 2 + Y ** 2 <= r * r
            disk = np.column_stack([X[keep], Y[keep]])
            caps = [np.column_stack([disk, np.full(len(disk), zc)]) for zc in (0.0, h)]
            return np.vstack([side] + caps)
        r = s[0]
        n = int(max(60, 4.0 * math.pi * r * r / sp ** 2 * 2))
        unit = fibonacci_sphere(n) * r
        return (self._centers[:, None, :] + unit[None, :, :]).reshape(-1, 3)

    # --------------------------------------------------------------- misc
    def copy(self) -> 'SimObject':
        """Return a deep copy."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary (pose as x, y, bottom z, yaw)."""
        c = self.center_xy()
        return {'name': self.name, 'label': self.label, 'shape': self._shape,
                'center_xy': [round(float(c[0]), 4), round(float(c[1]), 4)],
                'bottom_z': round(self.bottom_z(), 4), 'top_z': round(self.top_z(), 4),
                'yaw': round(self.yaw(), 4)}


def _box_surface(half: np.ndarray, spacing: float) -> np.ndarray:
    """Return grid samples on the 6 faces of a centered box (edges included)."""
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / spacing)) + 1)) for h in half]
    pts = []
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        A, B = np.meshgrid(axes[i], axes[j])
        for sgn in (-1.0, 1.0):
            P = np.zeros((A.size, 3))
            P[:, i] = A.ravel()
            P[:, j] = B.ravel()
            P[:, k] = sgn * half[k]
            pts.append(P)
    return np.vstack(pts)


class SimZone:
    """
    Place zone: a flat colored mat (wall_height 0) or a walled bin.

    center: (x, y, floor_z) in the base frame (m); size: (sx, sy) along the
    zone's own x / y (m); yaw (rad); wall_height (m) above the floor. Walls
    are WALL_THICKNESS thick and stand outside the size rectangle.
    """

    def __init__(self, name: str, center: Sequence[float], size: Sequence[float],
                 yaw: float = 0.0, color_bgr: Sequence[Any] = (90, 90, 90),
                 wall_height: float = 0.0, label: str = ''):
        if not name:
            raise ValueError('SimZone needs a name')
        self.name = str(name)
        c = [float(v) for v in center]
        if len(c) == 2:
            c.append(0.0)
        self.center = _finite_array(c, (3,), 'zone %s center' % name)
        sz = [float(v) for v in size]
        if len(sz) != 2:
            raise ValueError('zone %s: size needs 2 values' % name)
        self.size = (_positive(sz[0], 'zone size x'), _positive(sz[1], 'zone size y'))
        self.yaw = float(yaw)
        wh = float(wall_height)
        if not math.isfinite(self.yaw) or not math.isfinite(wh) or wh < 0.0:
            raise ValueError('zone %s: bad yaw / wall_height' % name)
        self.wall_height = wh
        self.color_bgr = _check_color(color_bgr, 'zone %s' % name)
        self.label = str(label)

    @property
    def floor_z(self) -> float:
        """Return the base z of the zone floor (m)."""
        return float(self.center[2])

    def to_zone(self) -> Zone:
        """Return the equivalent core Zone."""
        return Zone(name=self.name, center=self.center.copy(), size=tuple(self.size),
                    yaw=self.yaw, wall_height=self.wall_height, label=self.label)

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check containment with core Zone.contains_xy semantics."""
        return self.to_zone().contains_xy(x, y, margin)

    def corners_xy(self, margin: float = 0.0) -> np.ndarray:
        """Return 4x2 corners in base XY (counter-clockwise)."""
        return self.to_zone().corners_xy(margin)

    def to_config(self) -> Dict[str, Any]:
        """Return the zones.<name> config mapping (core.zones.parse_zone format)."""
        return {'center': [float(v) for v in self.center],
                'size': [float(v) for v in self.size],
                'yaw': float(self.yaw), 'wall_height': float(self.wall_height),
                'label': self.label}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary."""
        d = self.to_config()
        d['name'] = self.name
        return d

    def copy(self) -> 'SimZone':
        """Return a deep copy."""
        return copy.deepcopy(self)


def gripper_boxes(T_base_tcp: np.ndarray, width: float) -> List[Tuple[np.ndarray, np.ndarray,
                                                                      Tuple[int, int, int]]]:
    """
    Return the gripper as oriented boxes [(T_base_boxcenter, half sizes, color_bgr)].

    width: finger opening (m), inner face to inner face along tool y.
    """
    T = np.asarray(T_base_tcp, dtype=float)
    w = max(0.0, float(width))
    fx, fy, fl = FINGER_SIZE
    out = []
    for sgn in (-1.0, 1.0):
        off = make_T(np.eye(3), (0.0, sgn * (w / 2.0 + fy / 2.0), -fl / 2.0))
        out.append((T @ off, np.array(FINGER_SIZE) / 2.0, FINGER_COLOR_BGR))
    palm = make_T(np.eye(3), (0.0, 0.0, -(PALM_BOTTOM + PALM_SIZE[2] / 2.0)))
    out.append((T @ palm, np.array(PALM_SIZE) / 2.0, GRIPPER_COLOR_BGR))
    return out


class SimWorld:
    """
    Table, objects, zones and the gripper state.

    table_z: base z of the table top (m). Objects keep their identity across
    grasps; .held is the object in the gripper (or None). .tcp_pose /
    .gripper_width are the last values given to update_tcp (None before), used
    to draw the gripper. .last_grasp / .last_release describe the last attempt
    (keys ok, reason, object, ...) for logs and tests.
    """

    def __init__(self, table_z: float = 0.0, objects: Optional[List[SimObject]] = None,
                 zones: Optional[List[SimZone]] = None,
                 table_color_bgr: Sequence[Any] = (168, 172, 176), seed: int = 0):
        tz = float(table_z)
        if not math.isfinite(tz):
            raise ValueError('table_z must be finite')
        self.table_z = tz
        self.objects: List[SimObject] = list(objects or [])
        self.zones: List[SimZone] = list(zones or [])
        self.table_color_bgr = _check_color(table_color_bgr, 'table')
        self.seed = int(seed)
        self.held: Optional[SimObject] = None
        self.held_width: Optional[float] = None
        self._T_tcp_obj: Optional[np.ndarray] = None
        self.tcp_pose: Optional[np.ndarray] = None
        self.gripper_width: Optional[float] = None
        self.grasp_tol_x = GRASP_TOL_X
        self.grasp_tol_y_frac = GRASP_TOL_Y_FRAC
        self.last_grasp: Dict[str, Any] = {}
        self.last_release: Dict[str, Any] = {}
        self._check_unique()

    def _check_unique(self) -> None:
        for kind, items in (('object', self.objects), ('zone', self.zones)):
            names = [it.name for it in items]
            if len(set(names)) != len(names):
                raise ValueError('duplicate %s names: %s' % (kind, names))

    def add_object(self, obj: SimObject) -> None:
        """Add an object (names must be unique)."""
        if any(o.name == obj.name for o in self.objects):
            raise ValueError('object %s already exists' % obj.name)
        self.objects.append(obj)

    def object(self, name: str) -> SimObject:
        """Return the object with this name (KeyError if missing)."""
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def zone(self, name: str) -> SimZone:
        """Return the zone with this name (KeyError if missing)."""
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)

    def zone_at(self, x: float, y: float) -> Optional[SimZone]:
        """Return the zone containing base (x, y) (nearest center on overlap) or None."""
        hits = [z for z in self.zones if z.contains_xy(x, y)]
        if not hits:
            return None
        return min(hits, key=lambda z: math.hypot(x - z.center[0], y - z.center[1]))

    def zone_of(self, obj: SimObject) -> Optional[SimZone]:
        """Return the zone holding obj (by its footprint center), None if not in one or held."""
        if obj is self.held:
            return None
        c = obj.center_xy()
        return self.zone_at(float(c[0]), float(c[1]))

    # -------------------------------------------------------------- gripper
    def try_grasp(self, T_base_tcp: Any, open_width: float,
                  close_width: float) -> Optional[float]:
        """
        Close the gripper at T_base_tcp; attach and return the held width (m) or None.

        The fingers are assumed to have come straight down along tool z, open
        to open_width. See check_grasp for the success rule. On success the
        object is centered between the fingers and then follows the TCP
        rigidly; the returned width is its extent between the fingers, at
        least close_width. While an object is held, the held width is returned.
        """
        if self.held is not None:
            check_pose(T_base_tcp, 'try_grasp T_base_tcp')
            self.last_grasp = {'ok': True, 'reason': 'already_holding',
                               'object': self.held.name, 'width': self.held_width}
            return self.held_width
        ok, info = self.check_grasp(T_base_tcp, open_width, close_width)
        self.last_grasp = info
        if not ok:
            _log.info('sim grasp failed: %s', info)
            return None
        T = check_pose(T_base_tcp, 'try_grasp T_base_tcp')
        target = self.object(info['object'])
        shift = make_T(np.eye(3), -T[:3, 1] * info['offset_v'])
        target.T_base_obj = shift @ target.T_base_obj
        self.held = target
        self._T_tcp_obj = inv_T(T) @ target.T_base_obj
        self.held_width = float(info['width'])
        self.tcp_pose = T
        self.gripper_width = self.held_width
        _log.debug('sim grasp ok: %s', info)
        return self.held_width

    def check_grasp(self, T_base_tcp: Any, open_width: float,
                    close_width: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate a grasp without changing the world. Return (ok, info).

        Success needs an object (not held) whose
          * base z range contains the TCP z (bottom < z < top),
          * footprint center is within grasp_tol_x (8 mm) of the TCP along
            tool x and within grasp_tol_y_frac (30 %) of its half extent
            along tool y,
        no object with material in the column swept by either finger (inner
        faces at +-open_width / 2 along tool y, FINGER_SIZE, from the
        fingertips upwards) or under the palm, and a target extent along tool
        y between fingertips and finger tops < open_width.
        info: ok, reason ('' | 'no_object' | 'collision_finger' |
        'collision_palm' | 'no_material_between_fingers' | 'too_wide'),
        object, du / dv (m, footprint center minus TCP along tool x / y),
        width (held width, m) and offset_v (m) on success.
        """
        T = check_pose(T_base_tcp, 'grasp T_base_tcp')
        ow, cw = float(open_width), float(close_width)
        if not (math.isfinite(ow) and math.isfinite(cw)) or ow <= 0.0 or cw < 0.0:
            raise ValueError('grasp: bad widths open=%r close=%r' % (open_width, close_width))
        R, p = T[:3, :3], T[:3, 3]
        cands = []
        for obj in self.objects:
            if obj is self.held or not obj.bottom_z() < p[2] < obj.top_z():
                continue
            ulo, uhi = obj.extent_along(R[:, 0])
            vlo, vhi = obj.extent_along(R[:, 1])
            du = (ulo + uhi) / 2.0 - float(p @ R[:, 0])
            dv = (vlo + vhi) / 2.0 - float(p @ R[:, 1])
            tol_y = self.grasp_tol_y_frac * (vhi - vlo) / 2.0
            if abs(du) <= self.grasp_tol_x and abs(dv) <= tol_y:
                cands.append((math.hypot(du, dv), obj, du, dv))
        if not cands:
            return False, {'ok': False, 'reason': 'no_object', 'object': None}
        cands.sort(key=lambda c: c[0])
        _, target, du, dv = cands[0]
        info: Dict[str, Any] = {'ok': False, 'reason': '', 'object': target.name,
                                'du': round(du, 4), 'dv': round(dv, 4)}
        for obj in self.objects:
            if obj is self.held:
                continue
            hit = self._gripper_collision(obj, T, ow)
            if hit:
                info.update(reason='collision_%s' % hit, collided=obj.name)
                return False, info
        loc = (target.surface_points() - p) @ R
        above = -loc[:, 2]
        sl = ((np.abs(loc[:, 0]) <= FINGER_SIZE[0] / 2.0) & (above >= 0.0)
              & (above <= FINGER_SIZE[2]))
        if not np.any(sl):
            info.update(reason='no_material_between_fingers')
            return False, info
        vmin, vmax = float(loc[sl, 1].min()), float(loc[sl, 1].max())
        extent = vmax - vmin
        if extent >= ow:
            info.update(reason='too_wide', extent=round(extent, 4))
            return False, info
        info.update(ok=True, width=max(extent, cw), offset_v=(vmin + vmax) / 2.0)
        return True, info

    @staticmethod
    def _gripper_collision(obj: SimObject, T: np.ndarray, open_width: float) -> str:
        """Return 'finger' / 'palm' if obj has material in the gripper's swept volume, else ''."""
        R, p = T[:3, :3], T[:3, 3]
        loc = (obj.surface_points() - p) @ R
        above = -loc[:, 2]
        av = np.abs(loc[:, 1])
        finger = ((np.abs(loc[:, 0]) <= FINGER_SIZE[0] / 2.0) & (above >= 0.0)
                  & (av >= open_width / 2.0) & (av <= open_width / 2.0 + FINGER_SIZE[1]))
        if np.any(finger):
            return 'finger'
        palm = ((np.abs(loc[:, 0]) <= PALM_SIZE[0] / 2.0) & (above >= PALM_BOTTOM)
                & (av <= PALM_SIZE[1] / 2.0))
        if np.any(palm):
            return 'palm'
        return ''

    def update_tcp(self, T_base_tcp: Any, width: float) -> None:
        """Store the gripper pose / opening (m) for rendering; a held object follows rigidly."""
        T = check_pose(T_base_tcp, 'update_tcp T_base_tcp')
        w = float(width)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError('update_tcp: bad width %r' % width)
        self.tcp_pose = T
        self.gripper_width = w
        if self.held is not None and self._T_tcp_obj is not None:
            self.held.T_base_obj = T @ self._T_tcp_obj

    def release(self, T_base_tcp: Any = None) -> Optional[str]:
        """
        Open the gripper; drop the held object straight down. Return the zone name or None.

        The object keeps its footprint center xy and yaw, becomes upright and
        rests on the floor of the zone containing that xy (zone floor z =
        center[2]) or on the table. A partly outside object counts by its
        center. Returns None when nothing was held or it landed on the table;
        .last_release tells the two apart.
        """
        if T_base_tcp is not None:
            self.update_tcp(T_base_tcp, self.gripper_width or 0.0)
        obj = self.held
        if obj is None:
            self.last_release = {'ok': False, 'reason': 'nothing_held', 'object': None}
            return None
        self.held = None
        self._T_tcp_obj = None
        self.held_width = None
        c = obj.center_xy()
        yaw = obj.yaw()
        bottom_before = obj.bottom_z()
        zone = self.zone_at(float(c[0]), float(c[1]))
        floor = zone.floor_z if zone is not None else self.table_z
        obj.place_upright(float(c[0]), float(c[1]), yaw, floor)
        drop = bottom_before - floor
        self.last_release = {'ok': True, 'reason': '', 'object': obj.name,
                             'zone': None if zone is None else zone.name,
                             'xy': [round(float(c[0]), 4), round(float(c[1]), 4)],
                             'drop_height': round(drop, 4)}
        if drop > 0.15 or drop < -0.01:
            _log.warning('sim release of %s with drop height %.3f m', obj.name, drop)
        return None if zone is None else zone.name

    # -------------------------------------------------------------- scoring
    def placements(self) -> Dict[str, List[str]]:
        """Return zone name -> labels of the objects resting in it (every zone listed)."""
        out: Dict[str, List[str]] = {z.name: [] for z in self.zones}
        for obj in self.objects:
            z = self.zone_of(obj)
            if z is not None:
                out[z.name].append(obj.label)
        return out

    def remaining(self, pick_polygon: Optional[Sequence[Sequence[float]]] = None
                  ) -> List[SimObject]:
        """Return objects on the table (not held, not in a zone), optionally in pick_polygon."""
        out = []
        for obj in self.objects:
            if obj is self.held or self.zone_of(obj) is not None:
                continue
            if pick_polygon is not None:
                c = obj.center_xy()
                if not point_in_polygon(float(c[0]), float(c[1]), pick_polygon):
                    continue
            out.append(obj)
        return out

    def score(self, class_to_zone: Dict[str, str]) -> Dict[str, int]:
        """
        Count correct / wrong placements and remaining targets.

        correct: in zone class_to_zone[label]. wrong: in any other zone
        (including objects whose label is not a target). remaining: target
        objects (label in class_to_zone) not in a zone, on the table or held.
        Non-target objects left on the table are not counted.
        """
        res = {'correct': 0, 'wrong': 0, 'remaining': 0}
        for obj in self.objects:
            z = self.zone_of(obj)
            if z is None:
                if obj.label in class_to_zone:
                    res['remaining'] += 1
            elif class_to_zone.get(obj.label) == z.name:
                res['correct'] += 1
            else:
                res['wrong'] += 1
        return res

    # --------------------------------------------------------------- misc
    def copy(self) -> 'SimWorld':
        """Return a deep copy; a held object stays held in it."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary for logs."""
        tcp = None
        if self.tcp_pose is not None:
            tcp = [round(float(v), 4) for v in self.tcp_pose[:3, 3]]
        return {'table_z': self.table_z, 'seed': self.seed,
                'objects': [o.to_dict() for o in self.objects],
                'zones': [z.to_dict() for z in self.zones],
                'held': None if self.held is None else self.held.name,
                'held_width': self.held_width, 'tcp_xyz': tcp,
                'gripper_width': self.gripper_width,
                'placements': self.placements()}

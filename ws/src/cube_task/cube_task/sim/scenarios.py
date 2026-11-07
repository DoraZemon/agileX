"""
Seeded competition scenarios (L1 blocks, L2 fruit / vegetables, L3 instructions).

Base frame: PiPER base at the origin, z up, table top at z = 0 (the robot
stands on the table). The pick area is DEFAULT_PICK_POLYGON (same as config
perception.roi.pick_polygon). Zones lie outside it, inside the top-down reach
of the arm (see piper_fk / piper_ik). make_scenario() returns the world plus
an info dict; scenario_config_overlay() turns the info into a config overlay
for core.config.deep_merge.

Note on reach: with joint5 limited to +-70 deg, PiPER can hold the tool
vertical only at low TCP heights (about 0.13 m at r = 0.3 m with the 0.14 m
TCP offset, less further out). That is why the eye-in-hand observe pose is
lower than one might expect and why zones sit at r <= ~0.40 m.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import inv_T, make_T, pose6_to_T, rotation_angle, top_down_T
from ..core.types import CameraIntrinsics
from .world import SimObject, SimWorld, SimZone

# PiPER modified-DH table (d, a, alpha, theta_offset), copied from pyAgxArm
# api/constants.py ROBOT_MDH_PRESET['piper']; link transform
# Rx(alpha) Tx(a) Rz(q + theta_offset) Tz(d) (pyAgxArm utiles/mdh_kinematics.py).
PIPER_MDH = (
    (0.123, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.5707963267948966, -3.0058060377846343),
    (0.0, 0.28502999999999995, 0.0, -1.793849405199772),
    (0.25075, -0.02198, 1.5707963267948966, 0.0),
    (0.0, 0.0, -1.5707963267948966, 0.0),
    (0.091, 0.0, 1.5707963267948966, 0.0),
)
# Joint limits (rad), pyAgxArm ROBOT_JOINT_LIMIT_PRESET_RAD['piper'].
PIPER_LIMITS = (
    (-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
    (-1.745330, 1.745330), (-1.221730, 1.221730), (-2.967060, 2.967060),
)
# Same as config/default.yaml robot.tcp_offset (only used for info / reach checks).
DEFAULT_TCP_OFFSET = (0.0, 0.0, 0.140, 0.0, 0.0, 0.0)
GRIPPER_MAX_WIDTH = 0.070            # config robot.gripper.max_width (standard gripper)

DEFAULT_PICK_POLYGON = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]
TASKS = ('L1', 'L2', 'L3')
MOUNTS = ('eye_to_hand', 'eye_in_hand')
DIFFICULTIES = ('normal', 'hard')

# Orbbec-like wide-FOV color camera.
DEFAULT_INTRINSICS = dict(fx=380.0, fy=380.0, cx=319.5, cy=239.5, width=640, height=480)
EYE_TO_HAND_CAMERA_POS = (0.70, 0.0, 0.62)
EYE_TO_HAND_LOOK_AT = (0.30, 0.0, 0.0)
# Wrist camera: 0.07 m from the gripper axis on the flange -x side ("on top"
# of the gripper in the home pose), 0.03 m in front of the flange face,
# optical axis parallel to the tool z. Image down = flange x, right = -flange y.
EYE_IN_HAND_OFFSET = (-0.07, 0.0, 0.03)
# Eye-to-hand park pose: home with joint1 turned toward +y so the gripper is
# raised, outside the pick polygon in XY and outside the camera's view of it.
EYE_TO_HAND_PARK_JOINTS = (1.1, 0.3, -0.3, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- kinematics
def _mdh_link(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [ca * st, ca * ct, -sa, -sa * d],
                     [sa * st, sa * ct, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


def _fk_frames(q: Sequence[float]) -> List[np.ndarray]:
    T = np.eye(4)
    frames = []
    for (d, a, alpha, off), qi in zip(PIPER_MDH, q):
        T = T @ _mdh_link(d, a, alpha, float(qi) + off)
        frames.append(T)
    return frames


def piper_fk(q: Sequence[float]) -> np.ndarray:
    """Return T_base_flange (4x4) for joint angles q (6, rad)."""
    if len(q) != 6:
        raise ValueError('piper_fk needs 6 joint angles')
    return _fk_frames(q)[-1]


def _rot_log(R: np.ndarray) -> np.ndarray:
    """Return the rotation vector (axis * angle, rad) of R."""
    ang = rotation_angle(R)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-9:
        return 0.5 * w
    if ang > math.pi - 1e-4:
        axis = np.sqrt(np.maximum(0.0, (np.diag(R) + 1.0) / 2.0))
        k = int(np.argmax(axis))
        axis[[i for i in range(3) if i != k]] *= np.sign(R[k, [i for i in range(3) if i != k]])
        return axis / np.linalg.norm(axis) * ang
    return w / (2.0 * math.sin(ang)) * ang


def within_limits(q: Sequence[float], margin: float = 0.0) -> bool:
    """Check q against PIPER_LIMITS shrunk by margin (rad)."""
    return all(lo + margin <= float(v) <= hi - margin for v, (lo, hi) in zip(q, PIPER_LIMITS))


def piper_ik(T_base_flange: np.ndarray, seeds: Optional[Sequence[Sequence[float]]] = None,
             limit_margin: float = 0.02, max_iter: int = 300
             ) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Solve flange IK by damped least squares from several seeds.

    Returns (q or None, position error m, rotation error rad) of the best
    solution; q is None unless the errors are below 0.1 mm / 0.05 deg with
    every joint at least limit_margin (rad) inside PIPER_LIMITS.
    """
    Tt = np.asarray(T_base_flange, dtype=float)
    lo = np.array([v[0] for v in PIPER_LIMITS]) + limit_margin
    hi = np.array([v[1] for v in PIPER_LIMITS]) - limit_margin
    if seeds is None:
        yaw = math.atan2(Tt[1, 3], Tt[0, 3])
        seeds = [[yaw, 1.2, -1.0, 0.0, 1.0, 0.0], [yaw, 1.6, -1.2, 0.0, 1.1, 0.0],
                 [yaw, 0.8, -0.6, 0.0, 1.1, 0.0], [yaw, 1.9, -1.6, 0.0, 1.0, 0.0],
                 [yaw, 0.4, -0.3, 0.0, 0.3, 0.0]]
    best: Tuple[Optional[np.ndarray], float, float] = (None, float('inf'), float('inf'))
    for seed in seeds:
        q = np.clip(np.array(seed, dtype=float), lo, hi)
        for _ in range(max_iter):
            frames = _fk_frames(q)
            Tc = frames[-1]
            e = np.concatenate([Tt[:3, 3] - Tc[:3, 3], _rot_log(Tt[:3, :3] @ Tc[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 1e-7 and np.linalg.norm(e[3:]) < 1e-7:
                break
            J = np.zeros((6, 6))
            for i, Fi in enumerate(frames):
                z, p = Fi[:3, 2], Fi[:3, 3]
                J[:3, i] = np.cross(z, Tc[:3, 3] - p)
                J[3:, i] = z
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, lo, hi)
        Tc = piper_fk(q)
        ep = float(np.linalg.norm(Tt[:3, 3] - Tc[:3, 3]))
        er = rotation_angle(Tt[:3, :3].T @ Tc[:3, :3])
        if ep + er < best[1] + best[2]:
            best = (q.copy(), ep, er)
    q, ep, er = best
    if q is None or ep > 1e-4 or er > math.radians(0.05):
        return None, ep, er
    return q, ep, er


def top_down_reachable(x: float, y: float, z: float, closing_angle: Optional[float] = None,
                       tcp_offset: Sequence[float] = DEFAULT_TCP_OFFSET) -> bool:
    """Check that a vertical top-down TCP pose at (x, y, z) (m) has an IK solution."""
    if closing_angle is None:
        closing_angle = math.atan2(y, x) + math.pi / 2.0
    T_flange = top_down_T(x, y, z, closing_angle) @ inv_T(pose6_to_T(tcp_offset))
    return piper_ik(T_flange)[0] is not None


# -------------------------------------------------------------------- camera
def look_at(eye: Sequence[float], target: Sequence[float],
            down_hint: Sequence[float] = (0.0, 0.0, -1.0)) -> np.ndarray:
    """
    Return T_base_camera of an optical frame at eye looking at target.

    Image y (down) is the component of down_hint perpendicular to the view
    axis, so with the default hint the image is upright.
    """
    c = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - c
    z /= np.linalg.norm(z)
    y = np.asarray(down_hint, dtype=float)
    y = y - (y @ z) * z
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('look_at: down_hint parallel to the view axis')
    y /= np.linalg.norm(y)
    return make_T(np.column_stack([np.cross(y, z), y, z]), c)


def default_intrinsics() -> CameraIntrinsics:
    """Return the simulated Orbbec-like color intrinsics (640x480, fx = fy = 380)."""
    d = DEFAULT_INTRINSICS
    return CameraIntrinsics(d['fx'], d['fy'], d['cx'], d['cy'], d['width'], d['height'])


def eye_in_hand_T_flange_camera() -> np.ndarray:
    """
    Return T_flange_camera of the simulated wrist camera.

    Optical z = flange z (approach), image down = flange x, right = -flange y.
    """
    R = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return make_T(R, EYE_IN_HAND_OFFSET)


@functools.lru_cache(maxsize=4)
def _eye_in_hand_observe() -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    """
    Return (observe joints, T_base_camera) of the eye-in-hand observe pose.

    A straight-down view is only reachable from about 0.21 m (the pick area
    does not fit the image), so the camera looks at (0.30, 0, 0) tilted
    30 deg from vertical toward +x from 0.32 m away (camera at about
    (0.14, 0, 0.28)). The fingers then show in the bottom ~33 image rows,
    just below the pick polygon.
    """
    T_bc = look_at((0.0, 0.0, 0.0), (math.sin(math.radians(30.0)), 0.0,
                                     -math.cos(math.radians(30.0))),
                   down_hint=(-1.0, 0.0, 0.0))
    T_bc[:3, 3] = np.array([0.30, 0.0, 0.0]) - 0.32 * T_bc[:3, 2]
    T_bf = T_bc @ inv_T(eye_in_hand_T_flange_camera())
    q, ep, er = piper_ik(T_bf)
    if q is None:
        raise RuntimeError('eye-in-hand observe pose unreachable (err %.4f m %.4f rad)'
                           % (ep, er))
    T_true = piper_fk(q) @ eye_in_hand_T_flange_camera()
    return (tuple(float(v) for v in q), tuple(tuple(float(v) for v in row) for row in T_true))


@functools.lru_cache(maxsize=2)
def _views() -> Tuple[Tuple[np.ndarray, int], ...]:
    """Return (T_camera_base, max image row) of both camera mounts' observe views."""
    eih = np.array(_eye_in_hand_observe()[1])
    eth = look_at(EYE_TO_HAND_CAMERA_POS, EYE_TO_HAND_LOOK_AT)
    h = DEFAULT_INTRINSICS['height']
    return ((inv_T(eth), h - VIEW_MARGIN_PX), (inv_T(eih), h - EYE_IN_HAND_BOTTOM_PX))


def fully_visible(obj: SimObject) -> bool:
    """Check that the object's base AABB lies inside both mounts' observe images."""
    d = DEFAULT_INTRINSICS
    lo, hi = obj.aabb()
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])])
    for T_cb, v_max in _views():
        pc = pts @ T_cb[:3, :3].T + T_cb[:3, 3]
        if np.any(pc[:, 2] <= 0.05):
            return False
        u = d['fx'] * pc[:, 0] / pc[:, 2] + d['cx']
        v = d['fy'] * pc[:, 1] / pc[:, 2] + d['cy']
        if (u.min() < VIEW_MARGIN_PX or u.max() > d['width'] - VIEW_MARGIN_PX
                or v.min() < VIEW_MARGIN_PX or v.max() > v_max):
            return False
    return True


# ------------------------------------------------------------------- catalog
def _arc_points(radius: float, length: float, r_tube: float, n: int = 9) -> List[List[float]]:
    """Planar arc in local XY (lying on the table), footprint centered, bottom at z = 0."""
    ang = length / radius
    a = np.linspace(-ang / 2.0, ang / 2.0, n)
    pts = np.column_stack([radius * np.sin(a), radius * np.cos(a), np.full(n, r_tube)])
    pts[:, 1] -= (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    return pts.tolist()


def _line_points(length: float, r_tube: float) -> List[List[float]]:
    h = length / 2.0 - r_tube
    return [[-h, 0.0, r_tube], [h, 0.0, r_tube]]


def _grape_points(r: float) -> List[List[float]]:
    ring = 1.9 * r
    pts = [[ring * math.cos(k * math.pi / 3.0), ring * math.sin(k * math.pi / 3.0), r]
           for k in range(6)]
    pts.append([0.0, 0.0, r + 1.4 * r])
    return pts


# label -> (category, shape, size, color BGR). Plastic-model sizes (m). Every
# narrow axis is <= GRIPPER_MAX_WIDTH - 8 mm so the standard 0.07 m gripper
# has >= 4 mm clearance per side.
CATALOG: Dict[str, Tuple[str, str, tuple, Tuple[int, int, int]]] = {
    'apple': ('fruit', 'sphere', (0.031,), (45, 45, 185)),
    'orange': ('fruit', 'sphere', (0.031,), (30, 130, 235)),
    'banana': ('fruit', 'capsule_chain', (0.017, _arc_points(0.15, 0.17 - 0.034, 0.017)),
               (50, 205, 230)),
    'lemon': ('fruit', 'ellipsoid', (0.035, 0.025, 0.025), (60, 220, 235)),
    'grape': ('fruit', 'sphere_cluster', (0.010, _grape_points(0.010)), (110, 40, 95)),
    'carrot': ('vegetable', 'capsule_chain', (0.015, _line_points(0.14, 0.015)),
               (25, 110, 225)),
    'eggplant': ('vegetable', 'ellipsoid', (0.07, 0.03, 0.03), (95, 35, 75)),
    'cucumber': ('vegetable', 'capsule_chain', (0.018, _line_points(0.16, 0.018)),
                 (50, 135, 45)),
    'corn': ('vegetable', 'ellipsoid', (0.08, 0.027, 0.027), (70, 195, 225)),
    'tomato': ('vegetable', 'ellipsoid', (0.028, 0.028, 0.022), (35, 45, 205)),
    'green_pepper': ('vegetable', 'ellipsoid', (0.036, 0.031, 0.031), (45, 160, 55)),
}
FRUITS = tuple(k for k, v in CATALOG.items() if v[0] == 'fruit')
VEGETABLES = tuple(k for k, v in CATALOG.items() if v[0] == 'vegetable')

BLOCK_COLORS = {'red': (40, 40, 200), 'green': (60, 170, 60), 'blue': (200, 90, 40),
                'yellow': (40, 200, 220)}
MAT_COLORS = {'red': (70, 70, 150), 'green': (90, 140, 90), 'blue': (150, 110, 80),
              'yellow': (90, 170, 180)}

# Zones (name, center, size, wall_height, color BGR, label). All centers are
# within the vertical-tool reach at release height (TCP z 0.06: r <= 0.38 m)
# with x >= 0.08 (config robot.workspace.x) and walls >= 2 cm outside the pick polygon.
L1_ZONES = (
    ('zone_red', (0.21, 0.29, 0.0), (0.10, 0.10), 0.0, MAT_COLORS['red'], 'red'),
    ('zone_green', (0.10, 0.29, 0.0), (0.10, 0.10), 0.0, MAT_COLORS['green'], 'green'),
    ('zone_blue', (0.16, -0.29, 0.0), (0.10, 0.10), 0.0, MAT_COLORS['blue'], 'blue'),
)
L2_ZONES = (
    ('fruit_bin', (0.19, 0.315, 0.0), (0.14, 0.14), 0.05, (120, 120, 125), 'fruit'),
    ('veg_bin', (0.19, -0.315, 0.0), (0.14, 0.14), 0.05, (95, 115, 95), 'vegetable'),
)
L3_ZONES = (
    ('zone_blue', (0.22, 0.305, 0.0), (0.12, 0.12), 0.015, (170, 110, 60), 'blue'),
    ('zone_red', (0.09, 0.305, 0.0), (0.12, 0.12), 0.015, (60, 60, 170), 'red'),
    ('zone_green', (0.22, -0.305, 0.0), (0.12, 0.12), 0.015, (70, 150, 70), 'green'),
)
L3_LABEL_ALIASES = {
    'apple': ['苹果', 'apple'], 'orange': ['橙子', '橘子', 'orange'], 'banana': ['香蕉', 'banana'],
    'lemon': ['柠檬', 'lemon'], 'grape': ['葡萄', 'grape'], 'carrot': ['胡萝卜', 'carrot'],
    'eggplant': ['茄子', 'eggplant'], 'cucumber': ['黄瓜', 'cucumber'],
    'corn': ['玉米', 'corn'], 'tomato': ['西红柿', '番茄', 'tomato'],
    'green_pepper': ['青椒', 'green pepper'],
}
L3_ZONE_ALIASES = {
    'zone_blue': ['蓝色托盘', '蓝盘', '蓝色区域', 'blue tray'],
    'zone_red': ['红色托盘', '红盘', '红色区域', 'red tray'],
    'zone_green': ['绿色托盘', '绿盘', '绿色区域', 'green tray'],
}


# ------------------------------------------------------------------ sampling
EDGE_MARGIN = 0.01        # m, object footprint corners stay this far inside the pick polygon
REACH_R = (0.17, 0.43)    # m, object centers within the vertical-tool grasp reach
VIEW_MARGIN_PX = 4        # objects stay this far inside both observe images
EYE_IN_HAND_BOTTOM_PX = 40  # bottom image rows of the wrist view that show the fingers


def _footprint(obj: SimObject) -> np.ndarray:
    """Return the 4x2 base-XY corners of the object's yawed local AABB."""
    lo, hi = obj.local_aabb()
    loc = np.array([[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0], [hi[0], hi[1], 0.0],
                    [lo[0], hi[1], 0.0]])
    T = obj.T_base_obj
    return (loc @ T[:3, :3].T + T[:3, 3])[:, :2]


def _edge_normals(poly: np.ndarray) -> np.ndarray:
    """Return unit normals (N x 2) of the polygon edges."""
    e = np.roll(poly, -1, axis=0) - poly
    n = np.column_stack([-e[:, 1], e[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _gap_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the separation (m) of two convex polygons along their edge normals (<0 overlap)."""
    axes = np.vstack([_edge_normals(a), _edge_normals(b)])
    pa, pb = a @ axes.T, b @ axes.T
    return float(np.max(np.maximum(pb.min(axis=0) - pa.max(axis=0),
                                   pa.min(axis=0) - pb.max(axis=0))))


def _inward(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (inward unit normals, offsets) so p is inside iff normals @ p - offsets >= 0."""
    n = _edge_normals(poly)
    x, y = poly[:, 0], poly[:, 1]
    if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0.0:
        n = -n
    return n, np.einsum('ij,ij->i', n, poly)


def _inside(pts: np.ndarray, poly: np.ndarray, margin: float) -> bool:
    """Check every point (N x 2 or (2,)) is inside the convex polygon by >= margin (m)."""
    n, off = _inward(poly)
    d = np.atleast_2d(pts) @ n.T - off
    return bool(np.all(d >= margin))


def _place_random(obj: SimObject, placed: List[SimObject], rng: np.random.Generator,
                  polygon: np.ndarray, gap: float, min_center_dist: float = 0.0,
                  tries: int = 4000) -> None:
    """Place obj upright at a random pose inside polygon, gap (m) from every placed object."""
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    llo, lhi = obj.local_aabb()
    ext = lhi[:2] - llo[:2]
    elongated = float(ext.max()) > 2.0 * float(ext.min())
    long_axis = 0.0 if ext[0] >= ext[1] else math.pi / 2.0
    others = []
    for o in placed:
        fp = _footprint(o)
        c = o.center_xy()
        others.append((c, fp, float(np.max(np.linalg.norm(fp - c, axis=1)))))
    for _ in range(tries):
        x, y = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
        yaw = rng.uniform(-math.pi, math.pi)
        if elongated and rng.random() < 0.5:
            # Half of the tries lay long objects roughly along base y, the
            # long direction of the pick area, which makes dense layouts feasible.
            yaw = math.pi / 2.0 - long_axis + rng.normal(0.0, 0.35)
        if not REACH_R[0] <= math.hypot(x, y) <= REACH_R[1]:
            continue
        obj.place_upright(x, y, yaw, 0.0)
        fp = _footprint(obj)
        if not _inside(fp, polygon, EDGE_MARGIN) or not fully_visible(obj):
            continue
        c = obj.center_xy()
        rad = float(np.max(np.linalg.norm(fp - c, axis=1)))
        ok = True
        for oc, ofp, orad in others:
            d = float(np.linalg.norm(oc - c))
            if d < min_center_dist:
                ok = False
            elif d < rad + orad + gap and _gap_between(fp, ofp) < gap:
                ok = False
            if not ok:
                break
        if ok:
            return
    raise RuntimeError('could not place %s without overlap' % obj.name)


def _jitter(color: Tuple[int, int, int], rng: np.random.Generator, amount: int = 6
            ) -> Tuple[int, int, int]:
    c = np.clip(np.array(color) + rng.integers(-amount, amount + 1, 3), 0, 255)
    return int(c[0]), int(c[1]), int(c[2])


def _make_zones(spec) -> List[SimZone]:
    return [SimZone(n, c, s, 0.0, col, wh, lab) for n, c, s, wh, col, lab in spec]


def _catalog_object(name: str, label: str, rng: np.random.Generator) -> SimObject:
    _cat, shape, size, color = CATALOG[label]
    return SimObject(name, label, shape, size, _jitter(color, rng), np.eye(4))


def find_top_down_grasp(world: SimWorld, obj: SimObject, max_width: float = GRIPPER_MAX_WIDTH,
                        open_margin: float = 0.02, steps: int = 24
                        ) -> Optional[Tuple[float, float]]:
    """
    Return (closing angle rad, width m) of the narrowest working top-down grasp, or None.

    Tries closing angles k * pi / steps at the footprint center, TCP half the
    object height below the top (clamped to 8..30 mm), opening
    min(width + open_margin, max_width), like the default grasp config.
    """
    c, top, h = obj.center_xy(), obj.top_z(), obj.height()
    z = top - min(max(0.5 * h, 0.008), 0.03)
    best: Optional[Tuple[float, float]] = None
    for k in range(steps):
        a = k * math.pi / steps
        w = obj.footprint_extent((math.cos(a), math.sin(a)))
        if w > max_width - 0.005 or (best is not None and w >= best[1]):
            continue
        ok, _ = world.check_grasp(top_down_T(c[0], c[1], z, a), min(w + open_margin, max_width),
                                  0.0)
        if ok:
            best = (a, w)
    return best


def _check_graspable(objects: List[SimObject]) -> List[SimObject]:
    world = SimWorld(objects=objects)
    for obj in objects:
        if find_top_down_grasp(world, obj) is None:
            raise RuntimeError('%s has no collision-free grasp' % obj.name)
    return objects


def _draw_labels(task: str, hard: bool, rng: np.random.Generator) -> List[str]:
    """Draw the 6 object labels of an L2 / L3 scene in a random order."""
    if task == 'L2' and hard:
        chosen = [str(v) for v in rng.choice(sorted(CATALOG), 6, replace=True)]
    elif task == 'L2':
        chosen = ([str(v) for v in rng.choice(FRUITS, 3, replace=False)]
                  + [str(v) for v in rng.choice(VEGETABLES, 3, replace=False)])
    else:
        others = [k for k in sorted(CATALOG) if k != 'apple']
        chosen = ['apple', 'apple'] + [str(v) for v in rng.choice(others, 4, replace=False)]
    return [chosen[i] for i in rng.permutation(len(chosen))]


def _retry_layout(build, attempts: int = 150) -> List[SimObject]:
    """
    Call build(gap_scale) until it places every object (it raises RuntimeError on failure).

    After every third of the attempts the required gaps shrink (1.0, 0.8, 0.6).
    """
    for k in range(attempts):
        try:
            return build((1.0, 0.8, 0.6)[min(2, 3 * k // attempts)])
        except RuntimeError:
            continue
    raise RuntimeError('could not lay out the scenario objects')


def _l1_objects(rng: np.random.Generator, polygon: np.ndarray, hard: bool,
                gap_scale: float = 1.0) -> List[SimObject]:
    names = ['red', 'green', 'blue']
    cubes = [SimObject('%s_cube' % c, c, 'box', (e, e, e), _jitter(BLOCK_COLORS[c], rng),
                       np.eye(4))
             for c, e in zip(names, rng.uniform(0.03, 0.04, 3))]
    objects: List[SimObject] = []
    rest = cubes
    if hard:
        a, b = cubes[0], cubes[1]
        _place_random(a, objects, rng, polygon, 0.025 * gap_scale, 0.06, tries=300)
        objects.append(a)
        side = float(rng.choice([-1.0, 1.0]))
        lat = float(rng.uniform(-0.3, 0.3)) * a.size[0]
        off = a.T_base_obj[:3, :3] @ np.array([side * (a.size[0] + b.size[0]) / 2.0, lat, 0.0])
        c = a.center_xy() + off[:2]
        b.place_upright(c[0], c[1], a.yaw(), 0.0)
        if (not _inside(_footprint(b), polygon, EDGE_MARGIN)
                or not REACH_R[0] <= math.hypot(c[0], c[1]) <= REACH_R[1]
                or not fully_visible(b)):
            raise RuntimeError('touching cube outside the pick area')
        objects.append(b)
        rest = [cubes[2], SimObject('yellow_cube', 'yellow', 'box', (0.035,) * 3,
                                    _jitter(BLOCK_COLORS['yellow'], rng), np.eye(4))]
    for obj in rest:
        _place_random(obj, objects, rng, polygon, 0.025 * gap_scale, 0.06, tries=300)
        objects.append(obj)
    return _check_graspable(objects)


def _catalog_objects(chosen: List[str], rng: np.random.Generator, polygon: np.ndarray,
                     gap: float, gap_scale: float = 1.0) -> List[SimObject]:
    objs = [_catalog_object('%s_%d' % (lab, i), lab, rng) for i, lab in enumerate(chosen)]
    placed: List[SimObject] = []
    for obj in sorted(objs, key=lambda o: -float(np.max(o.local_aabb()[1][:2]))):
        _place_random(obj, placed, rng, polygon, gap * gap_scale, tries=150)
        placed.append(obj)
    return _check_graspable(objs)


# ----------------------------------------------------------------- scenarios
def make_scenario(task: str, seed: int, mount: str = 'eye_to_hand',
                  difficulty: str = 'normal') -> Tuple[SimWorld, Dict[str, Any]]:
    """
    Build a seeded scene. Return (SimWorld, info).

    task: 'L1' (3 colored cubes, 3 colored mats), 'L2' (6 fruit / vegetable
    models, 2 walled bins), 'L3' (L2-like objects with 2 apples, 3 colored
    trays, info['instructions']). difficulty 'hard': L1 adds two touching
    cubes and a yellow distractor; L2 uses smaller gaps and every catalog
    label. info keys: task, seed, mount, difficulty, labels, class_to_zone,
    categories, zones (name -> core Zone), pick_polygon, intrinsics
    (CameraIntrinsics), T_base_camera (eye_to_hand) or T_flange_camera and
    T_base_camera_observe (eye_in_hand), observe_joints, objects,
    required, time_limit_s, plus for L3 instructions / instruction_specs /
    label_aliases / zone_aliases. Deterministic for a given argument set.
    """
    if task not in TASKS:
        raise ValueError('task must be one of %s, got %r' % (TASKS, task))
    if mount not in MOUNTS:
        raise ValueError('mount must be one of %s, got %r' % (MOUNTS, mount))
    if difficulty not in DIFFICULTIES:
        raise ValueError('difficulty must be one of %s, got %r' % (DIFFICULTIES, difficulty))
    rng = np.random.default_rng([int(seed), TASKS.index(task), DIFFICULTIES.index(difficulty)])
    polygon = np.array(DEFAULT_PICK_POLYGON, dtype=float)
    hard = difficulty == 'hard'
    info: Dict[str, Any] = {'task': task, 'seed': int(seed), 'mount': mount,
                            'difficulty': difficulty}
    if task == 'L1':
        zones = _make_zones(L1_ZONES)
        class_to_zone = {'red': 'zone_red', 'green': 'zone_green', 'blue': 'zone_blue'}
        labels = ['red', 'green', 'blue'] + (['yellow'] if hard else [])
        categories: Dict[str, str] = {}
        objects = _retry_layout(lambda g: _l1_objects(rng, polygon, hard, g))
        info['time_limit_s'] = 180.0
    else:
        zones = _make_zones(L2_ZONES if task == 'L2' else L3_ZONES)
        gap = 0.018 if hard else 0.025
        for _draw in range(5):
            chosen = _draw_labels(task, hard, rng)
            try:
                objects = _retry_layout(lambda g: _catalog_objects(chosen, rng, polygon, gap, g))
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('could not lay out %s seed %s' % (task, seed))
        used = sorted(CATALOG) if (hard and task == 'L2') else sorted(set(chosen))
        labels = used
        categories = {k: CATALOG[k][0] for k in used}
        if task == 'L2':
            class_to_zone = {k: ('fruit_bin' if CATALOG[k][0] == 'fruit' else 'veg_bin')
                             for k in used}
        else:
            class_to_zone = {}
        info['time_limit_s'] = 300.0
    world = SimWorld(table_z=0.0, objects=objects, zones=zones, seed=int(seed))
    info.update(labels=labels, class_to_zone=class_to_zone, categories=categories,
                zones={z.name: z.to_zone() for z in zones},
                pick_polygon=[list(p) for p in DEFAULT_PICK_POLYGON],
                intrinsics=default_intrinsics(),
                objects=[{'name': o.name, 'label': o.label} for o in objects],
                required=sum(1 for o in objects if o.label in class_to_zone))
    if task == 'L3':
        info['required'] = 5
        info.update(_l3_instructions(world, rng))
    if mount == 'eye_to_hand':
        info['T_base_camera'] = look_at(EYE_TO_HAND_CAMERA_POS, EYE_TO_HAND_LOOK_AT)
        info['observe_joints'] = [float(v) for v in EYE_TO_HAND_PARK_JOINTS]
    else:
        q, T_obs = _eye_in_hand_observe()
        info['T_flange_camera'] = eye_in_hand_T_flange_camera()
        info['T_base_camera_observe'] = np.array(T_obs)
        info['observe_joints'] = list(q)
    return world, info


def _l3_instructions(world: SimWorld, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Return referee-style instructions for an L3 world and the expected parse.

    'Left' is base +y (config tasks.L3.left_axis default [0, 1]); 'nearest'
    is the smallest horizontal distance to the robot base.
    """
    zh_zone = {'zone_blue': '蓝色托盘', 'zone_red': '红色托盘', 'zone_green': '绿色托盘'}
    apples = sorted((o for o in world.objects if o.label == 'apple'),
                    key=lambda o: float(o.center_xy()[1]))
    others = [o for o in world.objects if o.label != 'apple']
    specs = [
        dict(text='把最左边的苹果放到蓝色托盘', label='apple', selector='leftmost',
             zone='zone_blue', count=1, object=apples[-1].name),
        dict(text='把最右边的苹果放到红色托盘', label='apple', selector='rightmost',
             zone='zone_red', count=1, object=apples[0].name),
    ]
    zones_cycle = ['zone_green', 'zone_blue', 'zone_red']
    for k, obj in enumerate(others[:3]):
        zh = L3_LABEL_ALIASES[obj.label][0]
        same = [o for o in others if o.label == obj.label]
        zone = zones_cycle[k % 3]
        if len(same) == 1 and k == 1:
            text, sel = '把离机器人最近的%s放到%s' % (zh, zh_zone[zone]), 'nearest'
        elif len(same) == 1:
            text, sel = '把%s放到%s' % (zh, zh_zone[zone]), 'any'
        else:
            target = min(same, key=lambda o: float(np.hypot(*o.center_xy())))
            text, sel = '把离机器人最近的%s放到%s' % (zh, zh_zone[zone]), 'nearest'
            obj = target
        specs.append(dict(text=text, label=obj.label, selector=sel, zone=zone, count=1,
                          object=obj.name))
    return {'instructions': [s['text'] for s in specs], 'instruction_specs': specs,
            'label_aliases': {k: list(v) for k, v in L3_LABEL_ALIASES.items()},
            'zone_aliases': {k: list(v) for k, v in L3_ZONE_ALIASES.items()}}


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def perturb_pose(T: np.ndarray, trans_err: float, rot_err_deg: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Return T with a translation error of trans_err m and a rotation error of rot_err_deg deg.

    Directions are random (seeded); the rotation is applied about the pose's
    own origin, so only the frame orientation changes.
    """
    t, r = float(trans_err), float(rot_err_deg)
    if not (math.isfinite(t) and math.isfinite(r)) or t < 0.0 or r < 0.0:
        raise ValueError('extrinsics_error must be two finite values >= 0, got %r'
                         % ((trans_err, rot_err_deg),))
    d = rng.normal(size=3)
    ax = rng.normal(size=3)
    out = np.array(T, dtype=float).copy()
    out[:3, 3] += d / np.linalg.norm(d) * t
    out[:3, :3] = _axis_angle(ax, math.radians(r)) @ out[:3, :3]
    return out


def _as_list(T: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if T is None:
        return None
    return [[float(v) for v in row] for row in np.asarray(T, dtype=float)]


def scenario_config_overlay(info: Dict[str, Any],
                            extrinsics_error: Sequence[float] = (0.0, 0.0)) -> Dict[str, Any]:
    """
    Return a config overlay (for core.config.deep_merge) describing the scenario.

    extrinsics_error: [translation m, rotation deg] applied in a direction
    seeded by info['seed'] to the extrinsics the robot code is given
    (camera.extrinsics_inline); sim.T_base_camera / sim.T_flange_camera keep
    the true pose used for rendering. Merge it over a config loaded without
    site overlays, or site zones are merged with the sim zones.
    """
    if len(extrinsics_error) != 2:
        raise ValueError('extrinsics_error must be [translation m, rotation deg]')
    task, mount = info['task'], info['mount']
    rng = np.random.default_rng([int(info['seed']), 7919])
    if mount == 'eye_to_hand':
        T_true = np.asarray(info['T_base_camera'], dtype=float)
        sim_cam = {'T_base_camera': _as_list(T_true), 'T_flange_camera': None}
    else:
        T_true = np.asarray(info['T_flange_camera'], dtype=float)
        sim_cam = {'T_base_camera': None, 'T_flange_camera': _as_list(T_true)}
    T_est = perturb_pose(T_true, extrinsics_error[0], extrinsics_error[1], rng)
    zones = {}
    for name, z in info['zones'].items():
        zones[name] = {'center': [float(v) for v in z.center], 'size': [float(v) for v in z.size],
                       'yaw': float(z.yaw), 'wall_height': float(z.wall_height),
                       'label': str(z.label)}
    task_cfg: Dict[str, Any] = {'labels': list(info['labels']),
                                'class_to_zone': dict(info['class_to_zone'])}
    if task == 'L3':
        task_cfg['label_aliases'] = info.get('label_aliases', {})
        task_cfg['zone_aliases'] = info.get('zone_aliases', {})
    sim = {'intrinsics': info['intrinsics'].to_dict(), 'seed': int(info['seed']),
           'extrinsics_error': [float(extrinsics_error[0]), float(extrinsics_error[1])]}
    sim.update(sim_cam)
    return {
        'zones': zones,
        'tasks': {task: task_cfg},
        'perception': {'roi': {'pick_polygon': [list(map(float, p))
                                                for p in info['pick_polygon']]}},
        'camera': {'backend': 'sim', 'mount': mount,
                   'extrinsics_inline': {'mount': mount, 'matrix': _as_list(T_est)}},
        'robot': {'backend': 'sim',
                  'poses': {'observe_joints': [float(v) for v in info['observe_joints']]}},
        'sim': sim,
    }

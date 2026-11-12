"""
Seeded competition scenarios (L1 blocks, L2 fruit / vegetables).

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
from .world import (FINGER_SIZE, GRASP_PAD_HALF, OFFICIAL_MIN_ZONE_FRACTION,
                    OFFICIAL_ZONE_FRACTION_BY_LABEL, SimObject, SimWorld, SimZone)

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
GRIPPER_MAX_WIDTH = 0.100            # config robot.gripper.max_width (100 mm stroke)

DEFAULT_PICK_POLYGON = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]
TASKS = ('L1', 'L2')
MOUNTS = ('eye_in_hand',)               # the competition has only the wrist camera
DIFFICULTIES = ('normal', 'hard')
MATERIALS = ('official', 'legacy')
LAYOUTS = ('A', 'B', 'S')

# Orbbec-like wide-FOV color camera.
DEFAULT_INTRINSICS = dict(fx=380.0, fy=380.0, cx=319.5, cy=239.5, width=640, height=480)
# Wrist camera: 0.07 m from the gripper axis on the flange -x side ("on top"
# of the gripper in the home pose), 0.03 m in front of the flange face,
# optical axis parallel to the tool z. Image down = flange x, right = -flange y.
EYE_IN_HAND_OFFSET = (-0.07, 0.0, 0.03)


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
    """Return (T_camera_base, max image row) of the legacy wrist observe view."""
    eih = np.array(_eye_in_hand_observe()[1])
    h = DEFAULT_INTRINSICS['height']
    return ((inv_T(eih), h - EYE_IN_HAND_BOTTOM_PX),)


def fully_visible(obj: SimObject, views: Optional[Sequence[Tuple[np.ndarray, int]]] = None,
                  exact: bool = False, intrinsics: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check that the object lies inside every given observe image (wrist camera views).

    views: [(T_camera_base, max image row)] (default: the legacy views).
    exact False: the corners of the base AABB must project inside; True:
    the corners of every part's AABB ('compound'; other shapes: every surface
    sample), so a long object lying diagonally is not penalized for its
    empty AABB corners. intrinsics: DEFAULT_INTRINSICS keys (default those).
    """
    d = DEFAULT_INTRINSICS if intrinsics is None else intrinsics
    if exact and obj.shape == 'compound':
        pts = np.array([[x, y, z] for lo, hi in obj.part_aabbs() for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    elif exact:
        pts = obj.surface_points()
    else:
        lo, hi = obj.aabb()
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    for T_cb, v_max in (_views() if views is None else views):
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


# ------------------------------------------------------- official materials
# Organizers' materials (2026-11): flat 0.20 m colored hard-card zones (no walls),
# 50 mm EVA foam cubes (L1) and six plastic fruit / vegetable models (L2).
CARD_SIZE = 0.20                     # m, card side
CUBE_EDGE = 0.050                    # m, L1 foam cube edge (+- CUBE_EDGE_TOL)
CUBE_EDGE_TOL = 0.002
CUBE_MASS = 0.010                    # kg
WIDE_GRIPPER_WIDTH = 0.100           # m, max_width of the new 100 mm stroke gripper
L1_TARGETS = ('red', 'green', 'yellow')
L1_DISTRACTORS = ('pink', 'orange', 'blue', 'purple')
# Distractor colors with a range in config perception.classifier.color.classes; only
# these join tasks.L1.labels (pink has none: it is the dangerous look-alike of red).
L1_KNOWN_DISTRACTORS = ('orange', 'blue', 'purple')
CUBE_COLORS = {'red': (40, 40, 200), 'green': (60, 170, 60), 'yellow': (40, 200, 220),
               'pink': (170, 120, 245), 'orange': (30, 130, 240), 'blue': (200, 90, 40),
               'purple': (150, 50, 120)}
CARD_COLORS = {'green': (70, 165, 60), 'yellow': (50, 205, 230), 'red': (55, 55, 205)}
OFFICIAL_ZONE_NAMES = ('zone_green', 'zone_yellow', 'zone_red')
OFFICIAL_L1_CLASS_TO_ZONE = {'red': 'zone_red', 'green': 'zone_green', 'yellow': 'zone_yellow'}
OFFICIAL_LABELS = ('cucumber', 'pear', 'banana', 'potato', 'persimmon', 'bell_pepper')
OFFICIAL_CLASS_TO_ZONE = {'cucumber': 'zone_green', 'pear': 'zone_green',
                          'banana': 'zone_yellow', 'potato': 'zone_yellow',
                          'persimmon': 'zone_red', 'bell_pepper': 'zone_red'}
OFFICIAL_MASS = {'cucumber': 0.043, 'pear': 0.100, 'banana': 0.050, 'potato': 0.057,
                 'persimmon': 0.100, 'bell_pepper': 0.082}


def _arc_capsules(length: float, r: float, radius: float, n: int) -> List[Dict[str, Any]]:
    """
    Return capsule parts along a planar arc lying on the table (local XY).

    length: overall length including the end caps (m); r: tube radius;
    radius: arc radius of the center line (bows toward +y); n segments. The
    footprint is centered on the origin, the bottom at z = 0.
    """
    ang = (length - 2.0 * r) / radius
    a = np.linspace(-ang / 2.0, ang / 2.0, n + 1)
    pts = np.column_stack([radius * np.sin(a), radius * np.cos(a)])
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    parts = []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d = p1 - p0
        mid = (p0 + p1) / 2.0
        parts.append({'shape': 'capsule', 'size': [r, float(np.linalg.norm(d)) / 2.0],
                      'pos': [float(mid[0]), float(mid[1]), r],
                      'yaw': float(math.atan2(d[1], d[0]))})
    return parts


def _banana_parts() -> List[Dict[str, Any]]:
    """Return the banana: 180 mm curved tube (r 20 mm) with a brown stem tip."""
    r = 0.02
    parts = _arc_capsules(0.15, r, 0.12, 4)
    last = parts[-1]
    yaw = last['yaw']
    t = np.array([math.cos(yaw), math.sin(yaw)])
    end = np.array(last['pos'][:2]) + t * last['size'][1]
    tip = end + t * 0.018
    parts.append({'shape': 'capsule', 'size': [0.006, 0.006],
                  'pos': [float(tip[0]), float(tip[1]), r], 'yaw': yaw,
                  'color_bgr': (40, 75, 95)})
    lo = np.min([np.array(p['pos'][:2]) - p['size'][0] - p['size'][1] for p in parts], axis=0)
    hi = np.max([np.array(p['pos'][:2]) + p['size'][0] + p['size'][1] for p in parts], axis=0)
    shift = (lo + hi) / 2.0
    for p in parts:
        p['pos'] = [p['pos'][0] - float(shift[0]), p['pos'][1] - float(shift[1]), p['pos'][2]]
    return parts


def official_parts(label: str) -> List[Dict[str, Any]]:
    """
    Return the 'compound' parts of an official L2 model (object frame: bottom center, z up).

    Dimensions from the organizers' table: cucumber 260 x 40 mm slightly
    curved; pear 125 mm tall (stem included), 85 mm bulb, narrower neck;
    banana 180 x 40 mm curved; potato 100 x 70 x ~55 mm; persimmon 70 mm
    wide, 50 mm tall with a flat calyx; bell pepper 90 mm tall (stem
    included), 70 mm wide.
    """
    green_stem = (40, 115, 55)
    if label == 'cucumber':
        return _arc_capsules(0.26, 0.02, 0.5, 4)
    if label == 'banana':
        return _banana_parts()
    if label == 'pear':
        return [{'shape': 'sphere', 'size': [0.0425], 'pos': [0.0, 0.0, 0.0425]},
                {'shape': 'ellipsoid', 'size': [0.027, 0.027, 0.030], 'pos': [0.0, 0.0, 0.078]},
                {'shape': 'ellipsoid', 'size': [0.020, 0.020, 0.020], 'pos': [0.0, 0.0, 0.097]},
                {'shape': 'cylinder', 'size': [0.0025, 0.012], 'pos': [0.0, 0.0, 0.119],
                 'color_bgr': (40, 75, 95)}]
    if label == 'potato':
        return [{'shape': 'ellipsoid', 'size': [0.050, 0.035, 0.0275], 'pos': [0.0, 0.0, 0.0275]}]
    if label == 'persimmon':
        return [{'shape': 'ellipsoid', 'size': [0.035, 0.035, 0.0235], 'pos': [0.0, 0.0, 0.0235]},
                {'shape': 'cylinder', 'size': [0.018, 0.003], 'pos': [0.0, 0.0, 0.0485],
                 'color_bgr': green_stem}]
    if label == 'bell_pepper':
        return [{'shape': 'ellipsoid', 'size': [0.032, 0.032, 0.032], 'pos': [0.0, 0.0, 0.032]},
                {'shape': 'ellipsoid', 'size': [0.035, 0.035, 0.026], 'pos': [0.0, 0.0, 0.052]},
                {'shape': 'cylinder', 'size': [0.012, 0.004], 'pos': [0.0, 0.0, 0.077],
                 'color_bgr': green_stem},
                {'shape': 'cylinder', 'size': [0.005, 0.013], 'pos': [0.0, 0.0, 0.0835],
                 'color_bgr': green_stem}]
    raise ValueError('unknown official label %r (expected one of %s)' % (label, OFFICIAL_LABELS))


OFFICIAL_COLORS = {'cucumber': (35, 95, 30), 'pear': (60, 190, 170), 'banana': (40, 200, 230),
                   'potato': (95, 180, 215), 'persimmon': (30, 120, 235),
                   'bell_pepper': (30, 30, 190)}

# Table layouts. cards: zone name -> card center (x, y) in base XY (m); pick:
# pick polygon (>= 2 cm from every card); eye_in_hand: main observe joints (rad) of
# the wrist camera. Card corners (TCP 0.085 m) and the release / retreat poses are
# reachable with the tool leaned outward <= 30 deg (layout_reach_report). The wrist
# views see the pick area (cards are taught zones, not observed).
OFFICIAL_LAYOUTS: Dict[str, Dict[str, Any]] = {
    # A: green card left of the pick area, red card right, yellow in front.
    # TEMPORARY (11/11): fits a 1000 x 500 mm table with the base at the middle of a
    # long edge; the pick area is 0.28 x 0.50 m so that 6 L2 models fit. Replace with
    # the real layout once the organisers give it.
    'A': {'cards': {'zone_green': (0.17, 0.35), 'zone_yellow': (0.49, 0.0),
                    'zone_red': (0.17, -0.35)},
          'pick': [[0.12, -0.25], [0.40, -0.25], [0.40, 0.25], [0.12, 0.25]],
          'eye_in_hand': (0.0, 0.486, -0.47, 0.0, 1.168, 0.0),
          # wrist observe views: eye_in_hand joints with joint1 turned by these (rad), in
          # scan order (right, middle, left); one view does not cover the pick area
          'wrist_yaws': (-0.5, 0.0, 0.5), 'wrist_lift': 0.0},
    # B: the three cards in a row in front of the pick area.
    'B': {'cards': {'zone_green': (0.43, 0.205), 'zone_yellow': (0.43, 0.0),
                    'zone_red': (0.43, -0.205)},
          'pick': [[0.16, -0.18], [0.305, -0.23], [0.305, 0.23], [0.16, 0.18]],
          'eye_in_hand': (0.0, 0.486, -0.495, 0.0, 1.168, 0.0),
          'wrist_yaws': (-0.45, 0.0, 0.45), 'wrist_lift': 0.1},
    # S: the debug station of 11/12 (cards taught with teach_poses, far from the base:
    # red and green centers are out of reach, releases go to their near part). The
    # wrist camera is the hand-eye calibrated one with the camera_info intrinsics;
    # the gripper hides the image rows below 320. No pick area is marked on the
    # table: objects lie between the base and the cards and the three views (joint1
    # turned) are scanned; an L1 object only has to be whole in one of them.
    'S': {'cards': {'zone_green': (0.5731, -0.2824), 'zone_yellow': (0.5837, 0.0147),
                    'zone_red': (0.5979, 0.3105)},
          'pick': [[0.20, -0.22], [0.42, -0.22], [0.42, 0.22], [0.20, 0.22]],
          'eye_in_hand': (0.0, 0.15, -0.5, 0.0, 1.1, 0.375),
          'views': ((-0.6, 0.15, -0.5, 0.0, 1.1, 0.375), (0.0, 0.15, -0.5, 0.0, 1.1, 0.375),
                    (0.6, 0.15, -0.5, 0.0, 1.1, 0.375)),
          'l1_any_view': True,
          'T_flange_camera': ((0.01685, 0.952586, 0.303803, -0.074811),
                              (-0.999745, 0.011473, 0.019475, -0.010645),
                              (0.015066, -0.304053, 0.952536, 0.039922),
                              (0.0, 0.0, 0.0, 1.0)),
          'intrinsics': dict(fx=488.6, fy=488.6, cx=319.8, cy=209.9, width=640, height=480),
          'bottom_px': 160},
}


def layout_T_flange_camera(layout: str = 'A') -> np.ndarray:
    """Return the wrist camera pose in the flange frame of a table layout."""
    T = OFFICIAL_LAYOUTS[layout].get('T_flange_camera')
    if T is None:
        return eye_in_hand_T_flange_camera()
    R, _ = np.linalg.qr(np.array(T, dtype=float)[:3, :3])
    R *= np.sign(np.diag(R))
    return make_T(R, np.array(T, dtype=float)[:3, 3])


def layout_intrinsics(layout: str = 'A') -> Dict[str, Any]:
    """Return the color intrinsics (DEFAULT_INTRINSICS keys) of a table layout."""
    return dict(OFFICIAL_LAYOUTS[layout].get('intrinsics') or DEFAULT_INTRINSICS)


def layout_bottom_px(layout: str = 'A') -> int:
    """Return the bottom image rows of a layout's wrist view that show the gripper."""
    return int(OFFICIAL_LAYOUTS[layout].get('bottom_px', EYE_IN_HAND_BOTTOM_PX))


# ------------------------------------------------------------------ sampling
EDGE_MARGIN = 0.01        # m, object footprint corners stay this far inside the pick polygon
REACH_R = (0.17, 0.43)    # m, object centers within the vertical-tool grasp reach
VIEW_MARGIN_PX = 4        # objects stay this far inside the observe images
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


def parts_gap(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    """Return the smallest separation (m, < 0 overlap) between two sets of convex footprints."""
    return min(_gap_between(p, q) for p in a for q in b)


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
                  tries: int = 4000, visible: Any = None,
                  reach: Tuple[float, float] = REACH_R) -> None:
    """
    Place obj upright at a random pose inside polygon, gap (m) from every placed object.

    visible: obj -> bool (default fully_visible, the legacy views); reach:
    (min, max) horizontal radius of the center (m).
    """
    visible = fully_visible if visible is None else visible
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
        if not reach[0] <= math.hypot(x, y) <= reach[1]:
            continue
        obj.place_upright(x, y, yaw, 0.0)
        fp = _footprint(obj)
        if not _inside(fp, polygon, EDGE_MARGIN) or not visible(obj):
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
    if obj.shape == 'compound':
        g = find_grasp(world, obj, max_width, open_margin, steps)
        return None if g is None else (g['closing'], g['width'])
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


def find_grasp(world: SimWorld, obj: SimObject, max_width: float = GRIPPER_MAX_WIDTH,
               open_margin: float = 0.02, steps: int = 24, min_depth: float = 0.008,
               max_depth: float = 0.030, z_step: float = 0.0025, max_checks: int = 48
               ) -> Optional[Dict[str, Any]]:
    """
    Return the narrowest working top-down grasp of obj, or None.

    Result keys: closing (rad), width (m, the section between the pads), x, y,
    z (TCP, m), depth (m below the top), open (opening used, m). Other shapes:
    find_top_down_grasp's single height. 'compound' objects: TCP heights
    min_depth .. max_depth below the top (grasp.min_grasp_depth /
    max_grasp_depth, at least 8 mm above the bottom) every z_step, closing
    angles k * pi / steps, the TCP centered on the pad-band section (see
    SimWorld.check_grasp); a section must fit max_width - 5 mm
    (grasp.max_object_width default), opening min(width + open_margin,
    max_width). Narrowest first, then lowest; at most max_checks candidates
    are checked against the world (neighbors, collisions, slipping).
    """
    if obj.shape != 'compound':
        g = find_top_down_grasp(world, obj, max_width, open_margin, steps)
        if g is None:
            return None
        c, top, h = obj.center_xy(), obj.top_z(), obj.height()
        z = top - min(max(0.5 * h, 0.008), 0.03)
        return {'closing': g[0], 'width': g[1], 'x': float(c[0]), 'y': float(c[1]),
                'z': z, 'depth': top - z, 'open': min(g[1] + open_margin, max_width)}
    c, top = obj.center_xy(), obj.top_z()
    cands = [t for t in compound_sections(obj, steps, min_depth, max_depth, z_step)
             if t[0] <= max_width - 0.005]
    cands.sort(key=lambda t: (t[0], t[1]))
    for w, z, a, off in cands[:max_checks]:
        x = float(c[0] + off * math.cos(a))
        y = float(c[1] + off * math.sin(a))
        ow = min(w + open_margin, max_width)
        ok, info = world.check_grasp(top_down_T(x, y, z, a), ow, 0.0)
        if ok and info['object'] == obj.name:
            return {'closing': a, 'width': float(info['width']), 'x': x, 'y': y, 'z': z,
                    'depth': top - z, 'open': ow}
    return None


def compound_sections(obj: SimObject, steps: int = 24, min_depth: float = 0.008,
                      max_depth: float = 0.030, z_step: float = 0.0025
                      ) -> List[Tuple[float, float, float, float]]:
    """
    Return the pad-band sections of an object as [(width m, TCP z m, closing rad, offset m)].

    For every TCP z min_depth .. max_depth below the top (at least 8 mm above
    the bottom, every z_step) and closing angle k * pi / steps through the
    footprint center: the extent along the closing axis of the material
    within the pad width (tool x) and GRASP_PAD_HALF of the TCP z, and the
    offset of its middle from the footprint center along the closing axis.
    """
    c, top, bottom = obj.center_xy(), obj.top_z(), obj.bottom_z()
    pts = obj.surface_points()
    z_hi = top - min_depth
    z_lo = max(top - max_depth, bottom + 0.008)
    n = int(math.floor((z_hi - z_lo) / z_step + 1e-9)) + 1 if z_hi >= z_lo else 0
    zs = [z_lo + k * z_step for k in range(n)]
    rel = pts[:, :2] - c
    out = []
    for k in range(steps):
        a = k * math.pi / steps
        along_x = rel @ np.array([-math.sin(a), math.cos(a)])   # tool x of top_down_T(.., a)
        along_y = rel @ np.array([math.cos(a), math.sin(a)])    # tool y (closing axis)
        in_pad = np.abs(along_x) <= FINGER_SIZE[0] / 2.0
        for z in zs:
            sel = in_pad & (np.abs(pts[:, 2] - z) <= GRASP_PAD_HALF)
            if np.any(sel):
                lo, hi = float(along_y[sel].min()), float(along_y[sel].max())
                out.append((round(hi - lo, 6), z, a, (lo + hi) / 2.0))
    return out


def _check_graspable(objects: List[SimObject]) -> List[SimObject]:
    world = SimWorld(objects=objects)
    for obj in objects:
        if find_top_down_grasp(world, obj) is None:
            raise RuntimeError('%s has no collision-free grasp' % obj.name)
    return objects


def _draw_labels(task: str, hard: bool, rng: np.random.Generator) -> List[str]:
    """Draw the 6 object labels of an L2 scene in a random order."""
    if hard:
        chosen = [str(v) for v in rng.choice(sorted(CATALOG), 6, replace=True)]
    else:
        chosen = ([str(v) for v in rng.choice(FRUITS, 3, replace=False)]
                  + [str(v) for v in rng.choice(VEGETABLES, 3, replace=False)])
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


# ------------------------------------------------------- official scenarios
OFFICIAL_GAP = {'normal': 0.015, 'hard': 0.008}   # m, footprint gap between L2 models
OFFICIAL_ATTEMPTS = 45               # layout attempts (gaps shrink after each third)
OFFICIAL_TRIES = 200                 # random poses tried per object and attempt
PACK_JITTER = 0.03                   # m, random slack added when packing L2 models
CUBE_GAP = 0.03                      # m, footprint gap between separate L1 cubes
CUBE_MIN_CENTER_DIST = 0.08          # m
OFFICIAL_REACH_R = (0.15, 0.45)      # m, horizontal radius of the object centers
CATEGORIES = {'cucumber': 'vegetable', 'pear': 'fruit', 'banana': 'fruit',
              'potato': 'vegetable', 'persimmon': 'fruit', 'bell_pepper': 'vegetable'}
# Default config values used by the reach report (place.release_height,
# place.retreat_height, grasp.approach_height) and the TCP z of a 50 mm cube released
# there, used for the card corners.
RELEASE_HEIGHT = 0.06
RETREAT_HEIGHT = 0.10
APPROACH_HEIGHT = 0.08
CARD_CHECK_Z = RELEASE_HEIGHT + CUBE_EDGE / 2.0
MAX_TILT_DEG = 30.0                  # config robot.workspace.max_tilt_from_down_deg


def official_zones(layout: str = 'A') -> List[SimZone]:
    """Return the three flat 0.20 m cards of a table layout (OFFICIAL_LAYOUTS)."""
    lay = OFFICIAL_LAYOUTS[layout]
    out = []
    for name in OFFICIAL_ZONE_NAMES:
        color = name[len('zone_'):]
        c = lay['cards'][name]
        yaw = float(c[2]) if len(c) > 2 else 0.0
        out.append(SimZone(name, (c[0], c[1], 0.0), (CARD_SIZE, CARD_SIZE), yaw,
                           CARD_COLORS[color], 0.0, color))
    return out


@functools.lru_cache(maxsize=4)
def official_observe(layout: str = 'A'
                     ) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    """Return (observe joints, T_base_camera) of the wrist camera of a table layout."""
    q = tuple(float(v) for v in OFFICIAL_LAYOUTS[layout]['eye_in_hand'])
    if not within_limits(q, 0.02):
        raise RuntimeError('layout %s: eye-in-hand observe joints outside the limits' % layout)
    T = piper_fk(q) @ layout_T_flange_camera(layout)
    return q, tuple(tuple(float(v) for v in row) for row in T)


@functools.lru_cache(maxsize=4)
def official_wrist_views(layout: str = 'A'
                         ) -> Tuple[Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]], ...]:
    """
    Return ((observe joints, T_base_camera), ...) of the wrist observe views of a layout.

    The layout's explicit views, else the eye_in_hand joints with joint1 turned by
    each of the layout's wrist_yaws and joint2 raised by wrist_lift, in scan order
    (robot.poses.observe_views).
    """
    lay = OFFICIAL_LAYOUTS[layout]
    out = []
    if lay.get('views'):
        joints = [list(float(v) for v in q) for q in lay['views']]
    else:
        joints = []
        for yaw in lay.get('wrist_yaws', (0.0,)):
            q = list(float(v) for v in lay['eye_in_hand'])
            q[0] += float(yaw)
            if yaw:
                q[1] += float(lay.get('wrist_lift', 0.0))
            joints.append(q)
    for q in joints:
        if not within_limits(tuple(q), 0.02):
            raise RuntimeError('layout %s: wrist view %s outside the joint limits' % (layout, q))
        T = piper_fk(tuple(q)) @ layout_T_flange_camera(layout)
        out.append((tuple(q), tuple(tuple(float(v) for v in row) for row in T)))
    return tuple(out)


def official_visible(obj: SimObject, layout: str = 'A', any_wrist_view: bool = True) -> bool:
    """
    Return True when obj is whole in the wrist camera image.

    any_wrist_view True (L2: objects may lie outside one view): in at least one of
    the wrist observe views; False (L1: objects inside the camera view): in the
    main wrist observe view (official_observe).
    """
    intr = layout_intrinsics(layout)
    h = intr['height']

    def seen(views) -> bool:
        return (fully_visible(obj, views, intrinsics=intr)
                or fully_visible(obj, views, exact=True, intrinsics=intr))

    wrist = ([T for _q, T in official_wrist_views(layout)] if any_wrist_view
             else [official_observe(layout)[1]])
    return any(seen(((inv_T(np.array(T)), h - layout_bottom_px(layout)),)) for T in wrist)


@functools.lru_cache(maxsize=4)
def official_views(layout: str = 'A') -> Tuple[Tuple[np.ndarray, int], ...]:
    """Return ((T_camera_base, max image row),) of the main wrist observe view of a layout."""
    eih = np.array(official_observe(layout)[1])
    return ((inv_T(eih), layout_intrinsics(layout)['height'] - layout_bottom_px(layout)),)


def _image_margin(pts: np.ndarray, T_cb: np.ndarray, v_max: float) -> float:
    """Return the smallest distance (px) of projected points inside the image (< 0 outside)."""
    d = DEFAULT_INTRINSICS
    pc = np.asarray(pts, dtype=float) @ T_cb[:3, :3].T + T_cb[:3, 3]
    if np.any(pc[:, 2] <= 0.05):
        return -math.inf
    u = d['fx'] * pc[:, 0] / pc[:, 2] + d['cx']
    v = d['fy'] * pc[:, 1] / pc[:, 2] + d['cy']
    return float(min(u.min(), d['width'] - 1 - u.max(), v.min(), v_max - v.max()))


def layout_view_report(layout: str = 'A') -> Dict[str, Dict[str, float]]:
    """
    Return the image margins (px, < 0 = outside) of the pick area and the cards.

    {'eye_in_hand': {'pick': pick polygon corners on the table, 'cards': every card
    corner}} for the main wrist view. It excludes its bottom EYE_IN_HAND_BOTTOM_PX rows
    (fingers); from a reachable observe pose it cannot also see the cards,
    which are taught zones there.
    """
    pick = np.array([[x, y, 0.0] for x, y in OFFICIAL_LAYOUTS[layout]['pick']])
    cards = np.array([[c[0], c[1], 0.0] for z in official_zones(layout)
                      for c in z.corners_xy()])
    out = {}
    for mount, (T_cb, v_max) in zip(MOUNTS, official_views(layout)):
        out[mount] = {'pick': round(_image_margin(pick, T_cb, v_max), 1),
                      'cards': round(_image_margin(cards, T_cb, v_max), 1)}
    return out


def official_cube(color: str, rng: Optional[np.random.Generator] = None,
                  edge: Optional[float] = None, name: Optional[str] = None) -> SimObject:
    """Return an L1 foam cube (edge CUBE_EDGE +- CUBE_EDGE_TOL drawn from rng, else edge)."""
    if edge is None:
        edge = (CUBE_EDGE if rng is None else
                float(rng.uniform(CUBE_EDGE - CUBE_EDGE_TOL, CUBE_EDGE + CUBE_EDGE_TOL)))
    base = CUBE_COLORS[color]
    return SimObject(name or '%s_cube' % color, color, 'box', (edge, edge, edge),
                     base if rng is None else _jitter(base, rng), np.eye(4), mass=CUBE_MASS,
                     material='foam')


def official_model(name: str, label: str, rng: Optional[np.random.Generator] = None
                   ) -> SimObject:
    """Return an official L2 plastic model ('compound'; color jittered with rng)."""
    base = OFFICIAL_COLORS[label]
    return SimObject(name, label, 'compound', official_parts(label),
                     base if rng is None else _jitter(base, rng), np.eye(4),
                     mass=OFFICIAL_MASS[label], material='plastic')


# Grasp candidates checked per object when deciding graspability; a class and its
# objects in a scene must use the same count or a class rated graspable alone
# (e.g. the bell pepper, first grasp after ~50 candidates) fails every scene.
CLASS_GRASP_CHECKS = 400


def _probe(label: str) -> SimObject:
    """Return an isolated object of a class at (0.30, 0) (L1 colors: a 52 mm cube)."""
    if label in OFFICIAL_LABELS:
        obj = official_model('probe', label)
    else:
        obj = official_cube(label if label in CUBE_COLORS else 'red',
                            edge=CUBE_EDGE + CUBE_EDGE_TOL, name='probe')
    obj.place_upright(0.30, 0.0, 0.0, 0.0)
    return obj


@functools.lru_cache(maxsize=128)
def _class_grasp(label: str, max_width: float) -> Tuple[Optional[Tuple[Tuple[str, float], ...]],
                                                        str]:
    obj = _probe(label)
    world = SimWorld(objects=[obj])
    g = find_grasp(world, obj, max_width, max_checks=CLASS_GRASP_CHECKS)
    if g is not None:
        g = dict(g, held_offset=g['z'] - obj.bottom_z(), height=obj.height(),
                 footprint=float(np.max(obj.aabb()[1][:2] - obj.aabb()[0][:2])))
        return tuple(sorted((k, float(v)) for k, v in g.items())), ''
    return None, _no_grasp_reason(world, obj, max_width)


def class_grasp(label: str, max_width: float = GRIPPER_MAX_WIDTH
                ) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Return (grasp of an isolated object of a class or None, reason).

    The grasp is find_grasp's dict plus held_offset (TCP height above the
    object bottom, m), height and footprint (largest horizontal extent, m).
    L1 color labels mean a 52 mm (largest tolerated) foam cube. reason is ''
    when a grasp exists, else why none does with this max_width (m).
    """
    g, why = _class_grasp(str(label), round(float(max_width), 6))
    return (None if g is None else dict(g)), why


def _no_grasp_reason(world: SimWorld, obj: SimObject, max_width: float) -> str:
    """Return why find_grasp found no grasp of an isolated object."""
    fit = max_width - 0.005
    if obj.shape != 'compound':
        w = min(obj.footprint_extent((math.cos(a), math.sin(a)))
                for a in np.linspace(0.0, math.pi, 24, endpoint=False))
        return ('too_wide: narrowest width %.3f m > %.3f m (max_width %.3f m - 5 mm)'
                % (w, fit, max_width))
    secs = compound_sections(obj)
    if not secs:
        return 'no_material: nothing between the pads at any grasp height'
    w_min = min(t[0] for t in secs)
    if w_min > fit:
        return ('too_wide: narrowest section between the pads %.3f m > %.3f m (max_width '
                '%.3f m - 5 mm) at every grasp height (8-30 mm below the top)'
                % (w_min, fit, max_width))
    w, z, a, off = min((t for t in secs if t[0] <= fit), key=lambda t: (t[0], t[1]))
    c = obj.center_xy()
    T = top_down_T(c[0] + off * math.cos(a), c[1] + off * math.sin(a), z, a)
    _ok, info = world.check_grasp(T, min(w + 0.02, max_width), 0.0)
    keys = ('extent', 'taper_deg', 'collided')
    detail = ', '.join('%s %s' % (k, info[k]) for k in keys if k in info) or 'see check_grasp'
    return ('%s: the sections that fit (narrowest %.3f m, %.0f mm below the top) are not '
            'held (%s)' % (info.get('reason') or 'no_grasp', w, 1000.0 * (obj.top_z() - z),
                           detail))


def official_graspability(max_width: float = GRIPPER_MAX_WIDTH) -> Dict[str, str]:
    """Return label -> '' (an isolated object can be grasped top-down) or the reason not."""
    return {lab: class_grasp(lab, max_width)[1] for lab in ('cube',) + OFFICIAL_LABELS}


def _require_grasps(objects: List[SimObject], max_width: float) -> List[SimObject]:
    """Raise RuntimeError unless every object of a graspable class has a grasp among the others."""
    world = SimWorld(objects=objects)
    for obj in objects:
        if class_grasp(obj.label, max_width)[0] is None:
            continue
        if find_grasp(world, obj, max_width, max_checks=CLASS_GRASP_CHECKS) is None:
            raise RuntimeError('%s has no collision-free grasp' % obj.name)
    return objects


def _official_l1_objects(rng: np.random.Generator, polygon: np.ndarray, hard: bool,
                         max_width: float, visible: Any, gap_scale: float = 1.0
                         ) -> List[SimObject]:
    """Return the cubes of an official L1 scene ('hard': distractors, two touching)."""
    cubes = [official_cube(str(c), rng) for c in rng.permutation(L1_TARGETS)]
    extra: List[SimObject] = []
    if hard:
        n = int(rng.integers(1, 3))
        extra = [official_cube(str(c), rng)
                 for c in rng.choice(L1_DISTRACTORS, n, replace=False)]
    gap = CUBE_GAP * gap_scale
    kw = dict(tries=300, visible=visible, reach=OFFICIAL_REACH_R)
    objects: List[SimObject] = []
    rest = cubes + extra
    if hard:
        a, b = cubes[0], cubes[1]
        _place_random(a, objects, rng, polygon, gap, CUBE_MIN_CENTER_DIST, **kw)
        objects.append(a)
        side = float(rng.choice([-1.0, 1.0]))
        lat = float(rng.uniform(-0.3, 0.3)) * a.size[0]
        off = a.T_base_obj[:3, :3] @ np.array([side * (a.size[0] + b.size[0]) / 2.0, lat, 0.0])
        c = a.center_xy() + off[:2]
        b.place_upright(c[0], c[1], a.yaw(), 0.0)
        if (not _inside(_footprint(b), polygon, EDGE_MARGIN)
                or not OFFICIAL_REACH_R[0] <= math.hypot(c[0], c[1]) <= OFFICIAL_REACH_R[1]
                or not visible(b)):
            raise RuntimeError('touching cube outside the pick area')
        objects.append(b)
        rest = cubes[2:] + extra
    for obj in rest:
        _place_random(obj, objects, rng, polygon, gap, CUBE_MIN_CENTER_DIST, **kw)
        objects.append(obj)
    return _require_grasps(objects, max_width)


def _place_packed(obj: SimObject, placed: List[SimObject], rng: np.random.Generator,
                  polygon: np.ndarray, gap: float, visible: Any,
                  reach: Tuple[float, float] = OFFICIAL_REACH_R, tries: int = OFFICIAL_TRIES,
                  jitter: float = PACK_JITTER) -> None:
    """
    Place obj upright at the tightest of `tries` random valid poses.

    A pose is valid inside polygon (EDGE_MARGIN), within reach, visible and
    gap (m) from every placed footprint. Its slack is the smallest clearance
    to the polygon edge or a placed footprint beyond the required margin;
    the pose with the least slack plus a random jitter (m) wins, so objects
    pack against the edges and each other and dense official layouts fit.
    Elongated objects take a yaw along the polygon's longer side (+- 0.35
    rad) in half of the tries. Raises RuntimeError when no pose is valid.
    """
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    llo, lhi = obj.local_aabb()
    ext = lhi[:2] - llo[:2]
    elongated = float(ext.max()) > 2.0 * float(ext.min())
    long_axis = 0.0 if ext[0] >= ext[1] else math.pi / 2.0
    side = 0.0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else math.pi / 2.0
    n_in, off_in = _inward(polygon)
    others = []
    for o in placed:
        fps = o.footprint_polygons()
        c = o.center_xy()
        others.append((c, fps, float(np.max(np.linalg.norm(np.vstack(fps) - c, axis=1)))))
    best = None
    for _ in range(tries):
        x, y = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
        yaw = rng.uniform(-math.pi, math.pi)
        if elongated and rng.random() < 0.5:
            yaw = side - long_axis + rng.normal(0.0, 0.35)
        noise = jitter * rng.random()
        if not reach[0] <= math.hypot(x, y) <= reach[1]:
            continue
        obj.place_upright(x, y, yaw, 0.0)
        fps = obj.footprint_polygons()
        allp = np.vstack(fps)
        edge = float(np.min(allp @ n_in.T - off_in)) - EDGE_MARGIN
        if edge < 0.0:
            continue
        c = obj.center_xy()
        rad = float(np.max(np.linalg.norm(allp - c, axis=1)))
        slack = edge
        for oc, ofps, orad in others:
            d = float(np.linalg.norm(oc - c))
            if d >= rad + orad + gap + slack:
                continue
            slack = min(slack, parts_gap(fps, ofps) - gap)
            if slack < 0.0:
                break
        if slack < 0.0 or (best is not None and slack + noise >= best[0]):
            continue
        if not visible(obj):
            continue
        best = (slack + noise, x, y, yaw)
    if best is None:
        raise RuntimeError('could not place %s without overlap' % obj.name)
    obj.place_upright(best[1], best[2], best[3], 0.0)


def _official_models(chosen: List[str], rng: np.random.Generator, polygon: np.ndarray,
                     gap: float, max_width: float, visible: Any, gap_scale: float = 1.0
                     ) -> List[SimObject]:
    """Return the L2 models of chosen (labels), packed largest first (_place_packed)."""
    objs = [official_model('%s_%d' % (lab, i), lab, rng) for i, lab in enumerate(chosen)]
    placed: List[SimObject] = []
    size = {o.name: float(np.prod(o.local_aabb()[1][:2] - o.local_aabb()[0][:2])) for o in objs}
    for obj in sorted(objs, key=lambda o: -size[o.name]):
        _place_packed(obj, placed, rng, polygon, gap * gap_scale, visible)
        placed.append(obj)
    return _require_grasps(objs, max_width)


def _official_scenario(task: str, seed: int, mount: str, difficulty: str, layout: str,
                       gripper_width: Optional[float]) -> Tuple[SimWorld, Dict[str, Any]]:
    """Build a scene from the official materials (see make_scenario)."""
    width = GRIPPER_MAX_WIDTH if gripper_width is None else float(gripper_width)
    rng = np.random.default_rng([int(seed), TASKS.index(task), DIFFICULTIES.index(difficulty),
                                 LAYOUTS.index(layout), 2026])
    lay = OFFICIAL_LAYOUTS[layout]
    polygon = np.array(lay['pick'], dtype=float)
    hard = difficulty == 'hard'

    def visible(obj: SimObject) -> bool:
        return official_visible(obj, layout,
                                any_wrist_view=task == 'L2' or bool(lay.get('l1_any_view')))

    zones = official_zones(layout)
    info: Dict[str, Any] = {'task': task, 'seed': int(seed), 'mount': mount,
                            'difficulty': difficulty, 'materials': 'official',
                            'layout': layout, 'gripper_width': gripper_width}
    if task == 'L1':
        objects = _retry_layout(lambda g: _official_l1_objects(rng, polygon, hard, width,
                                                               visible, g), OFFICIAL_ATTEMPTS)
        present = {o.label for o in objects}
        labels = list(L1_TARGETS) + [c for c in L1_KNOWN_DISTRACTORS if c in present]
        class_to_zone = dict(OFFICIAL_L1_CLASS_TO_ZONE)
        categories: Dict[str, str] = {}
        info['time_limit_s'] = 180.0
    else:
        chosen = [str(v) for v in rng.permutation(OFFICIAL_LABELS)]
        gap = OFFICIAL_GAP[difficulty]
        objects = _retry_layout(lambda g: _official_models(chosen, rng, polygon, gap, width,
                                                           visible, g), OFFICIAL_ATTEMPTS)
        labels = [k for k in OFFICIAL_LABELS if k in chosen]
        categories = {k: CATEGORIES[k] for k in labels}
        class_to_zone = {k: OFFICIAL_CLASS_TO_ZONE[k] for k in labels}
        info['time_limit_s'] = 300.0
    world = SimWorld(table_z=0.0, objects=objects, zones=zones, seed=int(seed),
                     min_zone_fraction=OFFICIAL_MIN_ZONE_FRACTION,
                     zone_fraction_by_label=OFFICIAL_ZONE_FRACTION_BY_LABEL)
    why = {o.label: class_grasp(o.label, width)[1] for o in objects}
    info.update(labels=labels, class_to_zone=class_to_zone, categories=categories,
                zones={z.name: z.to_zone() for z in zones},
                pick_polygon=[[float(v) for v in p] for p in lay['pick']],
                intrinsics=CameraIntrinsics(**{k: layout_intrinsics(layout)[k] for k in
                                               ('fx', 'fy', 'cx', 'cy', 'width', 'height')}),
                objects=[{'name': o.name, 'label': o.label} for o in objects],
                required=sum(1 for o in objects if o.label in class_to_zone),
                ungraspable={k: v for k, v in sorted(why.items()) if v})
    info['graspable_required'] = sum(1 for o in objects
                                     if o.label in class_to_zone and not why[o.label])
    q, T_obs = official_observe(layout)
    info['T_flange_camera'] = layout_T_flange_camera(layout)
    info['T_base_camera_observe'] = np.array(T_obs)
    info['observe_joints'] = list(q)
    info['observe_views'] = [list(v) for v, _T in official_wrist_views(layout)]
    return world, info


@functools.lru_cache(maxsize=16)
def _reach_tilt(x: float, y: float, z: float, tcp_offset: Tuple[float, ...],
                step_deg: float) -> Optional[float]:
    """Return the least outward lean (deg, multiple of step_deg) that makes a pose reachable."""
    from ..robot.kinematics import PiperKinematics
    T = top_down_T(x, y, z, math.atan2(y, x) + math.pi / 2.0)
    res = PiperKinematics().solve_tcp_with_tilt(T, pose6_to_T(tcp_offset),
                                                max_tilt=math.radians(MAX_TILT_DEG),
                                                step=math.radians(step_deg))
    return None if res is None else round(math.degrees(res[2]), 3)


def _release_xys(zone: SimZone, footprint: float) -> List[Tuple[float, float]]:
    """Return the zone center and the PlacePlanner release slots for this footprint."""
    from ..perception.grasp import release_slots
    c = (float(zone.center[0]), float(zone.center[1]))
    return [c] + [(float(x), float(y)) for x, y in release_slots(zone.to_zone(), footprint)]


def layout_reach_report(layout: str = 'A', gripper_width: Optional[float] = None,
                        tcp_offset: Sequence[float] = DEFAULT_TCP_OFFSET,
                        step_deg: float = 5.0) -> Dict[str, Any]:
    """
    Return the reach margins of a table layout (SafeArm lean rule, PiperKinematics).

    Every pose is top-down with the closing axis tangential, leaned outward
    by the least multiple of step_deg up to MAX_TILT_DEG that is reachable.
    Checked: each card's corners at CARD_CHECK_Z; for every class mapped to
    the card (L1 cube and L2 models that gripper_width can grasp) the zone
    center and release slots at the release TCP z (RELEASE_HEIGHT + held
    offset) and the retreat (+ RETREAT_HEIGHT); the pick polygon corners at
    the highest grasp TCP z and at the highest pre-grasp z (top +
    APPROACH_HEIGHT). Returns {'zones': {name: entry}, 'pick': entry, 'ok':
    bool, 'n_poses'} with entry {'min_margin_deg': MAX_TILT_DEG - largest
    lean needed (None if a pose is unreachable), 'worst': [x, y, z, lean or
    None], 'unreachable': [[x, y, z], ...]}.
    """
    width = GRIPPER_MAX_WIDTH if gripper_width is None else float(gripper_width)
    off = tuple(float(v) for v in tcp_offset)
    grasps = {lab: class_grasp(lab, width)[0] for lab in ('red',) + OFFICIAL_LABELS}
    by_zone: Dict[str, List[Dict[str, float]]] = {n: [] for n in OFFICIAL_ZONE_NAMES}
    for lab, zname in list(OFFICIAL_L1_CLASS_TO_ZONE.items()) + list(
            OFFICIAL_CLASS_TO_ZONE.items()):
        g = grasps['red' if lab in L1_TARGETS else lab]
        if g is not None:
            by_zone[zname].append(g)
    n = [0]

    def entry(poses: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        res = []
        for x, y, z in dict.fromkeys((round(x, 4), round(y, 4), round(z, 4))
                                     for x, y, z in poses):
            res.append((x, y, z, _reach_tilt(x, y, z, off, float(step_deg))))
        n[0] += len(res)
        bad = [[x, y, z] for x, y, z, t in res if t is None]
        worst = max(res, key=lambda r: math.inf if r[3] is None else r[3])
        return {'min_margin_deg': None if bad else round(MAX_TILT_DEG - worst[3], 1),
                'worst': [worst[0], worst[1], worst[2], worst[3]], 'unreachable': bad}

    out: Dict[str, Any] = {'layout': layout, 'gripper_width': width, 'tcp_offset': list(off),
                           'zones': {}}
    for zone in official_zones(layout):
        poses = [(float(c[0]), float(c[1]), CARD_CHECK_Z) for c in zone.corners_xy()]
        for g in by_zone[zone.name]:
            z_rel = zone.floor_z + RELEASE_HEIGHT + g['held_offset']
            for x, y in _release_xys(zone, g['footprint']):
                poses += [(x, y, z_rel), (x, y, z_rel + RETREAT_HEIGHT)]
        out['zones'][zone.name] = entry(poses)
    ok_g = [g for g in grasps.values() if g is not None]
    z_grasp = max(g['z'] for g in ok_g)
    z_pre = max(g['height'] for g in ok_g) + APPROACH_HEIGHT
    corners = OFFICIAL_LAYOUTS[layout]['pick']
    out['pick'] = entry([(float(x), float(y), z) for x, y in corners for z in (z_grasp, z_pre)])
    out['ok'] = all(e['min_margin_deg'] is not None
                    for e in list(out['zones'].values()) + [out['pick']])
    out['n_poses'] = n[0]
    return out


# ----------------------------------------------------------------- scenarios
def make_scenario(task: str, seed: int, mount: str = 'eye_in_hand',
                  difficulty: str = 'normal', materials: str = 'official', layout: str = 'A',
                  gripper_width: Optional[float] = None) -> Tuple[SimWorld, Dict[str, Any]]:
    """
    Build a seeded scene. Return (SimWorld, info).

    materials 'official' (default): the organizers' materials. L1: three
    50 mm foam cubes (red / green / yellow) for three flat 0.20 m cards
    zone_red / zone_green / zone_yellow; 'hard' adds 1-2 distractor cubes
    (pink / orange / blue / purple) and two touching cubes. L2: the six
    plastic models (cucumber, pear, banana, potato, persimmon, bell_pepper,
    'compound' shapes) for the cards (OFFICIAL_CLASS_TO_ZONE); 'hard' uses
    smaller gaps. layout: table layout 'A' or 'B' (LAYOUTS: card positions,
    pick polygon, cameras). gripper_width: robot.gripper.max_width (m) the
    scene is checked against (None = GRIPPER_MAX_WIDTH, the standard
    gripper); objects of a class that no top-down grasp fits are still
    placed and listed in info['ungraspable'] (label -> reason).

    materials 'legacy': the old made-up catalog, exactly as before (layout
    and gripper_width only go to info / the overlay): L1 3 colored cubes,
    3 mats ('hard': two touching cubes and a yellow distractor); L2 6 fruit /
    vegetable models, 2 walled bins ('hard': smaller gaps, every catalog
    label).

    info keys: task, seed, mount, difficulty, materials, layout,
    gripper_width, labels, class_to_zone, categories, zones (name -> core
    Zone), pick_polygon, intrinsics (CameraIntrinsics), T_flange_camera and
    T_base_camera_observe (the wrist camera; mount must be 'eye_in_hand'),
    observe_joints, objects, required, time_limit_s, plus for
    official scenes ungraspable and graspable_required (targets some
    top-down grasp fits). Deterministic for a given argument set.
    """
    if task not in TASKS:
        raise ValueError('task must be one of %s, got %r' % (TASKS, task))
    if mount not in MOUNTS:
        raise ValueError('mount must be one of %s, got %r' % (MOUNTS, mount))
    if difficulty not in DIFFICULTIES:
        raise ValueError('difficulty must be one of %s, got %r' % (DIFFICULTIES, difficulty))
    if materials not in MATERIALS:
        raise ValueError('materials must be one of %s, got %r' % (MATERIALS, materials))
    if layout not in LAYOUTS:
        raise ValueError('layout must be one of %s, got %r' % (LAYOUTS, layout))
    gw = None
    if gripper_width is not None:
        gw = float(gripper_width)
        if not math.isfinite(gw) or not 0.03 <= gw <= 0.20:
            raise ValueError('gripper_width must be in [0.03, 0.20] m, got %r' % (gripper_width,))
    if materials == 'legacy':
        world, info = _legacy_scenario(task, seed, mount, difficulty)
        info.update(materials='legacy', layout=layout, gripper_width=gw)
        return world, info
    return _official_scenario(task, int(seed), mount, difficulty, layout, gw)


def _legacy_scenario(task: str, seed: int, mount: str, difficulty: str
                     ) -> Tuple[SimWorld, Dict[str, Any]]:
    """Build a scene from the legacy catalog (see make_scenario)."""
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
        zones = _make_zones(L2_ZONES)
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
        class_to_zone = {k: ('fruit_bin' if CATALOG[k][0] == 'fruit' else 'veg_bin')
                         for k in used}
        info['time_limit_s'] = 300.0
    world = SimWorld(table_z=0.0, objects=objects, zones=zones, seed=int(seed))
    info.update(labels=labels, class_to_zone=class_to_zone, categories=categories,
                zones={z.name: z.to_zone() for z in zones},
                pick_polygon=[list(p) for p in DEFAULT_PICK_POLYGON],
                intrinsics=default_intrinsics(),
                objects=[{'name': o.name, 'label': o.label} for o in objects],
                required=sum(1 for o in objects if o.label in class_to_zone))
    q, T_obs = _eye_in_hand_observe()
    info['T_flange_camera'] = eye_in_hand_T_flange_camera()
    info['T_base_camera_observe'] = np.array(T_obs)
    info['observe_joints'] = list(q)
    return world, info


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
    (camera.extrinsics_inline); sim.T_flange_camera keeps the true pose used
    for rendering. Merge it over a config loaded without
    site overlays, or site zones are merged with the sim zones.
    """
    if len(extrinsics_error) != 2:
        raise ValueError('extrinsics_error must be [translation m, rotation deg]')
    task, mount = info['task'], info['mount']
    rng = np.random.default_rng([int(info['seed']), 7919])
    T_true = np.asarray(info['T_flange_camera'], dtype=float)
    sim_cam = {'T_flange_camera': _as_list(T_true)}
    T_est = perturb_pose(T_true, extrinsics_error[0], extrinsics_error[1], rng)
    zones = {}
    for name, z in info['zones'].items():
        zones[name] = {'center': [float(v) for v in z.center], 'size': [float(v) for v in z.size],
                       'yaw': float(z.yaw), 'wall_height': float(z.wall_height),
                       'label': str(z.label)}
    task_cfg: Dict[str, Any] = {'labels': list(info['labels']),
                                'class_to_zone': dict(info['class_to_zone'])}
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
                  'poses': {'observe_joints': [float(v) for v in info['observe_joints']],
                            'observe_views': [[float(v) for v in q]
                                              for q in info.get('observe_views', [])]}},
        'sim': sim,
    }

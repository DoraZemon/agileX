"""
Analytic RGB-D renderer used by the perception tests, plus its self-checks.

Pinhole camera (no distortion) ray-cast against a horizontal table plane,
oriented boxes standing on it and spheres resting on it. Depth is the
camera-frame z in meters (0 = no hit). Kept independent of cube_task.sim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np

from cube_task.core.transforms import rot_z
from cube_task.core.types import CameraIntrinsics, Frame, Zone

INTR = CameraIntrinsics(fx=600.0, fy=600.0, cx=319.5, cy=239.5, width=640, height=480)

RED = (30, 30, 200)
GREEN = (40, 170, 40)
BLUE = (190, 90, 20)
YELLOW = (30, 200, 220)
TABLE = (115, 120, 118)


@dataclass
class Box:
    """Box standing on the table: center (x, y), yaw, size (sx, sy, sz) in m."""

    x: float
    y: float
    yaw: float
    sx: float
    sy: float
    sz: float
    color: Tuple[int, int, int] = RED


@dataclass
class Sphere:
    """Sphere of radius r resting on the table at (x, y)."""

    x: float
    y: float
    r: float
    color: Tuple[int, int, int] = GREEN


def look_at(pos: Sequence[float], target: Sequence[float],
            up: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Return T_base_camera of an optical frame at pos looking at target."""
    pos = np.asarray(pos, dtype=float)
    z = np.asarray(target, dtype=float) - pos
    z /= np.linalg.norm(z)
    upv = np.asarray(up, dtype=float)
    y = -(upv - np.dot(upv, z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


def overhead_camera(cx: float = 0.30, cy: float = 0.0, height: float = 0.60) -> np.ndarray:
    """Camera straight above (cx, cy), image up = base +x."""
    return look_at((cx, cy, height), (cx, cy, 0.0), up=(1.0, 0.0, 0.0))


def oblique_camera(target: Sequence[float] = (0.30, 0.0, 0.0), elev_deg: float = 45.0,
                   dist: float = 0.65) -> np.ndarray:
    """Camera behind the target (towards -x) looking down at elev_deg."""
    e = math.radians(elev_deg)
    t = np.asarray(target, dtype=float)
    pos = t + dist * np.array([-math.cos(e), 0.0, math.sin(e)])
    return look_at(pos, t, up=(0.0, 0.0, 1.0))


def _rays(intr: CameraIntrinsics, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(intr.width, dtype=np.float64),
                       np.arange(intr.height, dtype=np.float64))
    rc = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)],
                  axis=-1).reshape(-1, 3)
    return T[:3, 3].copy(), rc @ T[:3, :3].T


def render(objects: Sequence[object], T_base_camera: np.ndarray,
           intr: CameraIntrinsics = INTR, table_z: float = 0.0,
           zones: Sequence[Zone] = (), zone_color: Tuple[int, int, int] = (200, 200, 60),
           noise_std: float = 0.0, flying: float = 0.0, dropout: float = 0.0,
           seed: int = 0, table: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray-cast the scene; return (color BGR uint8, depth float32 m, object index).

    object index per pixel: i for objects[i], -1 table, -2 nothing.
    flying: probability that a background pixel next to a nearer surface
    (step > 2 cm) gets a mixed depth between both surfaces.
    """
    rng = np.random.default_rng(seed)
    o, d = _rays(intr, np.asarray(T_base_camera, dtype=float))
    n = d.shape[0]
    t_best = np.full(n, np.inf)
    idx = np.full(n, -2, dtype=np.int64)
    shade = np.ones(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        if table:
            tt = np.where(d[:, 2] < -1e-12, (table_z - o[2]) / d[:, 2], np.inf)
            hit = (tt > 0) & (tt < t_best)
            t_best[hit] = tt[hit]
            idx[hit] = -1
        for k, ob in enumerate(objects):
            if isinstance(ob, Box):
                R = rot_z(ob.yaw)
                ol = R.T @ (o - np.array([ob.x, ob.y, table_z]))
                dl = d @ R
                lo = np.array([-ob.sx / 2, -ob.sy / 2, 0.0])
                hi = np.array([ob.sx / 2, ob.sy / 2, ob.sz])
                t1 = (lo - ol) / dl
                t2 = (hi - ol) / dl
                tmin = np.fmin(t1, t2)
                tmax = np.fmax(t1, t2)
                tnear = np.nanmax(tmin, axis=1)
                tfar = np.nanmin(tmax, axis=1)
                hit = (tnear <= tfar) & (tnear > 0) & (tnear < t_best)
                face = np.argmax(np.nan_to_num(tmin, nan=-np.inf), axis=1)
                t_best[hit] = tnear[hit]
                idx[hit] = k
                shade[hit] = np.array([0.8, 0.65, 1.0])[face[hit]]
            elif isinstance(ob, Sphere):
                c = np.array([ob.x, ob.y, table_z + ob.r])
                oc = o - c
                a = np.sum(d * d, axis=1)
                b = d @ oc
                cc = float(oc @ oc) - ob.r ** 2
                disc = b * b - a * cc
                ts = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
                hit = (disc >= 0) & (ts > 0) & (ts < t_best)
                t_best[hit] = ts[hit]
                idx[hit] = k
                p = o + ts[hit, None] * d[hit]
                nz = (p[:, 2] - c[2]) / ob.r
                shade[hit] = 0.7 + 0.3 * np.clip(nz, 0.0, 1.0)
    H, W = intr.height, intr.width
    depth = np.where(np.isfinite(t_best), t_best, 0.0)
    color = np.zeros((n, 3))
    color[idx == -1] = TABLE
    if zones and np.any(idx == -1):
        tb = np.nonzero(idx == -1)[0]
        p = o + t_best[tb, None] * d[tb]
        for z in zones:
            inside = np.array([z.contains_xy(px, py) for px, py in p[:, :2]])
            color[tb[inside]] = zone_color
    for k, ob in enumerate(objects):
        sel = idx == k
        color[sel] = np.asarray(ob.color, dtype=float)[None, :] * shade[sel, None]
    color += rng.normal(0.0, 2.0, size=color.shape)
    color = np.clip(np.round(color), 0, 255).astype(np.uint8).reshape(H, W, 3)
    depth = depth.reshape(H, W)
    if flying > 0.0:
        valid = depth > 0
        near = np.full((H, W), np.inf)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(np.where(valid, depth, np.inf), (dy, dx), axis=(0, 1))
            near = np.minimum(near, sh)
        cand = valid & np.isfinite(near) & (depth - near > 0.02)
        pick = cand & (rng.random((H, W)) < flying)
        alpha = rng.uniform(0.2, 0.8, size=(H, W))
        depth = np.where(pick, near + alpha * (depth - near), depth)
    if noise_std > 0.0:
        depth = np.where(depth > 0, depth + rng.normal(0.0, noise_std, size=depth.shape), 0.0)
    if dropout > 0.0:
        depth = np.where(rng.random((H, W)) < dropout, 0.0, depth)
    return color, depth.astype(np.float32), idx.reshape(H, W)


def make_frame(objects: Sequence[object], T_base_camera: np.ndarray, stamp: float = 1.0,
               **kw) -> Frame:
    """Render a scene and wrap it in a core Frame."""
    color, depth, _ = render(objects, T_base_camera, **kw)
    intr = kw.get('intr', INTR)
    return Frame(color=color, depth=depth, intrinsics=intr, stamp=stamp, source='test')


def box_corners(b: Box) -> np.ndarray:
    """Return the 4x2 base-XY footprint corners of a box."""
    R = rot_z(b.yaw)[:2, :2]
    loc = np.array([[-b.sx, -b.sy], [b.sx, -b.sy], [b.sx, b.sy], [-b.sx, b.sy]]) / 2.0
    return loc @ R.T + np.array([b.x, b.y])


def angle_diff_mod(a: float, b: float, period: float) -> float:
    """Return the smallest |a - b| modulo period."""
    d = (a - b) % period
    return min(d, period - d)


def scene_boxes() -> List[Box]:
    """Four separated boxes of different sizes and yaws in the pick area."""
    return [Box(0.25, -0.10, 0.3, 0.040, 0.030, 0.030, RED),
            Box(0.36, -0.06, -0.6, 0.050, 0.035, 0.045, GREEN),
            Box(0.26, 0.08, 1.1, 0.030, 0.030, 0.030, BLUE),
            Box(0.38, 0.10, 0.0, 0.060, 0.025, 0.025, YELLOW)]


# --------------------------------------------------------------------------- self-checks


def test_look_at_is_rotation():
    for T in (overhead_camera(), oblique_camera()):
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_overhead_table_depth():
    _, depth, idx = render([], overhead_camera(height=0.6))
    assert np.all(idx == -1)
    assert abs(float(depth[240, 320]) - 0.6) < 1e-6


def test_box_top_depth():
    b = Box(0.30, 0.0, 0.0, 0.05, 0.05, 0.04)
    _, depth, idx = render([b], overhead_camera(height=0.6))
    assert idx[240, 320] == 0
    assert abs(float(depth[240, 320]) - 0.56) < 1e-6

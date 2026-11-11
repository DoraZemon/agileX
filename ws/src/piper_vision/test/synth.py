"""Tiny ray-cast RGB-D renderer for tests: a table, flat cards and axis-aligned-in-yaw cubes."""
from __future__ import annotations

import math

import numpy as np

from piper_vision.camera import Intrinsics
from piper_vision.transforms import inv_T, make_T, rpy_to_rot

BGR = {'red': (30, 30, 200), 'green': (40, 160, 40), 'yellow': (30, 210, 230),
       'table': (170, 170, 170)}


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """Return T_base_camera for an optical camera at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack((x, y, z)), eye)


def default_intrinsics() -> Intrinsics:
    """Return 640x480 intrinsics similar to the Orbbec color stream."""
    return Intrinsics(600.0, 600.0, 319.5, 239.5, 640, 480)


def render(intr: Intrinsics, T_base_camera: np.ndarray, table_z: float, cubes, cards,
           noise_std: float = 0.0, seed: int = 0):
    """
    Render (color BGR uint8, depth float32 m).

    cubes: list of (label, x, y, yaw, edge). cards: list of (label, x, y, yaw, size).
    """
    h, w = intr.height, intr.width
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays_c = np.stack(((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)), -1)
    R = T_base_camera[:3, :3]
    o = T_base_camera[:3, 3]
    rays = rays_c @ R.T
    best_t = np.full((h, w), np.inf)
    color = np.zeros((h, w, 3), np.uint8)
    # table
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (table_z - o[2]) / rays[..., 2]
    hit = t > 0
    best_t[hit] = t[hit]
    color[hit] = BGR['table']
    p_table = o + rays * np.where(hit, t, 0)[..., None]
    for label, x, y, yaw, size in cards:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p_table[..., 0] - x, p_table[..., 1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        m = hit & (np.abs(lx) <= size / 2) & (np.abs(ly) <= size / 2)
        color[m] = BGR[label]
    for label, x, y, yaw, edge in cubes:
        T_bc = make_T(rpy_to_rot(0, 0, yaw), (x, y, table_z + edge / 2))
        T_cb = inv_T(T_bc)
        oc = T_cb[:3, :3] @ o + T_cb[:3, 3]
        rc = rays @ T_cb[:3, :3].T
        half = edge / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-half - oc) / rc
            t2 = (half - oc) / rc
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        m = (tmax >= tmin) & (tmin > 0) & (tmin < best_t)
        best_t[m] = tmin[m]
        # Shade side faces darker so the color still matches but is not flat.
        p = oc + rc * tmin[..., None]
        top = np.abs(p[..., 2] - half) < 1e-4
        base = np.array(BGR[label], float)
        color[m & top] = base.astype(np.uint8)
        color[m & ~top] = (base * 0.75).astype(np.uint8)
    depth = (best_t * rays_c[..., 2]).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        depth = depth + rng.normal(0, noise_std, depth.shape).astype(np.float32)
    return color, depth

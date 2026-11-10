"""
Vectorized RGB-D ray casting of a SimWorld (pinhole camera, no distortion).

Scene: a bounded table rectangle (TABLE_SIZE centered at TABLE_CENTER in base
XY, at world.table_z) above a darker floor FLOOR_DROP lower, zone mats 1 mm
above their floor with optional walls, the objects and the gripper. Every
primitive is intersected in its own frame (slab test for boxes, scaled unit
sphere for ellipsoids, quadratic + caps for cylinders, spheres for chains,
cylinder + end spheres for the capsule parts of compound objects) and only
inside the image rectangle covered by its projected local bounds.

Depth is the z coordinate in the camera optical frame (x right, y down, z
forward), float32 meters, 0 = invalid, exactly as a registered depth camera
delivers it. Color is uint8 BGR.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.transforms import inv_T, make_T, rot_z
from ..core.types import CameraIntrinsics
from .world import WALL_THICKNESS, SimWorld, check_pose, gripper_boxes

TABLE_SIZE = (1.2, 1.4)          # m, along base x / y
TABLE_CENTER = (0.3, 0.0)        # m, base XY
FLOOR_DROP = 0.75                # m, floor below the table top
BACKGROUND_BGR = (55, 55, 58)    # rays that hit nothing (depth 0)
LIGHT_DIR = np.array([0.45, 0.15, 1.0]) / np.linalg.norm([0.45, 0.15, 1.0])
AMBIENT = 0.42
DIFFUSE = 0.68
SHININESS = 40.0
COLOR_NOISE_STD = 2.5            # uint8 levels, only when rng is given
DISCONTINUITY_JUMP = 0.02        # m, depth jump treated as an edge
DISCONTINUITY_DROP_P = 0.3       # probability an edge pixel is invalid

_RAY_CACHE: Dict[Tuple[float, ...], np.ndarray] = {}


def camera_rays(intr: CameraIntrinsics) -> np.ndarray:
    """
    Return HxWx3 ray directions in the camera frame with z = 1 (cached).

    Pixel (u, v) has its center at integer coordinates (OpenCV convention),
    so the ray parameter t of a hit equals its depth z.
    """
    key = (float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy),
           int(intr.width), int(intr.height))
    rays = _RAY_CACHE.get(key)
    if rays is None:
        u = (np.arange(intr.width, dtype=float) - intr.cx) / intr.fx
        v = (np.arange(intr.height, dtype=float) - intr.cy) / intr.fy
        rays = np.empty((intr.height, intr.width, 3))
        rays[:, :, 0] = u[None, :]
        rays[:, :, 1] = v[:, None]
        rays[:, :, 2] = 1.0
        if len(_RAY_CACHE) > 8:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = rays
    return rays


def _check_intrinsics(intr: CameraIntrinsics) -> None:
    vals = [intr.fx, intr.fy, intr.cx, intr.cy]
    if not all(math.isfinite(float(v)) for v in vals) or intr.fx <= 0 or intr.fy <= 0:
        raise ValueError('bad intrinsics %r' % (vals,))
    if int(intr.width) <= 0 or int(intr.height) <= 0:
        raise ValueError('bad image size %dx%d' % (intr.width, intr.height))


# ------------------------------------------------------------------ kernels
def _hit_box(o: np.ndarray, d: np.ndarray, half: np.ndarray):
    """Slab test of rays o + t d (local frame) against a centered box."""
    dd = np.where(np.abs(d) < 1e-12, np.copysign(1e-12, d), d)
    inv = 1.0 / dd
    t1 = (-half - o) * inv
    t2 = (half - o) * inv
    tmin_ax = np.minimum(t1, t2)
    tn = tmin_ax.max(axis=1)
    tf = np.maximum(t1, t2).min(axis=1)
    hit = (tf >= tn) & (tn > 1e-6)
    t = np.where(hit, tn, np.inf)
    ax = tmin_ax.argmax(axis=1)
    n = np.zeros_like(d)
    idx = np.arange(len(d))
    n[idx, ax] = -np.sign(dd[idx, ax])
    return t, n


def _hit_ellipsoid(o: np.ndarray, d: np.ndarray, center: np.ndarray, radii: np.ndarray):
    oc = (o - center) / radii
    dd = d / radii
    a = np.einsum('ij,ij->i', dd, dd)
    b = dd @ oc
    c = float(oc @ oc) - 1.0
    disc = b * b - a * c
    hit = disc >= 0.0
    t = (-b - np.sqrt(np.where(hit, disc, 0.0))) / a
    hit &= t > 1e-6
    t = np.where(hit, t, np.inf)
    p = o + np.where(hit, t, 0.0)[:, None] * d
    n = (p - center) / (radii * radii)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return t, n


def _polyline_normals(p: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Return unit vectors from the closest polyline point to each p (smooth tube normals)."""
    best = np.full(len(p), np.inf)
    out = np.zeros_like(p)
    segs = [(line[i], line[i + 1]) for i in range(len(line) - 1)] or [(line[0], line[0])]
    for a, b in segs:
        ab = b - a
        den = float(ab @ ab)
        s = np.zeros(len(p)) if den < 1e-12 else np.clip((p - a) @ ab / den, 0.0, 1.0)
        v = p - (a + s[:, None] * ab)
        dist = np.einsum('ij,ij->i', v, v)
        sel = dist < best
        best[sel] = dist[sel]
        out[sel] = v[sel]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _hit_spheres(o: np.ndarray, d: np.ndarray, r: float, centers: np.ndarray,
                 axis: Optional[np.ndarray] = None):
    """Union of spheres; with axis (polyline), shade as a smooth tube around it."""
    a = np.einsum('ij,ij->i', d, d)
    best = np.full(len(d), np.inf)
    best_i = np.zeros(len(d), dtype=int)
    for i, c in enumerate(centers):
        oc = o - c
        b = d @ oc
        cc = float(oc @ oc) - r * r
        disc = b * b - a * cc
        ok = disc >= 0.0
        t = (-b - np.sqrt(np.where(ok, disc, 0.0))) / a
        ok &= (t > 1e-6) & (t < best)
        best = np.where(ok, t, best)
        best_i[ok] = i
    hit = np.isfinite(best)
    p = o + np.where(hit, best, 0.0)[:, None] * d
    if axis is not None and np.any(hit):
        n = np.zeros_like(p)
        n[hit] = _polyline_normals(p[hit], axis)
    else:
        n = (p - centers[best_i]) / r
    return best, n


def _hit_cylinder(o: np.ndarray, d: np.ndarray, r: float, h: float):
    """Vertical cylinder along local z from 0 to h, with caps."""
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (o[0] * d[:, 0] + o[1] * d[:, 1])
    c = o[0] ** 2 + o[1] ** 2 - r * r
    disc = b * b - 4.0 * a * c
    side_ok = (disc >= 0.0) & (a > 1e-12)
    ts = (-b - np.sqrt(np.where(side_ok, disc, 0.0))) / (2.0 * np.where(a > 1e-12, a, 1.0))
    zs = o[2] + ts * d[:, 2]
    side_ok &= (ts > 1e-6) & (zs >= 0.0) & (zs <= h)
    t = np.where(side_ok, ts, np.inf)
    kind = np.zeros(len(d), dtype=int)
    dz = np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
    for k, zc in ((1, h), (2, 0.0)):
        tc = (zc - o[2]) / dz
        px = o[0] + tc * d[:, 0]
        py = o[1] + tc * d[:, 1]
        ok = (tc > 1e-6) & (px * px + py * py <= r * r) & (tc < t)
        t = np.where(ok, tc, t)
        kind[ok] = k
    p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
    n = np.zeros_like(d)
    n[:, 0] = np.where(kind == 0, p[:, 0] / r, 0.0)
    n[:, 1] = np.where(kind == 0, p[:, 1] / r, 0.0)
    n[:, 2] = np.where(kind == 1, 1.0, np.where(kind == 2, -1.0, 0.0))
    return t, n


def _hit_capsule(o: np.ndarray, d: np.ndarray, r: float, hl: float):
    """Capsule along local x: segment -hl..hl, radius r."""
    a = d[:, 1] ** 2 + d[:, 2] ** 2
    b = 2.0 * (o[1] * d[:, 1] + o[2] * d[:, 2])
    c = o[1] ** 2 + o[2] ** 2 - r * r
    disc = b * b - 4.0 * a * c
    ok = (disc >= 0.0) & (a > 1e-12)
    ts = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * np.where(a > 1e-12, a, 1.0))
    xs = o[0] + ts * d[:, 0]
    ok &= (ts > 1e-6) & (np.abs(xs) <= hl)
    t = np.where(ok, ts, np.inf)
    cap = np.zeros(len(d), dtype=int)
    aa = np.einsum('ij,ij->i', d, d)
    for k, cx in ((1, hl), (2, -hl)):
        oc = o - np.array([cx, 0.0, 0.0])
        bb = d @ oc
        cc = float(oc @ oc) - r * r
        dd = bb * bb - aa * cc
        hit = dd >= 0.0
        tc = (-bb - np.sqrt(np.where(hit, dd, 0.0))) / aa
        hit &= (tc > 1e-6) & (tc < t)
        t = np.where(hit, tc, t)
        cap[hit] = k
    p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
    n = p.copy()
    n[:, 0] = np.where(cap == 1, p[:, 0] - hl, np.where(cap == 2, p[:, 0] + hl, 0.0))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return t, n


# --------------------------------------------------------------- primitives
class _Prim:
    """One renderable primitive in its own frame T (base <- prim)."""

    __slots__ = ('kind', 'T', 'params', 'color', 'spec', 'tex_amp', 'tex_seed', 'lo', 'hi')

    def __init__(self, kind: str, T: np.ndarray, params: Tuple[Any, ...],
                 color: Tuple[int, int, int], lo: np.ndarray, hi: np.ndarray,
                 spec: float = 0.0, tex_amp: float = 0.05, tex_seed: int = 0):
        self.kind = kind
        self.T = T
        self.params = params
        self.color = np.array(color, dtype=float)
        self.spec = spec
        self.tex_amp = tex_amp
        self.tex_seed = tex_seed
        self.lo = lo
        self.hi = hi


def _box_prim(T: np.ndarray, half: np.ndarray, color, spec=0.0, tex_amp=0.04,
              tex_seed=0) -> _Prim:
    half = np.asarray(half, dtype=float)
    return _Prim('box', T, (half,), color, -half, half, spec, tex_amp, tex_seed)


def _zone_prims(world: SimWorld) -> List[_Prim]:
    out = []
    for z in world.zones:
        R = rot_z(z.yaw)
        cx, cy, fz = (float(v) for v in z.center)
        hx, hy = z.size[0] / 2.0, z.size[1] / 2.0
        seed = sum(ord(ch) for ch in z.name)
        out.append(_box_prim(make_T(R, (cx, cy, fz + 0.0005)), (hx, hy, 0.0005),
                             z.color_bgr, 0.03, 0.04, seed))
        wt = WALL_THICKNESS
        wall_color = tuple(int(0.8 * c) for c in z.color_bgr)
        if z.wall_height > 0.0:
            wh = z.wall_height
            for lx, ly, half in (
                    (0.0, hy + wt / 2.0, (hx + wt, wt / 2.0, wh / 2.0)),
                    (0.0, -hy - wt / 2.0, (hx + wt, wt / 2.0, wh / 2.0)),
                    (hx + wt / 2.0, 0.0, (wt / 2.0, hy, wh / 2.0)),
                    (-hx - wt / 2.0, 0.0, (wt / 2.0, hy, wh / 2.0))):
                c = R @ np.array([lx, ly, 0.0]) + np.array([cx, cy, fz + wh / 2.0])
                out.append(_box_prim(make_T(R, c), half, wall_color, 0.05, 0.03, seed + 1))
        base_h = fz - world.table_z
        if base_h > 0.001:
            e = wt if z.wall_height > 0.0 else 0.0
            c = (cx, cy, world.table_z + base_h / 2.0)
            out.append(_box_prim(make_T(R, c), (hx + e, hy + e, base_h / 2.0),
                                 wall_color, 0.05, 0.03, seed + 2))
    return out


def _object_prims(world: SimWorld) -> List[_Prim]:
    out = []
    for obj in world.objects:
        T = obj.T_base_obj
        for prim in obj.primitives():
            kind = prim[0]
            if kind == 'part':
                out.append(_part_prim(T, prim, obj.texture_seed))
            elif kind == 'box':
                _, center, half = prim
                out.append(_box_prim(T @ make_T(np.eye(3), center), half, obj.color_bgr,
                                     0.22, 0.06, obj.texture_seed))
            elif kind == 'ellipsoid':
                _, center, radii = prim
                out.append(_Prim('ellipsoid', T, (center, radii), obj.color_bgr,
                                 center - radii, center + radii, 0.25, 0.06,
                                 obj.texture_seed))
            elif kind == 'cylinder':
                _, r, h = prim
                out.append(_Prim('cylinder', T, (r, h), obj.color_bgr,
                                 np.array([-r, -r, 0.0]), np.array([r, r, h]), 0.22, 0.06,
                                 obj.texture_seed))
            else:
                _, r, centers = prim
                axis = (np.array(obj.size[1], dtype=float)
                        if obj.shape == 'capsule_chain' else None)
                out.append(_Prim('spheres', T, (r, centers, axis), obj.color_bgr,
                                 centers.min(axis=0) - r, centers.max(axis=0) + r, 0.22,
                                 0.06, obj.texture_seed))
    return out


def _part_prim(T_base_obj: np.ndarray, prim: Tuple[Any, ...], seed: int) -> _Prim:
    """Return the _Prim of one compound part ('part', kind, T_obj_part, params, color)."""
    _, kind, T_op, prm, color = prim
    T = T_base_obj @ T_op
    if kind == 'box':
        return _box_prim(T, np.array(prm) / 2.0, color, 0.22, 0.06, seed)
    if kind == 'ellipsoid':
        radii = np.array(prm, dtype=float)
        return _Prim('ellipsoid', T, (np.zeros(3), radii), color, -radii, radii, 0.25, 0.06,
                     seed)
    if kind == 'cylinder':
        r, h = prm
        return _Prim('cylinder', T, (r, h), color, np.array([-r, -r, 0.0]),
                     np.array([r, r, h]), 0.22, 0.06, seed)
    r, hl = prm
    return _Prim('capsule', T, (r, hl), color, np.array([-hl - r, -r, -r]),
                 np.array([hl + r, r, r]), 0.25, 0.06, seed)


def _gripper_prims(world: SimWorld, gripper: Any) -> List[_Prim]:
    if gripper is False:
        return []
    if gripper is None:
        if world.tcp_pose is None:
            return []
        T, width = world.tcp_pose, world.gripper_width or 0.0
    else:
        if len(gripper) != 2:
            raise ValueError('gripper must be (T_base_tcp, width), None or False')
        T = check_pose(gripper[0], 'gripper T_base_tcp')
        width = float(gripper[1])
    return [_box_prim(Tb, half, color, 0.15, 0.02, 7)
            for Tb, half, color in gripper_boxes(T, width)]


def _texture(p: np.ndarray, amp: float, seed: int) -> np.ndarray:
    """Return N x 3 deterministic color multipliers from local hit points p (m)."""
    if amp <= 0.0 or len(p) == 0:
        return np.ones((len(p), 3))
    s = float(seed % 997)
    k1 = np.array([620.0 + s * 0.3, 410.0, 270.0 - s * 0.2])
    k2 = np.array([-330.0, 560.0 + s * 0.1, 450.0])
    wave = np.sin(p @ k1 + s) * np.sin(p @ k2 + 0.7 * s)
    q = np.floor(p / 0.0009) @ np.array([12.9898, 78.233, 37.719])
    speckle = np.modf(np.abs(np.sin(q + s) * 43758.5453))[0] - 0.5
    m = 1.0 + amp * (0.6 * wave + 0.9 * speckle)
    hue = 0.02 * np.sin(p @ np.array([150.0, -120.0, 90.0]) + s)
    return m[:, None] * (1.0 + hue[:, None] * np.array([1.0, -0.5, 0.3]))


_KERNELS = {
    'box': lambda o, d, prm: _hit_box(o, d, prm[0]),
    'ellipsoid': lambda o, d, prm: _hit_ellipsoid(o, d, prm[0], prm[1]),
    'cylinder': lambda o, d, prm: _hit_cylinder(o, d, prm[0], prm[1]),
    'spheres': lambda o, d, prm: _hit_spheres(o, d, prm[0], prm[1], prm[2]),
    'capsule': lambda o, d, prm: _hit_capsule(o, d, prm[0], prm[1]),
}


def _pixel_rect(prim: _Prim, T_cam_base: np.ndarray, intr: CameraIntrinsics
                ) -> Optional[Tuple[int, int, int, int]]:
    """Return the image rectangle (u0, u1, v0, v1) that can see prim, or None."""
    lo, hi = prim.lo, prim.hi
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    Tc = T_cam_base @ prim.T
    pc = corners @ Tc[:3, :3].T + Tc[:3, 3]
    z = pc[:, 2]
    W, H = int(intr.width), int(intr.height)
    if np.all(z <= 1e-4):
        return None
    if np.any(z <= 1e-4):
        return 0, W, 0, H
    u = intr.fx * pc[:, 0] / z + intr.cx
    v = intr.fy * pc[:, 1] / z + intr.cy
    u0, u1 = max(0, int(math.floor(u.min()))), min(W, int(math.ceil(u.max())) + 1)
    v0, v1 = max(0, int(math.floor(v.min()))), min(H, int(math.ceil(v.max())) + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    return u0, u1, v0, v1


def render_rgbd(world: SimWorld, intrinsics: CameraIntrinsics, T_base_camera: Any,
                gripper: Any = None, rng: Any = None, noise_std: float = 0.0,
                dropout: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render (color uint8 BGR HxWx3, depth float32 m HxW with 0 = invalid).

    T_base_camera: pose of the color optical frame in the base frame.
    gripper: None draws world.tcp_pose / world.gripper_width when set,
    (T_base_tcp, width) draws that, False draws no gripper.
    rng: numpy Generator (or int seed). With rng, gaussian color noise is
    added. noise_std: depth noise std (m) at 0.5 m, growing with (z / 0.5)^2.
    dropout: probability of a random invalid depth pixel; with dropout > 0,
    pixels at depth jumps > DISCONTINUITY_JUMP are also invalidated with
    probability DISCONTINUITY_DROP_P. Noise without an rng uses seed 0.
    Deterministic for a given rng state.
    """
    _check_intrinsics(intrinsics)
    T = check_pose(T_base_camera, 'T_base_camera')
    ns, dp = float(noise_std), float(dropout)
    if not math.isfinite(ns) or ns < 0.0 or not math.isfinite(dp) or not 0.0 <= dp < 1.0:
        raise ValueError('bad noise_std %r / dropout %r' % (noise_std, dropout))
    if rng is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(int(rng))
    H, W = int(intrinsics.height), int(intrinsics.width)
    R, o = T[:3, :3], T[:3, 3]
    D = camera_rays(intrinsics).reshape(-1, 3) @ R.T
    N = H * W
    zbuf = np.full(N, np.inf)
    col = np.zeros((N, 3))
    nrm = np.zeros((N, 3))
    nrm[:, 2] = 1.0
    tex = np.ones((N, 3))
    spec = np.zeros(N)

    # Table (bounded rectangle) and the floor below it.
    tz = world.table_z
    dz = D[:, 2]
    down = dz < -1e-9
    safe_dz = np.where(down, dz, -1.0)
    for z_plane, is_table in ((tz, True), (tz - FLOOR_DROP, False)):
        t = (z_plane - o[2]) / safe_dz
        ok = down & (t > 1e-6) & (t < zbuf)
        px = o[0] + t * D[:, 0]
        py = o[1] + t * D[:, 1]
        if is_table:
            ok &= ((np.abs(px - TABLE_CENTER[0]) <= TABLE_SIZE[0] / 2.0)
                   & (np.abs(py - TABLE_CENTER[1]) <= TABLE_SIZE[1] / 2.0))
            color, amp, seed = np.array(world.table_color_bgr, dtype=float), 0.05, 3
        else:
            color, amp, seed = 0.45 * np.array(world.table_color_bgr, dtype=float), 0.08, 5
        zbuf[ok] = t[ok]
        col[ok] = color
        pts = np.column_stack([px[ok], py[ok], np.zeros(int(ok.sum()))])
        tex[ok] = _texture(pts, amp, seed)

    prims = _zone_prims(world) + _object_prims(world) + _gripper_prims(world, gripper)
    T_cb = inv_T(T)
    for prim in prims:
        rect = _pixel_rect(prim, T_cb, intrinsics)
        if rect is None:
            continue
        u0, u1, v0, v1 = rect
        idx = (np.arange(v0, v1)[:, None] * W + np.arange(u0, u1)[None, :]).ravel()
        Rp, tp = prim.T[:3, :3], prim.T[:3, 3]
        o_l = Rp.T @ (o - tp)
        d_l = D[idx] @ Rp
        t, n_l = _KERNELS[prim.kind](o_l, d_l, prim.params)
        better = t < zbuf[idx]
        if not np.any(better):
            continue
        sel = idx[better]
        tb = t[better]
        zbuf[sel] = tb
        nrm[sel] = n_l[better] @ Rp.T
        col[sel] = prim.color
        spec[sel] = prim.spec
        tex[sel] = _texture(o_l + tb[:, None] * d_l[better], prim.tex_amp, prim.tex_seed)

    hit = np.isfinite(zbuf)
    V = -D / np.linalg.norm(D, axis=1, keepdims=True)
    facing = np.einsum('ij,ij->i', nrm, V)
    nrm = np.where((facing < 0.0)[:, None], -nrm, nrm)
    diff = np.clip(nrm @ LIGHT_DIR, 0.0, 1.0)
    half = V + LIGHT_DIR
    half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-12)
    hl = np.clip(np.einsum('ij,ij->i', nrm, half), 0.0, 1.0) ** SHININESS
    img = col * tex * (AMBIENT + DIFFUSE * diff)[:, None] + (255.0 * spec * hl)[:, None]
    img[~hit] = BACKGROUND_BGR
    if rng is not None:
        img += rng.normal(0.0, COLOR_NOISE_STD, img.shape)
    color = np.clip(np.rint(img), 0, 255).astype(np.uint8).reshape(H, W, 3)
    depth = np.where(hit, zbuf, 0.0).reshape(H, W)

    if ns > 0.0 or dp > 0.0:
        g = rng if rng is not None else np.random.default_rng(0)
        valid = depth > 0.0
        edge = _discontinuities(depth, DISCONTINUITY_JUMP) if dp > 0.0 else None
        if ns > 0.0:
            z = depth[valid]
            noisy = z + g.normal(0.0, 1.0, z.shape) * ns * (z / 0.5) ** 2
            depth[valid] = np.maximum(noisy, 1e-4)
        if dp > 0.0:
            drop = g.random(depth.shape) < dp
            drop |= edge & (g.random(depth.shape) < DISCONTINUITY_DROP_P)
            depth[drop] = 0.0
    return color, depth.astype(np.float32)


def _discontinuities(depth: np.ndarray, jump: float) -> np.ndarray:
    """Mark pixels whose valid 4-neighbor differs by more than jump (m)."""
    edge = np.zeros(depth.shape, dtype=bool)
    v = depth > 0.0
    dx = (np.abs(depth[:, 1:] - depth[:, :-1]) > jump) & v[:, 1:] & v[:, :-1]
    dy = (np.abs(depth[1:, :] - depth[:-1, :]) > jump) & v[1:, :] & v[:-1, :]
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    return edge

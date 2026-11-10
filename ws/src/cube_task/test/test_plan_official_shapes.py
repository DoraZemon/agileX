"""Synthetic point sets of the official L1 / L2 objects and Detections built from them."""
from __future__ import annotations

import math

from cube_task.core.types import Detection
from cube_task.perception.segment import MAX_INSTANCE_POINTS, rect_geometry, subsample_points
import numpy as np
import pytest

#: camera position (base frame, m) used for the visibility of the sampled surface.
CAMERA = np.array([0.30, -0.45, 0.65])
#: pear radius (m) at height (m) above the table: lower bulb, neck, top, stem.
PEAR_H = [0.0, 0.004, 0.012, 0.025, 0.040, 0.055, 0.068, 0.080, 0.092, 0.102, 0.110, 0.115,
          0.125]
PEAR_R = [0.0, 0.022, 0.033, 0.040, 0.0425, 0.039, 0.030, 0.025, 0.023, 0.020, 0.012, 0.003,
          0.003]
#: bell pepper (90 mm incl. stem, 70 mm wide) and persimmon (70 x 50 mm, oblate).
PEPPER_H = [0.0, 0.006, 0.02, 0.04, 0.06, 0.072, 0.078, 0.080, 0.081, 0.090]
PEPPER_R = [0.0, 0.026, 0.033, 0.035, 0.035, 0.031, 0.022, 0.010, 0.005, 0.005]
OFFICIAL = ('cube', 'cucumber', 'pear', 'banana', 'potato', 'persimmon', 'bell_pepper')


def _revolution(hs, rs, n_h=120, n_a=90):
    """Return (points, normals) of a surface of revolution r(h) about the z axis."""
    h = np.linspace(hs[0], hs[-1], n_h)
    r = np.interp(h, hs, rs)
    dr = np.gradient(r, h)
    a = np.linspace(0.0, 2.0 * math.pi, n_a, endpoint=False)
    H, A = np.meshgrid(h, a, indexing='ij')
    R = np.repeat(r[:, None], n_a, axis=1)
    D = np.repeat(dr[:, None], n_a, axis=1)
    P = np.stack([R * np.cos(A), R * np.sin(A), H], axis=-1).reshape(-1, 3)
    N = np.stack([np.cos(A), np.sin(A), -D], axis=-1).reshape(-1, 3)
    top = np.stack([np.zeros(1), np.zeros(1), [hs[-1]]], axis=-1)
    return np.vstack([P, top]), np.vstack([N, [[0.0, 0.0, 1.0]]])


def _ellipsoid(a, b, c, n=12000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u * [a, b, c] + [0.0, 0.0, c]
    N = u / np.array([a, b, c])
    return P, N


def _box(sx, sy, sz, step=0.002):
    pts, nrm = [], []
    gx = np.arange(-sx / 2, sx / 2 + 1e-9, step)
    gy = np.arange(-sy / 2, sy / 2 + 1e-9, step)
    gz = np.arange(0.0, sz + 1e-9, step)
    X, Y = np.meshgrid(gx, gy)
    pts.append(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, sz)]))
    nrm.append(np.tile([0.0, 0.0, 1.0], (X.size, 1)))
    for sgn in (-1.0, 1.0):
        Y, Z = np.meshgrid(gy, gz)
        pts.append(np.column_stack([np.full(Y.size, sgn * sx / 2), Y.ravel(), Z.ravel()]))
        nrm.append(np.tile([sgn, 0.0, 0.0], (Y.size, 1)))
        X, Z = np.meshgrid(gx, gz)
        pts.append(np.column_stack([X.ravel(), np.full(X.size, sgn * sy / 2), Z.ravel()]))
        nrm.append(np.tile([0.0, sgn, 0.0], (X.size, 1)))
    return np.vstack(pts), np.vstack(nrm)


def _tube(length, radius, bow, n_s=160, n_a=48):
    """Curved tube lying on the table along x (bow: sagitta of the arc in +y, m)."""
    s = np.linspace(-length / 2 + radius, length / 2 - radius, n_s)
    half = length / 2 - radius
    yc = bow * (1.0 - (s / half) ** 2) if half > 0 else np.zeros_like(s)
    a = np.linspace(0.0, 2.0 * math.pi, n_a, endpoint=False)
    S, A = np.meshgrid(s, a, indexing='ij')
    Y = np.repeat(yc[:, None], n_a, axis=1)
    P = np.stack([S, Y + radius * np.cos(A), radius + radius * np.sin(A)], axis=-1)
    N = np.stack([np.zeros_like(S), np.cos(A), np.sin(A)], axis=-1)
    pts, nrm = [P.reshape(-1, 3)], [N.reshape(-1, 3)]
    for end, sgn in ((s[0], -1.0), (s[-1], 1.0)):
        u = np.random.default_rng(1).normal(size=(800, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u = u[u[:, 0] * sgn > 0]
        pts.append(u * radius + [end, 0.0, radius])
        nrm.append(u)
    return np.vstack(pts), np.vstack(nrm)


def shape_points(kind):
    """Return (points, normals) of one official object in its own frame (bottom at z 0)."""
    if kind == 'cube':
        return _box(0.05, 0.05, 0.05)
    if kind == 'cucumber':
        return _tube(0.26, 0.02, 0.012)
    if kind == 'banana':
        return _tube(0.18, 0.02, 0.03)
    if kind == 'potato':
        return _ellipsoid(0.05, 0.035, 0.0275)
    if kind == 'persimmon':
        return _ellipsoid(0.035, 0.035, 0.025)
    if kind == 'pear':
        return _revolution(PEAR_H, PEAR_R)
    if kind == 'bell_pepper':
        return _revolution(PEPPER_H, PEPPER_R)
    raise ValueError(kind)


def make_object_det(kind, x=0.30, y=0.0, yaw=0.3, table_z=0.0, camera=CAMERA, label=None,
                    with_points=True, noise=0.0005, seed=0):
    """
    Return a Detection of an official object at (x, y) with the perception geometry.

    Points facing the camera only (convex self-occlusion), depth noise, subsampled
    like segment._make_detection; center / yaw / length / width from rect_geometry,
    top_z = 90th percentile, cap_center_xy from the top band.
    """
    P, N = shape_points(kind)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    P = P @ R.T + [x, y, table_z]
    N = N @ R.T
    vis = np.einsum('ij,ij->i', N, np.asarray(camera)[None, :] - P) > 0.0
    vis &= P[:, 2] > table_z + 0.008
    P = P[vis]
    rng = np.random.default_rng(seed)
    P = P + rng.normal(scale=noise, size=P.shape)
    cxy, dyaw, length, width = rect_geometry(P[:, :2])
    top = float(np.percentile(P[:, 2], 90))
    band = P[P[:, 2] >= top - max(0.002, 0.05 * (top - table_z))]
    extra = {'table_z': float(table_z), '_points_xy': P[:, :2].astype(np.float32),
             'cap_center_xy': [float(band[:, 0].mean()), float(band[:, 1].mean())]}
    if with_points:
        extra['_points_base'] = subsample_points(P)
    return Detection(label=label or kind, score=0.9, score_kind='knn', bbox=(10, 10, 40, 40),
                     center_base=np.array([cxy[0], cxy[1], top]), top_z=top,
                     height=top - table_z, yaw=dyaw, length=length, width=width,
                     n_points=int(P.shape[0]), extra=extra)


@pytest.mark.parametrize('kind', OFFICIAL)
def test_sampled_shapes_have_the_official_size(kind):
    d = make_object_det(kind, yaw=0.0)
    size = {'cube': (0.05, 0.05), 'cucumber': (0.26, 0.04), 'banana': (0.18, 0.04),
            'potato': (0.10, 0.07), 'persimmon': (0.07, 0.07), 'bell_pepper': (0.07, 0.07),
            'pear': (0.085, 0.085)}[kind]
    # The hidden far side shortens the visible footprint a little.
    assert d.length == pytest.approx(size[0], abs=0.012)
    assert d.extra['_points_base'].dtype == np.float32
    assert d.extra['_points_base'].shape[0] <= MAX_INSTANCE_POINTS


def test_subsample_points_caps_uniformly():
    pts = np.arange(36000, dtype=np.float64).reshape(-1, 3)
    out = subsample_points(pts, 4000)
    assert out.shape == (4000, 3) and out.dtype == np.float32
    assert np.allclose(out[1] - out[0], [9.0, 9.0, 9.0])          # every third row
    assert subsample_points(pts[:30], 4000).shape == (30, 3)

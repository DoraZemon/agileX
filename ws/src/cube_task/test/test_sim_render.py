"""Tests for sim/render.py: exact geometry, occlusion, noise and speed."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from cube_task.core.transforms import inv_T, make_T, top_down_T
from cube_task.core.types import CameraIntrinsics
from cube_task.sim.render import render_rgbd
from cube_task.sim.scenarios import look_at, make_scenario
from cube_task.sim.world import FINGER_SIZE, SimObject, SimWorld, SimZone

INTR = CameraIntrinsics(380.0, 380.0, 319.5, 239.5, 640, 480)
SMALL = CameraIntrinsics(190.0, 190.0, 159.5, 119.5, 320, 240)
DOWN = look_at((0.30, 0.0, 0.50), (0.30, 0.0, 0.0), down_hint=(-1.0, 0.0, 0.0))
OBLIQUE = look_at((0.70, 0.0, 0.62), (0.30, 0.0, 0.0))


def project(T_base_camera, intr, p):
    pc = inv_T(T_base_camera)[:3, :3] @ np.asarray(p) + inv_T(T_base_camera)[:3, 3]
    return intr.fx * pc[0] / pc[2] + intr.cx, intr.fy * pc[1] / pc[2] + intr.cy, pc[2]


def obj(shape, size, x, y, color=(40, 40, 200), name='o', yaw=0.0):
    o = SimObject(name, 'x', shape, size, color, np.eye(4))
    o.place_upright(x, y, yaw, 0.0)
    return o


def test_table_depth_exact_for_many_pixels():
    w = SimWorld()
    _, depth = render_rgbd(w, INTR, OBLIQUE)
    R, o = OBLIQUE[:3, :3], OBLIQUE[:3, 3]
    for u, v in [(319, 239), (320, 240), (100, 400), (600, 300), (50, 470)]:
        d = R @ np.array([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, 1.0])
        t = -o[2] / d[2]
        hit = o + t * d
        assert abs(hit[0] - 0.3) <= 0.6 and abs(hit[1]) <= 0.7
        assert depth[v, u] == pytest.approx(t, abs=1e-3)


def test_primitive_top_depths_straight_down():
    w = SimWorld(objects=[obj('box', (0.04, 0.04, 0.03), 0.30, 0.0, name='b'),
                          obj('sphere', (0.025,), 0.38, 0.10, name='s'),
                          obj('cylinder', (0.02, 0.05), 0.22, -0.10, name='c'),
                          obj('ellipsoid', (0.04, 0.02, 0.015), 0.38, -0.10, name='e')],
                 zones=[SimZone('m', (0.22, 0.12, 0.0), (0.08, 0.08))])
    _, depth = render_rgbd(w, INTR, DOWN)
    for p, top in [((0.30, 0.0), 0.03), ((0.38, 0.10), 0.05), ((0.22, -0.10), 0.05),
                   ((0.38, -0.10), 0.03), ((0.22, 0.12), 0.001)]:
        u, v, _ = project(DOWN, INTR, (p[0], p[1], top))
        ui, vi = int(round(u)), int(round(v))
        # The true depth of the ray through the pixel center, within 1 mm.
        assert depth[vi, ui] == pytest.approx(0.50 - top, abs=1e-3), p


def test_cube_center_inside_silhouette_and_colors():
    c = obj('box', (0.035, 0.035, 0.035), 0.33, 0.05, yaw=0.4)
    w = SimWorld(objects=[c])
    color, depth = render_rgbd(w, INTR, OBLIQUE)
    _, depth0 = render_rgbd(SimWorld(), INTR, OBLIQUE)
    mask = np.abs(depth - depth0) > 1e-4
    u, v, _ = project(OBLIQUE, INTR, (0.33, 0.05, 0.0175))
    assert mask[int(round(v)), int(round(u))]
    ys, xs = np.nonzero(mask)
    assert 150 < mask.sum() < 2500
    assert xs.min() <= u <= xs.max() and ys.min() <= v <= ys.max()
    px = color[mask].astype(float).mean(axis=0)
    assert px[2] > px[0] + 40 and px[2] > px[1] + 40        # red object (BGR)


def test_occlusion_order():
    # A tall box between the camera and a sphere along the same ray.
    cam = OBLIQUE[:3, 3]
    far = np.array([0.25, 0.0, 0.03])
    near = cam + 0.55 * (far - cam)
    tall = SimObject('tall', 'x', 'box', (0.03, 0.03, near[2] + 0.02), (200, 90, 40), np.eye(4))
    tall.place_upright(near[0], near[1], 0.0, 0.0)
    ball = obj('sphere', (0.03,), far[0], far[1], color=(40, 40, 200), name='ball')
    w = SimWorld(objects=[ball, tall])
    color, depth = render_rgbd(w, INTR, OBLIQUE)
    u, v, z_near = project(OBLIQUE, INTR, near)
    ui, vi = int(round(u)), int(round(v))
    assert depth[vi, ui] < z_near + 0.02
    assert color[vi, ui, 0] > color[vi, ui, 2]                # blue box in front
    _, depth_ball = render_rgbd(SimWorld(objects=[ball]), INTR, OBLIQUE)
    assert depth_ball[vi, ui] > depth[vi, ui] + 0.1


def test_gripper_drawn_and_occludes():
    w = SimWorld()
    T = top_down_T(0.30, 0.0, 0.10, 0.0)
    _, d0 = render_rgbd(w, INTR, DOWN)
    _, d1 = render_rgbd(w, INTR, DOWN, gripper=(T, 0.05))
    # Finger tip at tool y = +-(0.025 + 0.006) from the TCP, top-down: base y.
    u, v, _ = project(DOWN, INTR, (0.30, 0.031, 0.10 + FINGER_SIZE[2] / 2.0))
    assert d1[int(round(v)), int(round(u))] < d0[int(round(v)), int(round(u))] - 0.05
    w.update_tcp(T, 0.05)
    _, d2 = render_rgbd(w, INTR, DOWN)
    assert np.array_equal(d1, d2)
    _, d3 = render_rgbd(w, INTR, DOWN, gripper=False)
    assert np.array_equal(d0, d3)


def test_walled_bin_and_floor():
    z = SimZone('bin', (0.30, 0.0, 0.0), (0.14, 0.14), wall_height=0.05, color_bgr=(0, 200, 0))
    _, depth = render_rgbd(SimWorld(zones=[z]), INTR, DOWN)
    u, v, _ = project(DOWN, INTR, (0.30 + 0.072, 0.0, 0.05))
    assert depth[int(round(v)), int(round(u))] == pytest.approx(0.45, abs=1e-3)
    u, v, _ = project(DOWN, INTR, (0.30, 0.0, 0.001))
    assert depth[int(round(v)), int(round(u))] == pytest.approx(0.499, abs=1e-3)
    # A camera looking past the table edge sees the floor 0.75 m lower.
    T = look_at((1.2, 0.0, 0.3), (2.0, 0.0, -0.75))
    _, d = render_rgbd(SimWorld(), INTR, T)
    ui, vi = 320, 240
    ray = T[:3, :3] @ np.array([(ui - INTR.cx) / INTR.fx, (vi - INTR.cy) / INTR.fy, 1.0])
    t = (-0.75 - T[2, 3]) / ray[2]
    assert T[0, 3] + t * ray[0] > 0.9                        # past the table edge
    assert d[vi, ui] == pytest.approx(t, abs=1e-3)


def test_rays_missing_everything_are_invalid():
    T = look_at((0.3, 0.0, 0.5), (0.3, 0.0, 1.5), down_hint=(1.0, 0.0, 0.0))
    color, depth = render_rgbd(SimWorld(), SMALL, T)
    assert np.all(depth == 0.0)
    assert color.dtype == np.uint8 and color.shape == (240, 320, 3)


def test_noise_deterministic_and_bounded():
    w, info = make_scenario('L1', 0)
    T = info['T_base_camera']
    c1, d1 = render_rgbd(w, SMALL, T, rng=np.random.default_rng(5), noise_std=0.002,
                         dropout=0.01)
    c2, d2 = render_rgbd(w, SMALL, T, rng=np.random.default_rng(5), noise_std=0.002,
                         dropout=0.01)
    c3, d3 = render_rgbd(w, SMALL, T, rng=np.random.default_rng(6), noise_std=0.002,
                         dropout=0.01)
    assert np.array_equal(c1, c2) and np.array_equal(d1, d2)
    assert not np.array_equal(d1, d3)
    c0, d0 = render_rgbd(w, SMALL, T)
    both = (d0 > 0) & (d1 > 0)
    err = (d1 - d0)[both]
    expected = 0.002 * (d0[both] / 0.5) ** 2
    assert 0.8 < float(np.std(err / expected)) < 1.2
    lost = float(((d0 > 0) & (d1 == 0)).mean())
    assert 0.008 < lost < 0.05
    assert np.abs(c1.astype(int) - c0.astype(int)).mean() < 5.0
    assert d1.dtype == np.float32


def test_invalid_inputs_raise():
    w = SimWorld()
    with pytest.raises(ValueError):
        render_rgbd(w, SMALL, np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        render_rgbd(w, SMALL, DOWN, noise_std=-1.0)
    with pytest.raises(ValueError):
        render_rgbd(w, SMALL, DOWN, dropout=1.0)
    with pytest.raises(ValueError):
        render_rgbd(w, CameraIntrinsics(0.0, 1.0, 1.0, 1.0, 10, 10), DOWN)


@pytest.mark.parametrize('task,mount', [('L2', 'eye_to_hand'), ('L2', 'eye_in_hand')])
def test_render_speed(task, mount):
    w, info = make_scenario(task, 1, mount)
    if mount == 'eye_to_hand':
        T = info['T_base_camera']
    else:
        T = info['T_base_camera_observe']
    assert len(w.objects) == 6 and any(z.wall_height > 0 for z in w.zones)
    w.update_tcp(make_T(np.eye(3), (0.1, 0.2, 0.3)) @ top_down_T(0, 0, 0, 0), 0.07)
    times = []
    for k in range(3):
        t0 = time.perf_counter()
        render_rgbd(w, INTR, T, rng=np.random.default_rng(k), noise_std=0.0008, dropout=0.002)
        times.append(time.perf_counter() - t0)
    assert min(times) < 0.4, times
    assert math.isfinite(sum(times))

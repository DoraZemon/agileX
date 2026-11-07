"""Tests for tools/calib_points: pure helpers and scripted simulation runs of main()."""
from __future__ import annotations

import io
import json
import math

import cv2
import numpy as np
import pytest

from cube_task.core.extrinsics import Extrinsics, load_extrinsics
from cube_task.core.transforms import inv_T, make_T, pose6_to_T, pose_error, transform_points
from cube_task.core.types import Frame
from cube_task.sim.render import render_rgbd
from cube_task.sim.scenarios import default_intrinsics, look_at, make_scenario
from cube_task.sim.world import SimObject, SimWorld
from cube_task.tools import calib_points as cp
from cube_task.tools.ui import Ui

INTR = default_intrinsics()
T_BC = look_at((0.70, 0.0, 0.62), (0.30, 0.0, 0.0))     # the simulated eye-to-hand camera
T_FC = make_T(cv2.Rodrigues(np.array([0.02, -0.03, -1.55]))[0], (-0.07, 0.01, 0.03))


# ------------------------------------------------------------------ helpers
def _table_points(seed: int = 0, n: int = 8) -> np.ndarray:
    """Return n base-frame points spread over the pick area (z = block tops)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0.18, -0.20], [0.42, 0.20], size=(n, 2))
    return np.column_stack([xy, rng.uniform(0.0, 0.04, n)])


def _flange_poses(n: int, seed: int = 1) -> list:
    """Return n different eye-in-hand view poses (flange looking down at the table)."""
    rng = np.random.default_rng(seed)
    return [pose6_to_T([rng.uniform(0.10, 0.20), rng.uniform(-0.08, 0.08), rng.uniform(0.25, 0.32),
                        math.pi + rng.uniform(-0.1, 0.1), rng.uniform(-0.6, -0.4),
                        rng.uniform(-0.3, 0.3)]) for _ in range(n)]


def _pairs(mount: str, P: np.ndarray, noise: float = 0.0, seed: int = 2,
           views: list = None) -> list:
    """Return exact (or noisy camera-side) point pairs for the reference poses T_BC / T_FC."""
    rng = np.random.default_rng(seed)
    out = []
    poses = [None] * len(P)
    if mount == 'eye_in_hand':
        poses = _flange_poses(len(P)) if views is None else list(views)
    for i, (p, T_bf) in enumerate(zip(P, poses)):
        T_bc = T_BC if T_bf is None else T_bf @ T_FC
        p_cam = transform_points(inv_T(T_bc), p) + rng.normal(0.0, noise, 3)
        out.append(cp.PointPair(i + 1, p_cam, np.asarray(p, float), T_base_flange_cam=T_bf))
    return out


def _truth(mount: str) -> np.ndarray:
    """Return T_base_camera (eye_to_hand) or T_flange_camera (eye_in_hand) used for _pairs."""
    return T_BC if mount == 'eye_to_hand' else T_FC


def _render(objects, T_bc=T_BC, seed=None, noise=0.0, dropout=0.0) -> Frame:
    """Render a sim frame of the objects on the table (noise-free by default)."""
    world = SimWorld(objects=list(objects))
    rng = None if seed is None else np.random.default_rng(seed)
    color, depth = render_rgbd(world, INTR, T_bc, gripper=False, rng=rng, noise_std=noise,
                               dropout=dropout)
    return Frame(color, depth, INTR, 0.0, meta={'T_base_camera_true': T_bc})


def _cube(x: float, y: float, yaw: float = 0.3, size: float = 0.035) -> SimObject:
    obj = SimObject('cube', 'red', 'box', (size,) * 3, (40, 40, 200), np.eye(4))
    obj.place_upright(x, y, yaw, 0.0)
    return obj


def _top_center_cam(obj, T_bc=T_BC) -> np.ndarray:
    c = obj.center_xy()
    return transform_points(inv_T(T_bc), np.array([c[0], c[1], obj.top_z()]))


def _flat_frame(depth: np.ndarray) -> Frame:
    """Return a frame with a given depth image (camera looking straight down)."""
    return Frame(np.full(depth.shape + (3,), 128, np.uint8), depth.astype(np.float32), INTR, 0.0)


def _scripted(answers: dict):
    """Return an input_fn answering prompts by substring; each list is consumed in order."""
    queues = {k: list(v) for k, v in answers.items()}
    seen = []

    def fn(prompt):
        seen.append(prompt)
        for key, q in queues.items():
            if key in prompt:
                if not q:
                    raise AssertionError('no answer left for prompt %r' % prompt)
                return q.pop(0)
        raise AssertionError('unexpected prompt %r' % prompt)
    fn.seen = seen
    fn.queues = queues
    return fn


# ------------------------------------------------------------------ solving
@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_solve_extrinsics_exact(mount):
    ext, res = cp.solve_extrinsics(_pairs(mount, _table_points()), mount, stamp='now')
    dp, dr = pose_error(ext.T, _truth(mount))
    assert dp < 1e-9 and dr < 1e-9
    assert np.all(res < 1e-9) and ext.rms < 1e-9
    assert isinstance(ext, Extrinsics) and ext.mount == mount and ext.source == 'calib_points'
    assert ext.parent_frame == ('base' if mount == 'eye_to_hand' else 'flange')
    assert ext.stamp == 'now'


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_solve_extrinsics_noise(mount):
    ext, res = cp.solve_extrinsics(_pairs(mount, _table_points(n=10), noise=0.0005), mount)
    dp, dr = pose_error(ext.T, _truth(mount))
    assert dp < 0.002 and math.degrees(dr) < 0.3
    assert 0.0002 < ext.rms < 0.0015
    assert cp.find_outliers(res) == []


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_outlier_flagged_and_dropping_it_recovers(mount):
    pairs = _pairs(mount, _table_points(n=7), noise=0.0003)
    pairs[3].p_base = pairs[3].p_base + np.array([0.012, -0.006, 0.0])   # bad touch
    ext, res = cp.solve_extrinsics(pairs, mount)
    assert cp.find_outliers(res) == [3]
    assert ext.rms > cp.RMS_WARN_M - 0.001
    ext2, res2 = cp.solve_extrinsics(pairs[:3] + pairs[4:], mount)
    dp, dr = pose_error(ext2.T, _truth(mount))
    assert dp < 0.002 and math.degrees(dr) < 0.3 and cp.find_outliers(res2) == []


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_solve_refuses_too_few_or_collinear_points(mount):
    with pytest.raises(ValueError, match='at least 4'):
        cp.solve_extrinsics(_pairs(mount, _table_points(n=3)), mount)
    line = np.array([[0.2 + 0.05 * k, 0.1 * k - 0.15, 0.03] for k in range(6)])
    one_view = [_flange_poses(1)[0]] * len(line)
    with pytest.raises(ValueError, match='one line'):
        cp.solve_extrinsics(_pairs(mount, line, views=one_view), mount)
    if mount == 'eye_in_hand':
        # Different view poses make the flange-frame points non-collinear: solvable.
        ext, _res = cp.solve_extrinsics(_pairs(mount, line), mount)
        assert pose_error(ext.T, T_FC)[0] < 1e-9
    with pytest.raises(ValueError):
        cp.solve_extrinsics(_pairs(mount, _table_points()), 'on_the_wall')


def test_eye_in_hand_needs_the_flange_pose_of_each_measurement():
    pairs = _pairs('eye_in_hand', _table_points())
    pairs[2].T_base_flange_cam = None
    with pytest.raises(ValueError, match='#3'):
        cp.solve_extrinsics(pairs, 'eye_in_hand')


def test_find_outliers_rules():
    assert cp.find_outliers([0.001, 0.0012, 0.0009, 0.0011, 0.009]) == [4]
    # More than 3x the median but below 3 mm: kept.
    assert cp.find_outliers([0.0002, 0.0003, 0.0002, 0.0025]) == []
    # Large but not more than 3x the median: kept.
    assert cp.find_outliers([0.004, 0.005, 0.006, 0.010]) == []
    assert cp.find_outliers([0.001, 0.02, 0.0011, 0.0009, 0.015, 0.001]) == [1, 4]
    assert cp.find_outliers([0.001, 0.05]) == []


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_prediction_error_catches_a_bad_touch(mount):
    pairs = _pairs(mount, _table_points(n=5))
    assert cp.prediction_error(pairs[:2], pairs[2], mount) is None
    assert cp.prediction_error(pairs[:4], pairs[4], mount) < 1e-6
    bad = cp.PointPair(9, pairs[4].p_cam, pairs[4].p_base + np.array([0.0, 0.012, 0.0]),
                       T_base_flange_cam=pairs[4].T_base_flange_cam)
    assert cp.prediction_error(pairs[:4], bad, mount) == pytest.approx(0.012, abs=1e-6)


def test_line_offset_and_point_geometry():
    assert cp.line_offset([[0, 0, 0]]) == 0.0
    assert cp.line_offset([[0, 0, 0], [1, 1, 1], [2, 2, 2]]) < 1e-12
    square = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], dtype=float)
    assert cp.line_offset(square) == pytest.approx(0.05)
    assert cp.check_point_geometry(square, square) == (True, '')
    ok, why = cp.check_point_geometry(square[:3], square[:3])
    assert not ok and 'at least 4' in why
    thin = square * np.array([1.0, 0.1, 1.0])
    ok, why = cp.check_point_geometry(thin, thin)
    assert not ok and 'one line' in why


# ------------------------------------------------------------------ plane fit / geometry
def test_fit_plane_ransac_camera_frame():
    rng = np.random.default_rng(0)
    n_true = cp._unit([0.1, -0.8, -0.55])                # toward the camera at the origin
    e1, e2 = cp.plane_basis(n_true)
    uv = rng.uniform(-0.3, 0.3, size=(6000, 2))
    P = -0.6 * n_true + uv @ np.vstack([e1, e2]) + rng.normal(0, 0.001, (6000, 1)) * n_true
    clutter = -0.6 * n_true + rng.uniform(-0.1, 0.1, (1500, 3)) + 0.05 * n_true
    pts = np.vstack([P, clutter, np.full((50, 3), np.nan)])
    plane = cp.fit_plane_ransac(pts, rng=np.random.default_rng(1))
    assert plane.frame == 'camera' and plane.d > 0
    assert math.degrees(math.acos(min(1.0, float(plane.normal @ n_true)))) < 0.3
    assert plane.d == pytest.approx(0.6, abs=0.001)
    assert 0.7 < plane.inlier_ratio < 0.85 and plane.rms < 0.002
    assert cp.fit_plane_ransac(pts[:100]) is None


def test_plane_basis_is_orthonormal_and_right_handed():
    for n in ([0, 0, 1], [1, 0, 0], [0.3, -0.5, -0.8]):
        e1, e2 = cp.plane_basis(n)
        u = cp._unit(n)
        assert abs(e1 @ u) < 1e-12 and abs(e2 @ u) < 1e-12 and abs(e1 @ e2) < 1e-12
        assert np.allclose(np.cross(e1, e2), u)


def test_quad_center_is_the_projected_square_center():
    T = look_at((0.5, 0.2, 0.4), (0.3, 0.0, 0.0))
    c = np.array([0.3, 0.0, 0.0])
    sq = np.array([c + [-0.03, 0.03, 0], c + [0.03, 0.03, 0], c + [0.03, -0.03, 0],
                   c + [-0.03, -0.03, 0]])
    from cube_task.perception.geometry import project_points
    quad = project_points(sq, T, INTR)
    center = project_points(c, T, INTR)[0]
    assert np.linalg.norm(cp.quad_center(quad) - center) < 1e-6
    assert np.linalg.norm(quad.mean(axis=0) - center) > 0.05     # the corner mean is biased
    assert np.allclose(cp.quad_center(np.zeros((4, 2))), 0.0)


def test_pixel_ray_and_project_cam_round_trip():
    p = np.array([0.05, -0.02, 0.7])
    uv = cp.project_cam(p, INTR)[0]
    assert np.allclose(cp.pixel_ray(uv, INTR) * p[2], p)


def test_clean_depth_invalid_to_zero_and_median():
    d = np.full((20, 20), 0.5, np.float32)
    d[3, 3], d[5, 5], d[7, 7], d[9, 9] = np.nan, np.inf, -1.0, 0.9
    out = cp.clean_depth(d, 0)
    assert out.dtype == np.float32 and out[3, 3] == 0 and out[5, 5] == 0 and out[7, 7] == 0
    assert out[9, 9] == pytest.approx(0.9)
    med = cp.clean_depth(d, 3)
    assert med[9, 9] == pytest.approx(0.5) and med[3, 3] == pytest.approx(0.5)


# ------------------------------------------------------------------ block measurement
def test_measure_block_on_a_rendered_frame():
    cube = _cube(0.33, 0.07, 0.4)
    m = cp.measure_target_block(_render([cube]))
    assert m.ok, m.reason
    assert np.linalg.norm(m.p_cam - _top_center_cam(cube)) < 0.001
    assert m.height == pytest.approx(0.035, abs=0.001) and m.method == 'block'
    assert m.size == pytest.approx((0.035, 0.035), abs=0.004)
    assert m.mask.shape == (INTR.height, INTR.width) and m.mask.sum() == m.n_points
    assert m.outline.shape == (4, 2) and np.all(np.isfinite(m.pixel))
    assert cp.view_geometry(m.plane)[1] == pytest.approx(0.62, abs=0.002)


def test_measure_block_noisy_frames_averaged():
    cube = _cube(0.24, -0.12, 1.1)
    rng = np.random.default_rng(3)
    world = SimWorld(objects=[cube])
    ms = []
    for _ in range(4):
        color, depth = render_rgbd(world, INTR, T_BC, gripper=False, rng=rng, noise_std=0.0008,
                                   dropout=0.002)
        ms.append(cp.measure_target_block(Frame(color, depth, INTR, 0.0)))
    m = cp.average_measurements(ms)
    assert m.ok and m.n_frames == 4 and m.spread < 0.002
    assert np.linalg.norm(m.p_cam - _top_center_cam(cube)) < 0.0012


def test_measure_block_ignores_border_and_wide_components():
    depth = np.full((INTR.height, INTR.width), 0.6, np.float32)   # table 0.6 m below
    depth[0:60, :] = 0.58          # 20 mm high, touches the top image border (fingers)
    depth[300:470, 20:220] = 0.56  # 40 mm high, about 0.3 m wide (a bin)
    depth[200:240, 400:440] = 0.57  # the 30 mm block, 40 x 40 px
    m = cp.measure_target_block(_flat_frame(depth))
    assert m.ok, m.reason
    expect = np.array([(419.5 - INTR.cx) / INTR.fx, (219.5 - INTR.cy) / INTR.fy, 1.0]) * 0.57
    assert np.linalg.norm(m.p_cam - expect) < 1e-4
    assert m.height == pytest.approx(0.03, abs=1e-4)
    assert m.pixel == pytest.approx((419.5, 219.5), abs=0.01)
    depth[200:240, 400:440] = 0.6
    m = cp.measure_target_block(_flat_frame(depth))
    assert not m.ok
    assert 'no block' in m.reason and 'image border' in m.reason and 'wider than' in m.reason


def test_measure_block_failures():
    assert 'no depth' in cp.measure_target_block(Frame(np.zeros((4, 4, 3), np.uint8), None,
                                                       INTR, 0.0)).reason
    m = cp.measure_target_block(_flat_frame(np.zeros((INTR.height, INTR.width))))
    assert not m.ok and 'no table plane' in m.reason
    m = cp.measure_target_block(_render([]))
    assert not m.ok and 'no block found 8-80 mm above the table' in m.reason
    flat = SimObject('flat', 'x', 'box', (0.04, 0.04, 0.005), (40, 40, 200), np.eye(4))
    flat.place_upright(0.3, 0.0, 0.0, 0.0)
    assert not cp.measure_target_block(_render([flat])).ok      # 5 mm: below min_height


# ------------------------------------------------------------------ ArUco measurement
def _marker_frame(T_bc, size=0.04, marker_id=7, x=0.30, y=0.05, yaw=0.5):
    """Render a white plate and paint an ArUco marker on it; return (frame, true center cam)."""
    plate = cp.sim_target('aruco', size)
    plate.place_upright(x, y, yaw, 0.0)
    frame = _render([plate], T_bc)
    dic = cp.aruco_dictionary('DICT_4X4_50')
    px = cp.project_cam(transform_points(inv_T(T_bc), cp.marker_corners_base(plate, size)),
                        INTR)
    frame.color = cp.paint_marker(frame.color, px, cp.marker_image(dic, marker_id, 240))
    return frame, _top_center_cam(plate, T_bc), dic


def test_measure_aruco_with_depth_and_pnp_fallback():
    T_near = look_at((0.12, 0.0, 0.30), (0.30, 0.05, 0.0))
    frame, truth, dic = _marker_frame(T_near)
    m = cp.measure_target_aruco(frame, dic, 0.04)
    assert m.ok, m.reason
    assert m.method == 'aruco_depth' and m.marker_id == 7
    assert np.linalg.norm(m.p_cam - truth) < 0.001
    assert m.height == pytest.approx(cp.SIM_PLATE_M, abs=0.001)
    no_depth = Frame(frame.color, None, INTR, 0.0)
    m2 = cp.measure_target_aruco(no_depth, dic, 0.04)
    assert m2.ok and m2.method == 'aruco_pnp' and m2.plane is None
    assert np.linalg.norm(m2.p_cam - truth) < 0.008
    assert 'NO DEPTH' in cp.measurement_line(m2)


def test_measure_aruco_failures_and_dictionary():
    frame, _truth, dic = _marker_frame(T_BC)
    assert cp.measure_target_aruco(frame, dic, 0.04, marker_id=7).ok
    m = cp.measure_target_aruco(frame, dic, 0.04, marker_id=3)
    assert not m.ok and 'id 3 not found (seen: 7)' in m.reason
    m = cp.measure_target_aruco(_render([]), dic, 0.04)
    assert not m.ok and 'no ArUco marker' in m.reason
    with pytest.raises(ValueError, match='unknown ArUco dictionary'):
        cp.aruco_dictionary('DICT_NOPE')
    with pytest.raises(ValueError):
        cp.aruco_dictionary('CORNER_REFINE_SUBPIX')
    corners, ids = cp.detect_markers(cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY), dic)
    assert ids == [7] and corners[0].shape == (4, 2)


# ------------------------------------------------------------------ averaging, text, files
def _m(p, ok=True, reason='', height=0.03):
    return cp.Measurement(ok, reason, p_cam=None if p is None else np.asarray(p, float),
                          pixel=(10.0, 20.0), height=height)


def test_average_measurements():
    ms = [_m([0.0, 0.0, 0.5]), _m([0.001, 0.0, 0.5]), _m(None, False, 'no block'),
          _m([0.0, 0.001, 0.5])]
    m = cp.average_measurements(ms)
    assert m.ok and m.n_frames == 3
    assert np.allclose(m.p_cam, [0.001 / 3, 0.001 / 3, 0.5])
    assert m.spread == pytest.approx(np.linalg.norm([0.002 / 3, -0.001 / 3, 0.0]))
    assert m.pixel == (10.0, 20.0) and m.height == pytest.approx(0.03)
    spread = cp.average_measurements([_m([0.0, 0.0, 0.5]), _m([0.005, 0.0, 0.5])])
    assert not spread.ok and 'disagree by 2.5 mm' in spread.reason
    few = cp.average_measurements([_m([0, 0, 0.5]), _m(None, False, 'no block'),
                                   _m(None, False, 'no block'), _m(None, False, 'other')])
    assert not few.ok and few.reason == 'target found in 1 of 4 frames: no block'
    assert not cp.average_measurements([]).ok
    assert cp.average_measurements([_m([0, 0, 0.5]), _m(None, False, 'x')], min_ok=1).ok


def test_pairs_json_round_trip_and_validation():
    pairs = _pairs('eye_in_hand', _table_points(n=5))
    pairs[0].plane_normal, pairs[0].plane_d, pairs[0].spread = np.array([0, 0, -1.0]), 0.6, 0.0004
    data = json.loads(json.dumps(cp.pairs_to_json(pairs, 'eye_in_hand', 'block', {'x': 1})))
    assert data['tool'] == 'calib_points' and data['x'] == 1
    back = cp.pairs_from_json(data, 'eye_in_hand')
    assert [p.id for p in back] == [1, 2, 3, 4, 5]
    for a, b in zip(pairs, back):
        assert np.allclose(a.p_cam, b.p_cam) and np.allclose(a.T_base_flange_cam,
                                                             b.T_base_flange_cam)
    assert back[0].plane().d == pytest.approx(0.6) and back[1].plane() is None
    with pytest.raises(ValueError, match='mount'):
        cp.pairs_from_json(data, 'eye_to_hand')
    with pytest.raises(ValueError, match='session file'):
        cp.pairs_from_json({'tool': 'other', 'pairs': []})
    data['pairs'][1]['p_cam'] = [1.0, 2.0]
    with pytest.raises(ValueError, match='p_cam'):
        cp.pairs_from_json(data)
    data['pairs'][1] = dict(data['pairs'][2])
    with pytest.raises(ValueError, match='duplicate'):
        cp.pairs_from_json(data)


@pytest.mark.parametrize('text, expect', [
    ('', ('enter', None)), ('  ', ('enter', None)), ('m', ('measure', None)),
    ('S', ('solve', None)), ('u', ('undo', None)), ('quit', ('quit', None)),
    ('w', ('write', None)), ('a', ('add', None)), ('d 3', ('drop', 3)), ('d3', ('drop', 3)),
    ('drop 12', ('drop', 12)), ('d x3', ('unknown', None)), ('r', ('retouch', None)),
    ('x', ('discard', None)), ('help', ('unknown', None)), (None, ('enter', None))])
def test_parse_command(text, expect):
    assert cp.parse_command(text) == expect


def _planes_for(pairs, mount, T_true):
    """Attach the true table plane (z = 0 in base) in each measurement's camera frame."""
    for p in pairs:
        T_bc = T_true if mount == 'eye_to_hand' else p.T_base_flange_cam @ T_true
        n_c = T_bc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        p.plane_normal, p.plane_d = n_c, float(T_bc[2, 3])
    return pairs


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_sanity_report(mount):
    pairs = _planes_for(_pairs(mount, _table_points()), mount, _truth(mount))
    ext, _res = cp.solve_extrinsics(pairs, mount)
    poly = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]
    lines, warns = cp.sanity_report(ext, pairs, poly, expected_z=0.0)
    text = '\n'.join(lines).replace('-0.0', '0.0')
    assert warns == []
    assert 'tilt 0.00 deg from base z, z = 0.0 mm at the pick-area center' in text
    if mount == 'eye_to_hand':
        assert 'camera position in base: [700.0, 0.0, 620.0] mm' in text
        assert 'optical axis 32.8 deg from the table normal, camera 620 mm above' in text
    else:
        assert 'camera in the flange frame: [-70.0, 10.0, 30.0] mm' in text
    bad = Extrinsics(ext.T @ make_T(cv2.Rodrigues(np.array([0.09, 0.0, 0.0]))[0], (0, 0, 0)),
                     mount, rms=0.004)
    _lines, warns = cp.sanity_report(bad, pairs, poly, expected_z=0.0, expected_tol=0.01)
    assert any('tilted' in w for w in warns) and any('RMS 4.0 mm' in w for w in warns)
    assert any('expected_z' in w for w in warns)
    lines, _w = cp.sanity_report(ext, _pairs(mount, _table_points()))
    assert any('table checks skipped' in s for s in lines)


def test_compare_extrinsics():
    a = Extrinsics(T_BC, 'eye_to_hand', source='old.yaml')
    moved = T_BC.copy()
    moved[:3, 3] += [0.003, 0.0, -0.004]
    dp, dr, text = cp.compare_extrinsics(Extrinsics(moved, 'eye_to_hand'), a)
    assert dp == pytest.approx(0.005) and dr == pytest.approx(0.0, abs=1e-6)
    assert text == 'change from the previous extrinsics (old.yaml): 5.0 mm, 0.00 deg'
    other = Extrinsics(T_FC, 'eye_in_hand')
    assert cp.compare_extrinsics(other, a)[0] is None
    T_ft = pose6_to_T([0, 0, 0.14, 0, 0, 0])
    tcp = Extrinsics(inv_T(T_ft) @ T_FC, 'eye_in_hand', parent_frame='tcp')
    assert cp.compare_extrinsics(other, tcp, T_ft)[0] == pytest.approx(0.0, abs=1e-9)
    assert cp.compare_extrinsics(other, tcp)[0] is None


def test_text_helpers_and_preview():
    pairs = _pairs('eye_to_hand', _table_points(n=4))
    table = cp.residual_table(pairs, [0.0001, 0.0002, 0.0001, 0.009], [3])
    assert 'OUTLIER?' in table.splitlines()[5] and 'residual mm' in table
    assert 'RMS 4.50 mm, median 0.15 mm, max 9.00 mm over 4 points' in table
    assert cp.measurement_line(cp.Measurement(False, 'no block')) == \
        'measurement FAILED: no block'
    cube = _cube(0.3, 0.0)
    frame = _render([cube])
    m = cp.measure_target_block(frame)
    line = cp.measurement_line(cp.average_measurements([m, m]))
    assert 'top face 3' in line and '2 frames, spread 0.0 mm' in line and 'above the table' in line
    img = cp.draw_measurement(frame.color, m, 'point #1')
    assert img.shape == frame.color.shape and not np.array_equal(img, frame.color)
    u, v = (int(round(c)) for c in m.pixel)
    assert tuple(img[v, u]) == (0, 0, 255)               # red cross at the center


def test_sim_helpers():
    poly = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]
    spots = cp.sim_spots(poly)
    assert len(spots) == len(cp.SIM_GRID)
    assert all(0.18 <= x <= 0.42 and -0.19 <= y <= 0.19 for x, y in spots)
    assert spots[0] == pytest.approx((0.204, -0.152)) and spots[4] == pytest.approx((0.3, 0.0))
    cube = _cube(0.3, 0.0)
    assert cp.sim_visible(cube, T_BC, INTR)
    assert not cp.sim_visible(cube, look_at((2.0, 0.0, 0.6), (2.5, 0.0, 0.0)), INTR)
    T = cp.sim_touch_pose(cube)
    assert np.allclose(T[:3, 3], [0.3, 0.0, 0.035]) and T[2, 2] == pytest.approx(-1.0)
    plate = cp.sim_target('aruco', 0.04)
    assert plate.size == pytest.approx((0.06, 0.06, cp.SIM_PLATE_M))


# ------------------------------------------------------------------ scripted sim runs of main()
def _run(tmp_path, args, answers, name='run'):
    """Run main() in sim mode with scripted answers; return (code, printed text, input_fn)."""
    out = io.StringIO()
    fn = _scripted(answers)
    ui = Ui(headless=True, input_fn=fn, output=out, preview_dir=str(tmp_path / name))
    code = cp.main(['--mode', 'sim', '--headless', '--log-dir', str(tmp_path)] + args, ui=ui)
    return code, out.getvalue(), fn


def _result(tmp_path):
    return load_extrinsics(tmp_path / 'sim_site' / 'extrinsics.yaml')


def _last_session(tmp_path):
    files = sorted((tmp_path / 'calib_points').glob('*/pairs.json'))
    return json.loads(files[-1].read_text(encoding='utf-8'))


def _true_pose(mount):
    info = make_scenario('L1', 0, mount=mount)[1]
    return info['T_base_camera'] if mount == 'eye_to_hand' else info['T_flange_camera']


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_main_sim_recovers_the_true_camera_pose(tmp_path, mount):
    code, text, fn = _run(tmp_path, ['--mount', mount], {'w = write': ['w']})
    assert code == 0, text
    ext = _result(tmp_path)
    dp, dr = pose_error(ext.T, _true_pose(mount))
    assert dp < 0.002 and math.degrees(dr) < 0.3, (dp, math.degrees(dr))
    assert ext.mount == mount and ext.source == 'calib_points' and ext.rms < 0.0015
    assert ext.stamp and ext.parent_frame == ('base' if mount == 'eye_to_hand' else 'flange')
    assert 'keys written:' in text and '    matrix' in text and '    rms' in text
    assert 'RMS' in text and '[sim] error against the true camera pose' in text
    assert 'to use it: --set camera.extrinsics_file=' in text
    sess = _last_session(tmp_path)
    assert sess['mount'] == mount and len(sess['pairs']) == 6
    assert sess['result']['source'] == 'calib_points'
    if mount == 'eye_in_hand':        # the view pose changed between points
        views = {tuple(np.round(np.asarray(p['T_base_flange_cam'])[:3, 3], 4))
                 for p in sess['pairs']}
        assert len(views) > 3
    assert list((tmp_path / 'calib_points').glob('*/point_01.jpg'))
    assert (tmp_path / 'run' / 'calib_points.jpg').is_file()
    assert fn.queues['w = write'] == []


def test_main_sim_bad_touch_warned_dropped_and_replaced(tmp_path):
    answers = {'r = touch again': [''], 'Drop it': ['y'], 'w = write': ['a', 'w']}
    code, text, fn = _run(tmp_path, ['--sim-bad-touch', '5', '--frames', '2'], answers)
    assert code == 0, text
    assert 'WARNING: point #5: the touch is 1' in text and 'dropped point #5' in text
    assert 'OUTLIER?' in text
    dp, dr = pose_error(_result(tmp_path).T, _true_pose('eye_to_hand'))
    assert dp < 0.002 and math.degrees(dr) < 0.3
    assert [p['id'] for p in _last_session(tmp_path)['pairs']] == [1, 2, 3, 4, 6, 7]
    assert all(not q for q in fn.queues.values())


def test_main_sim_retouch_after_the_warning(tmp_path):
    code, text, _fn = _run(tmp_path, ['--sim-bad-touch', '4', '--frames', '2', '--points', '5'],
                           {'r = touch again': ['r'], 'w = write': ['w']})
    assert code == 0, text
    assert 'OUTLIER?' not in text
    assert [p['id'] for p in _last_session(tmp_path)['pairs']] == [1, 2, 3, 4, 5]


def test_main_sim_quit_then_resume(tmp_path):
    code, text, _fn = _run(tmp_path, ['--points', '4', '--frames', '2'],
                           {'w = write': ['d 9', 'd 2', 'q']})
    assert code == 1
    assert 'usage: d N with N one of 1, 2, 3, 4' in text and 'cannot drop' in text
    assert 'extrinsics NOT written. 4 points are saved: continue with --resume' in text
    assert not (tmp_path / 'sim_site' / 'extrinsics.yaml').exists()
    session = sorted((tmp_path / 'calib_points').glob('*/pairs.json'))[-1]
    code, text, _fn = _run(tmp_path, ['--points', '5', '--frames', '2', '--resume', str(session)],
                           {'w = write': ['w']}, 'run2')
    assert code == 0, text
    assert 'resumed 4 points' in text and 'point #5 recorded' in text
    assert len(_last_session(tmp_path)['pairs']) == 5
    assert _result(tmp_path).mount == 'eye_to_hand'


def test_main_sim_aruco_eye_in_hand(tmp_path):
    code, text, _fn = _run(tmp_path, ['--mount', 'eye_in_hand', '--target', 'aruco',
                                      '--frames', '2', '--points', '5'], {'w = write': ['w']})
    assert code == 0, text
    assert 'marker id 0' in text and 'CENTER OF THE MARKER' in text
    dp, dr = pose_error(_result(tmp_path).T, _true_pose('eye_in_hand'))
    assert dp < 0.002 and math.degrees(dr) < 0.3


def test_main_sim_unknown_aruco_dictionary(tmp_path):
    code, text, _fn = _run(tmp_path, ['--target', 'aruco', '--aruco-dict', 'DICT_FOO'], {})
    assert code == 2 and 'unknown ArUco dictionary' in text


@pytest.mark.parametrize('args', [['--points', '3'], ['--frames', '0'], ['--aruco-size', '40'],
                                  ['--max-spread', '0'], ['--sim-bad-touch', '2'],
                                  ['--mode', 'replay', '--replay-dir', '/tmp']])
def test_main_rejects_bad_arguments(args, capsys):
    with pytest.raises(SystemExit) as e:
        cp.main(args, ui=Ui(headless=True, input_fn=_scripted({})))
    assert e.value.code == 2
    assert 'calib_points: error:' in capsys.readouterr().err


def _fake_real_hardware(monkeypatch) -> dict:
    """Make main() run in real mode on the simulated arm / camera (SimClock); return state."""
    import argparse
    from cube_task.tools import common

    def as_sim(args):
        a = argparse.Namespace(**vars(args))
        a.mode = 'sim'
        return a

    state = {}

    def fake_open_session(args, **kw):
        s = common.open_session(as_sim(args), **kw)   # SimClock, sim arm and camera
        s.mode = 'real'
        state['s'] = s
        state['cube'] = _cube(0.3, 0.0)
        s.world.objects = [state['cube']]
        state['spots'] = iter(cp.sim_spots(s.cfg.get('perception.roi.pick_polygon')))
        return s

    monkeypatch.setattr(cp, 'load_cfg_for', lambda args, task=None: common.load_cfg_for(
        as_sim(args), task))
    monkeypatch.setattr(cp, 'open_session', fake_open_session)
    return state


def _place_next(state) -> None:
    """Operator: put the cube on the next spot and move the arm out of the view."""
    from cube_task.tools import onsite
    x, y = next(state['spots'])
    state['cube'].place_upright(x, y, 0.2, 0.0)
    onsite.sim_drag_joints(state['s'].arm, state['s'].cfg.get('robot.poses.observe_joints'))


def _touch(state) -> None:
    """Operator: drag the TCP onto the top-face center of the cube."""
    from cube_task.tools import onsite
    s = state['s']
    assert onsite.sim_drag_tcp(s.arm, s.cfg, cp.sim_touch_pose(state['cube']))


def test_main_real_mode_prompts_against_simulated_hardware(tmp_path, monkeypatch):
    """Drive the real-mode headless flow; the scripted 'operator' moves the sim block and arm."""
    state = _fake_real_hardware(monkeypatch)

    def operator(prompt):
        if 'put the block at a new spot' in prompt:
            answer = place_answers.pop(0)
            if answer == '':
                _place_next(state)
            return answer
        if 'touches the CENTER OF THE BLOCK TOP FACE' in prompt:
            answer = touch_answers.pop(0)
            if answer == '':
                _touch(state)
            return answer
        if prompt.startswith('w = write'):
            return 'w'
        raise AssertionError('unexpected prompt %r' % prompt)

    # place #1, #2, undo #2, place #3 (touch cancelled), 's' with 1 point, then #3, #4, #5
    place_answers = ['', '', 'u', '', 's', '', '', '']
    touch_answers = ['', '', 'q', '', '', '']
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=operator, output=out, preview_dir=str(tmp_path / 'p'))
    code = cp.main(['--points', '4', '--frames', '2', '--headless', '--log-dir', str(tmp_path),
                    '--site-dir', str(tmp_path / 'site')], ui=ui)
    text = out.getvalue()
    assert code == 0, text
    assert 'removed point #2' in text and 'point #3 cancelled' in text
    assert 'need at least 4 points to solve, have 1' in text
    assert 'The tool never moves the arm' in text and '[sim' not in text
    assert place_answers == [] and touch_answers == []
    ext = load_extrinsics(tmp_path / 'site' / 'extrinsics.yaml')
    dp, dr = pose_error(ext.T, _true_pose('eye_to_hand'))
    assert dp < 0.002 and math.degrees(dr) < 0.3
    assert [p['id'] for p in _last_session(tmp_path)['pairs']] == [1, 3, 4, 5]


class _WindowUi(Ui):
    """Ui in window mode without a display: records shown images, plays scripted keys."""

    def __init__(self, keys, answers, output):
        super().__init__(headless=True, input_fn=_scripted(answers), output=output)
        self.headless = False
        self.keys = list(keys)
        self.shown = 0

    def show(self, name, img, save_every=1):
        self.shown += 1
        return None

    def key(self, wait_ms=1):
        if wait_ms <= 1:              # a redraw after show(), not a wait for the operator
            return None
        k = self.keys.pop(0)
        return k() if callable(k) else k


def test_main_real_mode_window_keys(tmp_path, monkeypatch):
    state = _fake_real_hardware(monkeypatch)

    def place():
        _place_next(state)
        return None                   # one more live frame of the new spot

    def touch():
        _touch(state)
        return ' '

    keys = [place, '\n', touch] * 4
    out = io.StringIO()
    ui = _WindowUi(keys, {'w = write': ['w']}, out)
    code = cp.main(['--points', '4', '--frames', '2', '--log-dir', str(tmp_path),
                    '--site-dir', str(tmp_path / 'site')], ui=ui)
    text = out.getvalue()
    assert code == 0, text
    assert cp.WINDOW_PLACE in text and 'Enter or space in the preview window' in text
    assert ui.keys == [] and ui.shown >= 8
    dp, dr = pose_error(load_extrinsics(tmp_path / 'site' / 'extrinsics.yaml').T,
                        _true_pose('eye_to_hand'))
    assert dp < 0.002 and math.degrees(dr) < 0.3

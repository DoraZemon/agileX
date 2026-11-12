"""Hand-eye calibration: synthetic poses and a rendered ChArUco board."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from piper_vision.handeye import (BoardSpec, HandEyeCalibrator, board_corners_3d,
                                  coverage_warning, detect_board, find_charuco_corners,
                                  make_board, render_board)
from piper_vision.transforms import inv_T, make_T, pose6_to_T, rotation_angle, rpy_to_rot

from synth import default_intrinsics, look_at

T_FLANGE_CAM = make_T(rpy_to_rot(0.0, 0.0, -math.pi / 2), (0.06, 0.0, 0.03))
T_BASE_CAM = look_at((0.75, 0.0, 0.70), (0.25, 0.0, 0.0))


def _flange_poses(n, rng):
    out = []
    for _ in range(n):
        p = [rng.uniform(0.22, 0.38), rng.uniform(-0.12, 0.12), rng.uniform(0.25, 0.40),
             math.pi + rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-1.0, 1.0)]
        out.append(pose6_to_T(p))
    return out


def _noisy(T, rng, t_std=0.0005, r_std_deg=0.1):
    dR = rpy_to_rot(*np.radians(rng.normal(0, r_std_deg, 3)))
    return make_T(dR @ T[:3, :3], T[:3, 3] + rng.normal(0, t_std, 3))


def _views(mount, rng, n=12, T_flange_tcp=np.eye(4)):
    """Return (T_base_tcp, T_camera_board) pairs for the tutorial's two setups."""
    out = []
    for T_bf in _flange_poses(n, rng):
        T_bt = T_bf @ T_flange_tcp
        if mount == 'eye_in_hand':       # board still on the table, camera on the wrist
            T_base_board = make_T(np.eye(3), (0.30, 0.0, 0.0))
            T_cb = inv_T(T_bf @ T_FLANGE_CAM) @ T_base_board
        else:                            # board fixed on the flange, camera on a stand
            T_flange_board = make_T(rpy_to_rot(0, 0, 0.3), (0.0, 0.02, 0.05))
            T_cb = inv_T(T_BASE_CAM) @ T_bf @ T_flange_board
        out.append((T_bt, _noisy(T_cb, rng)))
    return out


@pytest.mark.parametrize('mount', ['eye_in_hand', 'eye_to_hand'])
def test_recovers_transform(mount):
    rng = np.random.default_rng(1)
    cal = HandEyeCalibrator(mount)
    for T_bt, T_cb in _views(mount, rng):
        cal.add(T_bt, T_cb)
    truth = T_FLANGE_CAM if mount == 'eye_in_hand' else T_BASE_CAM
    res = cal.solve()
    assert res.extrinsics.mount == mount
    assert res.extrinsics.parent_frame == ('flange' if mount == 'eye_in_hand' else 'base')
    assert np.linalg.norm(res.extrinsics.T[:3, 3] - truth[:3, 3]) < 0.004
    assert math.degrees(rotation_angle(res.extrinsics.T[:3, :3].T @ truth[:3, :3])) < 0.5
    # the residual is the spread of whatever the setup holds fixed, so it is
    # small for both setups when the solution is right
    assert res.rms_m < 0.004


@pytest.mark.parametrize('mount', ['eye_in_hand', 'eye_to_hand'])
def test_tcp_offset_poses_give_the_flange_result(mount):
    # agx_arm_ros launched with tcp_offset publishes T_base_tcp on /feedback/tcp_pose
    tcp = [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(2)
    cal = HandEyeCalibrator(mount, tcp_offset=tcp)
    for T_bt, T_cb in _views(mount, rng, T_flange_tcp=pose6_to_T(tcp)):
        cal.add(T_bt, T_cb)
    truth = T_FLANGE_CAM if mount == 'eye_in_hand' else T_BASE_CAM
    res = cal.solve()
    assert np.linalg.norm(res.extrinsics.T[:3, 3] - truth[:3, 3]) < 0.004
    assert res.rms_m < 0.004


def test_needs_enough_samples():
    cal = HandEyeCalibrator('eye_in_hand')
    with pytest.raises(ValueError, match='at least'):
        cal.solve()


def test_coverage_warning_for_pure_translation():
    cal = HandEyeCalibrator('eye_in_hand')
    for x in (0.2, 0.25, 0.3, 0.35):
        cal.add(pose6_to_T([x, 0, 0.3, math.pi, 0, 0]), np.eye(4))
    assert coverage_warning(cal.samples) is not None


def test_board_detection_pose():
    # The board frame differs between OpenCV 4.5 (origin bottom-left) and 4.7+
    # (top-left); only consistency matters. Check that every detected corner,
    # placed with the estimated pose, lands where the warped printout put it.
    spec = BoardSpec()
    img = render_board(spec, dpi=200)
    h, w = img.shape[:2]
    m_per_px = (spec.squares_x * spec.square_m) / w
    intr = default_intrinsics()
    T_cp = make_T(rpy_to_rot(0.35, -0.2, 0.1), (-0.07, -0.08, 0.50))   # printout plane
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], float) * m_per_px
    pc = corners @ T_cp[:3, :3].T + T_cp[:3, 3]
    uv = np.column_stack((intr.fx * pc[:, 0] / pc[:, 2] + intr.cx,
                          intr.fy * pc[:, 1] / pc[:, 2] + intr.cy)).astype(np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(src, uv)
    frame = cv2.warpPerspective(img, H, (intr.width, intr.height), borderValue=(255, 255, 255))
    det = detect_board(frame, intr, spec)
    assert det is not None
    T_est, n, err = det
    assert n >= 15 and err < 1.0
    board = make_board(spec)
    px, ids = find_charuco_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), board)
    truth = np.column_stack((px * m_per_px, np.zeros(len(px)))) @ T_cp[:3, :3].T + T_cp[:3, 3]
    est = board_corners_3d(board)[ids] @ T_est[:3, :3].T + T_est[:3, 3]
    assert np.max(np.linalg.norm(est - truth, axis=1)) < 0.004

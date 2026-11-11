"""Camera config loading, extrinsics formats and pose conversion."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from piper_vision.camera import (Extrinsics, camera_pose_in_base, extrinsics_from_dict,
                                 load_camera_config, points_base_to_camera,
                                 points_camera_to_base, write_extrinsics)
from piper_vision.transforms import (T_to_pose6, inv_T, make_T, pose6_to_T, rot_to_quat,
                                     rpy_to_rot)

ROOT = Path(__file__).resolve().parent.parent
T_EX = make_T(rpy_to_rot(0.3, -0.2, 1.1), (0.4, -0.1, 0.5))


def test_pose6_roundtrip():
    p = [0.1, -0.2, 0.3, 0.4, -0.5, 2.9]
    assert np.allclose(T_to_pose6(pose6_to_T(p)), p)


def test_example_config_loads():
    cfg = load_camera_config(str(ROOT / 'config' / 'camera.example.yaml'))
    assert cfg.mount == 'eye_to_hand'
    assert cfg.intrinsics.width == 640
    T = cfg.T_base_camera()
    assert T[:3, 3] == pytest.approx([0.70, 0.0, 0.62])
    # camera z points down and backward toward the arm
    assert T[2, 2] < -0.8


@pytest.mark.parametrize('d', [
    {'parent_frame': 'base', 'matrix': T_EX.tolist()},
    {'T_base_camera': {'matrix': T_EX.tolist()}},
    {'T_camera_base': {'matrix': inv_T(T_EX).tolist()}},
    {'mount': 'eye_to_hand', 'translation': T_EX[:3, 3].tolist(),
     'quaternion': rot_to_quat(T_EX[:3, :3])},
    {'parent_frame': 'base', 'xyz': T_EX[:3, 3].tolist(),
     'rpy_deg': [math.degrees(v) for v in T_to_pose6(T_EX)[3:]]},
    {'parent_frame': 'base', 'position': T_EX[:3, 3].tolist(),
     'orientation': rot_to_quat(T_EX[:3, :3])},
])
def test_extrinsics_formats_agree(d):
    ext = extrinsics_from_dict(d)
    assert ext.mount == 'eye_to_hand'
    assert np.allclose(ext.T, T_EX, atol=1e-6)


def test_tcp_parent_converted_to_flange():
    tcp = [0, 0, 0.1425, 0, 0, 0]
    ext = extrinsics_from_dict({'T_tcp_camera': {'matrix': T_EX.tolist()}, 'tcp_offset': tcp})
    assert ext.parent_frame == 'flange' and ext.mount == 'eye_in_hand'
    assert np.allclose(ext.T, pose6_to_T(tcp) @ T_EX)
    with pytest.raises(ValueError, match='tcp_offset'):
        extrinsics_from_dict({'T_tcp_camera': {'matrix': T_EX.tolist()}})


def test_camera_link_needs_link_optical():
    with pytest.raises(ValueError, match='T_link_optical'):
        extrinsics_from_dict({'parent_frame': 'base', 'matrix': T_EX.tolist(),
                              'child_frame': 'camera_link'})
    T_lo = make_T(rpy_to_rot(-math.pi / 2, 0, -math.pi / 2), (0, 0.015, 0))
    ext = extrinsics_from_dict({'parent_frame': 'base', 'matrix': T_EX.tolist(),
                                'child_frame': 'camera_link', 'T_link_optical': T_lo.tolist()})
    assert np.allclose(ext.T, T_EX @ T_lo)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match='disagree'):
        extrinsics_from_dict({'parent_frame': 'base', 'matrix': T_EX.tolist(),
                              'xyz': [0, 0, 0], 'rpy': [0, 0, 0]})
    big = T_EX.copy()
    big[:3, 3] = [400, 0, 500]   # mm by mistake
    with pytest.raises(ValueError, match='mm'):
        extrinsics_from_dict({'parent_frame': 'base', 'matrix': big.tolist()})
    with pytest.raises(ValueError, match='rpy_deg'):
        extrinsics_from_dict({'parent_frame': 'base', 'xyz': [0, 0, 0], 'rpy': [0, 0, 90]})
    bad = T_EX.copy()
    bad[:3, :3] *= 2
    with pytest.raises(ValueError, match='rotation'):
        extrinsics_from_dict({'parent_frame': 'base', 'matrix': bad.tolist()})


def test_eye_in_hand_chain_and_point_transform():
    ext = Extrinsics(T=T_EX, mount='eye_in_hand', parent_frame='flange')
    T_bf = pose6_to_T([0.3, 0.0, 0.4, math.pi, 0, 0])
    with pytest.raises(ValueError):
        camera_pose_in_base(ext)
    T_bc = camera_pose_in_base(ext, T_bf)
    assert np.allclose(T_bc, T_bf @ T_EX)
    pts = np.array([[0.1, 0.2, 0.5], [0.0, 0.0, 1.0]])
    back = points_base_to_camera(T_bc, points_camera_to_base(T_bc, pts))
    assert np.allclose(back, pts)


def test_write_extrinsics_keeps_other_sections(tmp_path):
    path = tmp_path / 'camera.yaml'
    path.write_text((ROOT / 'config' / 'camera.example.yaml').read_text(encoding='utf-8'))
    ext = Extrinsics(T=T_EX, mount='eye_in_hand', parent_frame='flange', rms=0.002)
    write_extrinsics(str(path), ext)
    cfg = load_camera_config(str(path))
    assert cfg.mount == 'eye_in_hand'
    assert np.allclose(cfg.extrinsics.T, T_EX, atol=1e-5)
    assert cfg.extrinsics.rms == pytest.approx(0.002)
    assert cfg.intrinsics.fx == 380.0
    assert cfg.board['squares_y'] == 7

"""
Camera config, hand-eye calibration and L1 / L2 detection for PiPER.

Independent of cube_task: depends only on numpy, OpenCV and PyYAML (rclpy and
pyAgxArm are imported lazily by the live tools).
"""
from __future__ import annotations

from .camera import (CameraConfig, Extrinsics, Intrinsics, camera_pose_in_base,
                     extrinsics_from_dict, load_camera_config, points_base_to_camera,
                     points_camera_to_base, save_camera_config, write_extrinsics)
from .handeye import BoardSpec, HandEyeCalibrator, HandEyeResult, detect_board
from .l1 import CubeDetection, L1Detector, L1Params, L1Result, ZoneDetection

__all__ = [
    'BoardSpec', 'CameraConfig', 'CubeDetection', 'Extrinsics', 'HandEyeCalibrator',
    'HandEyeResult', 'Intrinsics', 'L1Detector', 'L1Params', 'L1Result', 'ZoneDetection',
    'camera_pose_in_base', 'detect_board', 'extrinsics_from_dict', 'load_camera_config',
    'points_base_to_camera', 'points_camera_to_base', 'save_camera_config', 'write_extrinsics',
]

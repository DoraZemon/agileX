"""
Pose helpers.

Conventions (same as pyAgxArm and cube_task):

- ``T_a_b`` is a 4x4 float64 matrix that maps coordinates in frame b into
  frame a: ``p_a = T_a_b @ p_b``.
- pose6 is ``[x, y, z, roll, pitch, yaw]`` in m / rad with R = Rz(yaw) Ry(pitch) Rx(roll).
- Quaternions are ``[x, y, z, w]``.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    return make_T(R.T, -R.T @ T[:3, 3])


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of rpy_to_rot; pitch in [-pi/2, pi/2]."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put everything into yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def pose6_to_T(pose: Sequence[float]) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw] to 4x4."""
    if len(pose) != 6:
        raise ValueError('pose6 needs 6 values, got %d' % len(pose))
    x, y, z, r, p, yw = [float(v) for v in pose]
    return make_T(rpy_to_rot(r, p, yw), (x, y, z))


def T_to_pose6(T: np.ndarray) -> list:
    """Convert 4x4 to [x, y, z, roll, pitch, yaw]."""
    T = np.asarray(T, dtype=float)
    r, p, yw = rot_to_rpy(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), r, p, yw]


def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] to a rotation matrix."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError('zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R: np.ndarray) -> list:
    """Convert a rotation matrix to a quaternion [x, y, z, w] with w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    if w < 0:
        q = -q
    return [float(v) for v in q / np.linalg.norm(q)]


def rotation_angle(R: np.ndarray) -> float:
    """Return the rotation angle of R (rad, in [0, pi])."""
    c = (np.trace(np.asarray(R, dtype=float)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Return the closest rotation matrix (SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform Nx3 (or 3,) points with a 4x4 matrix."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(pts, dtype=float)
    out = p.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    return out.reshape(p.shape)


def wrap_angle(a: float, period: float = 2 * math.pi) -> float:
    """Wrap an angle into [-period/2, period/2)."""
    return float((a + period / 2.0) % period - period / 2.0)

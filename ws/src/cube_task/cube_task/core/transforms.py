"""
Rigid-body math. Conventions match the PiPER SDK (pyAgxArm).

Euler angles: pose6 = [x, y, z, roll, pitch, yaw] with
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (ZYX, same as pyAgxArm.utiles.tf.rpy_to_rot
and SciPy Rotation.from_euler('xyz', [roll, pitch, yaw])).
Quaternions are (x, y, z, w) like ROS geometry_msgs and SciPy.
T_a_b maps points in frame b to frame a: p_a = T_a_b @ [p_b, 1].
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of rpy_to_rot. pitch in [-pi/2, pi/2]; roll, yaw in [-pi, pi]."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock (pitch = +-90 deg): roll and yaw are coupled, put it all in yaw.
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return float(roll), float(pitch), float(yaw)


def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    """Quaternion (x, y, z, w) to rotation matrix. Normalizes the input."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError('zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (x, y, z, w) with w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(list(t), dtype=float)
    return T


def pose6_to_T(pose: Sequence[float]) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] (SDK convention) to 4x4."""
    if len(pose) != 6:
        raise ValueError('pose6 needs 6 values, got %d' % len(pose))
    x, y, z, r, p, yw = [float(v) for v in pose]
    return make_T(rpy_to_rot(r, p, yw), (x, y, z))


def T_to_pose6(T: np.ndarray) -> list:
    """4x4 to [x, y, z, roll, pitch, yaw] (SDK convention)."""
    T = np.asarray(T, dtype=float)
    r, p, yw = rot_to_rpy(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), r, p, yw]


def xyzquat_to_T(xyz: Sequence[float], q: Sequence[float]) -> np.ndarray:
    return make_T(quat_to_rot(q), xyz)


def T_to_xyzquat(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), rot_to_quat(T[:3, :3])


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 T to Nx3 points (or a single (3,) point)."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(pts, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (rad) of R, in [0, pi]."""
    c = (np.trace(np.asarray(R, dtype=float)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(position error m, rotation error rad) between two poses."""
    dp = float(np.linalg.norm(np.asarray(T_a)[:3, 3] - np.asarray(T_b)[:3, 3]))
    dR = np.asarray(T_a)[:3, :3].T @ np.asarray(T_b)[:3, :3]
    return dp, rotation_angle(dR)


def is_rotation(R: np.ndarray, tol: float = 1e-4) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < tol)


def top_down_R(closing_angle: float) -> np.ndarray:
    """
    TCP orientation for a vertical top-down grasp.

    Tool z (approach) points to base -Z. Tool y is the finger closing axis and
    lies in the base XY plane at closing_angle from base +X. Tool x = y cross z.
    """
    c, s = math.cos(closing_angle), math.sin(closing_angle)
    y = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def top_down_T(x: float, y: float, z: float, closing_angle: float) -> np.ndarray:
    return make_T(top_down_R(closing_angle), (x, y, z))


def closing_angle_of(T_base_tcp: np.ndarray) -> float:
    """Base-frame angle of the tool y axis projected on XY."""
    yv = np.asarray(T_base_tcp)[:3, 1]
    return float(math.atan2(yv[1], yv[0]))


def neutral_closing_angle(x: float, y: float) -> float:
    """
    Return the closing angle that needs no wrist twist for a target at (x, y).

    For PiPER with joint4 = joint6 = 0 in a top-down configuration the finger
    closing axis (flange y) is perpendicular to the azimuth of the target.
    """
    return wrap_pi(math.atan2(y, x) + math.pi / 2.0)


def choose_closing_angle(axis_angle: float, x: float, y: float,
                         symmetry: float = math.pi) -> float:
    """
    Pick the equivalent closing angle closest to the neutral wrist angle.

    axis_angle: any closing angle that grips the object correctly.
    symmetry: pi for a two-finger gripper on any object (theta == theta + pi);
              pi/2 when the object is also square (blocks).
    """
    neutral = neutral_closing_angle(x, y)
    best = axis_angle
    best_d = 1e9
    k_max = int(round(2.0 * math.pi / symmetry))
    for k in range(-k_max, k_max + 1):
        cand = axis_angle + k * symmetry
        d = abs(wrap_pi(cand - neutral))
        if d < best_d - 1e-12:
            best_d = d
            best = cand
    return wrap_pi(best)


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Least-squares rigid transform T with dst ~= T @ src (Kabsch, no scale).

    src, dst: Nx3, N >= 3 non-collinear. Returns (T, rms residual in meters).
    """
    P = np.asarray(src, dtype=float)
    Q = np.asarray(dst, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError('need two Nx3 arrays with N >= 3')
    cp = P.mean(axis=0)
    cq = Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    if np.linalg.matrix_rank(P - cp, tol=1e-6) < 2:
        raise ValueError('points are collinear or coincident')
    U, _S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    T = make_T(R, t)
    res = Q - transform_points(T, P)
    rms = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
    return T, rms

"""
Camera configuration and pose conversion.

One YAML file holds everything camera-related: which camera, the ROS topics,
the color intrinsics, the hand-eye extrinsics and the board used for
calibration. See config/camera.example.yaml for the full format.

Pose convention (same as pyAgxArm and cube_task): ``T_a_b`` maps coordinates
in frame b into frame a, so ``p_a = T_a_b @ p_b``. The camera frame is the
color optical frame (x right, y down, z forward).

    eye_to_hand (fixed camera): the file holds T_base_camera.
    eye_in_hand (wrist camera): the file holds T_flange_camera, and
        T_base_camera = T_base_flange(at capture) @ T_flange_camera.
    A calibration relative to the TCP is accepted with a tcp_offset pose6
    (the driver TCP offset used during calibration) and stored as flange.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .transforms import (T_to_pose6, inv_T, make_T, orthonormalize, pose6_to_T,
                         quat_to_rot, rot_to_quat, rotation_angle, rpy_to_rot)

LOG = logging.getLogger('piper_vision.camera')

MAX_TRANSLATION_M = 3.0
MAX_RPY_RAD = 2 * math.pi
_SUFFIXES = ('.yaml', '.yml', '.json')


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

@dataclass
class Intrinsics:
    """Pinhole intrinsics of the color image. dist all zeros = rectified."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # OpenCV order (k1, k2, p1, p2, k3).
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def K(self) -> np.ndarray:
        """Return the 3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    def scaled(self, width: int, height: int) -> 'Intrinsics':
        """Return intrinsics for the same camera resized to (width, height)."""
        sx = width / float(self.width)
        sy = height / float(self.height)
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                          int(width), int(height), np.array(self.dist, dtype=float))

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 camera-frame points to Nx2 pixels (no distortion)."""
        p = np.asarray(pts_cam, dtype=float).reshape(-1, 3)
        z = p[:, 2:3]
        uv = p[:, :2] / np.where(np.abs(z) < 1e-9, np.nan, z)
        return np.column_stack((self.fx * uv[:, 0] + self.cx,
                                self.fy * uv[:, 1] + self.cy))

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project Nx2 pixels at depth (m) to Nx3 camera-frame points."""
        u, v = np.asarray(uv, dtype=float).reshape(-1, 2).T
        z = np.asarray(depth, dtype=float).reshape(-1)
        return np.column_stack(((u - self.cx) / self.fx * z,
                                (v - self.cy) / self.fy * z, z))

    def to_dict(self) -> Dict[str, Any]:
        """Return a YAML-friendly dict."""
        return {'fx': float(self.fx), 'fy': float(self.fy), 'cx': float(self.cx),
                'cy': float(self.cy), 'width': int(self.width), 'height': int(self.height),
                'dist': [float(v) for v in np.asarray(self.dist).ravel()]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Intrinsics':
        """Build intrinsics from a dict (missing dist means rectified)."""
        try:
            intr = Intrinsics(float(d['fx']), float(d['fy']), float(d['cx']),
                              float(d['cy']), int(d['width']), int(d['height']),
                              np.array(d.get('dist', [0.0] * 5), dtype=float).ravel())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('intrinsics need fx, fy, cx, cy, width, height: %s' % exc)
        if not all(math.isfinite(v) for v in (intr.fx, intr.fy, intr.cx, intr.cy)):
            raise ValueError('intrinsics must be finite')
        if intr.width <= 0 or intr.height <= 0:
            raise ValueError('intrinsics width / height must be positive')
        return intr

    @staticmethod
    def from_camera_info(msg: Any) -> 'Intrinsics':
        """Build intrinsics from a ROS sensor_msgs/CameraInfo message."""
        K = list(msg.k)
        dist = list(msg.d) if msg.d else [0.0] * 5
        return Intrinsics(K[0], K[4], K[2], K[5], int(msg.width), int(msg.height),
                          np.array(dist, dtype=float))


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------

def is_optical_frame(name: str) -> bool:
    """Return whether a frame name ends in '_optical_frame'."""
    return str(name).lower().endswith('_optical_frame')


@dataclass
class Extrinsics:
    """
    Hand-eye result: the transform from the camera frame to its parent.

    ``T`` is T_<parent>_camera. ``mount`` is 'eye_to_hand' (parent 'base') or
    'eye_in_hand' (parent 'flange'). ``rms`` is the calibration residual in m.
    """

    T: np.ndarray
    mount: str
    parent_frame: str = 'base'
    child_frame: str = 'camera_color_optical_frame'
    source: str = ''
    rms: Optional[float] = None
    stamp: Optional[str] = None

    def __post_init__(self) -> None:
        self.T = _check_rigid(np.asarray(self.T, dtype=float), 'extrinsics')
        if self.mount not in ('eye_to_hand', 'eye_in_hand'):
            raise ValueError("mount must be 'eye_to_hand' or 'eye_in_hand', got %r" % self.mount)
        expected = 'base' if self.mount == 'eye_to_hand' else 'flange'
        if self.parent_frame != expected:
            raise ValueError('%s extrinsics must have parent_frame %r, got %r'
                             % (self.mount, expected, self.parent_frame))
        if not is_optical_frame(self.child_frame):
            raise ValueError('child_frame must end in _optical_frame, got %r. Give the '
                             'camera_link -> optical transform as T_link_optical instead.'
                             % self.child_frame)
        t = np.linalg.norm(self.T[:3, 3])
        if t >= MAX_TRANSLATION_M:
            raise ValueError('extrinsics translation %.2f m >= %.1f m; the matrix is '
                             'probably inverted or in mm' % (t, MAX_TRANSLATION_M))

    @property
    def name(self) -> str:
        """Return the transform name, 'T_<parent>_camera'."""
        return 'T_%s_camera' % self.parent_frame

    def to_dict(self) -> Dict[str, Any]:
        """Return a YAML-friendly dict."""
        d: Dict[str, Any] = {
            'mount': self.mount, 'parent_frame': self.parent_frame,
            'child_frame': self.child_frame, 'matrix': _matrix_list(self.T),
            'translation': [float(v) for v in self.T[:3, 3]],
            'quaternion_xyzw': rot_to_quat(self.T[:3, :3]),
            'pose6': T_to_pose6(self.T), 'source': self.source}
        if self.rms is not None:
            d['rms_m'] = float(self.rms)
        if self.stamp is not None:
            d['stamp'] = self.stamp
        return d

    def describe(self) -> str:
        """Return a one-line human description."""
        p = T_to_pose6(self.T)
        rms = 'none' if self.rms is None else '%.4f m' % self.rms
        return ('%s %s: xyz [%.4f %.4f %.4f] rpy_deg [%.2f %.2f %.2f] rms %s'
                % (self.mount, self.name, p[0], p[1], p[2],
                   math.degrees(p[3]), math.degrees(p[4]), math.degrees(p[5]), rms))


def _check_rigid(T: np.ndarray, what: str) -> np.ndarray:
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError('%s must be a finite 4x4, got shape %s' % (what, T.shape))
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('%s last row must be [0, 0, 0, 1]' % what)
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-3 or not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ValueError('%s rotation is not a proper rotation (det %.4f)'
                         % (what, np.linalg.det(R)))
    out = np.array(T, dtype=float)
    out[:3, :3] = orthonormalize(R)
    return out


def _matrix_list(T: np.ndarray) -> list:
    return [[round(float(v), 6) for v in row] for row in np.asarray(T)]


def _as_matrix(value: Any, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError('%s must be a 4x4 matrix, got shape %s' % (what, arr.shape))
    return _check_rigid(arr, what)


def _rotation_of(d: Dict[str, Any], what: str) -> np.ndarray:
    """Return the rotation described by one of quaternion / rotation / rpy / rpy_deg."""
    found = [k for k in ('quaternion', 'quaternion_xyzw', 'rotation', 'rpy', 'rpy_deg')
             if d.get(k) is not None]
    if len(found) != 1:
        raise ValueError('%s needs exactly one of quaternion / rotation / rpy / rpy_deg'
                         % what)
    key = found[0]
    if key in ('quaternion', 'quaternion_xyzw'):
        return quat_to_rot(d[key])
    if key == 'rotation':
        R = np.asarray(d[key], dtype=float)
        if R.shape != (3, 3):
            raise ValueError('%s rotation must be 3x3' % what)
        return R
    rpy = [float(v) for v in d[key]]
    if key == 'rpy_deg':
        rpy = [math.radians(v) for v in rpy]
    elif any(abs(v) > MAX_RPY_RAD for v in rpy):
        raise ValueError('%s rpy has an angle beyond +-2*pi; use rpy_deg for degrees' % what)
    return rpy_to_rot(*rpy)


def _translation_of(d: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ('translation', 'xyz', 'position'):
        if d.get(key) is not None:
            return np.asarray(d[key], dtype=float).reshape(3)
    return None


def _parse_transform(d: Dict[str, Any], what: str) -> np.ndarray:
    """
    Parse one transform from any supported representation.

    Accepted: ``matrix`` / ``T`` (4x4), ``translation`` + ``quaternion`` (xyzw) /
    ``rotation`` (3x3) / ``rpy`` / ``rpy_deg``, ``xyz`` + ``rpy``, a bare 4x4 or
    16 numbers, and the handeye_calibration_ros JSON ``position`` +
    ``orientation`` (xyzw). Several representations must agree.
    """
    if not isinstance(d, dict):
        return _as_matrix(d, what)
    matrices = []
    for key in ('matrix', 'T'):
        if d.get(key) is not None:
            matrices.append(_as_matrix(d[key], '%s.%s' % (what, key)))
    if isinstance(d.get('data'), list) and len(d['data']) == 16:
        matrices.append(_as_matrix(d['data'], '%s.data' % what))
    t = _translation_of(d)
    has_rot = any(d.get(k) is not None for k in
                  ('quaternion', 'quaternion_xyzw', 'orientation', 'rotation', 'rpy', 'rpy_deg'))
    if t is not None or has_rot:
        if t is None or not has_rot:
            raise ValueError('%s needs both a translation and a rotation' % what)
        if d.get('orientation') is not None:
            R = quat_to_rot(d['orientation'])
        else:
            R = _rotation_of(d, what)
        matrices.append(_check_rigid(make_T(R, t), what))
    if not matrices:
        raise ValueError('%s: no transform found' % what)
    for other in matrices[1:]:
        dt = np.linalg.norm(other[:3, 3] - matrices[0][:3, 3])
        dr = rotation_angle(other[:3, :3].T @ matrices[0][:3, :3])
        if dt > 1e-4 or dr > math.radians(0.05):
            raise ValueError('%s: the representations disagree by %.4f m / %.3f deg; '
                             'delete the ones you did not edit' % (what, dt, math.degrees(dr)))
    return matrices[0]


def extrinsics_from_dict(d: Dict[str, Any], default_mount: Optional[str] = None,
                         source: Optional[str] = None) -> Extrinsics:
    """
    Build extrinsics from a dict in any supported format.

    Key names select the direction: ``T_base_camera`` / ``T_flange_camera`` /
    ``T_tcp_camera`` are used directly; ``T_camera_base`` / ``T_camera_flange``
    / ``T_camera_tcp`` are inverted. A bare ``matrix`` needs ``parent_frame``
    (or ``mount``) to know the direction.
    """
    if not isinstance(d, dict):
        raise ValueError('extrinsics must be a mapping, got %r' % type(d).__name__)
    parent = None
    for key, frame in (('T_base_camera', 'base'), ('T_flange_camera', 'flange'),
                       ('T_tcp_camera', 'tcp')):
        if d.get(key) is not None:
            T, parent = _parse_transform(d[key], key), frame
            break
    else:
        for key, frame in (('T_camera_base', 'base'), ('T_camera_flange', 'flange'),
                           ('T_camera_tcp', 'tcp')):
            if d.get(key) is not None:
                T, parent = inv_T(_parse_transform(d[key], key)), frame
                break
        else:
            T = _parse_transform(d, 'extrinsics')
            parent = d.get('parent_frame')
    if parent == 'tcp':
        if d.get('tcp_offset') is None:
            raise ValueError('extrinsics relative to the TCP need tcp_offset: [x, y, z, '
                             'roll, pitch, yaw] (the driver TCP offset used during '
                             'calibration), so they can be stored relative to the flange')
        T_flange_tcp = pose6_to_T(d['tcp_offset'])
        T = T_flange_tcp @ T
        parent = 'flange'
    mount = d.get('mount', default_mount)
    if parent is None and mount == 'eye_to_hand':
        parent = 'base'
    if parent is None and mount == 'eye_in_hand':
        parent = 'flange'
    if parent is None:
        raise ValueError('extrinsics need parent_frame (base / flange) or mount')
    if mount is None:
        mount = 'eye_to_hand' if parent == 'base' else 'eye_in_hand'

    child = d.get('child_frame', 'camera_color_optical_frame')
    if not is_optical_frame(child):
        if d.get('T_link_optical') is None:
            raise ValueError('child_frame %r is not an optical frame; add T_link_optical '
                             '(the camera_link -> color optical transform, e.g. from '
                             'tf2_echo)' % child)
        T = T @ _parse_transform(d['T_link_optical'], 'T_link_optical')
        child = 'camera_color_optical_frame'
    rms = d.get('rms_m', d.get('rms'))
    return Extrinsics(T=T, mount=mount, parent_frame=parent, child_frame=child,
                      source=source if source is not None else str(d.get('source', '')),
                      rms=None if rms is None else float(rms), stamp=d.get('stamp'))


def camera_pose_in_base(ext: Extrinsics, T_base_flange: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return T_base_camera.

    Eye-in-hand needs the flange pose recorded at the moment of capture.
    """
    if ext.mount == 'eye_to_hand':
        return np.array(ext.T)
    if T_base_flange is None:
        raise ValueError('eye_in_hand needs the flange pose at capture time')
    return _check_rigid(np.asarray(T_base_flange, dtype=float), 'T_base_flange') @ ext.T


def points_base_to_camera(T_base_camera: np.ndarray, pts_base: np.ndarray) -> np.ndarray:
    """Transform Nx3 (or 3,) base-frame points into the camera frame."""
    return _apply(inv_T(T_base_camera), pts_base)


def points_camera_to_base(T_base_camera: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Transform Nx3 (or 3,) camera-frame points into the base frame."""
    return _apply(np.asarray(T_base_camera, dtype=float), pts_cam)


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    out = p.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    return out.reshape(p.shape)


# ---------------------------------------------------------------------------
# The camera config file
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Everything camera-related, loaded from one YAML file."""

    mount: str = 'eye_to_hand'
    topics: Dict[str, str] = field(default_factory=dict)
    depth_scale: float = 0.001
    depth_registered: bool = True
    intrinsics: Optional[Intrinsics] = None
    extrinsics: Optional[Extrinsics] = None
    board: Dict[str, Any] = field(default_factory=dict)
    tcp_offset: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    source: str = ''

    def T_base_camera(self, T_base_flange: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the camera pose in the base frame (needs the flange pose when wrist-mounted)."""
        if self.extrinsics is None:
            raise ValueError('no extrinsics in %s' % (self.source or 'camera config'))
        return camera_pose_in_base(self.extrinsics, T_base_flange)

    def T_flange_tcp(self) -> np.ndarray:
        """Return the configured TCP offset as a 4x4 transform."""
        return pose6_to_T(self.tcp_offset)


_DEFAULT_TOPICS = {
    'color': '/camera/color/image_raw',
    'depth': '/camera/depth/image_raw',
    'info': '/camera/color/camera_info',
}


def load_camera_config(path: str) -> CameraConfig:
    """Load a camera config YAML. Raises ValueError with the file name on bad content."""
    import yaml
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding='utf-8'))
    except OSError as exc:
        raise ValueError('cannot read camera config %s: %s' % (path, exc))
    except yaml.YAMLError as exc:
        raise ValueError('camera config %s is not valid YAML: %s' % (path, exc))
    if not isinstance(data, dict):
        raise ValueError('camera config %s must be a mapping' % path)
    mount = str(data.get('mount', 'eye_to_hand'))
    if mount not in ('eye_to_hand', 'eye_in_hand'):
        raise ValueError('%s: mount must be eye_to_hand or eye_in_hand' % path)
    topics = dict(_DEFAULT_TOPICS)
    topics.update(data.get('topics') or {})
    intr = None
    if data.get('intrinsics'):
        intr = Intrinsics.from_dict(data['intrinsics'])
    ext = None
    if data.get('extrinsics'):
        ext = extrinsics_from_dict(data['extrinsics'], default_mount=mount, source=str(p))
        if ext.mount != mount:
            raise ValueError('%s: mount is %s but extrinsics say %s'
                             % (path, mount, ext.mount))
    tcp = data.get('tcp_offset', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if len(tcp) != 6:
        raise ValueError('%s: tcp_offset needs 6 values' % path)
    return CameraConfig(mount=mount, topics=topics,
                        depth_scale=float(data.get('depth_scale', 0.001)),
                        depth_registered=bool(data.get('depth_registered', True)),
                        intrinsics=intr, extrinsics=ext,
                        board=dict(data.get('board') or {}),
                        tcp_offset=tuple(float(v) for v in tcp), source=str(p))


def save_camera_config(path: str, cfg: CameraConfig, header: str = '') -> Path:
    """Write a camera config atomically. Returns the path written."""
    import yaml
    data: Dict[str, Any] = {
        'mount': cfg.mount, 'topics': dict(cfg.topics),
        'depth_scale': cfg.depth_scale, 'depth_registered': cfg.depth_registered,
        'tcp_offset': [float(v) for v in cfg.tcp_offset]}
    if cfg.intrinsics is not None:
        data['intrinsics'] = cfg.intrinsics.to_dict()
    if cfg.extrinsics is not None:
        data['extrinsics'] = cfg.extrinsics.to_dict()
    if cfg.board:
        data['board'] = dict(cfg.board)
    text = (header.rstrip() + '\n\n' if header else '') + yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True)
    return _atomic_write(path, text)


def write_extrinsics(path: str, ext: Extrinsics, mount: Optional[str] = None) -> Path:
    """
    Write the extrinsics into an existing camera config, keeping the rest.

    A missing file is created with defaults. The written mount follows the
    extrinsics unless ``mount`` is given.
    """
    p = Path(path)
    cfg = load_camera_config(str(p)) if p.is_file() else CameraConfig()
    cfg.extrinsics = ext
    cfg.mount = mount or ext.mount
    return save_camera_config(str(p), cfg)


def _atomic_write(path: str, text: str) -> Path:
    import os
    import tempfile
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p

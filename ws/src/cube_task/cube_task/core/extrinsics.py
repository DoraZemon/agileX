"""
Camera extrinsics (hand-eye calibration): load, validate, save and compose.

``Extrinsics.T`` is ``T_<parent_frame>_<child_frame>``: it maps points in the
camera color optical frame (x right, y down, z forward) into parent_frame.

The competition has only the wrist camera, so the mount is always
'eye_in_hand'; a fixed-camera calibration (T_base_camera, mount eye_to_hand,
parent_frame base) is refused with a message saying so.

- mount 'eye_in_hand' (camera on the wrist): parent_frame 'flange' (T =
  T_flange_camera, flange = link6 as reported by the SDK) or 'tcp' (T =
  T_tcp_camera, TCP = flange @ pose6_to_T(robot.tcp_offset)). Files and
  config never give 'tcp': a T_tcp_camera must carry the driver tcp_offset
  used during the calibration and is converted to T_flange_camera on load.

child_frame must be an optical frame (name ending in _optical_frame). A
calibration of a ROS body frame (camera_link, REP-103 x forward, z up) needs
T_link_optical to be composed. Translations are meters, quaternions
(x, y, z, w), Euler angles follow core/transforms.py (R = Rz(yaw) Ry(pitch)
Rx(roll)). See load_extrinsics() for the accepted file formats.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import yaml

from .logger import atomic_write_text
from .transforms import (inv_T, make_T, pose6_to_T, quat_to_rot, rot_to_quat, rot_to_rpy,
                         rotation_angle, rpy_to_rot)

_LOG = logging.getLogger('cube_task.extrinsics')

MOUNTS = ('eye_in_hand',)
#: Allowed parent frames per mount.
PARENTS = {'eye_in_hand': ('flange', 'tcp')}
WRIST_ONLY = ('the competition camera is the wrist camera (eye_in_hand): give its hand-eye '
              'calibration T_flange_camera (scripts/handeye.sh)')
DEFAULT_CHILD_FRAME = 'camera_color_optical_frame'
#: Any extrinsic translation at least this long (m) is rejected (unit error).
MAX_TRANSLATION_M = 3.0
#: Max |R^T R - I| and |det R - 1| accepted for a rotation matrix.
ROTATION_TOL = 1e-3
#: Largest |Euler angle| (rad) accepted under a radian key; larger values are degrees.
MAX_RPY_RAD = 2.0 * math.pi + 1e-6
#: Longest translation (m) accepted in T_link_optical and in a file's tcp_offset.
MAX_LINK_OPTICAL_M = 0.2
MAX_TCP_OFFSET_M = 0.5
OPTICAL_SUFFIX = '_optical_frame'
# Agreement required between two representations of the same transform.
_AGREE_POS_M = 1e-4
_AGREE_ROT_RAD = 1e-3

_PARENT_ALIASES = {'base': 'base', 'base_link': 'base', 'flange': 'flange',
                   'link6': 'flange', 'tcp': 'tcp', 'tcp_link': 'tcp'}
_NAMED_KEYS = {  # key -> (parent frame, stored as T_camera_parent?)
    'T_base_camera': ('base', False), 'T_flange_camera': ('flange', False),
    'T_tcp_camera': ('tcp', False), 'T_camera_base': ('base', True),
    'T_camera_flange': ('flange', True), 'T_camera_tcp': ('tcp', True),
}


def normalize_mount(mount: Any) -> str:
    """Return 'eye_in_hand' ('-' and ' ' accepted for '_'); ValueError for anything else."""
    m = str(mount).strip().lower().replace('-', '_').replace(' ', '_')
    if m == 'eye_to_hand':
        raise ValueError('camera mount eye_to_hand (fixed camera) is not supported: '
                         + WRIST_ONLY)
    if m not in MOUNTS:
        raise ValueError('unknown camera mount %r: use eye_in_hand' % (mount,))
    return m


def normalize_parent_frame(parent: Any) -> str:
    """Return 'base', 'flange' or 'tcp' (aliases base_link, link6, tcp_link)."""
    p = str(parent).strip().lower()
    if p not in _PARENT_ALIASES:
        raise ValueError('unknown parent_frame %r: use base, flange or tcp' % (parent,))
    return _PARENT_ALIASES[p]


def is_optical_frame(name: Any) -> bool:
    """Return True when name is a camera optical frame (ends in _optical_frame, any case)."""
    return str(name or '').strip().lower().endswith(OPTICAL_SUFFIX)


def non_optical_frame_message(child: Any) -> str:
    """Return why a non-optical child_frame is refused and how to fix it."""
    name = str(child).strip() or 'camera_link'
    return ('extrinsics child_frame %r is not a camera optical frame. cube_task needs the pose '
            'of the color OPTICAL frame (x right, y down, z forward), the frame of the image '
            'and depth. ROS body frames such as camera_link or camera_color_frame follow '
            'REP-103 (x forward, y left, z up) and differ from it by a fixed rotation (and the '
            'color sensor offset), so using them directly rotates every point cloud. If the '
            'transform really is for the optical frame, set child_frame: '
            'camera_color_optical_frame. Otherwise, with the camera driver running, run '
            '"ros2 run tf2_ros tf2_echo %s camera_color_optical_frame" and add its 4x4 '
            'matrix to the file as T_link_optical (T_parent_optical = T_parent_link @ '
            'T_link_optical)' % (name, name))


def validate_transform(T: Any, what: str = 'transform',
                       max_translation: float = MAX_TRANSLATION_M) -> np.ndarray:
    """
    Check a rigid transform and return it as a clean 4x4 float64 copy.

    Accepts 4x4, 3x4 (bottom row added) or 16 / 12 flat numbers in row-major
    order. Requires finite values, bottom row [0 0 0 1], a proper rotation
    (|R^T R - I| and |det R - 1| <= ROTATION_TOL) and |t| < max_translation
    meters. Tiny orthonormality errors are removed with an SVD projection (a
    warning is logged when the error exceeds 1e-6). Raises ValueError with a
    message naming ``what`` otherwise.
    """
    try:
        a = np.array(T, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('%s: not a numeric matrix (%s)' % (what, e))
    if a.shape in ((16,), (12,)):
        a = a.reshape(-1, 4)
    if a.shape == (3, 4):
        a = np.vstack([a, [0.0, 0.0, 0.0, 1.0]])
    if a.shape != (4, 4):
        raise ValueError('%s: expected a 4x4 matrix, got shape %s' % (what, a.shape))
    if not np.all(np.isfinite(a)):
        raise ValueError('%s: contains NaN or inf' % what)
    if not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError('%s: bottom row must be [0, 0, 0, 1], got %s'
                         % (what, np.round(a[3], 6).tolist()))
    R = a[:3, :3]
    det = float(np.linalg.det(R))
    if det < 0:
        raise ValueError('%s: rotation part is a reflection (det = %.4f). Check for a '
                         'mirrored axis or a left-handed frame.' % (what, det))
    err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if err > ROTATION_TOL or abs(det - 1.0) > ROTATION_TOL:
        raise ValueError('%s: rotation part is not orthonormal (max |R^T R - I| = %.2e, '
                         'det = %.4f, tolerance %.0e). Expected a row-major 4x4 rigid '
                         'transform.' % (what, err, det, ROTATION_TOL))
    if err > 1e-12:
        U, _s, Vt = np.linalg.svd(R)
        a[:3, :3] = U @ Vt
        level = logging.WARNING if err > 1e-6 else logging.DEBUG
        _LOG.log(level, '%s: re-orthonormalized rotation (error %.2e)', what, err)
    tn = float(np.linalg.norm(a[:3, 3]))
    if tn >= max_translation:
        raise ValueError('%s: translation length %.3f m >= %.2f m. Units must be meters '
                         '(millimeters?).' % (what, tn, max_translation))
    return a


@dataclass(eq=False)
class Extrinsics:
    """
    Calibrated camera pose: ``T`` = T_<parent_frame>_<child_frame> (meters).

    Validated on construction (see validate_transform); invalid values raise
    ValueError. parent_frame None means 'flange'. Instances compare by identity
    (compare ``T`` with numpy).

    Attributes
    ----------
    T
        4x4 float64.
    mount
        'eye_in_hand'.
    parent_frame
        'flange' or 'tcp'.
    child_frame
        Camera frame name (color optical frame).
    source
        Where the calibration came from (tool or file).
    rms
        Calibration residual in meters, None if unknown.
    stamp
        Calibration time as free text (ISO date), None if unknown.

    """

    T: np.ndarray
    mount: str
    parent_frame: Optional[str] = None
    child_frame: str = DEFAULT_CHILD_FRAME
    source: str = ''
    rms: Optional[float] = None
    stamp: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize names and validate the transform."""
        self.mount = normalize_mount(self.mount)
        if self.parent_frame is None or str(self.parent_frame).strip() == '':
            self.parent_frame = PARENTS[self.mount][0]
        else:
            self.parent_frame = normalize_parent_frame(self.parent_frame)
        if self.parent_frame == 'base':
            raise ValueError('parent_frame base (T_base_camera, a fixed camera) is not '
                             'supported: ' + WRIST_ONLY)
        if self.parent_frame not in PARENTS[self.mount]:
            raise ValueError('mount %s needs parent_frame %s, got %r'
                             % (self.mount, ' or '.join(PARENTS[self.mount]),
                                self.parent_frame))
        self.source = '' if self.source is None else str(self.source)
        self.child_frame = str(self.child_frame or '').strip() or DEFAULT_CHILD_FRAME
        if not is_optical_frame(self.child_frame):
            raise ValueError(non_optical_frame_message(self.child_frame))
        self.T = validate_transform(self.T, 'extrinsics T_%s_camera%s'
                                    % (self.parent_frame,
                                       ' (%s)' % self.source if self.source else ''))
        if self.rms is not None:
            rms = float(self.rms)
            if not math.isfinite(rms) or rms < 0:
                raise ValueError('extrinsics rms must be a finite value >= 0 (m), got %r'
                                 % (self.rms,))
            self.rms = rms
        if self.stamp is not None:
            self.stamp = str(self.stamp)

    @property
    def name(self) -> str:
        """Return the transform name, e.g. 'T_flange_camera'."""
        return 'T_%s_camera' % self.parent_frame

    def to_dict(self) -> Dict[str, Any]:
        """Return our file format as a plain dict (see save_extrinsics)."""
        t, q = self.T[:3, 3], rot_to_quat(self.T[:3, :3])
        return {'mount': self.mount, 'parent_frame': self.parent_frame,
                'child_frame': self.child_frame,
                'matrix': [[float(v) for v in row] for row in self.T],
                'translation': [float(v) for v in t],
                'quaternion': [float(v) for v in q],
                'source': self.source, 'rms': self.rms, 'stamp': self.stamp}

    def describe(self) -> str:
        """Return a one-line human-readable summary (mm and degrees)."""
        t = self.T[:3, 3] * 1000.0
        rpy = [math.degrees(v) for v in rot_to_rpy(self.T[:3, :3])]
        s = '%s %s: t = [%.1f, %.1f, %.1f] mm, rpy = [%.2f, %.2f, %.2f] deg' % (
            self.mount, self.name, t[0], t[1], t[2], rpy[0], rpy[1], rpy[2])
        if self.rms is not None:
            s += ', rms = %.1f mm' % (self.rms * 1000.0)
        if self.source:
            s += ', source = %s' % self.source
        return s


# ---------------------------------------------------------------- parsing
_FORMAT_HANDEYE = 'handeye_calibration_ros'
_NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_PLAIN_RE = re.compile(r'[\s\d.eE+\-,;\[\]()]+')


def _numbers(value: Any, what: str) -> np.ndarray:
    try:
        a = np.array(value, dtype=float).ravel()
    except (TypeError, ValueError) as e:
        raise ValueError('%s: expected numbers, got %r (%s)' % (what, value, e))
    if not np.all(np.isfinite(a)):
        raise ValueError('%s: contains NaN or inf' % what)
    return a


def _vector(value: Any, n: int, what: str) -> np.ndarray:
    """Parse n numbers from a list (nested [[...]] allowed) or an {x, y, z[, w]} dict."""
    if isinstance(value, Mapping):
        keys = ('x', 'y', 'z', 'w')[:n]
        missing = [k for k in keys if k not in value]
        if missing:
            raise ValueError('%s: missing keys %s' % (what, missing))
        value = [value[k] for k in keys]
    a = _numbers(value, what)
    if a.size != n:
        raise ValueError('%s: expected %d numbers, got %d' % (what, n, a.size))
    return a


def _quat_rot(value: Any, what: str) -> np.ndarray:
    q = _vector(value, 4, what + ' (quaternion x, y, z, w)')
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 0.01:
        raise ValueError('%s: quaternion norm is %.4f, expected 1 (order x, y, z, w)'
                         % (what, norm))
    return quat_to_rot(q)


def _rotation_any(value: Any, what: str) -> np.ndarray:
    """'rotation': 3x3 matrix, quaternion (x, y, z, w) list or {x, y, z, w} dict."""
    if isinstance(value, Mapping):
        return _quat_rot(value, what)
    a = _numbers(value, what)
    if a.size == 4:
        return _quat_rot(a, what)
    if a.size == 9:
        return validate_transform(make_T(a.reshape(3, 3), (0.0, 0.0, 0.0)), what)[:3, :3]
    raise ValueError('%s: expected a 3x3 matrix or a quaternion (x, y, z, w), got %d '
                     'numbers (use rpy for roll / pitch / yaw in rad)' % (what, a.size))


def _check_radians(angles: np.ndarray, what: str, hint: str) -> None:
    """Raise ValueError when an angle given in rad is larger than 2*pi (degrees pasted)."""
    if np.any(np.abs(angles) > MAX_RPY_RAD):
        raise ValueError('%s %s: |angle| > 2*pi, these look like degrees (%s)'
                         % (what, [round(float(v), 4) for v in angles], hint))


def _parse_transform(d: Mapping, what: str) -> Tuple[np.ndarray, Optional[str], str]:
    """Return (T, parent frame implied by a T_<a>_<b> key or None, format label)."""
    trans: List[Tuple[str, np.ndarray]] = []
    rots: List[Tuple[str, np.ndarray]] = []
    named = [k for k in _NAMED_KEYS if k in d]
    if len(named) > 1:
        raise ValueError('%s: several transform keys %s; keep one' % (what, named))
    implied = None
    for key in named + [k for k in ('matrix', 'T') if k in d]:
        M = validate_transform(d[key], '%s %r' % (what, key))
        if key in _NAMED_KEYS:
            implied, inverted = _NAMED_KEYS[key]
            if inverted:
                M = inv_T(M)
        trans.append((key, M[:3, 3]))
        rots.append((key, M[:3, :3]))
    for key in ('translation', 'xyz', 'position'):
        if key in d:
            trans.append((key, _vector(d[key], 3, '%s %r' % (what, key))))
    for key in ('quaternion', 'orientation'):
        if key in d:
            rots.append((key, _quat_rot(d[key], '%s %r' % (what, key))))
    if 'rotation' in d:
        rots.append(('rotation', _rotation_any(d['rotation'], '%s %r' % (what, 'rotation'))))
    if 'rpy' in d:
        rpy = _vector(d['rpy'], 3, '%s rpy (rad)' % what)
        _check_radians(rpy, '%s rpy' % what, 'use rpy_deg for degrees')
        rots.append(('rpy', rpy_to_rot(*rpy)))
    if 'rpy_deg' in d:
        r = np.radians(_vector(d['rpy_deg'], 3, '%s rpy_deg' % what))
        rots.append(('rpy_deg', rpy_to_rot(*r)))
    if not trans and not rots:
        raise ValueError(
            '%s: no transform found. Expected one of: matrix (4x4), T (4x4), '
            'T_base_camera / T_flange_camera / T_tcp_camera (4x4), translation + '
            'quaternion (x, y, z, w) | rotation (3x3) | rpy, xyz + rpy, or '
            'handeye_calibration_ros position + orientation' % what)
    if not trans:
        raise ValueError('%s: rotation (%s) given without translation'
                         % (what, rots[0][0]))
    if not rots:
        raise ValueError('%s: translation (%s) given without rotation (quaternion, '
                         'rotation or rpy)' % (what, trans[0][0]))
    k0, t0 = trans[0]
    for k, t in trans[1:]:
        dp = float(np.linalg.norm(t - t0))
        if dp > _AGREE_POS_M:
            raise ValueError('%s: %r and %r disagree by %.2f mm. The file was edited '
                             'inconsistently; keep only one representation.'
                             % (what, k0, k, dp * 1000.0))
    k0, R0 = rots[0]
    for k, R in rots[1:]:
        da = rotation_angle(R0.T @ R)
        if da > _AGREE_ROT_RAD:
            raise ValueError('%s: %r and %r disagree by %.3f deg. The file was edited '
                             'inconsistently; keep only one representation.'
                             % (what, k0, k, math.degrees(da)))
    keys = set(d.keys())
    if {'position', 'orientation'} <= keys and not keys & {'matrix', 'T', 'mount'}:
        fmt = _FORMAT_HANDEYE
    else:
        fmt = '+'.join(sorted({trans[0][0], rots[0][0]}))
    return make_T(R0, t0), implied, fmt


def _tcp_to_flange(d: Mapping, T: np.ndarray, what: str) -> Tuple[np.ndarray, str]:
    """Return (T_flange_camera, source note) for a T_tcp_camera that carries tcp_offset."""
    raw = d.get('tcp_offset')
    if raw in (None, '', []):
        raise ValueError(
            '%s: parent_frame tcp (T_tcp_camera) needs "tcp_offset: [x, y, z, roll, pitch, '
            'yaw]" (m, rad) in the file: the TCP offset of the driver whose TCP pose was used '
            'during the calibration (agx_arm_ros launch parameter tcp_offset). It is converted '
            'to T_flange_camera, so robot.tcp_offset never moves the camera estimate. If the '
            'driver ran with tcp_offset all zero, use parent_frame: flange.' % what)
    off = _vector(raw, 6, '%s tcp_offset [x, y, z, roll, pitch, yaw]' % what)
    _check_radians(off[3:], '%s tcp_offset angles' % what, 'tcp_offset angles are rad')
    T_ft = validate_transform(pose6_to_T(off), '%s tcp_offset' % what,
                              max_translation=MAX_TCP_OFFSET_M)
    note = 'T_tcp_camera -> T_flange_camera with tcp_offset %s' % [
        round(float(v), 6) for v in off]
    return T_ft @ T, note


def _link_to_optical(d: Mapping, T: np.ndarray, child: str,
                     what: str) -> Tuple[np.ndarray, str, str]:
    """Return (T_parent_optical, child frame, source note) composed with T_link_optical."""
    if is_optical_frame(child):
        raise ValueError('%s: T_link_optical is given but child_frame %r is already an optical '
                         'frame. Set child_frame to the body frame the transform is for (for '
                         'example camera_link) or remove T_link_optical.' % (what, child))
    raw = d['T_link_optical']
    w = '%s T_link_optical' % what
    T_lo = _parse_transform(raw, w)[0] if isinstance(raw, Mapping) else raw
    T_lo = validate_transform(T_lo, w, max_translation=MAX_LINK_OPTICAL_M)
    note = '%s composed with T_link_optical' % child
    return T @ T_lo, DEFAULT_CHILD_FRAME, note


def extrinsics_from_dict(d: Mapping, default_mount: Optional[str] = None,
                         source: Optional[str] = None,
                         what: str = 'extrinsics') -> Extrinsics:
    """
    Build Extrinsics from a parsed mapping in any supported format.

    The mount is always eye_in_hand: a 'mount' (or 'mode') key, default_mount or
    parent_frame base (T_base_camera) saying otherwise raises ValueError. A transform
    without parent_frame is taken as T_flange_camera (with a warning for a
    handeye_calibration_ros result). parent_frame tcp needs tcp_offset (the
    driver's TCP offset during the calibration) and is converted to
    T_flange_camera. A non-optical child_frame (camera_link, ...) needs
    T_link_optical (4x4), which is composed; ValueError otherwise.
    """
    if not isinstance(d, Mapping):
        raise ValueError('%s: expected a mapping, got %s' % (what, type(d).__name__))
    T, implied, fmt = _parse_transform(d, what)
    parent = d.get('parent_frame')
    parent = None if parent in (None, '') else normalize_parent_frame(parent)
    if implied is not None and parent is not None and implied != parent:
        raise ValueError('%s: key T_..._camera implies parent_frame %r but parent_frame '
                         'is %r' % (what, implied, parent))
    parent = parent or implied
    if parent == 'base':
        raise ValueError('%s: T_base_camera / parent_frame base is a fixed-camera calibration: %s'
                         % (what, WRIST_ONLY))
    mount_raw = d.get('mount', d.get('mode'))
    if mount_raw not in (None, ''):
        mount = normalize_mount(mount_raw)
    else:
        if default_mount not in (None, ''):
            normalize_mount(default_mount)
        mount = 'eye_in_hand'
    if parent is None:
        parent = PARENTS[mount][0]
        if mount == 'eye_in_hand':
            if fmt == _FORMAT_HANDEYE:
                _LOG.warning(
                    '%s: handeye_calibration_ros eye_in_hand result is T_<pose topic '
                    'frame>_camera. /feedback/tcp_pose equals the flange only when '
                    'agx_arm_ros ran with tcp_offset all zero. Assuming parent_frame '
                    '"flange". If the driver ran with a non-zero tcp_offset, add '
                    '"parent_frame": "tcp" and "tcp_offset": [x, y, z, roll, pitch, yaw] (the '
                    'driver value, m / rad) to the file.', what)
            else:
                # Documented convention (docs/CONTRACTS.md): eye_in_hand holds T_flange_camera.
                _LOG.info('%s: eye_in_hand without parent_frame; using T_flange_camera '
                          '(for a T_tcp_camera set parent_frame: tcp and tcp_offset)', what)
    notes = []
    if parent == 'tcp' and mount == 'eye_in_hand':
        T, note = _tcp_to_flange(d, T, what)
        parent = 'flange'
        notes.append(note)
    elif d.get('tcp_offset') not in (None, '', []):
        _LOG.warning('%s: tcp_offset is only used with parent_frame tcp; ignored (parent_frame '
                     '%s)', what, parent)
    child = str(d.get('child_frame') or '').strip() or DEFAULT_CHILD_FRAME
    if d.get('T_link_optical') is not None:
        T, child, note = _link_to_optical(d, T, child, what)
        notes.append(note)
    elif not is_optical_frame(child):
        raise ValueError('%s: %s' % (what, non_optical_frame_message(child)))
    src = d.get('source') or source or ''
    if not d.get('source') and fmt:
        src = '%s [%s]' % (src, fmt) if src else fmt
    if notes:
        src = '%s [%s]' % (src, '; '.join(notes)) if src else '; '.join(notes)
    return Extrinsics(T=T, mount=mount, parent_frame=parent, child_frame=child,
                      source=str(src), rms=d.get('rms'), stamp=d.get('stamp'))


def _plain_numbers(text: str) -> Optional[List[float]]:
    if not _PLAIN_RE.fullmatch(text.strip()):
        return None
    return [float(v) for v in _NUM_RE.findall(text)]


def _json_file_as_yaml(text: str, where: str, err: Exception) -> Any:
    """Read a .json file that is not JSON as YAML (older save_extrinsics wrote YAML there)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if not isinstance(data, (Mapping, list)):
        raise ValueError('%s: invalid JSON (%s)' % (where, err))
    _LOG.warning('%s: the file is YAML, not JSON (calib_verify --apply / calib_points w wrote '
                 'YAML into .json files before); read as YAML', where)
    return data


def _parse_text(text: str, suffix: str, where: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError('%s: file is empty' % where)
    if suffix == '.json' or (suffix not in ('.yaml', '.yml') and stripped[0] in '{['):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if suffix == '.json':
                return _json_file_as_yaml(text, where, e)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        nums = _plain_numbers(text)
        if nums is not None:
            return nums
        raise ValueError('%s: invalid YAML (%s)' % (where, e))


def load_extrinsics(path: Union[str, Path], default_mount: Optional[str] = None) -> Extrinsics:
    """
    Load and validate extrinsics from a file.

    Formats (YAML or JSON, chosen by extension, else by content; a .json
    file that is not JSON is read as YAML with a warning):

    - ours (save_extrinsics): mount, parent_frame, child_frame, matrix 4x4,
      translation, quaternion (x, y, z, w), source, rms, stamp. All
      representations present must agree.
    - {matrix: 4x4} or {T: 4x4}; {T_base_camera | T_flange_camera |
      T_tcp_camera: 4x4} (sets parent_frame; T_base_camera is refused: a fixed
      camera); T_camera_* keys are inverted.
    - {translation, quaternion | rotation (3x3) | rpy}, {xyz, rpy} (rad,
      SDK ZYX, |angle| <= 2*pi), rpy_deg (degrees). Vectors may be lists or
      {x, y, z[, w]}.
    - handeye_calibration_ros JSON {position, orientation (x, y, z, w), rpy}:
      the eye_in_hand result T_<arm pose topic frame>_camera (flange assumed
      with a warning unless parent_frame is set).
    - a bare 4x4 / 16-number list, a plain text matrix, or a .npy array.

    parent_frame tcp also needs tcp_offset [x, y, z, roll, pitch, yaw] (the
    driver TCP offset during the calibration); the result is T_flange_camera.
    child_frame must end in _optical_frame unless T_link_optical (4x4, from
    tf2_echo <child_frame> camera_color_optical_frame) is given to compose.
    The mount must be eye_in_hand (file, default_mount). Raises
    FileNotFoundError, or ValueError with the reason.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError('extrinsics file not found: %s' % p)
    suffix = p.suffix.lower()
    if suffix == '.npy':
        data: Any = np.load(str(p), allow_pickle=False).tolist()
    else:
        data = _parse_text(p.read_text(encoding='utf-8'), suffix, str(p))
    if isinstance(data, str):
        nums = _plain_numbers(data)
        if nums is None:
            raise ValueError('%s: not a mapping or a 4x4 matrix' % p)
        data = nums
    if isinstance(data, (list, tuple)):
        data = {'matrix': data}
    if not isinstance(data, Mapping):
        raise ValueError('%s: expected a mapping or a 4x4 matrix, got %s'
                         % (p, type(data).__name__))
    return extrinsics_from_dict(data, default_mount=default_mount, source=str(p),
                                what=str(p))


def save_extrinsics(path: Union[str, Path], ext: Extrinsics) -> Path:
    """
    Write ext in our format atomically (temp file + rename); return the path.

    A .json path gets JSON (the header lines in "_comment"), any other path
    YAML with a commented header; load_extrinsics reads both back. Raises
    ValueError for a .npy path (load-only format) and for parent_frame tcp
    (the file could not record the TCP offset: save T_flange_camera).
    """
    if not isinstance(ext, Extrinsics):
        raise TypeError('save_extrinsics needs an Extrinsics, got %s' % type(ext).__name__)
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == '.npy':
        raise ValueError('%s: extrinsics cannot be written as .npy (a load-only format). Write '
                         'a .yaml file instead, e.g. %s, and set camera.extrinsics_file to it.'
                         % (p, p.with_suffix('.yaml')))
    if ext.parent_frame == 'tcp':
        raise ValueError('%s: a T_tcp_camera cannot be saved without the TCP offset it was '
                         'calibrated with; save T_flange_camera = T_flange_tcp @ T_tcp_camera '
                         '(parent_frame flange) instead' % p)
    rpy = [math.degrees(v) for v in rot_to_rpy(ext.T[:3, :3])]
    t = ext.T[:3, 3]
    header = [
        'cube_task camera extrinsics, written %s'
        % datetime.datetime.now().isoformat(timespec='seconds'),
        'matrix = %s: maps points in the camera color optical frame (x right,'
        % ext.name,
        '  y down, z forward) into the %s frame. Units: meters.' % ext.parent_frame,
        'mount eye_in_hand (wrist camera) -> T_flange_camera.',
        'matrix, translation and quaternion (x, y, z, w) must agree. When editing by',
        '  hand, change all of them or delete translation and quaternion.',
        'For reading only: t = [%.1f, %.1f, %.1f] mm, rpy (R = Rz Ry Rx) = '
        '[%.3f, %.3f, %.3f] deg' % (t[0] * 1000, t[1] * 1000, t[2] * 1000,
                                    rpy[0], rpy[1], rpy[2]),
    ]
    if suffix == '.json':
        data: Dict[str, Any] = {'_comment': header}
        data.update(ext.to_dict())
        atomic_write_text(p, json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        return p
    body = yaml.safe_dump(ext.to_dict(), allow_unicode=True, sort_keys=False,
                          default_flow_style=None)
    atomic_write_text(p, ''.join('# %s\n' % h for h in header) + body)
    return p


def extrinsics_from_cfg(cfg: Any) -> Optional[Extrinsics]:
    """
    Return the configured extrinsics, or None when none are configured.

    camera.extrinsics_inline ({mount, matrix, parent_frame?, ...}) wins over
    camera.extrinsics_file (resolved with cfg.path). A missing file returns
    None (with a warning). Raises ValueError for invalid data and when the
    extrinsics mount differs from camera.mount.
    """
    mount_raw = cfg.get('camera.mount', None)
    cfg_mount = None if mount_raw in (None, '') else normalize_mount(mount_raw)
    inline = cfg.get('camera.extrinsics_inline', None)
    if inline not in (None, '', {}):
        if not isinstance(inline, Mapping):
            raise ValueError('camera.extrinsics_inline must be a mapping {mount, matrix}, '
                             'got %s' % type(inline).__name__)
        ext = extrinsics_from_dict(inline, default_mount=cfg_mount,
                                   source='config camera.extrinsics_inline',
                                   what='camera.extrinsics_inline')
    else:
        p = cfg.path('camera.extrinsics_file')
        if p is None:
            _LOG.warning('no camera extrinsics configured (camera.extrinsics_file is empty)')
            return None
        if not p.exists():
            _LOG.warning('camera extrinsics file %s does not exist', p)
            return None
        ext = load_extrinsics(p, default_mount=cfg_mount)
    if cfg_mount is not None and ext.mount != cfg_mount:
        raise ValueError('extrinsics (%s) are for mount %s but camera.mount is %s'
                         % (ext.source, ext.mount, cfg_mount))
    return ext


def camera_pose_in_base(ext: Extrinsics, T_base_flange: Optional[np.ndarray] = None,
                        T_flange_tcp: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return T_base_camera (4x4) for a frame captured now.

    Needs T_base_flange at capture time (arm stationary); parent_frame 'tcp' (only
    from Extrinsics built in code: loaded files are converted to 'flange')
    also needs T_flange_tcp = pose6_to_T(robot.tcp_offset). Raises ValueError
    when a required pose is missing or invalid.
    """
    if T_base_flange is None:
        raise ValueError('eye_in_hand extrinsics need T_base_flange at capture time')
    T_bf = validate_transform(T_base_flange, 'T_base_flange')
    if ext.parent_frame == 'flange':
        return T_bf @ ext.T
    if T_flange_tcp is None:
        raise ValueError('extrinsics are T_tcp_camera: T_flange_tcp '
                         '(pose6_to_T(robot.tcp_offset)) is required')
    return T_bf @ validate_transform(T_flange_tcp, 'T_flange_tcp') @ ext.T

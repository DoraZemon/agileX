"""
Shared data types. Every module exchanges data through these classes.

Units: meters, radians, seconds. Poses are 4x4 homogeneous numpy matrices
(float64). ``T_a_b`` maps coordinates expressed in frame b into frame a.
Images: color is HxWx3 uint8 BGR; depth is HxW float32 meters aligned to the
color image, with 0 / NaN / inf meaning invalid.
Time stamps are host monotonic seconds from the injected Clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # OpenCV order (k1, k2, p1, p2, k3). All zeros means rectified image.
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    def scaled(self, width: int, height: int) -> 'CameraIntrinsics':
        """Intrinsics for the same camera resized to (width, height)."""
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx,
                                self.cy * sy, int(width), int(height),
                                np.array(self.dist, dtype=float))

    def to_dict(self) -> Dict[str, Any]:
        return {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy,
                'width': self.width, 'height': self.height,
                'dist': [float(v) for v in np.asarray(self.dist).ravel()]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'CameraIntrinsics':
        return CameraIntrinsics(float(d['fx']), float(d['fy']), float(d['cx']),
                                float(d['cy']), int(d['width']), int(d['height']),
                                np.array(d.get('dist', [0.0] * 5), dtype=float))


@dataclass
class Frame:
    color: np.ndarray                      # HxWx3 uint8 BGR
    depth: Optional[np.ndarray]            # HxW float32 meters, aligned to color
    intrinsics: CameraIntrinsics           # intrinsics of the color image
    stamp: float                           # host monotonic time of capture/receipt
    frame_id: str = 'camera_color_optical_frame'
    source: str = 'unknown'                # 'sim' | 'ros' | 'file' | ...
    meta: Dict[str, Any] = field(default_factory=dict)

    def has_depth(self) -> bool:
        return self.depth is not None


@dataclass
class TablePlane:
    """Plane n.p + d = 0 with unit normal n pointing up (away from table)."""

    normal: np.ndarray                     # (3,)
    d: float
    frame: str                             # 'base' or 'camera'
    inlier_ratio: float = 0.0
    rms: float = 0.0

    def signed_height(self, pts: np.ndarray) -> np.ndarray:
        """Height of points (Nx3, same frame) above the plane, meters."""
        return pts @ self.normal + self.d

    def z_at(self, x: float, y: float) -> float:
        """Plane z at (x, y). Only meaningful for frame == 'base'."""
        n = self.normal
        if abs(n[2]) < 1e-6:
            raise ValueError('plane is vertical')
        return float(-(n[0] * x + n[1] * y + self.d) / n[2])


@dataclass
class Detection:
    """One object instance seen in one frame."""

    label: str                             # class name; 'unknown' if not classified
    score: float                           # classifier quality in [0, 1]
    score_kind: str = 'none'               # 'prob' | 'knn' | 'color' | 'none'
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x0, y0, x1, y1 pixels
    center_px: Tuple[float, float] = (0.0, 0.0)
    mask: Optional[np.ndarray] = None      # HxW bool
    # Geometry in robot base frame. None when depth or extrinsics are missing.
    center_base: Optional[np.ndarray] = None   # (3,) grasp center, z = top surface
    top_z: Optional[float] = None          # robust top height (base z)
    height: Optional[float] = None         # top_z - table z under the object
    yaw: Optional[float] = None            # major-axis angle in base XY, [-pi/2, pi/2)
    length: Optional[float] = None         # extent along major axis (m)
    width: Optional[float] = None          # extent along minor axis (m)
    n_points: int = 0
    valid: bool = True
    reject_reason: str = ''
    track_id: int = -1                     # stable id within one run, -1 if unassigned
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary without the mask."""
        c = None if self.center_base is None else [round(float(v), 4) for v in self.center_base]
        return {'label': self.label, 'score': round(float(self.score), 3),
                'score_kind': self.score_kind, 'bbox': [int(v) for v in self.bbox],
                'center_base': c,
                'top_z': None if self.top_z is None else round(float(self.top_z), 4),
                'height': None if self.height is None else round(float(self.height), 4),
                'yaw': None if self.yaw is None else round(float(self.yaw), 4),
                'length': None if self.length is None else round(float(self.length), 4),
                'width': None if self.width is None else round(float(self.width), 4),
                'n_points': int(self.n_points), 'valid': bool(self.valid),
                'reject_reason': self.reject_reason, 'track_id': int(self.track_id)}


@dataclass
class PerceptionResult:
    detections: List[Detection]
    table: Optional[TablePlane] = None
    debug_image: Optional[np.ndarray] = None   # BGR overlay for logs / viewer
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    frame_stamp: float = 0.0

    def valid_detections(self) -> List[Detection]:
        return [d for d in self.detections if d.valid]


@dataclass
class GraspPlan:
    """Top-down grasp. All poses are TCP poses in the base frame."""

    target: Detection
    T_pre: np.ndarray                      # above the object
    T_grasp: np.ndarray                    # fingers around the object
    T_lift: np.ndarray                     # after closing, lifted
    open_width: float                      # opening before descent (m)
    close_width: float                     # commanded width when closing (m)
    expected_width: float                  # object width along closing direction (m)
    closing_angle: float                   # base-frame angle of the closing (tool y) axis
    feasible: bool = True
    reason: str = ''


@dataclass
class PlacePlan:
    zone: str
    T_pre: np.ndarray                      # above the zone at transit height
    T_release: np.ndarray                  # where the gripper opens
    T_retreat: np.ndarray                  # after opening, moved up
    open_width: float = 0.07
    feasible: bool = True
    reason: str = ''


@dataclass
class MotionResult:
    ok: bool
    # '' | 'timeout' | 'unreachable' | 'arm_error' | 'comm' | 'rejected' | 'aborted' | 'stalled'
    error: str = ''
    detail: str = ''
    duration: float = 0.0
    final_tcp: Optional[np.ndarray] = None
    pos_err: Optional[float] = None        # meters, TCP position error at the end
    rot_err_deg: Optional[float] = None


@dataclass
class GripperResult:
    ok: bool
    width: Optional[float] = None          # measured opening after the command (m)
    error: str = ''                        # '' | 'timeout' | 'comm' | 'aborted' | 'fault'
    detail: str = ''


@dataclass
class ArmHealth:
    ok: bool
    reason: str = ''
    comm_ok: bool = True
    enabled: Optional[bool] = None
    ctrl_mode: Optional[int] = None
    arm_status: Optional[int] = None
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stamped:
    value: Any
    stamp: float                           # host monotonic seconds


@dataclass
class PickResult:
    """Outcome of one pick-and-place attempt on one detection."""

    ok: bool                               # object believed placed in the zone
    stage: str = ''                        # stage reached / failed at
    error: str = ''
    label: str = ''
    zone: str = ''
    duration: float = 0.0
    grasp_width: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskReport:
    task: str
    ok: bool                               # all required targets placed in time
    placed: int = 0
    required: int = 0
    attempts: int = 0
    elapsed: float = 0.0
    time_limit: float = 0.0
    # 'done' | 'timeout' | 'fault' | 'aborted' | 'no_targets' | 'self_check' | 'config'
    stop_reason: str = ''
    picks: List[PickResult] = field(default_factory=list)
    log_dir: str = ''


@dataclass
class Zone:
    """Place zone in the base frame (tray, bin or marked area)."""

    name: str
    center: np.ndarray                     # (3,) x, y, floor z (where the object rests)
    size: Tuple[float, float] = (0.10, 0.10)   # extent along the zone's own x / y (m)
    yaw: float = 0.0                       # zone x axis angle in base XY
    wall_height: float = 0.0               # rim height above floor (0 = flat mat)
    release_height: Optional[float] = None  # overrides place.release_height
    label: str = ''                        # free text (color / category) for logs and L3

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        dx, dy = x - float(self.center[0]), y - float(self.center[1])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return (abs(lx) <= self.size[0] / 2.0 + margin
                and abs(ly) <= self.size[1] / 2.0 + margin)

    def corners_xy(self, margin: float = 0.0) -> np.ndarray:
        """4x2 corner points in base XY (counter-clockwise)."""
        hx, hy = self.size[0] / 2.0 + margin, self.size[1] / 2.0 + margin
        loc = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        return loc @ R.T + np.asarray(self.center[:2], dtype=float)

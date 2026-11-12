"""
Hand-eye calibration.

Collect pairs of (TCP pose, board pose in the camera) and solve for the camera
mount transform with OpenCV's AX = XB solvers (cv2.calibrateHandEye), matching
agilexrobotics/handeye_calibration_ros (the agx_arm_calibration node described
in the AgileX ROS 2 hand-eye tutorial):

    eye_in_hand (wrist camera): the board lies still on the table, the camera
        moves with the arm. Solves T_tcp_camera, stored as T_flange_camera
        (T_flange_tcp @ T_tcp_camera, T_flange_tcp from the driver tcp_offset).
    eye_to_hand (fixed camera): the board is fixed on the flange and moves
        with the arm. Solves T_base_camera.

The gripper transform passed to cv2.calibrateHandEye is T_base_tcp for
eye_in_hand and its inverse for eye_to_hand, as the tutorial's node does.
The board pose comes from a ChArUco board seen by the color camera, so the
result is expressed in the color optical frame (x right, y down, z forward).

The arm is moved by hand in drag-teach mode. The TCP pose is read from the
agx_arm_ros topic /feedback/tcp_pose as in the tutorial (or with pyAgxArm
over CAN); no motion or mode command is ever sent.

Capture the board at 10+ poses that differ in orientation, not just position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import Extrinsics
from .transforms import inv_T, pose6_to_T, rotation_angle

LOG = logging.getLogger('piper_vision.handeye')

METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}
MIN_POSES = 4


@dataclass
class BoardSpec:
    """
    A ChArUco board.

    squares_x / squares_y count the chessboard squares; the marker dictionary
    defaults to DICT_5X5_100, which fits the small boards used on a desk.
    """

    squares_x: int = 5
    squares_y: int = 7
    square_m: float = 0.030
    marker_m: float = 0.022
    dictionary: str = 'DICT_5X5_100'

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError('a ChArUco board needs at least 3x3 squares')
        if not 0 < self.marker_m < self.square_m:
            raise ValueError('marker size must be smaller than the square size')

    @staticmethod
    def from_dict(d: dict) -> 'BoardSpec':
        """Build a board spec from the camera config's ``board`` section."""
        return BoardSpec(int(d.get('squares_x', 5)), int(d.get('squares_y', 7)),
                         float(d.get('square_m', 0.030)), float(d.get('marker_m', 0.022)),
                         str(d.get('dictionary', 'DICT_5X5_100')))


@dataclass
class Sample:
    """One calibration view: the TCP pose and the board pose in the camera."""

    T_base_tcp: np.ndarray
    T_camera_board: np.ndarray
    n_corners: int = 0
    reproj_px: float = 0.0


@dataclass
class HandEyeResult:
    """A solved hand-eye transform with its residual."""

    extrinsics: Extrinsics
    rms_m: float
    max_m: float
    per_sample_m: List[float] = field(default_factory=list)
    method: str = 'park'
    n_samples: int = 0


class HandEyeCalibrator:
    """Collects samples and solves AX = XB for one mount type."""

    def __init__(self, mount: str, method: str = 'park',
                 tcp_offset: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
        """
        tcp_offset is the agx_arm_ros launch argument used while recording the
        TCP poses (all zeros = the poses are flange poses).
        """
        if len(tcp_offset) != 6:
            raise ValueError('tcp_offset needs 6 values')
        self.T_flange_tcp = pose6_to_T(tcp_offset)
        if mount not in ('eye_to_hand', 'eye_in_hand'):
            raise ValueError("mount must be 'eye_to_hand' or 'eye_in_hand'")
        if method not in METHODS:
            raise ValueError('method must be one of %s' % sorted(METHODS))
        self.mount = mount
        self.method = method
        self.samples: List[Sample] = []

    def add(self, T_base_tcp: np.ndarray, T_camera_board: np.ndarray,
            n_corners: int = 0, reproj_px: float = 0.0) -> Sample:
        """Record one view. Both poses are 4x4; the arm pose is T_base_tcp."""
        sample = Sample(np.asarray(T_base_tcp, dtype=float),
                        np.asarray(T_camera_board, dtype=float),
                        int(n_corners), float(reproj_px))
        if sample.T_base_tcp.shape != (4, 4) or sample.T_camera_board.shape != (4, 4):
            raise ValueError('poses must be 4x4')
        self.samples.append(sample)
        return sample

    def solve(self) -> HandEyeResult:
        """
        Solve for the camera transform.

        Raises ValueError below 4 samples. The residual is the spread of the
        pose that the setup holds fixed: the board pose in the base frame for
        eye_in_hand, the board pose in the flange frame for eye_to_hand.
        """
        if len(self.samples) < MIN_POSES:
            raise ValueError('need at least %d samples, have %d' % (MIN_POSES, len(self.samples)))
        R_g, t_g, R_t, t_t = [], [], [], []
        for s in self.samples:
            # eye_to_hand passes the inverse, matching handeye_calibration_ros
            # (PoseMartix with check_mode, mode eye_to_hand).
            T_gripper = inv_T(s.T_base_tcp) if self.mount == 'eye_to_hand' else s.T_base_tcp
            R_g.append(T_gripper[:3, :3])
            t_g.append(T_gripper[:3, 3:])
            R_t.append(s.T_camera_board[:3, :3])
            t_t.append(s.T_camera_board[:3, 3:])
        R_x, t_x = cv2.calibrateHandEye(R_g, t_g, R_t, t_t, method=METHODS[self.method])
        T = np.eye(4)
        T[:3, :3] = R_x
        T[:3, 3] = np.asarray(t_x).reshape(3)
        per = self.residuals(T)
        rms = float(np.sqrt(np.mean(np.square(per))))
        if self.mount == 'eye_to_hand':
            parent, T_store = 'base', T
        else:
            parent, T_store = 'flange', self.T_flange_tcp @ T
        ext = Extrinsics(T=T_store, mount=self.mount, parent_frame=parent,
                         source='handeye %s, %d samples' % (self.method, len(self.samples)),
                         rms=rms)
        return HandEyeResult(ext, rms, float(np.max(per)), per, self.method, len(self.samples))

    def residuals(self, T_parent_camera: np.ndarray) -> List[float]:
        """
        Return per-sample position spread (m) of the pose the setup holds fixed.

        T_parent_camera is the raw solver output (T_tcp_camera for eye_in_hand,
        T_base_camera for eye_to_hand). eye_in_hand reconstructs the board in
        the base frame, eye_to_hand reconstructs it in the TCP frame; the
        spread is the distance of each reconstruction from the median.
        """
        boards = [self._fixed_pose(s, T_parent_camera) for s in self.samples]
        centers = np.array([b[:3, 3] for b in boards])
        median = np.median(centers, axis=0)
        return [float(np.linalg.norm(c - median)) for c in centers]

    def _fixed_pose(self, sample: Sample, T_parent_camera: np.ndarray) -> np.ndarray:
        if self.mount == 'eye_to_hand':
            # The board rides on the arm, so its pose relative to the TCP is constant.
            return inv_T(sample.T_base_tcp) @ T_parent_camera @ sample.T_camera_board
        return sample.T_base_tcp @ T_parent_camera @ sample.T_camera_board

    def orientation_spread_deg(self) -> float:
        """Return the mean pairwise TCP rotation (deg); small means the data is weak."""
        if len(self.samples) < 2:
            return 0.0
        angles = []
        for i in range(len(self.samples)):
            for j in range(i + 1, len(self.samples)):
                Ri = self.samples[i].T_base_tcp[:3, :3]
                Rj = self.samples[j].T_base_tcp[:3, :3]
                angles.append(rotation_angle(Rj.T @ Ri))
        return float(np.degrees(np.mean(angles)))


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

def _dictionary(name: str) -> Any:
    if not hasattr(cv2.aruco, name):
        raise ValueError('unknown aruco dictionary %r' % name)
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(spec: BoardSpec) -> Any:
    """
    Create the OpenCV board object for a spec.

    Works with OpenCV 4.5 (CharucoBoard_create) and 4.7+ (CharucoBoard).
    Keep squares_y odd: OpenCV 4.6 changed the marker layout of boards with
    an even number of rows, so such a board prints differently across versions.
    """
    d = _dictionary(spec.dictionary)
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        return cv2.aruco.CharucoBoard_create(spec.squares_x, spec.squares_y,
                                             spec.square_m, spec.marker_m, d)
    return cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y),
                                  spec.square_m, spec.marker_m, d)


def board_corners_3d(board: Any) -> np.ndarray:
    """Return the Nx3 chessboard corner positions in the board frame (m)."""
    if hasattr(board, 'getChessboardCorners'):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    return np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)


def find_charuco_corners(gray: np.ndarray, board: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Nx2 pixel corners, N ids); empty arrays when nothing is found."""
    empty = (np.zeros((0, 2)), np.zeros(0, dtype=int))
    if hasattr(cv2.aruco, 'CharucoDetector'):
        cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    else:
        dictionary = board.dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            return empty
        _, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if ci is None or len(ci) == 0:
        return empty
    return (np.asarray(cc, dtype=np.float64).reshape(-1, 2),
            np.asarray(ci, dtype=int).reshape(-1))


def detect_board(color: np.ndarray, intrinsics: Any, spec: BoardSpec,
                 min_corners: int = 6) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Detect the board and return (T_camera_board, n_corners, reprojection px).

    Returns None when fewer than ``min_corners`` corners are found. The pose
    maps board coordinates (origin at the board corner, z out of the board)
    into the color optical frame.
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    board = make_board(spec)
    px, ids = find_charuco_corners(gray, board)
    if len(ids) < max(4, min_corners):
        return None
    obj = board_corners_3d(board)[ids]
    K = intrinsics.K()
    dist = np.asarray(intrinsics.dist, dtype=np.float64).ravel()
    ok, rvec, tvec = cv2.solvePnP(obj, px, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - px, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T, int(len(ids)), err


def render_board(spec: BoardSpec, dpi: float = 200.0) -> np.ndarray:
    """
    Render the board as a print-ready BGR image.

    The image is sized so that printing at ``dpi`` gives the nominal square
    size. Print at 100% scale (no "fit to page") and measure one square with
    a ruler before calibrating.
    """
    px_per_m = dpi / 0.0254
    size = (int(round(spec.squares_x * spec.square_m * px_per_m)),
            int(round(spec.squares_y * spec.square_m * px_per_m)))
    board = make_board(spec)
    img = board.generateImage(size) if hasattr(board, 'generateImage') else board.draw(size)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def draw_detection(color: np.ndarray, T_camera_board: np.ndarray, intrinsics: Any,
                   axis_m: float = 0.03) -> np.ndarray:
    """Draw the detected board axes on a copy of the image."""
    out = color.copy()
    rvec, _ = cv2.Rodrigues(T_camera_board[:3, :3])
    cv2.drawFrameAxes(out, intrinsics.K(), np.asarray(intrinsics.dist, dtype=float).ravel(),
                      rvec, T_camera_board[:3, 3:], axis_m)
    return out


def coverage_warning(samples: Sequence[Sample], min_spread_deg: float = 15.0) -> Optional[str]:
    """Return a warning when the TCP orientations barely change, else None."""
    if len(samples) < 2:
        return None
    angles = [rotation_angle(samples[j].T_base_tcp[:3, :3].T @ samples[i].T_base_tcp[:3, :3])
              for i in range(len(samples)) for j in range(i + 1, len(samples))]
    spread = float(np.degrees(np.mean(angles)))
    if spread < min_spread_deg:
        return ('TCP orientations differ by only %.1f deg on average; move the arm to '
                'clearly different angles, not just translations' % spread)
    return None

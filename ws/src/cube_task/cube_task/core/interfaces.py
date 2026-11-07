"""
Abstract interfaces between camera, perception, robot and task layers.

Implementations: camera/ (ros, file), sim/ (simulated camera), robot/
(pyAgxArm real arm, simulated arm), perception/ (classifiers, detectors).
The task layer only talks to these interfaces, so the same task code runs in
simulation, replay and on the real robot.
"""
from __future__ import annotations

import abc
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import (ArmHealth, CameraIntrinsics, Detection, Frame, GripperResult,
                    MotionResult, Stamped)


class ArmError(RuntimeError):
    """Raised when the arm cannot be connected or is in an unusable state."""


class CameraError(RuntimeError):
    """Raised when the camera cannot be started."""


class Clock(abc.ABC):
    """Monotonic time source. SimClock advances virtual time on sleep()."""

    @abc.abstractmethod
    def now(self) -> float:
        """Return monotonic seconds."""

    @abc.abstractmethod
    def sleep(self, dt: float) -> None:
        """Sleep dt seconds (virtual in simulation)."""


class CameraSource(abc.ABC):
    """Source of synchronized, depth-aligned RGB-D frames."""

    name: str = 'camera'

    @abc.abstractmethod
    def start(self) -> None:
        """Open the device / subscriptions. Raise CameraError on failure."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Release resources. Must be safe to call twice."""

    @abc.abstractmethod
    def get_frame(self, timeout: float = 1.0,
                  newer_than: Optional[float] = None) -> Optional[Frame]:
        """
        Return the latest frame with stamp > newer_than, or None on timeout.

        The returned depth is float32 meters aligned to the color image, and
        intrinsics describe the color image.
        """

    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Return the color intrinsics once known, else None."""
        return None


class ArmInterface(abc.ABC):
    """
    Robot arm + parallel gripper.

    All poses are 4x4 matrices in the robot base frame. TCP = flange * T_flange_tcp,
    where T_flange_tcp comes from config robot.tcp_offset. Motion calls block until
    the motion is verified complete, failed, timed out or aborted; they never raise
    for motion failures and report them in MotionResult instead.
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """Open communication. Raise ArmError on failure."""

    @abc.abstractmethod
    def enable(self, timeout: float = 5.0) -> bool:
        """Enable motors. Return True when all joints report enabled."""

    @abc.abstractmethod
    def health(self) -> ArmHealth:
        """Communication and controller status. ok=False means do not move."""

    @abc.abstractmethod
    def get_tcp_pose(self) -> Optional[Stamped]:
        """Latest TCP pose (value: 4x4 T_base_tcp)."""

    @abc.abstractmethod
    def get_flange_pose(self) -> Optional[Stamped]:
        """Latest flange pose (value: 4x4 T_base_flange)."""

    @abc.abstractmethod
    def get_joints(self) -> Optional[Stamped]:
        """Latest joint angles (value: np.ndarray shape (6,), rad)."""

    @abc.abstractmethod
    def move_tcp(self, T_base_tcp: np.ndarray, linear: bool = False,
                 timeout: Optional[float] = None) -> MotionResult:
        """Move the TCP to the target pose (joint-space P2P, or Cartesian line)."""

    @abc.abstractmethod
    def move_joints(self, q: Sequence[float],
                    timeout: Optional[float] = None) -> MotionResult:
        """Move to joint angles (rad)."""

    @abc.abstractmethod
    def gripper(self, width: float, force: Optional[float] = None,
                wait: bool = True) -> GripperResult:
        """Command the opening width (m). With wait=True return after it settles."""

    @abc.abstractmethod
    def gripper_width(self) -> Optional[Stamped]:
        """Latest measured opening width (value: float, m)."""

    @abc.abstractmethod
    def set_speed_percent(self, percent: int) -> None:
        """Set the controller speed percentage for move_tcp / move_joints."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop issuing commands and hold the current position if possible."""

    @abc.abstractmethod
    def emergency_stop(self) -> None:
        """Trigger the controller electronic emergency stop (arm sinks slowly, needs reset)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Disconnect without disabling the motors (disable drops the arm)."""

    # Optional teach-mode support used by on-site tools.
    def supports_teach(self) -> bool:
        return False

    def set_abort_event(self, event: threading.Event) -> None:
        """Blocking waits return MotionResult(ok=False, error='aborted') once set."""
        self._abort_event = event

    @property
    def abort_event(self) -> threading.Event:
        ev = getattr(self, '_abort_event', None)
        if ev is None:
            ev = threading.Event()
            self._abort_event = ev
        return ev


class InstanceClassifier(abc.ABC):
    """Assign labels to class-agnostic instances found by depth segmentation."""

    labels: List[str] = []

    @abc.abstractmethod
    def classify(self, frame: Frame, detections: List[Detection]) -> None:
        """
        Fill label / score / score_kind of each detection in place.

        Use label 'unknown' with a low score when unsure; never guess.
        """


class Detector2D(abc.ABC):
    """Image-space object detector (for example a YOLO ONNX model)."""

    labels: List[str] = []

    @abc.abstractmethod
    def detect(self, color: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """Return [(bbox x0,y0,x1,y1 in pixels, label, probability), ...]."""

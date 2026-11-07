"""
Simulated RGB-D camera rendering a SimWorld (core.interfaces.CameraSource).

Eye-to-hand: fixed T_base_camera. Eye-in-hand: T_base_camera =
arm.get_flange_pose().value @ T_flange_camera at capture time, so the arm
must be stationary while capturing, as on the real robot. Time stamps come
from the injected clock; frames are produced at FRAME_PERIOD_S.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, List, Optional

import numpy as np

from ..core.interfaces import CameraSource, Clock
from ..core.types import CameraIntrinsics, Frame
from .render import render_rgbd
from .world import SimWorld, check_pose

_log = logging.getLogger('cube_task.sim.sim_camera')

FRAME_PERIOD_S = 1.0 / 30.0
FAULT_KINDS = ('none_frames', 'stale', 'no_depth')


class SimCamera(CameraSource):
    """
    Render frames of a SimWorld at the clock's current time.

    noise_std: depth noise std (m) at 0.5 m (grows with z^2); dropout:
    random invalid depth fraction; seed: noise generator seed. Exactly one of
    T_base_camera (eye-to-hand) or T_flange_camera + arm (eye-in-hand).
    """

    name = 'sim'

    def __init__(self, world: SimWorld, clock: Clock, intrinsics: CameraIntrinsics,
                 T_base_camera: Any = None, T_flange_camera: Any = None, arm: Any = None,
                 noise_std: float = 0.0008, dropout: float = 0.002, seed: int = 0):
        if (T_base_camera is None) == (T_flange_camera is None):
            raise ValueError('SimCamera needs exactly one of T_base_camera / T_flange_camera')
        if T_flange_camera is not None and arm is None:
            raise ValueError('eye-in-hand SimCamera needs the arm')
        ns, dp = float(noise_std), float(dropout)
        if not math.isfinite(ns) or ns < 0.0 or not math.isfinite(dp) or not 0.0 <= dp < 1.0:
            raise ValueError('bad noise_std %r / dropout %r' % (noise_std, dropout))
        self.world = world
        self.clock = clock
        self._intr = intrinsics
        self.mount = 'eye_to_hand' if T_base_camera is not None else 'eye_in_hand'
        self.T_base_camera = None if T_base_camera is None else check_pose(
            T_base_camera, 'T_base_camera')
        self.T_flange_camera = None if T_flange_camera is None else check_pose(
            T_flange_camera, 'T_flange_camera')
        self.arm = arm
        self.noise_std = ns
        self.dropout = dp
        self._rng = np.random.default_rng(int(seed))
        self._lock = threading.Lock()
        self._started = False
        self._faults: List[List[Any]] = []
        self._last: Optional[Frame] = None
        self._seq = 0
        self.frames_rendered = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Mark the camera as running (nothing to open)."""
        self._started = True

    def stop(self) -> None:
        """Stop delivering frames. Safe to call twice."""
        self._started = False

    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Return the color intrinsics."""
        return self._intr

    # -------------------------------------------------------------- faults
    def inject(self, kind: str, count: int = 1) -> None:
        """
        Queue a fault for the next count get_frame calls.

        kind: 'none_frames' (return None after waiting the timeout),
        'stale' (return the previous frame object with its old stamp),
        'no_depth' (a fresh frame with depth=None). Faults run in FIFO order.
        """
        if kind not in FAULT_KINDS:
            raise ValueError('fault kind must be one of %s, got %r' % (FAULT_KINDS, kind))
        n = int(count)
        if n < 1:
            raise ValueError('fault count must be >= 1')
        with self._lock:
            self._faults.append([kind, n])

    def clear_faults(self) -> None:
        """Drop all queued faults."""
        with self._lock:
            self._faults = []

    def _next_fault(self) -> Optional[str]:
        with self._lock:
            if not self._faults:
                return None
            kind = self._faults[0][0]
            self._faults[0][1] -= 1
            if self._faults[0][1] <= 0:
                self._faults.pop(0)
            return kind

    # ------------------------------------------------------------- capture
    def current_T_base_camera(self) -> Optional[np.ndarray]:
        """Return the true camera pose now, or None when the arm pose is unavailable."""
        if self.T_base_camera is not None:
            return self.T_base_camera.copy()
        st = self.arm.get_flange_pose()
        if st is None or st.value is None:
            return None
        try:
            T_bf = check_pose(st.value, 'flange pose')
        except ValueError as e:
            _log.error('sim camera: invalid flange pose: %s', e)
            return None
        return T_bf @ self.T_flange_camera

    def get_frame(self, timeout: float = 1.0,
                  newer_than: Optional[float] = None) -> Optional[Frame]:
        """
        Return a frame stamped clock.now() (> newer_than), or None.

        When clock.now() <= newer_than the clock advances by frame periods
        until it is newer (at most timeout), like waiting for the next frame.
        Returns None when stopped, on an injected 'none_frames' fault, when
        no newer frame arrives within timeout, or when the eye-in-hand flange
        pose is unavailable.
        """
        if not self._started:
            _log.error('sim camera: get_frame before start()')
            return None
        to = max(0.0, float(timeout))
        fault = self._next_fault()
        if fault == 'none_frames':
            self.clock.sleep(to)
            return None
        if fault == 'stale' and self._last is not None:
            return self._last
        if newer_than is not None:
            waited = 0.0
            while self.clock.now() <= newer_than:
                if waited + FRAME_PERIOD_S > to + 1e-9:
                    self.clock.sleep(max(0.0, to - waited))
                    return None
                self.clock.sleep(FRAME_PERIOD_S)
                waited += FRAME_PERIOD_S
        T_bc = self.current_T_base_camera()
        if T_bc is None:
            _log.warning('sim camera: no flange pose for eye-in-hand capture')
            return None
        color, depth = render_rgbd(self.world, self._intr, T_bc, rng=self._rng,
                                   noise_std=self.noise_std, dropout=self.dropout)
        self._seq += 1
        self.frames_rendered += 1
        frame = Frame(color=color, depth=None if fault == 'no_depth' else depth,
                      intrinsics=self._intr, stamp=float(self.clock.now()),
                      frame_id='camera_color_optical_frame', source='sim',
                      meta={'T_base_camera_true': T_bc, 'seq': self._seq,
                            'mount': self.mount, 'fault': fault or ''})
        self._last = frame
        return frame

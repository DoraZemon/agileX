"""
RGB-D camera rendered by MuJoCo (core.interfaces.CameraSource).

MujocoCamera renders the task camera of a MujocoWorld ('wrist_cam' on link6) at the
current physics state with mujoco.Renderer (offscreen, MUJOCO_GL=egl).
Depth is the z-depth in the optical frame in meters (MuJoCo's linearized
depth buffer), 0 where nothing was hit. The intrinsics are those of the
MuJoCo pinhole (MujocoWorld.intrinsics: fx = fy from fovy, cx = (W - 1) / 2,
cy = (H - 1) / 2), so pixel centers follow the OpenCV convention.

Sensor model as sim/render.py: gaussian color noise, depth noise
noise_std * (z / 0.5)^2, random dropout and dropout at depth jumps, from a
seeded generator (deterministic). Faults for tests and the bench as
sim_camera.SimCamera: inject('none_frames' | 'stale' | 'no_depth', count).

Rendering happens synchronously inside get_frame() in the calling thread;
the renderer (an EGL context) is created by the first get_frame() and
freed by stop() / close().
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, List, Optional

import numpy as np

from ...core.interfaces import CameraSource, Clock
from ...core.types import CameraIntrinsics, Frame
from ..render import _discontinuities, COLOR_NOISE_STD, DISCONTINUITY_DROP_P, DISCONTINUITY_JUMP
from ..sim_camera import FAULT_KINDS, FRAME_PERIOD_S
from .engine import require_mujoco

_LOG = logging.getLogger('cube_task.sim.mujoco.camera')

#: depth at or beyond this fraction of the far clipping plane is "nothing hit" (sky).
FAR_FRACTION = 0.999


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    v = cfg.get(key, default)
    return default if v is None else v


class MujocoCamera(CameraSource):
    """
    Render RGB-D frames of a MujocoWorld's task camera at the clock's time.

    world: scene.build_scene result, attached to clock (world.attach). cfg:
    optional Config for sim.noise_std (0.0008), sim.dropout (0.002) and
    sim.seed (0); explicit arguments win. camera: MuJoCo camera name
    (default world.task_camera). color_noise: gaussian color noise std in
    uint8 levels (default render.COLOR_NOISE_STD; 0 = none).
    """

    name = 'mujoco'

    def __init__(self, world: Any, clock: Clock, cfg: Any = None,
                 noise_std: Optional[float] = None, dropout: Optional[float] = None,
                 seed: Optional[int] = None, camera: Optional[str] = None,
                 color_noise: Optional[float] = None):
        self._mj = require_mujoco()
        if world.sim is None:
            world.attach(clock)
        elif world.sim.clock is not clock:
            raise ValueError('MujocoCamera: the world is attached to another clock')
        ns = float(_cfg_get(cfg, 'sim.noise_std', 0.0008) if noise_std is None else noise_std)
        dp = float(_cfg_get(cfg, 'sim.dropout', 0.002) if dropout is None else dropout)
        cn = float(COLOR_NOISE_STD if color_noise is None else color_noise)
        if not (math.isfinite(ns) and ns >= 0.0 and math.isfinite(dp) and 0.0 <= dp < 1.0
                and math.isfinite(cn) and cn >= 0.0):
            raise ValueError('bad noise_std %r / dropout %r / color_noise %r'
                             % (noise_std, dropout, color_noise))
        self.world = world
        self.clock = clock
        self.camera = str(camera or world.task_camera)
        self._cam_id = int(self._mj.mj_name2id(world.model, self._mj.mjtObj.mjOBJ_CAMERA,
                                               self.camera))
        if self._cam_id < 0:
            raise ValueError('MujocoCamera: no camera named %r in the scene' % self.camera)
        if self.camera == world.task_camera:
            self._intr = world.intrinsics
        else:
            from .scene import intrinsics_for_fovy
            base = world.intrinsics
            self._intr = intrinsics_for_fovy(float(world.model.cam_fovy[self._cam_id]),
                                             base.width, base.height)
        self.mount = str(world.info.get('mount', 'eye_in_hand'))
        self.noise_std = ns
        self.dropout = dp
        self.color_noise = cn
        self._rng = np.random.default_rng(int(_cfg_get(cfg, 'sim.seed', 0) if seed is None
                                              else seed))
        self._lock = threading.Lock()
        self._renderer = None
        self._thread: Optional[int] = None
        self._started = False
        self._faults: List[List[Any]] = []
        self._last: Optional[Frame] = None
        self._seq = 0
        self.frames_rendered = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Mark the camera as running (the renderer is created by the first get_frame)."""
        self._started = True

    def stop(self) -> None:
        """Stop delivering frames and free the renderer. Safe to call twice."""
        self._started = False
        self.close()

    def close(self) -> None:
        """Free the offscreen renderer (EGL context)."""
        r, self._renderer = self._renderer, None
        if r is not None:
            try:
                r.close()
            except Exception:  # noqa: B902 - teardown must not raise
                _LOG.exception('closing the MuJoCo renderer failed')

    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Return the color intrinsics of the rendered images."""
        return self._intr

    # -------------------------------------------------------------- faults
    def inject(self, kind: str, count: int = 1) -> None:
        """
        Queue a fault for the next count get_frame calls (as SimCamera.inject).

        kind: 'none_frames' (return None after waiting the timeout), 'stale'
        (return the previous frame object), 'no_depth' (depth=None).
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
    def current_T_base_camera(self) -> np.ndarray:
        """Return the true optical-frame camera pose in the base frame now."""
        return self.world.camera_pose(self.camera)

    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.world.model, int(self._intr.height),
                                               int(self._intr.width))
            self._thread = threading.get_ident()
        return self._renderer

    def render(self):
        """
        Render the current physics state without noise.

        Returns (color uint8 BGR HxWx3, depth float32 m HxW with 0 = invalid,
        T_base_camera optical 4x4). Must be called from the thread that
        rendered first (the EGL context is bound to it).
        """
        if self._renderer is not None and self._thread != threading.get_ident():
            raise RuntimeError('MujocoCamera renders only in the thread that created it')
        r = self._get_renderer()
        m = self.world.model
        with self.world.sim.lock:
            r.update_scene(self.world.data, camera=self._cam_id)
            rgb = r.render()
            r.enable_depth_rendering()
            try:
                depth = r.render()
            finally:
                r.disable_depth_rendering()
            T_bc = self.current_T_base_camera()
        far = float(m.vis.map.zfar) * float(m.stat.extent)
        near = float(m.vis.map.znear) * float(m.stat.extent)
        depth = np.where((depth >= FAR_FRACTION * far) | (depth <= near), 0.0,
                         depth).astype(np.float32)
        color = np.ascontiguousarray(rgb[:, :, ::-1])
        return color, depth, T_bc

    def _add_noise(self, color: np.ndarray, depth: np.ndarray):
        g = self._rng
        if self.color_noise > 0.0:
            img = color.astype(np.float32) + g.normal(0.0, self.color_noise,
                                                      color.shape).astype(np.float32)
            color = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        ns, dp = self.noise_std, self.dropout
        if ns > 0.0 or dp > 0.0:
            depth = depth.astype(np.float64)
            valid = depth > 0.0
            edge = _discontinuities(depth, DISCONTINUITY_JUMP) if dp > 0.0 else None
            if ns > 0.0:
                z = depth[valid]
                depth[valid] = np.maximum(z + g.normal(0.0, 1.0, z.shape) * ns * (z / 0.5) ** 2,
                                          1e-4)
            if dp > 0.0:
                drop = g.random(depth.shape) < dp
                drop |= edge & (g.random(depth.shape) < DISCONTINUITY_DROP_P)
                depth[drop] = 0.0
            depth = depth.astype(np.float32)
        return color, depth

    def get_frame(self, timeout: float = 1.0,
                  newer_than: Optional[float] = None) -> Optional[Frame]:
        """
        Return a frame stamped clock.now() (> newer_than), or None.

        When clock.now() <= newer_than the clock (and the physics) advances
        by frame periods (1/30 s) until it is newer, at most timeout. Returns
        None when stopped, on an injected 'none_frames' fault or when no newer
        frame arrives within timeout.
        """
        if not self._started:
            _LOG.error('mujoco camera: get_frame before start()')
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
        try:
            color, depth, T_bc = self.render()
        except RuntimeError as e:
            _LOG.error('mujoco camera: %s', e)
            return None
        color, depth = self._add_noise(color, depth)
        self._seq += 1
        self.frames_rendered += 1
        frame = Frame(color=color, depth=None if fault == 'no_depth' else depth,
                      intrinsics=self._intr, stamp=float(self.clock.now()),
                      frame_id='camera_color_optical_frame', source='mujoco',
                      meta={'T_base_camera_true': T_bc, 'seq': self._seq, 'mount': self.mount,
                            'camera': self.camera, 'fault': fault or ''})
        self._last = frame
        return frame

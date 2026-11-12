"""Create the configured CameraSource (camera.backend: ros | file | sim | mujoco)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.extrinsics import normalize_mount, validate_transform
from ..core.interfaces import CameraSource, Clock
from ..core.types import CameraIntrinsics

BACKENDS = ('ros', 'file', 'sim', 'mujoco')


def _pose(cfg: Any, key: str) -> Optional[np.ndarray]:
    v = cfg.get(key, None)
    if v is None:
        return None
    return validate_transform(v, key)


def make_camera(cfg: Any, clock: Clock, sim_world: Any = None,
                arm: Any = None) -> CameraSource:
    """
    Return an unstarted CameraSource for camera.backend; call start() on it.

    - 'ros': RosCamera(cfg, clock).
    - 'file': FileCamera(cfg.path('camera.file.dir'), clock, loop=camera.file.loop,
      repeat=camera.file.repeat (default 1)).
    - 'sim': cube_task.sim.sim_camera.SimCamera with sim_world, the intrinsics
      in sim.intrinsics and the true wrist camera pose sim.T_flange_camera
      (needs arm).
    - 'mujoco': cube_task.sim.mujoco.camera.MujocoCamera rendering the task
      camera of sim_world (a MujocoWorld) with the noise of sim.noise_std /
      sim.dropout / sim.seed; the arm is not needed (the wrist camera moves
      with the simulated link6).

    Raises ValueError for an unknown backend or missing parameters.
    """
    backend = str(cfg.get('camera.backend', '') or '').strip().lower()
    if backend == 'ros':
        from .ros_camera import RosCamera
        return RosCamera(cfg, clock)
    if backend == 'file':
        from .file_camera import FileCamera
        d = cfg.path('camera.file.dir')
        if d is None:
            raise ValueError('camera.backend is file but camera.file.dir is empty')
        repeat = int(cfg.get('camera.file.repeat', 1) or 1)
        return FileCamera(d, clock, loop=bool(cfg.get('camera.file.loop', True)),
                          repeat=repeat)
    if backend == 'sim':
        return _make_sim_camera(cfg, clock, sim_world, arm)
    if backend == 'mujoco':
        if sim_world is None or not hasattr(sim_world, 'camera_pose'):
            raise ValueError('camera.backend mujoco needs the MujocoWorld of '
                             'sim.mujoco.scene.build_scene as sim_world')
        from ..sim.mujoco.camera import MujocoCamera
        return MujocoCamera(sim_world, clock, cfg)
    raise ValueError('unknown camera.backend %r (use one of %s)'
                     % (cfg.get('camera.backend', None), ', '.join(BACKENDS)))


def _make_sim_camera(cfg: Any, clock: Clock, sim_world: Any, arm: Any) -> CameraSource:
    if sim_world is None:
        raise ValueError('camera.backend sim needs sim_world')
    intr_d = cfg.get('sim.intrinsics', None)
    if not intr_d:
        raise ValueError('camera.backend sim needs sim.intrinsics {fx, fy, cx, cy, width, '
                         'height} (merge sim.scenarios.scenario_config_overlay into the config)')
    try:
        intr = CameraIntrinsics.from_dict(intr_d)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('sim.intrinsics is invalid: %s' % e)
    mount_raw = cfg.get('camera.mount', None)
    if mount_raw not in (None, ''):
        normalize_mount(mount_raw)
    T_fc = _pose(cfg, 'sim.T_flange_camera')
    if T_fc is None:
        raise ValueError('camera.backend sim needs sim.T_flange_camera (the wrist camera)')
    if arm is None:
        raise ValueError('the sim wrist camera needs the arm (flange pose)')
    from ..sim.sim_camera import SimCamera
    return SimCamera(world=sim_world, clock=clock, intrinsics=intr,
                     T_flange_camera=T_fc, arm=arm,
                     noise_std=float(cfg.get('sim.noise_std', 0.0008)),
                     dropout=float(cfg.get('sim.dropout', 0.002)),
                     seed=int(cfg.get('sim.seed', 0)))

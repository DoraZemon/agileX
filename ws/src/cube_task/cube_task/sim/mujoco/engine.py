"""
MuJoCo physics engine shared by the MuJoCo arm, camera and video recorder.

Time: physics advances only through the injected SimClock. Every clock.sleep(dt)
(arm motion loops, runner settle waits, camera frame waits) calls
MujocoSim._on_advance, which runs the physics steps for that interval. Before
every physics step the registered controllers set data.ctrl (for example the
arm's trajectory interpolation); after the step the frame hooks run whenever
the virtual time crosses one of their frame times (video recording).

All mujoco imports are lazy so the package imports without MuJoCo installed.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

_LOG = logging.getLogger('cube_task.sim.mujoco')

ControllerFn = Callable[['MujocoSim', float], None]
FrameHook = Callable[['MujocoSim', float], None]


def require_mujoco():
    """Import and return the mujoco module, or raise ImportError with the fix."""
    try:
        import mujoco  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            'MuJoCo is not installed for this Python. Use the MuJoCo venv: '
            '~/.venvs/piper-mujoco/bin/python (see docs/仿真与测试.md)') from e
    return mujoco


class _Hook:
    def __init__(self, fn: FrameHook, period: float, t_next: float):
        self.fn = fn
        self.period = float(period)
        self.t_next = float(t_next)


class MujocoSim:
    """
    Own an MjModel / MjData pair and step it in lockstep with a SimClock.

    model, data: mujoco.MjModel / mujoco.MjData. clock: core.clock.SimClock
    (anything with now() and add_listener(fn(t_old, t_new))). meta: free-form
    dict from the scene builder (names of joints, actuators, sites, objects).
    """

    def __init__(self, model: Any, data: Any, clock: Any, meta: Optional[dict] = None):
        self.mujoco = require_mujoco()
        self.m = model
        self.d = data
        self.clock = clock
        self.meta = dict(meta or {})
        self.dt = float(model.opt.timestep)
        self.lock = threading.RLock()
        self._controllers: List[ControllerFn] = []
        self._hooks: List[_Hook] = []
        self._t = float(clock.now())          # virtual time of the physics state
        self.n_steps = 0
        self.mujoco.mj_forward(self.m, self.d)
        clock.add_listener(self._on_advance)

    # ----------------------------------------------------------- registration
    def add_controller(self, fn: ControllerFn) -> None:
        """Call fn(sim, t) before every physics step (set data.ctrl there)."""
        self._controllers.append(fn)

    def add_frame_hook(self, fn: FrameHook, fps: float) -> None:
        """Call fn(sim, t) every 1/fps seconds of virtual time (first call now)."""
        if fps <= 0:
            raise ValueError('fps must be > 0')
        self._hooks.append(_Hook(fn, 1.0 / float(fps), self._t))

    def remove_frame_hook(self, fn: FrameHook) -> None:
        self._hooks = [h for h in self._hooks if h.fn is not fn]

    # ------------------------------------------------------------------ time
    @property
    def time(self) -> float:
        """Return the virtual time of the physics state in seconds (clock time base)."""
        return self._t

    def _on_advance(self, t_old: float, t_new: float) -> None:
        with self.lock:
            while self._t + self.dt <= t_new + 1e-9:
                self.step_once()

    def step_once(self) -> None:
        """Run the controllers, one physics step, then due frame hooks."""
        mj = self.mujoco
        t = self._t
        for fn in self._controllers:
            fn(self, t)
        mj.mj_step(self.m, self.d)
        self._t = t + self.dt
        self.n_steps += 1
        for h in self._hooks:
            if self._t + 1e-9 >= h.t_next:
                try:
                    h.fn(self, self._t)
                except Exception:  # noqa: B902 - a broken recorder must not stop physics
                    _LOG.exception('frame hook failed; removing it')
                    h.period = float('inf')
                h.t_next += h.period
                if h.t_next <= self._t:
                    h.t_next = self._t + h.period

    def settle(self, seconds: float) -> None:
        """Advance the clock (and physics) by seconds, e.g. to let objects come to rest."""
        self.clock.sleep(float(seconds))

    # ---------------------------------------------------------------- lookup
    def body_id(self, name: str) -> int:
        i = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError('no body named %r' % name)
        return int(i)

    def joint_id(self, name: str) -> int:
        i = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError('no joint named %r' % name)
        return int(i)

    def actuator_id(self, name: str) -> int:
        i = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError('no actuator named %r' % name)
        return int(i)

    def site_id(self, name: str) -> int:
        i = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise KeyError('no site named %r' % name)
        return int(i)

    def camera_id(self, name: str) -> int:
        i = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_CAMERA, name)
        if i < 0:
            raise KeyError('no camera named %r' % name)
        return int(i)

    def body_pose(self, name_or_id) -> 'Any':
        """Return the 4x4 world pose of a body (numpy array)."""
        import numpy as np
        i = name_or_id if isinstance(name_or_id, int) else self.body_id(name_or_id)
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.d.xmat[i], dtype=float).reshape(3, 3)
        T[:3, 3] = np.asarray(self.d.xpos[i], dtype=float)
        return T

    def site_pose(self, name_or_id) -> 'Any':
        """Return the 4x4 world pose of a site (numpy array)."""
        import numpy as np
        i = name_or_id if isinstance(name_or_id, int) else self.site_id(name_or_id)
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.d.site_xmat[i], dtype=float).reshape(3, 3)
        T[:3, 3] = np.asarray(self.d.site_xpos[i], dtype=float)
        return T

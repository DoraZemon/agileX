"""
Shared helpers for the end-to-end simulation tests (this module has no tests).

SimRig builds a seeded scenario exactly like ``run_task --mode sim``
(tools.common.load_cfg_for + open_session): SimWorld, SimArm wrapped in
SafeArm, SimCamera, real perception, SimClock. The arm handed to the runner is
a CountingArm, so tests can see every command the task layer tried to send.
The simulation KNN models are trained once and cached by
tools.common.ensure_sim_knn (CUBE_TASK_CACHE, default ~/.cache/cube_task).
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cube_task.task.runner import TaskRunner
from cube_task.tools.common import add_common_args, load_cfg_for, open_session
from cube_task.tools.sim_bench import ground_truth
import pytest

#: run the extended parametrizations too (more seeds, hard L2): CUBE_TASK_SLOW=1
SLOW = bool(os.environ.get('CUBE_TASK_SLOW'))
slow = pytest.mark.skipif(not SLOW, reason='slow e2e case; set CUBE_TASK_SLOW=1 to run it')

MOTION_COMMANDS = ('move_tcp', 'move_joints', 'gripper')


def sim_args(task: str = 'L1', seed: int = 0, mount: str = 'eye_to_hand',
             difficulty: str = 'normal', extrinsics_error=(0.0, 0.0),
             sets: Sequence[str] = ()) -> argparse.Namespace:
    """Return the Namespace the tools build for ``--mode sim``."""
    p = argparse.ArgumentParser()
    add_common_args(p)
    argv = ['--task', task, '--mode', 'sim', '--seed', str(seed), '--mount', mount,
            '--difficulty', difficulty, '--extrinsics-error', str(extrinsics_error[0]),
            str(extrinsics_error[1])]
    for s in sets:
        argv += ['--set', s]
    return p.parse_args(argv)


class MemLogger:
    """In-memory RunLogger stand-in recording events, image names and JSON files."""

    dir = ''

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.images: List[str] = []
        self.json: Dict[str, Any] = {}

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append(dict(fields, kind=kind))

    def save_image(self, name: str, bgr: Any) -> None:
        self.images.append(name)

    def save_json(self, name: str, obj: Any) -> None:
        self.json[name] = obj

    def kinds(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e['kind'] == kind]

    def decisions(self, action: str) -> List[Dict[str, Any]]:
        return [e for e in self.kinds('decision') if e.get('action') == action]


class CountingArm:
    """Proxy around the SafeArm that records every command and its result."""

    def __init__(self, arm) -> None:
        self._arm = arm
        self.calls: List[tuple] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._arm, name)
        if name in MOTION_COMMANDS + ('stop', 'emergency_stop'):
            def wrapped(*a, **kw):
                res = attr(*a, **kw)
                self.calls.append((name, a, kw, res))
                return res
            return wrapped
        return attr

    def motions(self) -> List[tuple]:
        return [c for c in self.calls if c[0] in MOTION_COMMANDS]

    def first_error_index(self, error: str) -> Optional[int]:
        """Return the index in calls of the first command that failed with error."""
        for i, c in enumerate(self.calls):
            if c[0] in MOTION_COMMANDS and getattr(c[3], 'error', '') == error:
                return i
        return None


class SimRig:
    """One simulated round: session objects, counting arm, in-memory logger."""

    def __init__(self, task: str = 'L1', seed: int = 0, mount: str = 'eye_to_hand',
                 difficulty: str = 'normal', extrinsics_error=(0.0, 0.0),
                 sets: Sequence[str] = ()):
        self.task = task
        self.args = sim_args(task, seed, mount, difficulty, extrinsics_error, sets)
        self.cfg, self.world, self.info = load_cfg_for(self.args, task)
        self.session = open_session(self.args, task, cfg_world_info=(self.cfg, self.world,
                                                                     self.info))
        self.cfg = self.session.cfg
        self.clock = self.session.clock
        self.sim_arm = self.session.arm.inner
        self.arm = CountingArm(self.session.arm)
        self.camera = self.session.camera
        self.logger = MemLogger()
        self.initial_xy = {o.name: o.center_xy().copy() for o in self.world.objects}

    def runner(self, camera=None, arm=None, **kw) -> TaskRunner:
        return TaskRunner(self.cfg, self.task, arm or self.arm, camera or self.camera,
                          self.session.perception, self.clock, self.logger, mode='sim', **kw)

    def run(self, **kw):
        return self.runner(**kw).run()

    def truth(self, targets=None) -> Dict[str, int]:
        return ground_truth(self.world, self.info, self.task, targets=targets)

    def zone_of(self, name: str) -> Optional[str]:
        z = self.world.zone_of(self.world.object(name))
        return None if z is None else z.name

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> 'SimRig':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def describe(rep, rig: SimRig) -> str:
    """Return a one-line failure description for assertion messages."""
    return 'stop=%s placed=%d/%d attempts=%d truth=%s picks=%s' % (
        rep.stop_reason, rep.placed, rep.required, rep.attempts, rig.truth(),
        [(p.label, p.stage, p.error) for p in rep.picks])

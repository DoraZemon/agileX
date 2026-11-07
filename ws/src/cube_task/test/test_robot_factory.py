"""Tests for robot/factory.py."""
from __future__ import annotations

import numpy as np
import pytest

from cube_task.core import transforms as tf
from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.robot.factory import make_arm
from cube_task.robot.safety import SafeArm


class TinyWorld:
    held = None

    def __init__(self):
        self.updates = 0

    def try_grasp(self, T, open_width, close_width):
        return None

    def update_tcp(self, T, width):
        self.updates += 1

    def release(self, T):
        return None


def _cfg(**over):
    cfg = load_config(use_overlays=False)
    for k, v in over.items():
        cfg.set(k.replace('__', '.'), v)
    return cfg


def test_sim_backend_is_wrapped_and_uses_the_world():
    world = TinyWorld()
    arm = make_arm(_cfg(robot__backend='sim'), SimClock(), sim_world=world)
    assert isinstance(arm, SafeArm)
    assert type(arm.inner).__name__ == 'SimArm'
    arm.connect()
    assert arm.enable()
    assert arm.move_tcp(tf.top_down_T(0.3, 0.0, 0.05, 1.5708)).ok
    assert world.updates > 1
    res = arm.move_tcp(tf.top_down_T(0.9, 0.0, 0.05, 0.0))
    assert res.error == 'rejected'


def test_tcp_offset_from_config():
    cfg = _cfg(robot__backend='sim', robot__tcp_offset=[0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
    arm = make_arm(cfg, SimClock())
    arm.connect()
    T_f = arm.get_flange_pose().value
    T_t = arm.get_tcp_pose().value
    assert T_t[:3, 3] == pytest.approx(T_f[:3, 3] + 0.10 * T_f[:3, 2])
    assert np.allclose(T_t[:3, :3], T_f[:3, :3])


def test_pyagxarm_backend_builds_without_touching_hardware():
    arm = make_arm(_cfg(robot__backend='pyagxarm'), SimClock())
    assert isinstance(arm, SafeArm) and type(arm.inner).__name__ == 'PyAgxArm'
    assert not arm.health().ok                # not connected yet
    assert arm.move_tcp(tf.top_down_T(0.3, 0.0, 0.05, 1.5708)).error == 'comm'


def test_invalid_config():
    with pytest.raises(ValueError, match='backend'):
        make_arm(_cfg(robot__backend='gazebo'), SimClock())
    with pytest.raises(ValueError, match='tcp_offset'):
        make_arm(_cfg(robot__backend='sim', robot__tcp_offset=[0.0, 0.1]), SimClock())
    with pytest.raises(ValueError, match='model'):
        make_arm(_cfg(robot__backend='sim', robot__model='piper_xl'), SimClock())

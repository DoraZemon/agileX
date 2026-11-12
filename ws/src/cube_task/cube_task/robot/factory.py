"""Build the arm for the configured backend, always wrapped in SafeArm."""
from __future__ import annotations

import logging
from typing import Any

from ..core.interfaces import Clock
from .kinematics import flange_to_tcp_from_cfg, PiperKinematics
from .safety import SafeArm, SafetyGuard

LOG = logging.getLogger('cube_task.robot.factory')

BACKENDS = ('sim', 'pyagxarm', 'mujoco')


def make_arm(cfg, clock: Clock, sim_world: Any = None) -> SafeArm:
    """
    Return SafeArm(inner) for config robot.backend.

    'sim': robot.sim_arm.SimArm(cfg, clock, world=sim_world).
    'pyagxarm': robot.pyagxarm_arm.PyAgxArm(cfg, clock) (real arm over CAN;
    the SDK is imported on connect()). 'mujoco': sim.mujoco.arm.MujocoArm on
    the physics of sim_world (a sim.mujoco.world.MujocoWorld, attached to
    clock here if it is not yet). The SafeArm uses SafetyGuard(cfg),
    PiperKinematics(robot.model) and T_flange_tcp from robot.tcp_offset.
    Raises ValueError for an unknown backend or invalid robot config.
    The arm is not connected yet: call connect() and enable().
    """
    backend = str(cfg.get('robot.backend', None))
    if backend not in BACKENDS:
        raise ValueError('config robot.backend must be one of %s, got %r' % (BACKENDS, backend))
    T_flange_tcp = flange_to_tcp_from_cfg(cfg)
    kin = PiperKinematics(str(cfg.get('robot.model', None)))
    if backend == 'sim':
        from .sim_arm import SimArm
        inner = SimArm(cfg, clock, world=sim_world)
    elif backend == 'mujoco':
        if sim_world is None or not hasattr(sim_world, 'attach'):
            raise ValueError('robot.backend mujoco needs the MujocoWorld of '
                             'sim.mujoco.scene.build_scene as sim_world')
        from ..sim.mujoco.arm import MujocoArm
        inner = MujocoArm(cfg, clock, sim_world.attach(clock))
    else:
        if sim_world is not None:
            LOG.warning('sim_world ignored for robot.backend=pyagxarm')
        from .pyagxarm_arm import PyAgxArm
        inner = PyAgxArm(cfg, clock)
    return SafeArm(inner, SafetyGuard(cfg), kin=kin, T_flange_tcp=T_flange_tcp,
                   table_check_joint_moves=bool(
                       cfg.get('robot.safety.table_check_joint_moves', True)))

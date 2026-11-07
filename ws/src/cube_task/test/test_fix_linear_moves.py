"""robot.motion.linear_moves=false sends every short vertical move as P2P."""
from test_e2e_common import SimRig

from cube_task.core.config import Config


class _RecordingArm:
    """Arm proxy that records the linear flag of every move_tcp call."""

    def __init__(self, arm):
        self._arm = arm
        self.linear_flags = []

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def move_tcp(self, T, linear=False, timeout=None):
        self.linear_flags.append(bool(linear))
        return self._arm.move_tcp(T, linear=linear, timeout=timeout)


def _run(linear_moves, descent):
    rig = SimRig('L1', seed=1)
    rig.cfg.set('robot.motion.linear_moves', linear_moves)
    rig.cfg.set('grasp.linear_descent', descent)
    assert isinstance(rig.cfg, Config)
    arm = _RecordingArm(rig.arm)
    rep = rig.run(arm=arm)
    return rig, rep, arm


def test_all_p2p_when_linear_moves_and_descent_are_off():
    rig, rep, arm = _run(False, False)
    assert rep.ok and rep.placed == 3
    assert rig.world.score(rig.info['class_to_zone'])['wrong'] == 0
    assert arm.linear_flags and not any(arm.linear_flags)


def test_default_uses_linear_short_moves():
    rig, rep, arm = _run(True, True)
    assert rep.ok and rep.placed == 3
    assert any(arm.linear_flags)

"""motion_ctl: argument parsing, sim sub-commands, MoveIt waypoint resampling."""
from __future__ import annotations

from cube_task.robot.moveit_bridge import MAX_WAYPOINTS, resample_waypoints
from cube_task.tools import motion_ctl
import numpy as np
import pytest


def _line(n, total):
    return [np.full(6, total * i / (n - 1)) for i in range(n)]


def test_resample_empty_and_single():
    assert resample_waypoints([]) == []
    out = resample_waypoints([np.zeros(6)])
    assert len(out) == 1


def test_resample_keeps_the_last_point_and_thins_the_middle():
    pts = _line(101, 1.0)
    out = resample_waypoints(pts, step=0.15)
    assert np.allclose(out[-1], pts[-1])
    assert 2 <= len(out) <= 8


def test_resample_never_exceeds_max_points():
    out = resample_waypoints(_line(500, 6.0), step=0.01)
    assert len(out) <= MAX_WAYPOINTS
    assert np.allclose(out[-1], np.full(6, 6.0))


@pytest.mark.parametrize('step', [0.0, -1.0, float('nan')])
def test_resample_rejects_a_bad_step(step):
    with pytest.raises(ValueError):
        resample_waypoints(_line(5, 1.0), step=step)


def test_parser_pick_needs_label_and_xyz():
    p = motion_ctl.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(['pick', '--mode', 'sim'])
    a = p.parse_args(['pick', '--mode', 'sim', '--label', 'red', '--xyz', '0.3', '0.0', '0.05'])
    assert a.cmd == 'pick' and a.xyz == [0.3, 0.0, 0.05] and not a.dry_run


def test_parser_moveit_is_plan_only_by_default():
    a = motion_ctl.build_parser().parse_args(['moveit', '--mode', 'sim'])
    assert a.cmd == 'moveit' and a.execute is False and a.waypoint_step == 0.15


def test_moveit_refuses_mujoco_mode():
    assert motion_ctl.main(['moveit', '--mode', 'mujoco']) == motion_ctl.EXIT_REFUSED


def test_status_in_sim_succeeds():
    assert motion_ctl.main(['status', '--mode', 'sim']) == motion_ctl.EXIT_OK


def test_vision_list_only_in_sim_succeeds():
    assert motion_ctl.main(['vision', '--mode', 'sim', '--list-only']) == motion_ctl.EXIT_OK


def test_gripper_refuses_a_width_above_the_maximum():
    code = motion_ctl.main(['gripper', '--mode', 'sim', '--width', '0.5'])
    assert code == motion_ctl.EXIT_REFUSED

"""Tests for task/skills.py with a scripted fake arm."""
from __future__ import annotations

from cube_task.core.clock import SimClock
from cube_task.core.types import MotionResult
from cube_task.core.zones import parse_zones
from cube_task.perception.grasp import GraspPlanner, held_offset_of, PlacePlanner
from cube_task.task.skills import PickPlaceSkill
import numpy as np
from test_task_fakes import FakeArm, FakeLogger, FakeWorld, make_cfg, make_det


def _setup(label='red', x=0.30, y=0.05, zone='zone_red', task='L1'):
    cfg = make_cfg(task)
    clock = SimClock()
    zones = parse_zones(cfg)
    world = FakeWorld(zones)
    det = world.add(make_det(label, x, y))
    arm = FakeArm(clock, world)
    log = FakeLogger()
    skill = PickPlaceSkill(cfg, arm, clock, log)
    g = GraspPlanner(cfg, task).plan(det)
    p = PlacePlanner(cfg).plan(zones[zone], held_offset_of(g), 0.2)
    return cfg, clock, world, arm, log, skill, g, p


def _z(cmd):
    return float(cmd[1][2, 3])


def _fail_at(stage_index, error):
    def fn(kind, target, linear, idx):
        if idx == stage_index:
            return MotionResult(ok=False, error=error, detail='scripted')
        return None
    return fn


def test_success_sequence_and_logging():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    r = skill.execute(g, p)
    assert r.ok and r.stage == 'done', (r.stage, r.error)
    assert world.placed == {'zone_red': ['red']}
    kinds = [c[0] for c in arm.commands]
    assert kinds == ['gripper', 'move_tcp', 'move_tcp', 'gripper', 'move_tcp', 'move_tcp',
                     'move_tcp', 'gripper', 'move_tcp']
    m = arm.motions()
    assert [c[2] for c in m] == [False, True, True, False, True, True]
    assert np.allclose(m[0][1], g.T_pre) and np.allclose(m[1][1], g.T_grasp)
    assert np.allclose(m[2][1], g.T_lift) and np.allclose(m[3][1], p.T_pre)
    assert np.allclose(m[4][1], p.T_release) and np.allclose(m[5][1], p.T_retreat)
    assert arm.gripper_cmds() == [g.open_width, g.close_width, p.open_width]
    steps = log.kinds('skill_step')
    assert [s['stage'] for s in steps] == ['open', 'pre', 'descend', 'close', 'lift', 'transit',
                                           'release', 'open_release', 'retreat']
    assert all(s['duration'] > 0 for s in steps)
    assert r.duration > 0 and r.grasp_width == 0.03
    assert log.kinds('pick_result')[0]['ok'] is True


def test_grasp_miss_opens_and_retreats_without_transit():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    world.miss_next = 1
    r = skill.execute(g, p)
    assert not r.ok and r.stage == 'grasp_miss' and not r.detail['fault']
    m = arm.motions()
    assert len(m) == 3
    assert np.allclose(m[-1][1], g.T_pre) and m[-1][2] is True
    assert arm.gripper_cmds()[-1] == g.open_width
    assert not world.placed


def test_dropped_after_lift_and_in_transit():
    for drop_after, stage in ((3, 'dropped'), (4, 'dropped_in_transit')):
        cfg, clock, world, arm, log, skill, g, p = _setup()
        arm.width_fn = lambda a, n=drop_after: 0.0 if a.n_motion >= n else a.width
        r = skill.execute(g, p)
        assert not r.ok and r.stage == stage, (r.stage, r.error)
        assert not r.detail['fault']
        assert len(arm.motions()) == drop_after
        assert len(arm.gripper_cmds()) == 2              # no opening over a bin


def test_comm_fault_stops_immediately():
    for err in ('comm', 'arm_error', 'aborted', 'weird_new_error'):
        cfg, clock, world, arm, log, skill, g, p = _setup()
        arm.fail_fn = _fail_at(1, err)                  # descent
        r = skill.execute(g, p)
        assert not r.ok and r.detail['fault'] and r.stage == 'descend' and r.error == err
        assert len(arm.commands) == 3                   # open, pre, failed descent; nothing more


def test_recoverable_descent_failure_retreats_up_once():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.fail_fn = _fail_at(1, 'stalled')
    r = skill.execute(g, p)
    assert not r.ok and not r.detail['fault'] and r.error == 'stalled'
    m = arm.motions()
    assert len(m) == 3 and np.allclose(m[2][1], g.T_pre) and m[2][2] is True
    assert r.detail['recovery'] == 'retreated'


def test_failed_recovery_retreat_is_a_fault():
    cfg, clock, world, arm, log, skill, g, p = _setup()

    def fn(kind, target, linear, idx):
        return MotionResult(ok=False, error='timeout') if idx in (1, 2) else None
    arm.fail_fn = fn
    r = skill.execute(g, p)
    assert r.detail['fault'] and len(arm.motions()) == 3


def test_pre_failure_does_not_move_further():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.fail_fn = _fail_at(0, 'rejected')
    r = skill.execute(g, p)
    assert not r.ok and r.stage == 'pre' and not r.detail['fault']
    assert len(arm.motions()) == 1


def test_transit_failure_puts_object_back_without_opening_over_bin():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.fail_fn = _fail_at(3, 'timeout')                # move to place.T_pre
    r = skill.execute(g, p)
    assert not r.ok and not r.detail['fault'], r.detail
    m = arm.motions()
    assert np.allclose(m[4][1], g.T_lift) and m[4][2] is False
    assert np.allclose(m[5][1], g.T_grasp) and m[5][2] is True
    assert np.allclose(m[6][1], g.T_pre)
    assert r.detail['recovery'] == 'put back' and not r.detail['holding']
    assert not world.placed and world.dropped == ['red']   # back on the table


def test_lift_failure_puts_back_in_place():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.fail_fn = _fail_at(2, 'stalled')
    r = skill.execute(g, p)
    m = arm.motions()
    assert np.allclose(m[3][1], g.T_grasp) and np.allclose(m[4][1], g.T_pre)
    assert not r.ok and r.detail['recovery'] == 'put back'


def test_release_stalled_near_release_opens_there():
    cfg, clock, world, arm, log, skill, g, p = _setup()

    def fn(kind, target, linear, idx):
        if idx == 4:                                    # descent into the zone
            T = np.array(p.T_release)
            T[2, 3] += 0.02                             # stopped 2 cm above release
            arm.tcp = T
            return MotionResult(ok=False, error='stalled')
        return None
    arm.fail_fn = fn
    r = skill.execute(g, p)
    assert r.ok and world.placed == {'zone_red': ['red']}
    assert 'released near' in r.detail['recovery']


def test_release_stalled_high_does_not_open_over_bin():
    cfg, clock, world, arm, log, skill, g, p = _setup()

    def fn(kind, target, linear, idx):
        if idx == 4:
            return MotionResult(ok=False, error='timeout')   # still at place.T_pre
        return None
    arm.fail_fn = fn
    r = skill.execute(g, p)
    assert not r.ok
    assert not world.placed
    # The gripper opened only at the pick spot (put back), never above the zone.
    open_cmds = [c for c in arm.commands if c[0] == 'gripper'][2:]
    assert len(open_cmds) == 1
    m = arm.motions()
    assert np.allclose(m[-2][1], g.T_grasp)


def test_missing_gripper_feedback_is_a_fault():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.no_width = True
    r = skill.execute(g, p)
    assert r.detail['fault'] and r.error == 'no_gripper_feedback'
    assert len(arm.motions()) == 0


def test_stale_width_feedback_is_a_fault_while_holding():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    from cube_task.core.types import Stamped
    real = arm.gripper_width

    def stale():
        s = real()
        return Stamped(s.value, s.stamp - 10.0) if arm.n_motion >= 3 else s
    arm.gripper_width = stale
    r = skill.execute(g, p)
    assert r.detail['fault'] and r.stage == 'verify_dropped'
    assert len(arm.motions()) == 3


def test_dry_run_never_closes_gripper():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    r = skill.execute(g, p, dry_run=True)
    assert not r.ok and r.stage == 'dry_run_done'
    assert all(w >= g.open_width for w in arm.gripper_cmds())
    zs = [_z(c) for c in arm.motions()]
    assert min(zs) >= g.T_grasp[2, 3] + 0.05 - 1e-9
    assert any(abs(z - (p.T_release[2, 3] + 0.05)) < 1e-9 for z in zs)
    assert world.objects and not world.placed


def test_infeasible_plan_sends_nothing():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    g.feasible, g.reason = False, 'too_wide'
    r = skill.execute(g, p)
    assert not r.ok and r.stage == 'plan' and arm.commands == []


def test_abort_event_stops_before_motion():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    arm.abort_event.set()
    r = skill.execute(g, p)
    assert r.detail['fault'] and r.error == 'aborted'
    assert arm.motions() == []


def test_unhealthy_arm_gets_no_recovery_motion():
    cfg, clock, world, arm, log, skill, g, p = _setup()

    def fn(kind, target, linear, idx):
        if idx == 1:
            arm.healthy = False
            return MotionResult(ok=False, error='timeout')
        return None
    arm.fail_fn = fn
    r = skill.execute(g, p)
    assert r.detail['fault'] and len(arm.motions()) == 2


def test_gripper_comm_error_is_a_fault():
    cfg, clock, world, arm, log, skill, g, p = _setup()
    from cube_task.core.types import GripperResult
    arm.gripper = lambda w, force=None, wait=True: GripperResult(ok=False, error='comm')
    r = skill.execute(g, p)
    assert r.detail['fault'] and r.error == 'gripper_comm' and arm.motions() == []


def test_backend_exception_is_contained():
    cfg, clock, world, arm, log, skill, g, p = _setup()

    def boom(*a, **k):
        raise RuntimeError('driver bug')
    arm.move_tcp = boom
    r = skill.execute(g, p)
    assert r.detail['fault'] and 'driver bug' in r.error
    assert arm.commands[-1][0] == 'stop'

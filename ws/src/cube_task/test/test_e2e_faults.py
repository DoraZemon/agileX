"""End-to-end fault handling in simulation (L1, eye_to_hand): faults injected mid-run."""
from __future__ import annotations

from cube_task.tools.sim_bench import FaultCamera, GraspMissArm
from test_e2e_common import describe, MOTION_COMMANDS, SimRig

#: 2 preparation commands + 10 per pick: call 16 falls inside the second pick.
SECOND_PICK_CALL = 16


def test_comm_loss_mid_run_stops_as_fault_without_more_motion():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        rig.sim_arm.inject('comm_loss', at_call=SECOND_PICK_CALL)
        rep = rig.run()
        assert rep.stop_reason == 'fault' and not rep.ok, describe(rep, rig)
        i = rig.arm.first_error_index('comm')
        assert i is not None, 'the injected comm loss must reach the task layer'
        later = [c[0] for c in rig.arm.calls[i + 1:] if c[0] in MOTION_COMMANDS]
        assert later == [], 'motion commands after the comm loss: %s' % later
        truth = rig.truth()
        assert rep.placed == 1 and truth['correct'] == 1 and truth['wrong'] == 0
        assert not rig.session.arm.health().ok


def test_grasp_miss_is_retried_and_the_round_completes():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        arm = GraspMissArm(rig.arm, at_call=1, close_width=0.0)
        rep = rig.run(arm=arm)
        assert arm.fired and rig.sim_arm.fault_log[0]['kind'] == 'grasp_miss'
        assert rep.ok, describe(rep, rig)
        assert rep.attempts == 4 and rep.picks[0].stage == 'grasp_miss'
        assert rig.truth() == {'correct': 3, 'wrong': 0, 'remaining': 0}


def test_drop_is_retried_and_the_round_completes():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        rig.sim_arm.inject('drop')
        rep = rig.run()
        assert rig.sim_arm.fault_log and rig.sim_arm.fault_log[0]['kind'] == 'drop'
        assert rep.ok, describe(rep, rig)
        assert rep.attempts >= 4 and not rep.picks[0].ok
        assert rig.truth() == {'correct': 3, 'wrong': 0, 'remaining': 0}


def test_motion_timeout_is_recovered_and_the_round_completes():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        rig.sim_arm.inject('timeout', at_call=5)
        rep = rig.run()
        assert rig.sim_arm.fault_log, 'timeout fault was not used'
        assert rep.ok, describe(rep, rig)
        assert any(p.error == 'timeout' for p in rep.picks)
        assert rig.truth()['wrong'] == 0


def test_camera_none_frames_mid_run_recover():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        cam = FaultCamera(rig.camera, at_call=4, count=3)
        rep = rig.run(camera=cam)
        assert cam.fired
        empty = [e for e in rig.logger.kinds('observation') if not e.get('ok')]
        assert len(empty) >= 1
        assert rep.ok, describe(rep, rig)
        assert rig.truth() == {'correct': 3, 'wrong': 0, 'remaining': 0}

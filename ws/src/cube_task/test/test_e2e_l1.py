"""End-to-end L1 simulation: SimWorld + SimArm + SimCamera + real perception and runner."""
from __future__ import annotations

import numpy as np
import pytest
from test_e2e_common import describe, SimRig, slow

MOUNTS = ('eye_to_hand', 'eye_in_hand')


@pytest.mark.parametrize('mount', MOUNTS)
@pytest.mark.parametrize('seed', [0, 1])
def test_l1_completes_without_wrong_placements(seed, mount):
    with SimRig('L1', seed, mount) as rig:
        rep = rig.run()
        truth = rig.truth()
        assert rep.ok and rep.stop_reason == 'done', describe(rep, rig)
        assert truth == {'correct': 3, 'wrong': 0, 'remaining': 0}
        assert rep.placed == 3 and rep.elapsed <= rep.time_limit


@pytest.mark.parametrize('seed', [2, 3, 4, 5])
@slow
def test_l1_more_seeds(seed):
    for mount in MOUNTS:
        with SimRig('L1', seed, mount) as rig:
            rep = rig.run()
            assert rig.truth()['wrong'] == 0, describe(rep, rig)
            assert rep.ok, describe(rep, rig)


@pytest.mark.parametrize('seed,mount', [(0, 'eye_to_hand'), (1, 'eye_to_hand'),
                                        (0, 'eye_in_hand')])
def test_l1_hard_never_places_distractor_or_wrong_zone(seed, mount):
    # Touching blocks + a yellow distractor. The merged pair may be missed (being fixed
    # elsewhere), so only the safety properties are asserted here.
    with SimRig('L1', seed, mount, difficulty='hard') as rig:
        yellow = [o.name for o in rig.world.objects if o.label == 'yellow']
        assert yellow, 'hard scenario must contain the distractor'
        runner = rig.runner()
        rep = runner.run()
        assert rig.truth()['wrong'] == 0, describe(rep, rig)
        for name in yellow:
            assert rig.zone_of(name) is None
            assert np.allclose(rig.world.object(name).center_xy(), rig.initial_xy[name])
        assert all(p.label != 'yellow' for p in rep.picks)
        assert not any(e.get('label') == 'yellow' for e in rig.logger.decisions('pick'))
        assert not runner.fault_detail.startswith('exception')
        assert rep.stop_reason in ('done', 'no_targets', 'timeout'), describe(rep, rig)


@pytest.mark.parametrize('seed', [1, 2, 3])
@slow
def test_l1_hard_more_seeds(seed):
    for mount in MOUNTS:
        with SimRig('L1', seed, mount, difficulty='hard') as rig:
            rep = rig.run()
            assert rig.truth()['wrong'] == 0, describe(rep, rig)
            assert all(p.label != 'yellow' for p in rep.picks)


@pytest.mark.parametrize('mount', MOUNTS)
def test_l1_succeeds_with_extrinsics_error(mount):
    with SimRig('L1', 0, mount, extrinsics_error=(0.003, 0.5)) as rig:
        assert rig.cfg.get('sim.extrinsics_error') == [0.003, 0.5]
        rep = rig.run()
        assert rep.ok, describe(rep, rig)
        assert rig.truth() == {'correct': 3, 'wrong': 0, 'remaining': 0}


def test_dry_run_never_closes_the_gripper_and_places_nothing():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        rep = rig.run(dry_run=True)
        assert not rep.ok and rep.placed == 0
        assert [p.stage for p in rep.picks] == ['dry_run_done'] * 3, describe(rep, rig)
        widths = [c[1][0] if c[1] else c[2]['width'] for c in rig.arm.calls
                  if c[0] == 'gripper']
        assert widths and min(widths) > 0.03       # opened for 30-40 mm blocks, never closed
        assert rig.truth() == {'correct': 0, 'wrong': 0, 'remaining': 3}
        assert rig.world.held is None
        for o in rig.world.objects:
            assert np.allclose(o.center_xy(), rig.initial_xy[o.name])
        assert len(rig.logger.decisions('pick')) == 3     # each block visited once


def test_skills_dry_run_hover_poses_are_reachable_without_workaround():
    with SimRig('L1', 0, 'eye_to_hand') as rig:
        rep = rig.run(dry_run=True)
        assert [p.stage for p in rep.picks] == ['dry_run_done'] * 3, describe(rep, rig)


def test_deadline_stops_with_timeout_and_no_late_pick():
    with SimRig('L1', 0, 'eye_to_hand', sets=['tasks.L1.time_limit_s=35']) as rig:
        rep = rig.run()
        assert rep.stop_reason == 'timeout' and not rep.ok, describe(rep, rig)
        min_time = float(rig.cfg.get('runner.min_time_for_pick_s'))
        picks = rig.logger.decisions('pick')
        assert picks, 'at least one pick fits in 35 s'
        assert all(e['remaining'] >= min_time for e in picks)
        assert len(picks) < 3
        truth = rig.truth()
        assert truth['wrong'] == 0 and truth['correct'] == rep.placed


def test_targets_pick_only_the_given_labels_in_order():
    with SimRig('L1', 0, 'eye_to_hand', sets=['tasks.L1.required=2']) as rig:
        rep = rig.run(targets=['blue', 'red'])
        assert rep.ok, describe(rep, rig)
        assert [p.label for p in rep.picks] == ['blue', 'red']
        green = [o.name for o in rig.world.objects if o.label == 'green'][0]
        assert rig.zone_of(green) is None
        assert rig.truth(targets=['blue', 'red']) == {'correct': 2, 'wrong': 0, 'remaining': 0}

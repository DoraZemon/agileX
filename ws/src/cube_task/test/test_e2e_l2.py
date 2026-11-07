"""End-to-end L2 (eye_to_hand) and L3 simulation with the cached simulation KNN model."""
from __future__ import annotations

import sys

from cube_task.tools import run_task
from cube_task.tools.sim_bench import l3_expected
import pytest
from test_e2e_common import describe, SimRig, slow


@pytest.mark.parametrize('seed', [0, 1])
def test_l2_eye_to_hand_completes_without_wrong_placements(seed):
    with SimRig('L2', seed, 'eye_to_hand') as rig:
        rep = rig.run()
        truth = rig.truth()
        assert truth['wrong'] == 0, describe(rep, rig)
        assert rep.ok and truth == {'correct': 6, 'wrong': 0, 'remaining': 0}, \
            describe(rep, rig)
        assert rep.elapsed <= rep.time_limit


@pytest.mark.parametrize('seed', [0, 1, 2])
@slow
def test_l2_hard_never_places_wrongly(seed):
    with SimRig('L2', seed, 'eye_to_hand', difficulty='hard') as rig:
        rep = rig.run()
        assert rig.truth()['wrong'] == 0, describe(rep, rig)


def test_l3_runs_the_scenario_instructions_and_places_the_named_objects():
    with SimRig('L3', 0, 'eye_to_hand') as rig:
        expected = l3_expected(rig.info)
        assert len(expected) == len(rig.info['instructions']) == 5
        rep = rig.run()
        assert rep.required == 5
        got = {name: rig.zone_of(name) for name in expected}
        assert got == expected, describe(rep, rig)
        assert rig.truth() == {'correct': 5, 'wrong': 0, 'remaining': 0}
        assert rep.ok, describe(rep, rig)
        texts = [p.detail.get('instruction') for p in rep.picks]
        assert texts == list(rig.info['instructions'])


def test_run_task_main_l3_sim_uses_the_scenario_instructions(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['run_task', '--task', 'L3', '--mode', 'sim', '--seed',
                                      '1', '--log-dir', str(tmp_path)])
    assert run_task.main() == run_task.EXIT_OK
    out = capsys.readouterr().out
    assert '5 from sim scenario' in out
    assert 'ground truth (sim): correct 5, wrong 0, remaining 0' in out


@pytest.mark.parametrize('seed', [1, 2])
@slow
def test_l3_eye_in_hand(seed):
    with SimRig('L3', seed, 'eye_in_hand') as rig:
        rep = rig.run()
        assert rig.truth()['wrong'] == 0, describe(rep, rig)
        assert rep.ok, describe(rep, rig)

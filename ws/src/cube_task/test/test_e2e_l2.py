"""End-to-end L2 (eye_to_hand) simulation with the cached simulation KNN model."""
from __future__ import annotations

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

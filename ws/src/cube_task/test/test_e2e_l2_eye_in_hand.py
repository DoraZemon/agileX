"""End-to-end L2 simulation with the wrist (eye_in_hand) camera."""
from __future__ import annotations

import pytest
from test_e2e_common import describe, SimRig


@pytest.mark.parametrize('seed', [0, 1])
def test_l2_eye_in_hand_completes_without_wrong_placements(seed):
    with SimRig('L2', seed, 'eye_in_hand') as rig:
        rep = rig.run()
        truth = rig.truth()
        assert truth['wrong'] == 0, describe(rep, rig)
        assert rep.ok and truth == {'correct': 6, 'wrong': 0, 'remaining': 0}, \
            describe(rep, rig)
        assert rep.elapsed <= rep.time_limit

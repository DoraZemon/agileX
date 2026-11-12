"""End-to-end L2 hard simulation (wrist camera) with the cached simulation KNN model."""
from __future__ import annotations

import pytest
from test_e2e_common import describe, SimRig, slow


@pytest.mark.parametrize('seed', [0, 1, 2])
@slow
def test_l2_hard_never_places_wrongly(seed):
    with SimRig('L2', seed, 'eye_in_hand', difficulty='hard') as rig:
        rep = rig.run()
        assert rig.truth()['wrong'] == 0, describe(rep, rig)

"""L1 detection against rendered scenes with known ground truth."""
from __future__ import annotations

import math

import numpy as np
import pytest

from piper_vision.l1 import L1Detector, L1Params
from piper_vision.transforms import wrap_angle

from synth import default_intrinsics, look_at, render

TABLE_Z = -0.02
CARDS = [('green', 0.30, 0.30, 0.0, 0.20), ('yellow', 0.05, 0.32, 0.1, 0.20),
         ('red', -0.18, 0.30, 0.0, 0.20)]
CUBES = [('red', 0.32, -0.08, 0.3, 0.05), ('green', 0.22, 0.02, -0.5, 0.05),
         ('yellow', 0.40, 0.06, 0.0, 0.05)]


def _scene(cubes=CUBES, cards=CARDS, noise=0.001, eye=(0.75, 0.0, 0.70)):
    intr = default_intrinsics()
    T = look_at(eye, (0.15, 0.10, TABLE_Z))
    color, depth = render(intr, T, TABLE_Z, cubes, cards, noise_std=noise)
    return color, depth, intr, T


def _match(res, label, x, y):
    found = [c for c in res.cubes if c.label == label]
    assert found, 'no %s cube in %s' % (label, [c.summary() for c in res.cubes])
    return min(found, key=lambda c: math.hypot(c.center_base[0] - x, c.center_base[1] - y))


def test_cubes_and_cards_found_with_depth():
    color, depth, intr, T = _scene()
    res = L1Detector(L1Params()).detect(color, depth, intr, T)
    assert res.table_z == pytest.approx(TABLE_Z, abs=0.003)
    assert len(res.cubes) == 3
    for label, x, y, yaw, edge in CUBES:
        c = _match(res, label, x, y)
        assert math.hypot(c.center_base[0] - x, c.center_base[1] - y) < 0.006
        assert c.center_base[2] == pytest.approx(TABLE_Z + edge, abs=0.004)
        assert c.height == pytest.approx(edge, abs=0.005)
        assert abs(wrap_angle(c.yaw - yaw, math.pi / 2)) < math.radians(6)
        assert c.score > 0.8
    names = sorted(z.name for z in res.zones)
    assert names == ['zone_green', 'zone_red', 'zone_yellow']
    for label, x, y, _, _ in CARDS:
        z = res.zone(label)
        assert math.hypot(z.center_base[0] - x, z.center_base[1] - y) < 0.01


def test_card_color_not_mistaken_for_cube():
    # A red cube standing on the red card: the card is flat, the cube is raised.
    cubes = [('red', -0.18, 0.30, 0.2, 0.05)]
    color, depth, intr, T = _scene(cubes=cubes)
    res = L1Detector(L1Params()).detect(color, depth, intr, T)
    assert [c.label for c in res.cubes] == ['red']
    c = res.cubes[0]
    assert math.hypot(c.center_base[0] + 0.18, c.center_base[1] - 0.30) < 0.008


def test_touching_same_color_cubes_split():
    cubes = [('green', 0.30, -0.05, 0.0, 0.05), ('green', 0.30, 0.0, 0.0, 0.05)]
    color, depth, intr, T = _scene(cubes=cubes, cards=[])
    res = L1Detector(L1Params()).detect(color, depth, intr, T)
    greens = res.by_label('green')
    assert len(greens) == 2
    ys = sorted(c.center_base[1] for c in greens)
    assert ys == pytest.approx([-0.05, 0.0], abs=0.008)


def test_color_only_fallback_needs_table_z():
    color, _, intr, T = _scene(cubes=CUBES[:1], cards=CARDS[:1], noise=0.0)
    res = L1Detector(L1Params()).detect(color, None, intr, T)
    assert not res.cubes and any('table_z' in w for w in res.warnings)
    res = L1Detector(L1Params(table_z=TABLE_Z)).detect(color, None, intr, T)
    assert len(res.cubes) == 1 and res.cubes[0].source == 'color'
    c = res.cubes[0]
    # side faces bias the estimate; it must still land on the cube
    assert math.hypot(c.center_base[0] - 0.32, c.center_base[1] + 0.08) < 0.025
    assert res.zone('green') is not None


def test_roi_polygon_filters():
    color, depth, intr, T = _scene()
    p = L1Params(roi_polygon=[[0.15, -0.15], [0.45, -0.15], [0.45, 0.0], [0.15, 0.0]])
    res = L1Detector(p).detect(color, depth, intr, T)
    assert sorted(c.label for c in res.cubes) == ['red']
    assert not res.zones


def test_bad_inputs_do_not_raise():
    color, depth, intr, T = _scene()
    det = L1Detector(L1Params())
    assert det.detect(None, depth, intr, T).warnings
    assert det.detect(color, depth[:100], intr, T).warnings
    blank = np.zeros_like(color)
    res = det.detect(blank, np.zeros_like(depth), intr, T)
    assert res.cubes == []


def test_params_reject_typos():
    with pytest.raises(ValueError, match='cube_sise'):
        L1Params.from_dict({'cube_sise': 0.05})

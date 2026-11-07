"""Tests for task/selection.py."""
from __future__ import annotations

from cube_task.task.selection import clearances, is_blacklisted, select_target
import numpy as np
import pytest
from test_task_fakes import make_det


def test_nearest_to_base():
    a = make_det('red', 0.40, 0.0)
    b = make_det('blue', 0.20, 0.10)
    c = make_det('green', 0.30, -0.20)
    assert select_target([a, b, c], 'nearest') is b
    assert select_target([a, b, c], 'nearest', arm_xy=(0.45, 0.0)) is a


def test_score_strategy_with_distance_tie_break():
    a = make_det('red', 0.40, 0.0, score=0.95)
    b = make_det('red', 0.20, 0.0, score=0.70)
    c = make_det('red', 0.25, 0.0, score=0.95)
    assert select_target([a, b, c], 'score') is c


def test_order_picks_first_available_label():
    r = make_det('red', 0.20, 0.0)
    g = make_det('green', 0.40, 0.0)
    assert select_target([r, g], 'nearest', order=['green', 'red']) is g
    assert select_target([r], 'nearest', order=['blue', 'red']) is r
    assert select_target([r, g], 'nearest', order=['blue']) is None


def test_blacklisted_positions_are_skipped():
    a = make_det('red', 0.20, 0.0)
    b = make_det('red', 0.35, 0.0)
    assert select_target([a, b], 'nearest', already_failed=[(0.21, 0.005)]) is b
    assert select_target([a, b], 'nearest', already_failed=[(0.2, 0.0), (0.35, 0.0)]) is None
    # blacklist applies before the label order
    assert select_target([a, b], 'nearest', order=['red'],
                         already_failed=[(0.2, 0.0)]) is b
    assert is_blacklisted(a, [(0.2, 0.029)], 0.03)
    assert not is_blacklisted(a, [(0.2, 0.031)], 0.03)


def test_isolated_prefers_free_closing_direction():
    # yaw = 0 -> closing axis along base y.
    crowded = make_det('red', 0.30, 0.00, yaw=0.0)
    blocker = make_det('blue', 0.30, 0.045, yaw=0.0)      # 4.5 cm along closing axis
    free = make_det('green', 0.30, -0.15, yaw=0.0)
    side = make_det('blue', 0.345, -0.15, yaw=0.0)        # neighbor off the finger band
    dets = [crowded, blocker, free, side]
    c_crowded, _ = clearances(crowded, dets)
    c_free, o_free = clearances(free, dets)
    assert c_crowded < 0.02
    assert c_free == pytest.approx(0.15 - 0.03) and o_free < 0.02
    assert select_target([crowded, free], 'isolated', neighbors=dets) is free
    alone, _ = clearances(make_det('red', 0.2, 0.2), [])
    assert alone == float('inf')
    lonely = make_det('red', 0.20, 0.20)
    assert select_target([crowded, lonely], 'isolated', neighbors=dets + [lonely]) is lonely


def test_invalid_inputs():
    a = make_det('red', 0.2, 0.0)
    bad = make_det('red', 0.1, 0.0)
    bad.center_base = np.array([float('nan'), 0.0, 0.03])
    none = make_det('red', 0.1, 0.0)
    none.center_base = None
    assert select_target([bad, none, a], 'nearest') is a
    assert select_target([], 'nearest') is None
    with pytest.raises(ValueError):
        select_target([a], 'random')

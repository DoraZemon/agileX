"""Runner loop fixes: observation failures, pushed objects, label cap, re-measure, slots."""
from __future__ import annotations

from cube_task.core.zones import parse_zones
import numpy as np
import pytest
from test_task_fakes import FakePerception, FakeWorld, install_fake_modules, make_det, Rig


@pytest.fixture(autouse=True)
def fake_modules(monkeypatch):
    install_fake_modules(monkeypatch)


L1_BLOCKS = [('red', 0.25, -0.10), ('green', 0.30, 0.05), ('blue', 0.38, 0.12)]


def _l1(rig, blocks=L1_BLOCKS):
    return [rig.world.add(make_det(lbl, x, y)) for lbl, x, y in blocks]


def _picks(rig):
    return [e for e in rig.logger.kinds('decision') if e['action'] == 'pick']


# ------------------------------------------------------------- observation failures
def test_camera_outage_mid_run_does_not_end_the_round():
    rig = Rig('L1')
    _l1(rig)
    state = {'n': 0}

    def on_place(zone, label):
        state['n'] += 1
        if state['n'] == 1:
            rig.camera.none_frames = 12             # about 40 s without frames
    rig.world.on_place = on_place
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 3, rep.stop_reason
    failed = [e for e in rig.logger.kinds('observation') if not e['ok']]
    assert len(failed) == 12 and all(e['reason'] == 'no frame' for e in failed)
    waits = [e['wait'] for e in rig.logger.kinds('decision') if e['action'] == 'observe_retry']
    assert waits[:4] == [0.5] * 4 and waits[5:] == [2.0] * 7      # back-off after 5
    assert not [e for e in rig.logger.kinds('decision') if e['action'] == 'nothing_to_pick']


def test_camera_dead_until_the_deadline_names_the_camera():
    rig = Rig('L1')
    _l1(rig)
    rig.world.on_place = lambda zone, label: setattr(rig.camera, 'none_frames', 10 ** 6)
    runner = rig.runner()
    rep = runner.run()
    assert rep.stop_reason == 'timeout' and rep.placed == 1 and not rep.ok
    assert runner.fault_detail.startswith('observation failures: no frame (')
    assert rig.logger.json['report.json']['fault_detail'] == runner.fault_detail


class FlakyPerception(FakePerception):
    def __init__(self, *a, fail_calls=(), **kw):
        super().__init__(*a, **kw)
        self.fail_calls = set(fail_calls)
        self.n = 0

    def process(self, frame, T_base_camera, zones=None):
        self.n += 1
        if self.n in self.fail_calls:
            raise RuntimeError('boom %d' % self.n)
        return super().process(frame, T_base_camera, zones)


def test_perception_errors_are_retried_not_counted_as_empty():
    rig = Rig('L1')
    _l1(rig)
    rig.perception = FlakyPerception(rig.clock, rig.world, fail_calls=range(2, 9))
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 3
    failed = [e for e in rig.logger.kinds('observation') if not e['ok']]
    assert len(failed) == 7 and failed[0]['reason'].startswith('perception error: boom 2')


def test_l3_observation_failures_are_retried():
    rig = Rig('L3')
    rig.world.add(make_det('apple', 0.25, -0.10, size=0.05, length=0.055))
    rig.camera.none_frames = 6
    rep = rig.runner(instructions=['把苹果放到蓝色托盘']).run()
    assert rep.ok and rep.placed == 1, [p.error for p in rep.picks]


def test_empty_ok_observations_still_end_the_round():
    rig = Rig('L1')
    rig.world.add(make_det('unknown', 0.25, -0.10))
    rep = rig.runner().run()
    assert rep.stop_reason == 'no_targets'
    assert len([e for e in rig.logger.kinds('decision')
                if e['action'] == 'nothing_to_pick']) == 4


# ---------------------------------------------------------------- pushed objects
class PushWorld(FakeWorld):
    """A missed grasp of label `push` pushes that object 3.5 cm toward the base."""

    def __init__(self, zones, push='red', step=(-0.035, 0.0)):
        super().__init__(zones)
        self.push, self.step = push, np.array(step + (0.0,))

    def try_grasp(self, T):
        x, y = T[0, 3], T[1, 3]
        for d in self.objects:
            if d.label == self.push and np.hypot(d.center_base[0] - x,
                                                 d.center_base[1] - y) < 0.01:
                d.center_base = d.center_base + self.step
                return None
        return super().try_grasp(T)


def test_object_pushed_by_failed_grasps_keeps_its_attempt_count():
    rig = Rig('L1')
    rig.world = PushWorld(parse_zones(rig.cfg))
    rig.arm.world = rig.world
    rig.perception.world = rig.world
    _l1(rig)
    rep = rig.runner().run()
    red = [p for p in rep.picks if p.label == 'red']
    assert len(red) == 3 and rep.placed == 2, [(p.label, p.stage) for p in rep.picks]
    assert [e['attempt'] for e in _picks(rig) if e['label'] == 'red'] == [1, 2, 3]
    moved = [e for e in rig.logger.kinds('decision') if e['action'] == 'track_moved']
    assert len(moved) >= 2
    assert rep.stop_reason == 'no_targets'


def test_label_failure_cap_tries_other_labels_first():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 4)
    _l1(rig, [('red', 0.24, 0.0), ('red', 0.22, -0.12), ('red', 0.26, 0.12),
              ('green', 0.40, 0.0)])
    rig.world.always_miss = {'red'}
    rep = rig.runner().run()
    assert [p.label for p in rep.picks] == ['red'] * 6 + ['green'] + ['red'] * 3
    assert rep.placed == 1 and rep.stop_reason == 'no_targets'


def test_two_same_label_objects_keep_separate_records():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 2)
    _l1(rig, [('red', 0.25, -0.03), ('red', 0.25, 0.03)])     # 6 cm apart, both visible
    rig.world.miss_next = 2
    rep = rig.runner().run()
    assert rep.placed == 2
    assert not [e for e in rig.logger.kinds('decision') if e['action'] == 'track_moved']


# ------------------------------------------------------------- re-measure
class WidePerception(FakePerception):
    """Report label `label` 7 cm wide (too wide for the gripper) on the given calls."""

    def __init__(self, *a, label='red', wide_calls=(1,), **kw):
        super().__init__(*a, **kw)
        self.label, self.wide_calls, self.n = label, set(wide_calls), 0

    def process(self, frame, T_base_camera, zones=None):
        self.n += 1
        res = super().process(frame, T_base_camera, zones)
        if self.n in self.wide_calls or 'all' in self.wide_calls:
            for d in res.detections:
                if d.label == self.label:
                    d.length = d.width = 0.07
        return res


def test_one_too_wide_frame_does_not_blacklist():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    _l1(rig, L1_BLOCKS[:1])
    rig.perception = WidePerception(rig.clock, rig.world, wide_calls=(1,))
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 1 and rep.attempts == 1
    dec = rig.logger.kinds('decision')
    assert [e['action'] for e in dec if e['action'] in ('remeasure', 'blacklist')] == [
        'remeasure']


def test_persistently_too_wide_object_is_blacklisted_after_two_observations():
    rig = Rig('L1')
    _l1(rig, L1_BLOCKS[:1])
    rig.perception = WidePerception(rig.clock, rig.world, wide_calls=('all',))
    rep = rig.runner().run()
    assert rep.attempts == 0 and rep.stop_reason == 'no_targets'
    acts = [e for e in rig.logger.kinds('decision') if e['action'] in ('remeasure', 'blacklist')]
    assert [e['action'] for e in acts] == ['remeasure', 'blacklist']
    assert acts[1]['reason'].startswith('grasp infeasible: too_wide')
    assert rig.perception.n >= 2


def test_unreachable_still_blacklists_at_once():
    rig = Rig('L1')
    _l1(rig)
    rig.arm.reach_fn = lambda T: ((False, 'ik failed')
                                  if np.linalg.norm(T[:2, 3] - [0.30, 0.05]) < 0.01
                                  else (True, ''))
    rep = rig.runner().run()
    assert rep.placed == 2
    assert not [e for e in rig.logger.kinds('decision') if e['action'] == 'remeasure']


# ---------------------------------------------------------------- release slots
def _l2_shared_zone_rig(zone_size=0.16):
    rig = Rig('L2')
    rig.cfg.set('zones.zone_red.size', [zone_size, zone_size])
    rig.world = FakeWorld(parse_zones(rig.cfg))
    rig.arm.world = rig.world
    rig.perception.world = rig.world
    rig.cfg.set('tasks.L2.required', 2)
    rig.world.add(make_det('apple', 0.22, -0.15, size=0.06, length=0.065, height=0.07))
    rig.world.add(make_det('orange', 0.35, 0.15, size=0.055, length=0.058, height=0.055))
    return rig


def test_shared_zone_spreads_releases_and_clears_earlier_objects():
    rig = _l2_shared_zone_rig()
    rep = rig.runner().run()
    assert rep.ok and sorted(rig.world.placed['zone_red']) == ['apple', 'orange']
    p1, p2 = _picks(rig)
    r1, r2 = np.array(p1['release_xyz']), np.array(p2['release_xyz'])
    center = np.array([0.20, 0.30])
    # Two labels share zone_red: no center-first; both inside the zone, not overlapping.
    assert np.linalg.norm(r1[:2] - center) > 0.03
    assert np.linalg.norm(r1[:2] - r2[:2]) >= (0.065 + 0.058) / 2.0
    for r in (r1, r2):
        assert np.all(np.abs(r[:2] - center) <= 0.08 - 0.029)
    # The second object's bottom (TCP minus held offset) clears the 7 cm apple by 1 cm.
    held2 = p2['grasp_xyz'][2] - 0.0
    assert r2[2] - held2 >= 0.07 + 0.01 - 1e-9


def test_spread_release_out_of_reach_falls_back_to_the_zone_center():
    rig = _l2_shared_zone_rig()
    center = np.array([0.20, 0.30])

    def reach(T):
        xy = T[:2, 3]
        off = np.linalg.norm(xy - center)
        return (False, 'ik failed') if 0.005 < off < 0.08 else (True, '')
    rig.arm.reach_fn = reach
    rep = rig.runner().run()
    assert rep.ok
    for e in _picks(rig):
        assert np.allclose(e['release_xyz'][:2], center)


def test_single_label_zones_release_at_the_center():
    rig = Rig('L1')
    _l1(rig)
    rep = rig.runner().run()
    assert rep.ok
    zones = parse_zones(rig.cfg)
    for e in _picks(rig):
        assert np.allclose(e['release_xyz'][:2], zones[e['zone']].center[:2])


def test_spread_releases_can_be_switched_off():
    rig = _l2_shared_zone_rig()
    rig.cfg.set('place.spread_releases', False)
    rep = rig.runner().run()
    assert rep.ok
    for e in _picks(rig):
        assert np.allclose(e['release_xyz'][:2], [0.20, 0.30])

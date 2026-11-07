"""Runner preparation fixes: eye_in_hand view check, held object, start lift, keys, L3 parse."""
from __future__ import annotations

import io
import os
import threading

from cube_task.core.clock import SimClock
from cube_task.core.transforms import make_T, top_down_T
from cube_task.core.types import GripperResult, MotionResult
from cube_task.task.operator import Operator
from cube_task.task.selfcheck import check_pick_area, self_check
import numpy as np
import pytest
from test_task_fakes import (FakeCamera, FakeExt, FakeSafeArm, install_fake_modules, make_cfg,
                             make_det, Rig)

#: camera (= flange in the fakes) looking along base +x, level: the table is out of view.
R_FORWARD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
LOOK_AWAY = make_T(R_FORWARD, (0.05, 0.25, 0.30))
LOOK_DOWN = top_down_T(0.30, 0.0, 0.50, 0.0)


@pytest.fixture(autouse=True)
def fake_modules(monkeypatch):
    install_fake_modules(monkeypatch)


class ObserveArm(FakeSafeArm):
    """FakeSafeArm whose observe move (move_joints) ends at a chosen flange pose."""

    def __init__(self, *a, observe_T=LOOK_DOWN, **kw):
        super().__init__(*a, **kw)
        self.observe_T = np.array(observe_T, dtype=float)

    def _motion(self, kind, target, linear):
        r = super()._motion(kind, target, linear)
        if r.ok and kind == 'move_joints':
            self.tcp = self.observe_T.copy()
        return r


class HoldingArm(FakeSafeArm):
    """FakeSafeArm that starts with the gripper closed on an object of width w."""

    def __init__(self, *a, w=0.035, **kw):
        super().__init__(*a, **kw)
        self.width = w

    def gripper(self, width, force=None, wait=True):
        if width < self.width:              # fingers stop on the held object
            self.commands.append(('gripper', float(width), force))
            return GripperResult(ok=True, width=self.width)
        return super().gripper(width, force, wait)


def _eih_rig(observe_T, n_blocks=1):
    rig = Rig('L1', camera__mount='eye_in_hand')
    rig.cfg.set('tasks.L1.required', n_blocks)
    rig.arm = ObserveArm(rig.clock, rig.world, observe_T=observe_T)
    rig.ext = FakeExt(np.eye(4), 'eye_in_hand')
    for lbl, x, y in [('red', 0.25, -0.10), ('green', 0.30, 0.05)][:n_blocks]:
        rig.world.add(make_det(lbl, x, y))
    return rig


# ------------------------------------------------------- eye_in_hand view check
def test_self_check_defers_eye_in_hand_pick_area_away_from_observe_pose():
    clock = SimClock()
    cfg = make_cfg('L1', camera__mount='eye_in_hand')
    arm = FakeSafeArm(clock)
    arm.tcp = LOOK_AWAY.copy()                       # rest pose: the wrist camera looks forward
    ext = FakeExt(np.eye(4), 'eye_in_hand')
    ok, msgs = self_check(cfg, 'L1', arm=arm, camera=FakeCamera(clock), mode='real',
                          extrinsics=ext, clock=clock)
    assert ok, msgs
    assert any(m.startswith('[WARN]') and '(eye_in_hand: checked at the observe pose)' in m
               for m in msgs), msgs
    # Already at the observe pose (joints within 2 deg): the failure is real.
    cfg.set('robot.poses.observe_joints', [0.0] * 6)
    ok, msgs = self_check(cfg, 'L1', arm=arm, camera=FakeCamera(clock), mode='real',
                          extrinsics=ext, clock=clock)
    assert not ok and any(m.startswith('[FAIL]') and 'pick polygon' in m for m in msgs), msgs


def test_self_check_eye_to_hand_pick_area_failure_stays_a_failure():
    clock = SimClock()
    cfg = make_cfg('L1')
    ext = FakeExt(LOOK_AWAY, 'eye_to_hand')
    ok, msgs = self_check(cfg, 'L1', arm=FakeSafeArm(clock), camera=FakeCamera(clock),
                          mode='real', extrinsics=ext, clock=clock)
    assert not ok and any(m.startswith('[FAIL]') and 'pick polygon' in m for m in msgs)


def test_check_pick_area_needs_a_frame_and_a_view():
    clock = SimClock()
    cfg = make_cfg('L1', camera__mount='eye_in_hand')
    arm = FakeSafeArm(clock)
    ext = FakeExt(np.eye(4), 'eye_in_hand')
    cam = FakeCamera(clock)
    arm.tcp = LOOK_DOWN.copy()
    ok, msg = check_pick_area(cfg, cam, ext, arm, newer_than=clock.now())
    assert ok and 'valid depth' in msg
    arm.tcp = LOOK_AWAY.copy()
    ok, msg = check_pick_area(cfg, cam, ext, arm)
    assert not ok and 'pick polygon' in msg
    cam.none_frames = 1
    ok, msg = check_pick_area(cfg, cam, ext, arm)
    assert not ok and 'no camera frame' in msg


def test_real_eye_in_hand_run_checks_the_view_at_the_observe_pose():
    rig = _eih_rig(LOOK_DOWN)
    rig.arm.tcp = LOOK_AWAY.copy()                   # starts at rest, looking away
    rep = rig.runner(mode='real').run()
    assert rep.ok and rep.placed == 1, (rep.stop_reason, rig.logger.kinds('refused'))
    view = rig.logger.kinds('view_check')
    assert len(view) == 1 and view[0]['ok']


def test_real_eye_in_hand_view_check_survives_one_missing_frame():
    rig = _eih_rig(LOOK_DOWN)

    def drop_next_frame(kind, T, lin, idx):
        if idx == 0:                                # the observe move: next frame is lost
            rig.camera.none_frames = 1
        return None
    rig.arm.fail_fn = drop_next_frame
    rep = rig.runner(mode='real').run()
    assert rep.ok, rig.logger.kinds('refused')
    assert rig.logger.kinds('view_check')[0]['ok']


def test_real_eye_in_hand_refuses_when_the_observe_pose_does_not_see_the_area():
    rig = _eih_rig(LOOK_AWAY)
    runner = rig.runner(mode='real')
    rep = runner.run()
    assert rep.stop_reason == 'fault' and rep.attempts == 0
    assert runner.fault_detail.startswith('pick area not visible from the observe pose: ')
    # Moved to the observe pose and opened the gripper; then held there.
    assert [c[0] for c in rig.arm.commands] == ['move_joints', 'gripper']


# ----------------------------------------------------------- held object at start
def test_refuses_when_the_gripper_holds_an_object_before_any_motion():
    rig = Rig('L1')
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.arm = HoldingArm(rig.clock, rig.world, w=0.035)
    runner = rig.runner()
    rep = runner.run()
    assert rep.stop_reason == 'config' and rep.attempts == 0
    assert runner.fault_detail.startswith('gripper is closed on something (0.035 m)')
    assert rig.arm.motions() == []
    assert rig.arm.gripper_cmds() == [pytest.approx(0.035 - 0.008)]   # the closing probe only


@pytest.mark.parametrize('width', [0.0, 0.055, 0.07])
def test_empty_gripper_at_any_width_starts(width):
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.arm.width = width
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 1
    probes = rig.logger.kinds('gripper_probe')
    assert len(probes) == (1 if width == 0.055 else 0)  # only the ambiguous width is probed


# ------------------------------------------------------------- lift before observe
def test_low_start_goes_straight_up_before_the_observe_move():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.arm.tcp = top_down_T(0.30, 0.05, 0.05, 0.3)      # stopped low after a fault
    rep = rig.runner().run()
    assert rep.ok
    m = rig.arm.motions()
    assert m[0][0] == 'move_tcp' and m[0][2] is True            # linear
    assert np.allclose(m[0][1][:3, 3], [0.30, 0.05, 0.15])     # transit height, same xy
    assert np.allclose(m[0][1][:3, :3], top_down_T(0, 0, 0, 0.3)[:3, :3])
    assert m[1][0] == 'move_joints'


def test_high_start_and_refused_lift_go_to_observe_directly():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.arm.reach_fn = lambda T: ((False, 'ik failed')
                                  if np.linalg.norm(T[:2, 3] - [0.30, 0.05]) < 1e-6
                                  else (True, ''))
    rig.arm.tcp = top_down_T(0.30, 0.05, 0.05, 0.0)
    rep = rig.runner().run()
    assert rep.ok and rig.arm.motions()[0][0] == 'move_joints'
    assert rig.logger.kinds('start_lift')[0]['ok'] is False


def test_failed_start_lift_is_a_fault():
    rig = Rig('L1')
    rig.world.add(make_det('red', 0.25, -0.10))
    rig.arm.tcp = top_down_T(0.30, 0.05, 0.05, 0.0)
    rig.arm.fail_fn = lambda kind, T, lin, idx: MotionResult(ok=False, error='comm')
    runner = rig.runner()
    rep = runner.run()
    assert rep.stop_reason == 'fault'
    assert runner.fault_detail == 'cannot go up to the transit height: comm'
    assert len(rig.arm.motions()) == 1


# ------------------------------------------------------------------ operator keys
class _PromptOut:
    """Output stream that types Enter into the pipe once the start prompt is shown."""

    def __init__(self, op_ref, wfd):
        self.op_ref, self.wfd = op_ref, wfd
        self.cleared_at_prompt = None
        self.text = []

    def write(self, msg):
        self.text.append(msg)
        if 'Press Enter' in msg:
            self.cleared_at_prompt = not self.op_ref[0].start_event.is_set()
            os.write(self.wfd, b'\n')

    def flush(self):
        pass


def _pipe_operator(out=None):
    r, w = os.pipe()
    rf = os.fdopen(r, 'r')
    return Operator(enabled=True, stdin=rf, out=out), rf, w


@pytest.mark.parametrize('key', [b'p', b'e'])
def test_keys_work_during_the_first_observe_move(key):
    rig = Rig('L1')
    rig.world.add(make_det('red', 0.25, -0.10))
    op, rf, w = _pipe_operator(out=io.StringIO())
    seen = {}

    def fn(kind, T, lin, idx):
        if idx == 0:                                    # the move to the observe pose
            os.write(w, key)
            seen['abort'] = op.abort_event.wait(2.0)
            return MotionResult(ok=False, error='aborted')
        return None
    rig.arm.fail_fn = fn
    try:
        rep = rig.runner(operator=op).run()
    finally:
        op.close()
        rf.close()
        os.close(w)
    assert seen['abort'], 'the key reader must run during the first motion'
    assert rep.stop_reason == 'aborted' and rep.attempts == 0
    assert [c[0] for c in rig.arm.commands] == ['move_joints'] + (
        ['estop'] if key == b'e' else [])
    assert rig.arm.estopped == (key == b'e')


def test_enter_typed_during_the_observe_move_does_not_start_the_round():
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.world.add(make_det('red', 0.25, -0.10))
    ref = [None]
    r, w = os.pipe()
    rf = os.fdopen(r, 'r')
    out = _PromptOut(ref, w)
    op = Operator(enabled=True, stdin=rf, out=out)
    ref[0] = op
    early = threading.Event()

    def fn(kind, T, lin, idx):
        if idx == 0:
            os.write(w, b'\n')
            if op.start_event.wait(2.0):
                early.set()
        return None
    rig.arm.fail_fn = fn
    try:
        rep = rig.runner(operator=op).run()
    finally:
        op.close()
        rf.close()
        os.close(w)
    assert early.is_set(), 'the reader saw the early Enter during the move'
    assert out.cleared_at_prompt is True, 'the early Enter must be discarded at the prompt'
    assert rep.ok and rep.placed == 1


# ---------------------------------------------------------------- L3 parse timing
def test_l3_instructions_are_parsed_before_the_timer_starts(monkeypatch):
    import cube_task.task.runner as runner_mod
    rig = Rig('L3')
    rig.world.add(make_det('apple', 0.25, -0.10, size=0.05, length=0.055))
    rig.world.add(make_det('carrot', 0.22, 0.05, size=0.03, length=0.14))
    calls = []
    real = runner_mod.parse_instruction

    def recording(text, cfg, labels=None, zones=None):
        calls.append(rig.clock.now())
        return real(text, cfg, labels, zones)
    monkeypatch.setattr(runner_mod, 'parse_instruction', recording)
    texts = ['把苹果放到蓝色托盘', '把西瓜放到红色托盘', '把胡萝卜放到绿色区域']
    rep = rig.runner(instructions=texts).run()
    t0 = rig.logger.kinds('start')[0]['t0']
    assert len(calls) == 3 and max(calls) <= t0
    assert [p.ok for p in rep.picks] == [True, False, True]
    assert len(rig.logger.kinds('instruction')) == 3       # still logged per instruction
    assert len(rig.logger.kinds('instructions_parsed')[0]['specs']) == 3

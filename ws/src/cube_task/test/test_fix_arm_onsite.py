"""
Tests for the drag-teach exit guards and helpers (review findings arm-1 / arm-2).

Covers tools/onsite.py (exit_teach_mode_confirmed refuses a raised arm, the
c = take CAN control branch of ensure_motion_ready, warning texts), the
firmware / pose helpers of robot/pyagxarm_arm.py and check_setup._arm_fix.
No hardware: a scripted fake arm in drag-teach mode records every call.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.types import ArmHealth
from cube_task.robot import pyagxarm_arm as pa
from cube_task.tools import check_setup, onsite
from cube_task.tools.ui import Ui

RAISED = [0.1, 1.2, -0.9, 0.0, 0.8, 0.0]
FOLDED = [0.02, 0.0, -0.03, 0.01, 0.04, -0.049]


class Script:
    """input_fn for Ui: returns the answers in order, records the prompts, then EOF."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class TeachArm:
    """Fake real arm in drag-teach mode; enable() switches it to CAN control if seamless."""

    def __init__(self, joints, firmware=None, seamless=True):
        self.joints = list(joints)
        self.teaching = True
        self.seamless = seamless
        self.calls = []
        if firmware is not None:
            self.firmware_info = {'software_version': firmware, 'hardware_version': 'H-V1.2-1'}

    def health(self):
        if self.teaching:
            return ArmHealth(ok=False, reason='controller in TEACHING_MODE mode', comm_ok=True,
                             enabled=True, ctrl_mode=onsite.CTRL_TEACHING)
        return ArmHealth(ok=True, comm_ok=True, enabled=True, ctrl_mode=1)

    def read_teach_pose(self):
        T = np.eye(4).tolist()
        return {'joints': list(self.joints), 'T_base_flange': T, 'T_base_tcp': T,
                'ctrl_mode': 2 if self.teaching else 1, 'teach_status': 0, 'age_s': 0.0}

    def enable(self, timeout=5.0):
        self.calls.append('enable')
        if self.seamless:
            self.teaching = False
        return True

    def exit_teach_mode(self):
        self.calls.append('exit_teach_mode')
        self.teaching = False
        return True

    def _motion(self, name):
        self.calls.append(name)
        raise AssertionError('%s must not be called in drag-teach mode' % name)

    def move_tcp(self, *_a, **_k):
        self._motion('move_tcp')

    def move_joints(self, *_a, **_k):
        self._motion('move_joints')

    def stop(self):
        self._motion('stop')


class Wrapped:
    """SafeArm-like wrapper: firmware_info lives on .inner."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _ui(answers):
    out = io.StringIO()
    script = Script(answers)
    return Ui(headless=True, input_fn=script, output=out), out, script


# ------------------------------------------------------------------ exit_teach_mode
def test_exit_teach_mode_refused_for_a_raised_arm_without_asking():
    arm = TeachArm(RAISED)
    ui, out, script = _ui(['EXIT'])
    assert onsite.exit_teach_mode_confirmed(ui, arm) is False
    assert arm.calls == [] and script.prompts == []          # no EXIT prompt at all
    text = out.getvalue()
    assert 'exit_teach_mode refused: not folded at joint zero' in text
    assert 'joint2 68.8 deg' in text and 'joint1 5.7 deg' in text
    assert 'drag the arm folded onto its support' in text


def test_exit_teach_mode_refused_without_joint_feedback():
    arm = TeachArm(FOLDED)
    arm.read_teach_pose = lambda: None
    ui, out, script = _ui(['EXIT'])
    assert onsite.exit_teach_mode_confirmed(ui, arm) is False
    assert arm.calls == [] and 'no fresh joint feedback' in out.getvalue()


@pytest.mark.parametrize('word, ran', [('EXIT', True), ('exit', False), ('y', False)])
def test_exit_teach_mode_runs_only_folded_and_after_the_typed_word(word, ran):
    arm = TeachArm(FOLDED)
    ui, out, script = _ui([word])
    assert onsite.exit_teach_mode_confirmed(ui, arm) is ran
    assert arm.calls == (['exit_teach_mode'] if ran else [])
    assert 'resting folded on its support and is nobody touching it' in script.prompts[0]
    text = out.getvalue()
    assert onsite.EXIT_TEACH_WARNING in text and onsite.EXIT_TEACH_WARNING_ZH in text


def test_exit_warning_states_the_real_order_and_no_holding():
    en, zh = onsite.EXIT_TEACH_WARNING, onsite.EXIT_TEACH_WARNING_ZH
    steps = ['move_js', 'e-stop', 'reset()', 'enable', 'move_j to joint zero']
    pos = [en.index(s) for s in steps]
    assert pos == sorted(pos)
    assert 'hands' in en.lower() and 'off the arm' in en
    for bad in ('held', 'holds', 'person'):
        assert bad not in en
    assert '托住' not in zh and '手不要碰' in zh
    zh_steps = ['move_js', '电子急停', 'reset()', '重新使能', 'move_j 到全零位']
    zpos = [zh.index(s) for s in zh_steps]
    assert zpos == sorted(zpos)


def test_teach_mode_reminder_x_is_refused_for_a_raised_arm():
    arm = TeachArm(RAISED)
    ui, out, _s = _ui(['x'])
    onsite.teach_mode_reminder(ui, arm)
    assert arm.calls == [] and 'exit_teach_mode refused' in out.getvalue()
    assert 'c = take CAN control' in out.getvalue()


# ------------------------------------------------------------------ c = take CAN control
def test_c_takes_can_control_with_enable_only():
    arm = Wrapped(TeachArm(RAISED, firmware='S-V1.8-5'))
    ui, out, script = _ui(['c', 'CAN'])
    assert onsite.ensure_motion_ready(ui, arm, purpose='move above the objects') is True
    assert arm.inner.calls == ['enable']
    text = out.getvalue()
    assert 'firmware S-V1.8-5 (S-V1.8-5 or later) switches seamlessly' in text
    assert 'NO motion command' in text and 'CAN control taken' in text
    first = script.prompts[0]
    assert first.startswith('Press the teach button on the arm')       # still the first advice
    assert first.index('c = take CAN control') < first.index('x = exit_teach_mode')
    assert 'Type CAN' in script.prompts[1]


def test_c_needs_the_typed_word():
    arm = TeachArm(RAISED, firmware='S-V1.8-8')
    ui, out, _s = _ui(['c', 'yes', 'q'])
    assert onsite.ensure_motion_ready(ui, arm) is False
    assert arm.calls == [] and 'CAN control not requested' in out.getvalue()


def test_c_on_old_firmware_explains_the_reset_path():
    arm = TeachArm(RAISED, firmware='S-V1.8-3', seamless=False)
    ui, out, _s = _ui(['c', 'CAN', 'q'])
    assert onsite.ensure_motion_ready(ui, arm) is False
    assert arm.calls == ['enable']
    text = out.getvalue()
    assert 'still not ready after enable' in text
    assert 'firmware S-V1.8-3 is below S-V1.8-5, only a reset ends TEACHING_MODE' in text
    assert 'option x' in text


def test_x_in_ensure_motion_ready_is_refused_for_a_raised_arm():
    arm = TeachArm(RAISED)
    ui, out, script = _ui(['x', 'q'])
    assert onsite.ensure_motion_ready(ui, arm) is False
    assert arm.calls == [] and 'exit_teach_mode refused' in out.getvalue()
    assert not any('Type EXIT' in p for p in script.prompts)


def test_unknown_firmware_gets_both_ways_out_once():
    arm = TeachArm(RAISED)
    ui, out, _s = _ui(['', '', 'q'])
    assert onsite.ensure_motion_ready(ui, arm) is False
    text = out.getvalue()
    assert text.count('If it stays in drag-teach mode after the teach button') == 1
    assert 'firmware S-V1.8-5 and later: take CAN control' in text
    assert 'older firmware: while drag mode is still on' in text
    assert arm.calls == []


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize('version, expect', [
    ('S-V1.8-5', True), ('S-V1.8-8', True), (' S-V1.8-9 ', True), ('S-V2.0-0', True),
    ('S-V1.8-4', False), ('S-V1.8-3', False), ('S-V1.7-9', False), ('S-V1.6-3', False),
    (None, None), ('', None), ('1.20', None), ('S-V1.8', None)])
def test_seamless_teach_switch_follows_the_official_node(version, expect):
    assert pa.seamless_teach_switch(version) is expect


def test_zero_pose_problem():
    assert pa.zero_pose_problem(FOLDED) == ''
    assert pa.zero_pose_problem([0.0] * 5) == 'need 6 finite joint angles'
    assert pa.zero_pose_problem([0.0, 0.0, float('nan'), 0, 0, 0]) == 'need 6 finite joint angles'
    msg = pa.zero_pose_problem([0, 0, 0, 0, 0, -0.051])
    assert msg == 'not folded at joint zero: joint6 -2.9 deg (limit 2.9 deg)'


def test_canonical_pose6():
    cur = [0.3, 0.0, 0.2, -3.1, 0.1, -3.0]
    out = pa.canonical_pose6([0.3, 0.0, 0.1, math.pi, 0.0, math.pi - 5e-7], cur)
    assert out[3] == -math.pi and out[5] == -math.pi
    assert out[:3] == [0.3, 0.0, 0.1] and out[4] == 0.0
    # same side, not near the wrap, or no feedback: unchanged
    assert pa.canonical_pose6([0, 0, 0, -math.pi, 0, 1.0], cur)[3] == -math.pi
    assert pa.canonical_pose6([0, 0, 0, 3.0, 0, 3.1], cur)[3:] == [3.0, 0.0, 3.1]
    assert pa.canonical_pose6([0, 0, 0, math.pi, 0, 0], None)[3] == math.pi
    assert pa.canonical_pose6([0, 0, 0, math.pi, 0, 0], [0.0] * 3)[3] == math.pi
    assert pa.canonical_pose6([0, 0, 0, -math.pi, 0, 0], [0, 0, 0, 0.5, 0, 0])[3] == math.pi


@pytest.mark.parametrize('value, expect', [(None, True), (True, True), (False, False),
                                           ('off', False), ('Yes', True), (0, False)])
def test_linear_moves_enabled(value, expect):
    cfg = load_config(use_overlays=False)
    if value is not None:
        cfg.set('robot.motion.linear_moves', value)
    assert pa.linear_moves_enabled(cfg) is expect


def test_linear_moves_enabled_rejects_junk():
    cfg = load_config(use_overlays=False)
    cfg.set('robot.motion.linear_moves', 'maybe')
    with pytest.raises(ValueError, match='linear_moves'):
        pa.linear_moves_enabled(cfg)


def test_arm_firmware_version_reads_the_wrapped_arm():
    assert onsite.arm_firmware_version(Wrapped(TeachArm(RAISED, 'S-V1.8-4'))) == 'S-V1.8-4'
    assert onsite.arm_firmware_version(TeachArm(RAISED)) is None


@pytest.mark.parametrize('version, part', [
    ('S-V1.8-8', 'switches seamlessly: take CAN control with option c'),
    ('S-V1.8-2', 'below S-V1.8-5, only a reset ends TEACHING_MODE'),
    (None, 'firmware S-V1.8-5 and later: take CAN control')])
def test_check_setup_teach_fix_depends_on_the_firmware(version, part):
    fix = check_setup._arm_fix('controller in TEACHING_MODE mode (drag-teach active)', version)
    assert fix.startswith('press the teach button on the arm again')
    assert part in fix
    assert check_setup._arm_fix('controller in TEACHING_MODE mode') == check_setup._arm_fix(
        'controller in TEACHING_MODE mode', None)
    assert 'CAN link' in check_setup._arm_fix('no joint angles feedback')

"""Review fixes in teach_poses: 'd <name>' deletes a zone; joint readings near a limit."""
from __future__ import annotations

import copy
import io
import math

import numpy as np
import pytest
import yaml

from cube_task.robot.kinematics import PiperKinematics
from cube_task.tools import teach_poses as tp
from cube_task.tools.onsite import ArmPose
from cube_task.tools.ui import Ui

KIN = PiperKinematics('piper')


def run_sim(tmp_path, answers, extra=()):
    out = io.StringIO()
    feed = list(answers)

    def fn(_prompt):
        if not feed:
            raise EOFError
        return feed.pop(0)

    ui = Ui(headless=True, preview_dir=str(tmp_path / 'preview'), input_fn=fn, output=out)
    rc = tp.main(['--mode', 'sim', '--task', 'L1', '--headless', '--log-dir', str(tmp_path)]
                 + list(extra), ui=ui)
    return rc, out.getvalue()


def poses_file(tmp_path):
    return tmp_path / 'sim_site' / 'poses.yaml'


def written(tmp_path):
    return yaml.safe_load(poses_file(tmp_path).read_text(encoding='utf-8'))


TEACH_TWO = ['z zone_red', '', '', 'n', 'z zone_green', '', '', 'n', 'w', 'q']


# ------------------------------------------------------------------ [onsite] arm-4: delete
def test_apply_changes_delete_marker_and_pop_dotted():
    assert copy.deepcopy(tp.DELETE) is tp.DELETE and copy.copy(tp.DELETE) is tp.DELETE
    existing = {'zones': {'a': {'center': [0.1, 0.2, 0.0]}, 'b': {'center': [0.3, 0.3, 0.0]}},
                'robot': {'poses': {'home_joints': [0.0] * 6}}}
    data = tp.apply_changes(existing, {'zones.a': tp.DELETE})
    assert set(data['zones']) == {'b'} and set(existing['zones']) == {'a', 'b'}
    data = tp.apply_changes(data, {'zones.b': tp.DELETE, 'zones.nope': tp.DELETE})
    assert 'zones' not in data and data['robot'] == existing['robot']   # empty mapping pruned
    # Changes apply in order: delete then re-teach keeps only the new keys.
    data = tp.apply_changes(existing, {'zones.a': tp.DELETE, 'zones.a.size': [0.2, 0.2]})
    assert data['zones']['a'] == {'size': [0.2, 0.2]}
    d = {'x': {'y': {'z': 1}, 'k': 2}}
    assert tp.pop_dotted(d, 'x.y.z') and d == {'x': {'k': 2}}
    assert not tp.pop_dotted(d, 'x.y.z') and not tp.pop_dotted(d, 'x.k.q')
    assert not tp.pop_dotted(d, 'nope')


def test_zone_sources_and_zone_users(tmp_path):
    default = tmp_path / 'default.yaml'
    default.write_text('zones: {}\n', encoding='utf-8')
    other = tmp_path / 'other.yaml'
    other.write_text('zones: {veg_bin: {center: [0.3, 0.2, 0.0]}}\n', encoding='utf-8')
    poses = tmp_path / 'poses.yaml'
    poses.write_text('zones: {veg_bin: {center: [0.3, 0.2, 0.0]}}\n', encoding='utf-8')
    broken = tmp_path / 'broken.yaml'
    broken.write_text('zones: [unclosed\n', encoding='utf-8')
    sources = [str(default), str(other), str(poses), str(broken), 'sim:L1:0',
               'override:zones.veg_bin.size=[0.2, 0.2]', 'override:zones={veg_bin: {}}',
               'override:zones={fruit_bin: {}}', 'override:robot.speed_percent=20']
    assert tp.zone_sources(sources, 'veg_bin', poses) == [
        str(other), '--set zones.veg_bin.size', '--set zones']
    assert tp.zone_sources(sources[:3], 'fruit_bin', poses) == []
    tasks = {'L1': {'class_to_zone': {'red': 'zone_red'}},
             'L2': {'class_to_zone': {'carrot': 'veg_bin', 'corn': 'veg_bin', 'apple': 'f'}},
             'L3': None}
    assert tp.zone_users(tasks, 'veg_bin') == ['tasks.L2.class_to_zone (carrot, corn)']
    assert tp.zone_users(tasks, 'zone_x') == [] and tp.zone_users(None, 'a') == []


def test_format_state_lists_pending_deletions():
    view = tp.apply_changes({'zones': {'a': {'center': [0.2, 0.2, 0.0]}}}, {})
    text = tp.format_state(view, {'zones.b': tp.DELETE}, None)
    assert 'deleted, not written yet: zones.b' in text and '1 unsaved change(s)' in text


def test_delete_writes_through_the_poses_file_and_warns(tmp_path):
    assert run_sim(tmp_path, TEACH_TWO)[0] == 0
    assert set(written(tmp_path)['zones']) == {'zone_red', 'zone_green'}
    rc, out = run_sim(tmp_path, ['d zone_green', 'l', 'w', 'q'])
    assert rc == 0
    data = written(tmp_path)
    assert set(data['zones']) == {'zone_red'}
    assert 'recorded delete zone zone_green' in out
    assert 'deleted, not written yet: zones.zone_green' in out
    assert 'keys deleted:\n    zones.zone_green' in out
    assert ('WARNING: tasks.L1.class_to_zone (green) still sends objects to zone_green'
            in out)
    # The sim scenario still defines zone_green: the merge cannot remove it.
    assert 'WARNING: zone zone_green is also defined in the merged configuration' in out
    assert list((tmp_path / 'sim_site').glob('poses.bak_*.yaml'))


def test_delete_undo_unknown_and_prompted_name(tmp_path):
    assert run_sim(tmp_path, TEACH_TWO)[0] == 0
    before = written(tmp_path)
    rc, out = run_sim(tmp_path, ['d zone_red', 'u', 'w', 'd zone_nowhere', 'd', 'q', 'q'])
    assert rc == 0 and written(tmp_path) == before
    assert 'undone: delete zone zone_red' in out
    assert 'nothing to write: no unsaved changes' in out
    assert 'no zone zone_nowhere in' in out and 'nothing deleted' in out


def test_delete_then_reteach_in_one_session(tmp_path):
    assert run_sim(tmp_path, TEACH_TWO)[0] == 0
    rc, out = run_sim(tmp_path, ['d zone_red', 'z zone_red', '0.12 0.12', '', 'n', 'w', 'q'])
    assert rc == 0
    z = written(tmp_path)['zones']['zone_red']
    assert z['size'] == [0.12, 0.12] and len(z['center']) == 3
    assert 'zone_green' in written(tmp_path)['zones']


@pytest.mark.parametrize('merged', [False, True])
def test_delete_a_zone_only_the_poses_file_has(tmp_path, merged):
    poses_file(tmp_path).parent.mkdir(parents=True)
    poses_file(tmp_path).write_text(yaml.safe_dump({'zones': {'old_bin': {
        'center': [0.30, 0.30, 0.0], 'size': [0.10, 0.10]}}}), encoding='utf-8')
    # merged: the poses file is also merged into the loaded config, like the real site overlay;
    # the checks after w must then drop the deleted zone from that config too.
    extra = ['--config', str(poses_file(tmp_path))] if merged else []
    rc, out = run_sim(tmp_path, ['d old_bin', 'w', 'q'], extra=extra)
    assert rc == 0
    assert 'zones' not in (written(tmp_path) or {})
    assert 'also defined' not in out and 'still sends' not in out
    after = out.split('keys deleted:', 1)[1]
    assert 'zone reachability' in after and 'old_bin' not in after.split('\n', 2)[2]


def test_delete_warns_when_another_file_defines_the_zone(tmp_path):
    assert run_sim(tmp_path, TEACH_TWO)[0] == 0
    extra = tmp_path / 'extra.yaml'
    extra.write_text('zones: {zone_red: {center: [0.25, 0.25, 0.0]}}\n', encoding='utf-8')
    rc, out = run_sim(tmp_path, ['d zone_red', 'w', 'q'], extra=['--config', str(extra)])
    assert rc == 0 and 'zone_red' not in written(tmp_path)['zones']
    assert 'WARNING: zone zone_red is also defined in %s' % extra in out
    # A zone only in the other file cannot be deleted here.
    rc, out = run_sim(tmp_path, ['d zone_red', 'q'], extra=['--config', str(extra)])
    assert 'zone zone_red is not in' in out and 'remove it there' in out


# ------------------------------------------------------------------ [onsite] arm-6: limits
def test_clip_to_limits_and_degree_messages():
    q = [0.0, -0.0017, 0.001, 0.0, 0.0, 0.0]          # folded rest pose read by the encoders
    assert tp.joints_problem(q, KIN).startswith(
        'outside the joint limits: joint2 = -0.097 deg < 0.000 deg, '
        'joint3 = 0.057 deg > 0.000 deg')
    assert 're-teach it' in tp.joints_problem(q, KIN)
    clipped, notes = tp.clip_to_limits(q, KIN)
    assert clipped == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] and len(notes) == 2
    assert notes[0].startswith('joint2 = -0.097 deg is 0.097 deg outside its limit: recorded '
                               'as 0.000 deg')
    assert tp.joints_problem(clipped, KIN) == ''
    far = [0.0, -0.02, 0.0, 0.0, 0.0, 0.0]               # 1.146 deg outside: not clipped
    assert tp.clip_to_limits(far, KIN) == (far, [])
    why = tp.joints_problem(far, KIN)
    assert 'joint2 = -1.146 deg < 0.000 deg' in why and 're-teach' not in why
    hi = KIN.limits[4, 1]
    clipped, _n = tp.clip_to_limits([0.0, 1.0, -1.0, 0.0, hi + 0.009, 0.0], KIN)
    assert clipped[4] == hi
    problems = tp.validate_poses({'robot': {'poses': {'home_joints': far}}}, KIN)
    assert problems == ['robot.poses.home_joints ' + why]
    assert tp.clip_to_limits([0.0] * 5, KIN) == ([0.0] * 5, [])


def test_home_pose_read_just_outside_the_limits_is_recorded_clipped(tmp_path, monkeypatch):
    real = tp.read_pose_averaged

    def reading(arm, clock, n, dt):
        pose, dev = real(arm, clock, n, dt)
        q = np.array([0.0, -0.0017, 0.0009, 0.0, 0.0, 0.0])
        return ArmPose(q, pose.T_base_flange, pose.T_base_tcp), dev

    monkeypatch.setattr(tp, 'read_pose_averaged', reading)
    rc, out = run_sim(tmp_path, ['h', 'w', 'q'])
    assert rc == 0
    assert written(tmp_path)['robot']['poses']['home_joints'] == [0.0] * 6
    assert 'note: joint2 = -0.097 deg is 0.097 deg outside its limit' in out
    assert 'home pose not recorded' not in out


@pytest.mark.parametrize('j2', [-0.05, math.radians(-3.0)])
def test_home_pose_far_outside_the_limits_is_refused(tmp_path, monkeypatch, j2):
    real = tp.read_pose_averaged

    def reading(arm, clock, n, dt):
        pose, dev = real(arm, clock, n, dt)
        return ArmPose(np.array([0.0, j2, 0.0, 0.0, 0.0, 0.0]), pose.T_base_flange,
                       pose.T_base_tcp), dev

    monkeypatch.setattr(tp, 'read_pose_averaged', reading)
    rc, out = run_sim(tmp_path, ['h', 'q'])
    assert rc == 0 and not poses_file(tmp_path).exists()
    assert 'home pose not recorded: the joints are outside the joint limits: joint2 = ' in out

"""teach_poses: pure helpers (zones, pick polygon, gripper view, checks) and scripted runs."""
from __future__ import annotations

import io
import math

from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.transforms import top_down_T
from cube_task.core.types import ArmHealth
from cube_task.core.zones import parse_zone
from cube_task.robot.kinematics import flange_to_tcp_from_cfg, PiperKinematics
from cube_task.robot.safety import SafeArm, SafetyGuard
from cube_task.robot.sim_arm import SimArm
from cube_task.sim.scenarios import EYE_TO_HAND_PARK_JOINTS, make_scenario, scenario_config_overlay
from cube_task.tools import common, onsite
from cube_task.tools import teach_poses as tp
from cube_task.tools.ui import Ui
import numpy as np
import pytest
import yaml

WS = {'x': [0.08, 0.60], 'y': [-0.40, 0.40], 'z': [-0.05, 0.50], 'r_min': 0.10, 'r_max': 0.62}
SQUARE = [[0.15, -0.22], [0.45, -0.22], [0.45, 0.22], [0.15, 0.22]]


def sim_cfg(task='L1', mount='eye_to_hand'):
    base = load_config(use_overlays=False)
    world, info = make_scenario(task, 0, mount=mount)
    cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, [])
    return cfg, world, info


class Script:
    """Scripted operator input for Ui(input_fn=...): the answers in order, then EOF."""

    def __init__(self, answers, on_prompt=None):
        self.answers = list(answers)
        self.prompts = []
        self.on_prompt = on_prompt

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.on_prompt is not None:
            self.on_prompt(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def make_test_ui(tmp_path, answers, on_prompt=None):
    out = io.StringIO()
    script = Script(answers, on_prompt)
    ui = Ui(headless=True, preview_dir=str(tmp_path / 'preview'), input_fn=script, output=out)
    return ui, out, script


def run_sim(tmp_path, answers, task='L1'):
    ui, out, script = make_test_ui(tmp_path, answers)
    rc = tp.main(['--mode', 'sim', '--task', task, '--headless', '--log-dir', str(tmp_path)],
                 ui=ui)
    return rc, out.getvalue(), script


def written(tmp_path):
    return yaml.safe_load((tmp_path / 'sim_site' / 'poses.yaml').read_text(encoding='utf-8'))


def assert_zones_match(zones, truth, tol=1e-3):
    assert set(zones) == set(truth)
    for name, z in truth.items():
        got = zones[name]
        assert np.allclose(got['center'], z.center, atol=tol), name
        assert np.allclose(got['size'], z.size, atol=tol), name
        assert abs(got['wall_height'] - z.wall_height) <= tol, name
        assert abs(tp.wrap_half_pi(got['yaw'] - z.yaw)) <= tol, name


# ------------------------------------------------------------------ zones
def test_zone_from_touches_center_size_wall():
    z = tp.zone_from_touches([0.21, 0.29, 0.0123456], (0.12, 0.10), 0.05)
    assert z == {'center': [0.21, 0.29, 0.0123], 'size': [0.12, 0.1], 'yaw': 0.0,
                 'wall_height': 0.05}
    assert all(type(v) is float for v in z['center'] + z['size'])
    assert yaml.safe_load(yaml.safe_dump(z)) == z
    assert set(tp.zone_changes('zone_red', z)) == {
        'zones.zone_red.center', 'zones.zone_red.size', 'zones.zone_red.yaw',
        'zones.zone_red.wall_height'}


@pytest.mark.parametrize('deg, expect', [(30, 30), (210, 30), (100, -80), (-90, -90),
                                         (90, -90), (180, 0), (-20, -20)])
def test_zone_yaw_from_a_second_touch(deg, expect):
    c = [0.2, 0.3, 0.0]
    a = math.radians(deg)
    p = [c[0] + 0.05 * math.cos(a), c[1] + 0.05 * math.sin(a), 0.001]
    z = tp.zone_from_touches(c, axis_point=p)
    assert math.degrees(z['yaw']) == pytest.approx(expect, abs=0.01)
    assert -math.pi / 2 - 1e-4 <= z['yaw'] < math.pi / 2


def test_zone_from_touches_keeps_a_given_yaw_and_rejects_bad_input():
    assert tp.zone_from_touches([0.2, 0.3, 0.0], yaw=0.5)['yaw'] == 0.5
    with pytest.raises(ValueError, match='at least'):
        tp.zone_from_touches([0.2, 0.3, 0.0], axis_point=[0.21, 0.3, 0.0])
    with pytest.raises(ValueError, match='meters'):
        tp.zone_from_touches([0.2, 0.3, 0.0], size=(100, 100))
    with pytest.raises(ValueError, match='wall'):
        tp.zone_from_touches([0.2, 0.3, 0.0], wall_height=-0.01)
    with pytest.raises(ValueError, match='center'):
        tp.zone_from_touches([0.2, float('nan'), 0.0])


def test_zone_names_suggestions_and_validation():
    tasks = {'L1': {'class_to_zone': {'red': 'zone_red', 'green': 'zone_green',
                                      'blue': 'zone_red'}},
             'L2': {'class_to_zone': {'apple': 'fruit_bin'}}, 'L3': {'class_to_zone': {}},
             'x': None}
    assert tp.task_zone_names(tasks) == ['zone_red', 'zone_green', 'fruit_bin']
    assert tp.suggest_zone_names(tasks, ['zone_red']) == ['zone_green', 'fruit_bin']
    assert tp.task_zone_names(None) == []
    assert tp.zone_name_problem('zone_red') == '' and tp.zone_name_problem('Bin-2') == ''
    for bad in ('', '1zone', 'a.b', 'zone red', 'x' * 41):
        assert tp.zone_name_problem(bad), bad


def test_zone_warnings():
    z = tp.zone_from_touches([0.30, 0.0, 0.05])
    w = tp.zone_warnings('z1', z, WS, SQUARE, 0.0)
    assert any('inside the pick polygon' in s for s in w)
    assert any('did the fingertips touch the zone floor' in s for s in w)
    far = tp.zone_from_touches([0.70, 0.30, 0.0])
    assert any('outside the workspace' in s for s in tp.zone_warnings('z2', far, WS))
    edge = tp.zone_from_touches([0.21, 0.25, 0.0])
    assert tp.zone_warnings('z3', edge, WS, SQUARE, 0.0) == ['zone z3 overlaps the pick polygon']
    assert tp.zone_warnings('ok', tp.zone_from_touches([0.21, 0.29, 0.0]), WS, SQUARE, 0.0) == []
    assert tp.zone_warnings('bad', edge, WS, [[0, 0], 'x'], 'nan') == []


# ------------------------------------------------------------------ pick polygon
def test_polygon_geometry():
    assert tp.polygon_area(SQUARE) == pytest.approx(0.132)
    cw = SQUARE[::-1]
    assert tp.polygon_area(cw) == pytest.approx(-0.132)
    assert tp.ccw_polygon(cw)[0] == cw[0] and tp.polygon_area(tp.ccw_polygon(cw)) > 0
    assert tp.ccw_polygon(SQUARE) == SQUARE
    assert tp.is_convex_polygon(SQUARE) and tp.is_convex_polygon(cw)
    bowtie = [SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]]
    assert not tp.is_convex_polygon(bowtie)
    assert not tp.is_convex_polygon([[0, 0], [1, 0], [2, 0], [1, 1]])
    star = [[math.cos(a), math.sin(a)] for a in np.radians([90, 234, 18, 162, 306])]
    assert not tp.is_convex_polygon(star)
    assert tp.point_in_polygon(0.3, 0.0, SQUARE) and not tp.point_in_polygon(0.5, 0.0, SQUARE)
    assert tp.distance_to_polygon(0.47, 0.0, SQUARE) == pytest.approx(0.02)
    assert tp.polygon_contains(SQUARE, 0.46, 0.0, margin=0.02)
    assert not tp.polygon_contains(SQUARE, 0.48, 0.0, margin=0.02)


def test_validate_pick_polygon():
    def has(problems, text):
        return any(text in p for p in problems)

    assert tp.validate_pick_polygon(SQUARE, WS) == []
    assert tp.validate_pick_polygon(SQUARE[::-1], WS) == []
    assert has(tp.validate_pick_polygon(SQUARE[:3], WS), 'need 4 corners')
    small = [[0.2, 0.0], [0.25, 0.0], [0.25, 0.05], [0.2, 0.05]]
    assert has(tp.validate_pick_polygon(small, WS), 'too small')
    bowtie = [SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]]
    assert has(tp.validate_pick_polygon(bowtie, WS), 'convex')
    far = [[0.15, -0.22], [0.70, -0.22], [0.70, 0.22], [0.15, 0.22]]
    assert has(tp.validate_pick_polygon(far, WS), 'corner 2 [700, -220] mm is outside')
    dup = [SQUARE[0], SQUARE[0], SQUARE[2], SQUARE[3]]
    assert has(tp.validate_pick_polygon(dup, WS), 'touched twice')
    assert has(tp.validate_pick_polygon([[0.2, 0.0], [0.3, float('nan')], [0.3, 0.1],
                                         [0.2, 0.1]], WS), 'finite')
    assert has(tp.validate_pick_polygon('abc', WS), 'numbers')
    assert tp.validate_pick_polygon(SQUARE[:3], WS, n_corners=0) == []


def test_pick_polygon_from_touches():
    cw = SQUARE[::-1]
    touches = [[x, y, 0.010 + 0.002 * k] for k, (x, y) in enumerate(cw)]
    changes, problems, notes = tp.pick_polygon_from_touches(touches, WS)
    assert problems == []
    assert changes[tp.KEY_TABLE_Z] == pytest.approx(0.013)
    poly = changes[tp.KEY_POLYGON]
    assert poly[0] == cw[0] and tp.polygon_area(poly) > 0
    assert any('clockwise' in n for n in notes) and not any('differ' in n for n in notes)
    tilted = [[x, y, 0.03 * k] for k, (x, y) in enumerate(SQUARE)]
    _c, _p, notes = tp.pick_polygon_from_touches(tilted, WS)
    assert any('differ by 90 mm' in n for n in notes)
    bad, problems, _n = tp.pick_polygon_from_touches([[x, y, -0.2] for x, y in SQUARE], WS)
    assert bad is None and any('table z -200 mm' in p for p in problems)


# ------------------------------------------------------------------ gripper in view
def test_table_hit_and_gripper_points():
    T_cam = np.eye(4)
    T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])      # looking straight down from z = 1 m
    T_cam[:3, 3] = [0.0, 0.0, 1.0]
    assert np.allclose(tp.table_hit_xy([0.1, 0.0, 0.5], T_cam, 0.0), [0.2, 0.0])
    assert tp.table_hit_xy([0.1, 0.0, 1.5], T_cam, 0.0) is None
    pts = tp.gripper_points(top_down_T(0.3, 0.0, 0.1, 0.0), 0.07)
    assert pts.shape == (9, 3) and np.allclose(pts[0], [0.3, 0.0, 0.1])
    assert sorted(set(np.round(pts[1:, 2], 6))) == [0.1, 0.15]
    assert sorted(np.round(np.ptp(pts[1:, :2], axis=0), 6)) == [0.02, 0.094]


def test_gripper_in_view_with_the_sim_camera():
    cfg, _w, info = sim_cfg()
    T_bc = info['T_base_camera']
    over = top_down_T(0.30, 0.0, 0.15, 0.0)
    assert tp.gripper_in_view(over, T_bc, SQUARE, 0.0, 0.07)
    ok, msg = tp.observe_view_check(over, T_bc, SQUARE, 0.0, 0.07)
    assert not ok and 'TCP' in msg and 'fingertip box corners' in msg
    park = PiperKinematics('piper').fk(EYE_TO_HAND_PARK_JOINTS) @ flange_to_tcp_from_cfg(cfg)
    assert tp.gripper_in_view(park, T_bc, SQUARE, 0.0, 0.07) == []
    assert tp.observe_view_report(cfg, park)[0] is True
    assert tp.observe_view_report(cfg, over)[0] is False
    beside = top_down_T(0.30, -0.33, 0.15, 0.0)      # over a zone, outside the pick area
    assert tp.observe_view_check(beside, T_bc, SQUARE, 0.0, 0.07)[0] is True
    c2 = cfg.copy()
    c2.set('camera.extrinsics_inline', None)
    c2.set('camera.extrinsics_file', '')
    ok, msg = tp.observe_view_report(c2, over)
    assert ok is None and 'skipped' in msg
    # A hand-edited, malformed polygon skips the check instead of failing the command.
    assert tp.observe_view_report(cfg, park, polygon=[[0, 0], [1, 'x'], [2, 2]])[0] is None
    assert tp.observe_view_report(cfg, park, polygon=5)[0] is None


# ------------------------------------------------------------------ YAML mapping / checks
def test_apply_changes_merges_and_validate_poses():
    existing = {'zones': {'zone_red': {'center': [0.1, 0.2, 0.0], 'label': 'red'},
                          'zone_x': {'center': [0.3, 0.3, 0.0]}}, 'other': {'k': 1}}
    changes = tp.zone_changes('zone_red', tp.zone_from_touches([0.21, 0.29, 0.0]))
    changes[tp.KEY_OBSERVE] = [1.1, 0.3, -0.3, 0.0, 0.0, 0.0]
    data = tp.apply_changes(existing, changes)
    assert data['zones']['zone_red']['label'] == 'red'
    assert data['zones']['zone_red']['center'] == [0.21, 0.29, 0.0]
    assert data['zones']['zone_x'] == existing['zones']['zone_x'] and data['other'] == {'k': 1}
    assert existing['zones']['zone_red']['center'] == [0.1, 0.2, 0.0]
    assert data['robot']['poses']['observe_joints'][0] == 1.1
    kin = PiperKinematics('piper')
    assert tp.validate_poses(data, kin, WS) == []
    bad = tp.apply_changes(data, {tp.KEY_HOME: [0, 0, 0, 0, 3.0, 0],
                                  'zones.zone_y.size': [0.1, 0.1],
                                  tp.KEY_POLYGON: SQUARE[:2], tp.KEY_TABLE_Z: 'x'})
    problems = tp.validate_poses(bad, kin, WS)
    assert len(problems) == 4
    assert problems[0].startswith('robot.poses.home_joints outside the joint limits: joint5')
    assert any(p.startswith('zones.zone_y: ') and 'center' in p for p in problems)
    assert any(p.startswith('perception.roi.pick_polygon: need at least 3') for p in problems)
    assert any(p.startswith('perception.table.expected_z must be a number') for p in problems)
    assert tp.effective(data, Config({'a': {'b': 2}}, '.', []), 'a.b') == 2
    assert tp.effective({'a': {'b': None}}, Config({'a': {'b': 2}}, '.', []), 'a.b') is None
    assert tp.get_dotted(data, 'zones.zone_x.center') == [0.3, 0.3, 0.0]
    assert tp.get_dotted(data, 'zones.nope.center', 'd') == 'd'


def test_zone_reach_rows_ok_tilt_and_unreachable():
    cfg, _w, info = sim_cfg()
    arm = onsite.kinematic_arm(cfg)
    transit, rows = tp.zone_reach_rows(cfg, arm)
    assert transit == pytest.approx(0.15)
    assert {r['zone'] for r in rows} == set(info['zones'])
    assert all(r['result'] == 'OK' or r['result'].startswith('tilt') for r in rows)
    assert all(r['release_z'] == pytest.approx(0.09) for r in rows)
    cfg.set('zones.far', {'center': [0.55, 0.30, 0.0]})
    _t, rows = tp.zone_reach_rows(cfg, onsite.kinematic_arm(cfg))
    far = [r for r in rows if r['zone'] == 'far'][0]
    assert far['result'] == 'UNREACHABLE' and far['transit'] == 'unreachable'
    lines = tp.reach_report(cfg, arm)
    assert any('UNREACHABLE' in s for s in lines)
    assert any(s.startswith('  far transit: tcp radius') for s in lines)
    cfg.set('zones.broken', {'size': [0.1, 0.1]})
    assert tp.reach_report(cfg, arm)[0].startswith('[FAIL] zone reachability not checked')


def test_pose_report_selfcheck_and_post_write_report():
    cfg, _w, _i = sim_cfg()
    kin = PiperKinematics('piper')
    assert tp.pose_report(cfg, kin) == [
        '[ OK ] observe pose: the gripper is outside the camera view of the pick area']
    assert tp.check_tasks(cfg) == ['L1'] and tp.check_tasks(cfg, 'L2') == ['L2']
    assert tp.selfcheck_failures(cfg, ['L1'], 'sim') == [
        'self check (L1, config only): no FAIL items']
    lines = tp.post_write_report(cfg, 'sim', ['L1'])
    assert lines[0].startswith('zone reachability') and '[FAIL]' not in '\n'.join(lines)
    assert lines[-1] == 'self check (L1, config only): no FAIL items'
    cfg.set(tp.KEY_OBSERVE, [0, 0, 0, 0, 3.0, 0])
    assert tp.pose_report(cfg, kin)[0].startswith('[FAIL] robot.poses.observe_joints outside')
    cfg.set('tasks.L1.class_to_zone.red', 'zone_missing')
    lines = tp.selfcheck_failures(cfg, ['L1'], 'sim')
    assert lines[0].endswith('FAIL items:') and any('zone_missing' in s for s in lines)


def test_format_state_and_map():
    view = tp.apply_changes({}, {tp.KEY_OBSERVE: [0.1] * 6,
                                 'zones.zone_red.center': [0.21, 0.29, 0.0],
                                 'zones.zone_red.size': [0.1, 0.1]})
    tasks = {'L1': {'class_to_zone': {'red': 'zone_red', 'blue': 'zone_blue'}}}
    text = tp.format_state(view, {'zones.zone_red.center': [0.21, 0.29, 0.0]}, tasks)
    assert '  observe joints  [5.7, 5.7, 5.7, 5.7, 5.7, 5.7] deg' in text
    assert '  home joints     not taught' in text
    assert any(s.startswith('*') and 'zone_red' in s and '[210.0, 290.0, 0.0]' in s
               for s in text.splitlines())
    assert 'task zones not taught yet: zone_blue' in text
    assert '1 unsaved change(s)' in text
    zones = {'zone_red': parse_zone('zone_red', view['zones']['zone_red'])}
    img = tp.draw_map(zones, ['zone_red'], SQUARE, WS, tcp=[0.3, 0.0, 0.1], lines=['a', 'b'])
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[1] == 630
    assert img.shape[0] > 490 and int(img.max()) > 200
    assert tp.map_px(0.0, 0.0) == (315, 455) and tp.map_px(0.1, 0.1) == (245, 385)


# ------------------------------------------------------------------ scripted runs
L1_SCRIPT = ['o', 'h', 'p',
             'z', '', '', '', 'y',          # zone_red (suggested), defaults, yaw from a touch
             'z zone_green', '0.10 0.10', '0', 'n',
             'z', '', '', '', '',            # zone_blue (suggested), Enter for every default
             'u',                            # undo zone_blue
             'l',
             'z zone_blue', '', '', 'n',
             'w', 'q']


def test_main_sim_l1_writes_the_scenario_poses_and_zones(tmp_path):
    rc, out, _s = run_sim(tmp_path, L1_SCRIPT)
    assert rc == 0
    data = written(tmp_path)
    _w, info = make_scenario('L1', 0)
    assert_zones_match(data['zones'], info['zones'])
    assert np.allclose(data['perception']['roi']['pick_polygon'], info['pick_polygon'],
                       atol=1e-3)
    assert data['perception']['table']['expected_z'] == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(data['robot']['poses']['observe_joints'], info['observe_joints'],
                       atol=1e-4)
    assert data['robot']['poses']['home_joints'] == [0.0] * 6
    assert 'undone: zone zone_blue' in out
    assert 'keys written:' in out
    for key in ('zones.zone_red.center', 'zones.zone_blue.yaw', 'robot.poses.observe_joints',
                'perception.roi.pick_polygon', 'perception.table.expected_z'):
        assert '\n    %s\n' % key in out
    assert 'OK: the gripper is outside the camera view of the pick area' in out
    assert 'zone reachability' in out and 'UNREACHABLE' not in out
    assert 'self check (L1, config only): no FAIL items' in out
    assert 'task zones not taught yet: zone_blue' in out
    assert (tmp_path / 'preview' / 'teach_poses.jpg').is_file()
    assert not list((tmp_path / 'sim_site').glob('poses.bak_*'))


def test_main_sim_merges_the_existing_file_and_keeps_a_backup(tmp_path):
    assert run_sim(tmp_path, L1_SCRIPT)[0] == 0
    first = written(tmp_path)
    # Re-teach zone_red only, then quit: the tool asks to write the unsaved change.
    rc, out, script = run_sim(tmp_path, ['z zone_red', '0.12 0.12', '', '', 'q', 'y'])
    assert rc == 0
    data = written(tmp_path)
    assert data['zones']['zone_red']['size'] == [0.12, 0.12]
    assert data['zones']['zone_red']['center'] == first['zones']['zone_red']['center']
    assert data['zones']['zone_green'] == first['zones']['zone_green']
    assert data['robot'] == first['robot'] and data['perception'] == first['perception']
    backups = list((tmp_path / 'sim_site').glob('poses.bak_*.yaml'))
    assert len(backups) == 1
    assert yaml.safe_load(backups[0].read_text(encoding='utf-8')) == first
    assert any(p.startswith('Write them to') for p in script.prompts)
    assert 'kept from the previous file: ' in out and 'previous file kept as' in out
    # Re-teaching offers the stored size as the default.
    assert 'zone zone_red size along its x and y in m [0.100 0.100]: ' in script.prompts


def test_main_sim_l2_bins_with_walls(tmp_path):
    rc, out, _s = run_sim(tmp_path, ['p', 'z', '', '0.14 0.14', '0.05', 'n',
                                     'z', '', '0.14 0.14', '0.05', 'y', 'w', 'q'], task='L2')
    assert rc == 0
    _w, info = make_scenario('L2', 0)
    data = written(tmp_path)
    assert_zones_match(data['zones'], info['zones'])
    assert set(data['zones']) == {'fruit_bin', 'veg_bin'}
    assert 'UNREACHABLE' not in out and 'zone reachability' in out


def test_quit_without_writing_and_end_of_input_writes(tmp_path):
    rc, out, _s = run_sim(tmp_path, ['h', 'q', 'n', 'y'])
    assert rc == 0 and not (tmp_path / 'sim_site' / 'poses.yaml').exists()
    assert 'were NOT written' in out and 'home_joints' in out
    # The input ends with an unsaved pose: the default answer writes it.
    rc, out, _s = run_sim(tmp_path, ['h'])
    assert rc == 0
    assert written(tmp_path)['robot']['poses']['home_joints'] == [0.0] * 6


def test_cancelled_and_invalid_input_records_nothing(tmp_path):
    rc, out, _s = run_sim(tmp_path, ['z zone_nowhere', 'z 9bad', 'z zone_red', 'abc',
                                     '100 100', '0.1', 'x', 'u', 'w', 'q'])
    assert rc == 0
    assert 'the scenario has no zone zone_nowhere' in out
    assert "zone name '9bad'" in out
    assert out.count('enter 2 number(s) in [0.02, 0.60] m') == 3
    assert 'zone zone_red not recorded' in out
    assert "unknown command 'x'" in out and 'nothing to undo' in out
    assert 'nothing to write: no unsaved changes' in out
    assert not (tmp_path / 'sim_site' / 'poses.yaml').exists()


class TeachModeSimArm(SimArm):
    """SimArm that reports drag-teach mode like the real arm and records any command."""

    commands = []

    def health(self):
        h = SimArm.health(self)
        if not h.comm_ok:
            return h
        return ArmHealth(ok=False, reason='drag-teach mode', comm_ok=True, enabled=True,
                         ctrl_mode=onsite.CTRL_TEACHING)

    def read_teach_pose(self):
        j, f, t = self.get_joints(), self.get_flange_pose(), self.get_tcp_pose()
        if j is None:
            return None
        return {'joints': [float(v) for v in j.value], 'T_base_flange': f.value.tolist(),
                'T_base_tcp': t.value.tolist(), 'ctrl_mode': onsite.CTRL_TEACHING,
                'teach_status': 1, 'age_s': 0.0}

    def _refuse(self, name):
        TeachModeSimArm.commands.append(name)
        raise AssertionError('teach_poses must not call %s' % name)

    def move_tcp(self, *_a, **_k):
        self._refuse('move_tcp')

    def move_joints(self, *_a, **_k):
        self._refuse('move_joints')

    def gripper(self, *_a, **_k):
        self._refuse('gripper')

    def enable(self, *_a, **_k):
        self._refuse('enable')

    def emergency_stop(self):
        self._refuse('emergency_stop')

    def exit_teach_mode(self):
        self._refuse('exit_teach_mode')


def test_main_real_mode_only_reads_the_arm(tmp_path, monkeypatch):
    """Real mode with a fake arm in drag-teach mode: the 'human' moves it between prompts."""
    held = {}
    TeachModeSimArm.commands = []

    def fake_open_session(args, task=None, arm=True, camera=True, perception=True,
                          logger=False, operator=False, connect=True, cfg_world_info=None):
        assert args.mode == 'real' and arm and not camera and not perception
        assert connect is False                      # never enabled
        cfg = cfg_world_info[0]
        clock = SimClock()
        s = common.Session(cfg, 'real', task, clock)
        s.arm = SafeArm(TeachModeSimArm(cfg, clock), SafetyGuard(cfg),
                        kin=PiperKinematics(str(cfg.get('robot.model'))),
                        T_flange_tcp=flange_to_tcp_from_cfg(cfg))
        held['arm'], held['cfg'] = s.arm, cfg
        return s

    def human(prompt):
        arm = held.get('arm')
        if 'Drag the arm to the observe pose' in prompt:
            arm.inner.set_joints(EYE_TO_HAND_PARK_JOINTS)
        elif 'center of zone zone_a' in prompt:
            assert onsite.sim_drag_tcp(arm, held['cfg'], top_down_T(0.20, 0.30, 0.004, 0.0))

    monkeypatch.setattr(tp, 'open_session', fake_open_session)
    site = tmp_path / 'site'
    ui, out, script = make_test_ui(tmp_path, ['o', '', 'z zone_a', '', '0.10 0.10', '0', 'n',
                                              'w', 'q'], on_prompt=human)
    rc = tp.main(['--headless', '--site-dir', str(site), '--log-dir', str(tmp_path)], ui=ui)
    text = out.getvalue()
    assert rc == 0
    assert TeachModeSimArm.commands == [] and 'failed' not in text
    data = yaml.safe_load((site / 'poses.yaml').read_text(encoding='utf-8'))
    assert np.allclose(data['zones']['zone_a']['center'], [0.20, 0.30, 0.004], atol=1e-4)
    assert np.allclose(data['robot']['poses']['observe_joints'], EYE_TO_HAND_PARK_JOINTS,
                       atol=1e-4)
    assert any(p.endswith('then Enter (q = cancel): ') for p in script.prompts)
    assert 'This tool only reads the arm; it never moves it.' in text
    assert 'this is not the site overlay' in text
    assert 'still in drag-teach mode' in text


def test_replay_mode_and_bad_options_are_refused():
    with pytest.raises(SystemExit):
        tp.main(['--mode', 'replay', '--replay-dir', '/tmp'])
    with pytest.raises(SystemExit):
        tp.main(['--mode', 'sim', '--samples', '0'])


def test_an_unreadable_poses_file_is_left_alone(tmp_path):
    site = tmp_path / 'sim_site'
    site.mkdir()
    (site / 'poses.yaml').write_text('zones: [unclosed\n', encoding='utf-8')
    rc, out, _s = run_sim(tmp_path, ['h', 'w', 'q'])
    assert rc == 2 and 'cannot read' in out
    assert (site / 'poses.yaml').read_text(encoding='utf-8') == 'zones: [unclosed\n'

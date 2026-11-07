"""Tests for tools/calib_verify: correction math, offset averaging and scripted sim runs."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.extrinsics import Extrinsics, load_extrinsics
from cube_task.core.transforms import (inv_T, make_T, pose6_to_T, rotation_angle, rpy_to_rot,
                                       top_down_T, transform_points)
from cube_task.core.types import Detection, TablePlane
from cube_task.perception.grasp import GraspPlanner
from cube_task.sim.scenarios import perturb_pose
from cube_task.sim.world import SimObject, SimWorld
from cube_task.tools import calib_verify as cv
from cube_task.tools.ui import Ui

HOVER = 0.03
# Objects on a table at z = 0 (base frame, m): top centers and the table under them.
P_TRUE = np.array([[0.30, -0.12, 0.030], [0.36, 0.02, 0.045], [0.24, 0.13, 0.030],
                   [0.33, 0.10, 0.060]])
T_TABLE = P_TRUE * np.array([1.0, 1.0, 0.0])


def _operator(P_true, P_est, table_est, table_true=0.0, hover=HOVER):
    """Hover above each estimate, measure the true offset; return Nx3 corrections."""
    D = []
    for pt, pe, tz in zip(P_true, P_est, table_est):
        tcp = np.array([pe[0], pe[1], pe[2] + hover])
        dx, dy = (pt[:2] - tcp[:2]) * 1e3
        h = (tcp[2] - table_true) * 1e3
        D.append(cv.sample_correction(pe[:2], tz, tcp, dx, dy, h))
    return np.array(D)


def _seen(T_true_base_cam, T_est_base_cam, P):
    """Return where a camera with estimate T_est puts the true base points P."""
    return transform_points(T_est_base_cam @ inv_T(T_true_base_cam), P)


# ------------------------------------------------------------------ correction math
def test_trans_is_pure_translation():
    T = cv.trans([0.001, -0.002, 0.003])
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [0.001, -0.002, 0.003])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_eye_to_hand_correction_restores_true_pose(seed):
    T_true = make_T(rpy_to_rot(-2.57, 0.0, 1.5708), (0.70, 0.0, 0.62))
    T_est = perturb_pose(T_true, 0.004, 0.0, np.random.default_rng(seed))
    P_est = _seen(T_true, T_est, P_TRUE)
    table_est = _seen(T_true, T_est, T_TABLE)[:, 2]
    D = _operator(P_TRUE, P_est, table_est)
    rep = cv.summarize_corrections(D)
    assert rep['outliers'] == [] and rep['n_z'] == len(P_TRUE)
    ext = Extrinsics(T=T_est, mount='eye_to_hand', source='perturbed')
    new = cv.corrected_extrinsics(ext, rep['mean_inliers'])
    assert np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) > 0.0039
    assert np.linalg.norm(new.T[:3, 3] - T_true[:3, 3]) < 0.0005
    assert rotation_angle(new.T[:3, :3].T @ T_true[:3, :3]) < 1e-6
    assert np.allclose(new.T, cv.trans(rep['mean_inliers']) @ T_est, atol=1e-7)
    assert new.mount == 'eye_to_hand' and new.parent_frame == 'base'


def _wrist_setup(seed):
    T_bf = top_down_T(0.22, 0.03, 0.30, 0.4) @ make_T(rpy_to_rot(0.2, -0.1, 0.3), (0, 0, 0))
    T_fc = make_T(rpy_to_rot(0.0, 0.0, 1.5708), (0.065, 0.0, 0.035))
    T_fc_est = perturb_pose(T_fc, 0.004, 0.0, np.random.default_rng(seed))
    return T_bf, T_fc, T_fc_est


@pytest.mark.parametrize('seed', [0, 5])
def test_eye_in_hand_correction_restores_true_pose(seed):
    T_bf, T_fc, T_fc_est = _wrist_setup(seed)
    T_bc, T_bc_est = T_bf @ T_fc, T_bf @ T_fc_est
    D = _operator(P_TRUE, _seen(T_bc, T_bc_est, P_TRUE), _seen(T_bc, T_bc_est, T_TABLE)[:, 2])
    d = cv.summarize_corrections(D)['mean_inliers']
    ext = Extrinsics(T=T_fc_est, mount='eye_in_hand', parent_frame='flange')
    new = cv.corrected_extrinsics(ext, d, T_base_flange=T_bf)
    assert new.parent_frame == 'flange'
    assert np.linalg.norm(new.T[:3, 3] - T_fc[:3, 3]) < 0.0005
    assert rotation_angle(new.T[:3, :3].T @ T_fc[:3, :3]) < 1e-6
    expected = inv_T(T_bf) @ cv.trans(d) @ T_bf @ T_fc_est
    assert np.allclose(new.T, expected, atol=1e-7)
    # The eye_to_hand rule applied to the flange extrinsics would be wrong here.
    assert np.linalg.norm((cv.trans(d) @ T_fc_est)[:3, 3] - T_fc[:3, 3]) > 0.001


def test_eye_in_hand_tcp_parent_correction():
    T_bf, T_fc, T_fc_est = _wrist_setup(3)
    T_ft = pose6_to_T([0.0, 0.0, 0.14, 0.0, 0.0, 0.0])
    T_tc, T_tc_est = inv_T(T_ft) @ T_fc, inv_T(T_ft) @ T_fc_est
    T_bc, T_bc_est = T_bf @ T_ft @ T_tc, T_bf @ T_ft @ T_tc_est
    D = _operator(P_TRUE, _seen(T_bc, T_bc_est, P_TRUE), _seen(T_bc, T_bc_est, T_TABLE)[:, 2])
    ext = Extrinsics(T=T_tc_est, mount='eye_in_hand', parent_frame='tcp')
    new = cv.corrected_extrinsics(ext, D.mean(axis=0), T_bf, T_ft)
    assert np.linalg.norm(new.T[:3, 3] - T_tc[:3, 3]) < 0.0005
    with pytest.raises(ValueError):
        cv.parent_pose_in_base(ext, T_bf, None)
    with pytest.raises(ValueError):
        cv.parent_pose_in_base(ext, None, T_ft)
    assert 'T_tcp_camera <- inv((T_base_flange_obs @ T_flange_tcp))' in cv.correction_formula(ext)


def test_corrected_extrinsics_metadata():
    ext = Extrinsics(T=make_T(np.eye(3), (0.5, 0.0, 0.6)), mount='eye_to_hand',
                     source='official', rms=0.002)
    new = cv.corrected_extrinsics(ext, [0.001, 0.0, -0.002], rms=0.0004, n=3, stamp='today')
    assert np.allclose(new.T[:3, 3], [0.501, 0.0, 0.598])
    assert new.rms == pytest.approx(0.0004) and new.stamp == 'today'
    assert new.source.startswith('calib_verify') and 'official' in new.source
    assert cv.corrected_extrinsics(ext, [0, 0, 0]).rms == pytest.approx(0.002)
    assert cv.correction_formula(ext) == 'eye_to_hand: T_base_camera <- Trans(d) @ T_base_camera'
    with pytest.raises(ValueError):
        cv.corrected_extrinsics(Extrinsics(T=np.eye(4), mount='eye_in_hand'), [0.001, 0, 0])


def test_sample_correction():
    d = cv.sample_correction([0.30, 0.10], 0.004, [0.302, 0.097, 0.064], 1.0, -2.0, 61.0)
    assert np.allclose(d, [0.003, -0.005, -0.001])
    d = cv.sample_correction([0.30, 0.10], 0.004, [0.30, 0.10, 0.06], 0.0, 0.0)
    assert np.allclose(d[:2], 0.0) and np.isnan(d[2])
    assert np.isnan(cv.sample_correction([0, 0], None, [0, 0, 0.1], 0, 0, 50.0)[2])


# ------------------------------------------------------------------ averaging
def test_summary_consistent_offsets():
    rng = np.random.default_rng(0)
    D = np.array([0.001, -0.002, 0.0005]) + rng.normal(0, 0.0002, size=(5, 3))
    rep = cv.summarize_corrections(D)
    assert rep['n'] == 5 and rep['n_z'] == 5 and rep['outliers'] == []
    assert np.allclose(rep['mean'], D.mean(axis=0)) and np.allclose(rep['mean_inliers'],
                                                                    rep['mean'])
    assert np.all(rep['std'] < 0.0005) and rep['rms_inliers'] < 0.0006
    assert rep['warnings'] == []
    text = '\n'.join(cv.summary_lines(rep))
    assert 'mean correction d' in text and 'WARNING' not in text


def test_summary_flags_outlier():
    D = np.array([[0.0010, -0.0020, 0.0], [0.0012, -0.0021, 0.0], [0.0009, -0.0018, 0.0],
                  [0.0011, -0.0019, 0.0], [0.0090, -0.0020, 0.0]])
    rep = cv.summarize_corrections(D, names=['#0', '#1', '#2', '#3', '#4 lemon'])
    assert rep['outliers'] == [4]
    assert np.allclose(rep['mean_inliers'], D[:4].mean(axis=0))
    assert rep['mean'][0] > rep['mean_inliers'][0] + 0.001
    assert any('#4 lemon' in w and 'median' in w for w in rep['warnings'])
    text = '\n'.join(cv.summary_lines(rep))
    assert 'outliers left out' in text and 'over 4 measurements' in text
    assert 'OUTLIER' in cv.samples_table([cv.Sample(k, 'x', np.zeros(3), np.zeros(3), 0.0, 0.0,
                                                    None, D[k]) for k in range(5)], [4])


def test_summary_missing_z_and_few_samples():
    nan = float('nan')
    rep = cv.summarize_corrections([[0.001, 0.002, nan], [0.0012, 0.0018, -0.003],
                                    [0.0011, 0.0021, nan]])
    assert rep['n_z'] == 1 and rep['mean'][2] == pytest.approx(-0.003)
    rep = cv.summarize_corrections([[0.001, 0.002, nan]])
    assert rep['n_z'] == 0 and rep['mean'][2] == 0.0
    assert any('only 1 measurement' in w for w in rep['warnings'])
    assert 'z not measured' in '\n'.join(cv.summary_lines(rep))
    rep = cv.summarize_corrections([[0.0, 0.0, 0.0], [0.006, 0.0, 0.0]])
    assert any('differ by 6.0 mm' in w for w in rep['warnings'])
    rep = cv.summarize_corrections(np.zeros((0, 3)))
    assert rep['n'] == 0 and cv.summary_lines(rep) == ['nothing measured']


def test_summary_spread_warning_and_small_correction():
    D = [[0.004, 0.0, 0.0], [-0.004, 0.0, 0.0], [0.0, 0.004, 0.0], [0.0, -0.004, 0.0]]
    rep = cv.summarize_corrections(D)
    assert rep['outliers'] == [] and rep['rms_inliers'] == pytest.approx(0.004)
    assert any('one translation cannot fix that' in w for w in rep['warnings'])
    text = '\n'.join(cv.summary_lines(rep))
    assert 'no correction needed' in text
    assert 'fingertip error' in '\n'.join(cv.summary_lines(rep, point_mode=True))


# ------------------------------------------------------------------ parsing / targets
@pytest.mark.parametrize('text, value', [
    ('2 -1', (2.0, -1.0, None)), ('2,-1,61', (2.0, -1.0, 61.0)), ('0 0', (0.0, 0.0, None)),
    ('-0.5  3.5 40', (-0.5, 3.5, 40.0)), ('', None), ('2', None), ('a b', None),
    ('1 2 3 4', None), ('150 0', None), ('1 1 0', None), ('1 1 nan', None)])
def test_parse_measurement(text, value):
    got, err = cv.parse_measurement(text)
    assert got == value
    assert (err == '') == (value is not None)


def _det(valid=True, reason=''):
    return SimpleNamespace(valid=valid, reject_reason=reason)


def test_select_targets():
    dets = [_det(), _det(False, 'too_small'), _det(), _det()]
    assert cv.select_targets(dets) == ([0, 2, 3], [])
    idx, msgs = cv.select_targets(dets, [3, 1, 3, 7, 0])
    assert idx == [3, 0]
    assert any('too_small' in m for m in msgs) and any('--index 7' in m for m in msgs)
    assert cv.select_targets([], None) == ([], [])


def test_approach_z():
    assert cv.approach_z(0.03, 0.03, 0.08) == pytest.approx(0.11)
    assert cv.approach_z(0.03, 0.10, 0.08) == pytest.approx(0.13)
    assert cv.approach_z(0.03, 0.03, 0.08, highest_top=0.09) == pytest.approx(0.14)


def _block(x=0.32, y=-0.05, top=0.034, yaw=0.3):
    return Detection(label='red', score=1.0, center_base=np.array([x, y, top]), top_z=top,
                     height=top - 0.004, yaw=yaw, length=0.031, width=0.030,
                     extra={'table_z': 0.004})


def test_hover_plan_above_grasp_point():
    cfg = load_config(use_overlays=False)
    planner = GraspPlanner(cfg, 'L1')
    det = _block()
    hp, why = cv.hover_plan(det, 2, planner, HOVER, 0.08, highest_top=0.09)
    assert why == '' and hp.index == 2 and hp.label == 'red'
    gp = planner.plan(det)
    assert np.allclose(hp.target, [gp.T_grasp[0, 3], gp.T_grasp[1, 3], 0.034])
    assert hp.T_hover[2, 3] == pytest.approx(0.034 + HOVER)
    assert np.allclose(hp.T_hover[:2, 3], hp.target[:2])
    assert np.allclose(hp.T_hover[:3, 2], [0, 0, -1])            # tool z points down
    assert hp.closing_angle == pytest.approx(gp.closing_angle)
    assert hp.T_approach[2, 3] == pytest.approx(0.14) and hp.table_z == pytest.approx(0.004)
    bad = Detection(label='red', score=1.0)
    hp, why = cv.hover_plan(bad, 0, planner, HOVER, 0.08)
    assert hp is None and why.startswith('missing_geometry')


def test_point_plan():
    hp = cv.point_plan([0.30, 0.05, 0.01], HOVER, 0.08)
    assert hp.index == -1 and hp.table_z == pytest.approx(0.01)
    assert np.allclose(hp.T_hover[:3, 3], [0.30, 0.05, 0.04])
    assert hp.T_approach[2, 3] == pytest.approx(0.09)


def test_table_z_estimate():
    plane = TablePlane(normal=np.array([0.0, 0.0, 1.0]), d=-0.004, frame='base')
    res = SimpleNamespace(table=plane, valid_detections=lambda: [])
    assert cv.table_z_estimate(res, [[0.1, -0.1], [0.4, -0.1], [0.4, 0.1]]) == pytest.approx(
        0.004)
    res = SimpleNamespace(table=None, valid_detections=lambda: [_block(), _block()])
    assert cv.table_z_estimate(res, None) == pytest.approx(0.004)
    assert cv.table_z_estimate(SimpleNamespace(table=None, valid_detections=list)) is None


def test_sim_measurement():
    T = np.eye(4)
    T[:3, 3] = [0.30, 0.10, 0.0]
    world = SimWorld(table_z=0.0, objects=[SimObject('b', 'red', 'box', (0.03, 0.03, 0.03),
                                                     (0, 0, 255), T)])
    dx, dy, h = cv.sim_measurement(world, [0.302, 0.097, 0.06])
    assert (dx, dy, h) == pytest.approx((-2.0, 3.0, 60.0))
    assert cv.sim_measurement(world, [0.0, 0.3, 0.06]) is None
    assert cv.sim_measurement(world, [0.30, 0.05, 0.05], point=[0.301, 0.05, 0.01]) == \
        pytest.approx((1.0, 0.0, 40.0))


def test_texts():
    h = cv.measure_help()
    assert '+x points forward, away from the robot' in h and "+y points to the robot's left" in h
    assert 'table' in h and 'point' in cv.measure_help(point_mode=True)
    t = cv.plan_text(None, HOVER, 20, 'eye_to_hand')
    assert '30 mm above its detected top' in t and 'speed 20 %' in t and 'e-stop' in t


@pytest.mark.parametrize('argv', [
    ['--point', '0.3', '0', '0', '--apply'], ['--hover', '0.001'], ['--hover', '0.5'],
    ['--all', '--index', '1'], ['--point', '0.3', '0', '0', '--all'], ['--index', '-1']])
def test_bad_arguments_exit(argv, capsys):
    with pytest.raises(SystemExit):
        cv.main(argv + ['--mode', 'sim'])


# ------------------------------------------------------------------ scripted sim runs
@pytest.fixture
def session(monkeypatch):
    """Capture the Session main() opens so the scripted operator can see the sim world."""
    box = {}
    real = cv.open_session

    def wrapper(*a, **k):
        box['s'] = real(*a, **k)
        return box['s']
    monkeypatch.setattr(cv, 'open_session', wrapper)
    return box


def _truth(s):
    """Measure like a careful operator: true object center and table vs the fingertips."""
    tcp = np.asarray(s.arm.get_tcp_pose().value, dtype=float)[:3, 3]
    obj = min(s.world.objects, key=lambda o: float(np.hypot(*(o.center_xy() - tcp[:2]))))
    dx, dy = (obj.center_xy() - tcp[:2]) * 1e3
    return '%.2f %.2f %.2f' % (dx, dy, (tcp[2] - s.world.table_z) * 1e3)


def _scripted(box, replies=(), confirm='y'):
    """Return (input_fn, log): replies answer the prompts in order, then the defaults."""
    log = {'prompts': [], 'at_confirm': None}
    queue = list(replies)

    def answer(prompt):
        log['prompts'].append(prompt)
        if len(log['prompts']) > 60:
            raise EOFError
        if queue:
            r = queue.pop(0)
            return _truth(box['s']) if r == 'TRUTH' else r
        if 'offset "dx dy [h]"' in prompt:
            return _truth(box['s'])
        if prompt.startswith('Write the corrected'):
            s = box['s']
            log['at_confirm'] = (np.asarray(s.arm.get_joints().value, dtype=float),
                                 float(s.arm.gripper_width().value))
            return confirm
        return 'y' if '(y/n)' in prompt else ''
    return answer, log


def _run(tmp_path, argv, answer):
    out = io.StringIO()
    ui = Ui(input_fn=answer, headless=True, preview_dir=str(tmp_path / 'preview'), output=out)
    code = cv.main(['--mode', 'sim', '--task', 'L1', '--headless', '--log-dir', str(tmp_path)]
                   + argv, ui=ui)
    return code, out.getvalue()


@pytest.mark.parametrize('mount, key', [('eye_to_hand', 'sim.T_base_camera'),
                                        ('eye_in_hand', 'sim.T_flange_camera')])
def test_sim_apply_restores_true_extrinsics(tmp_path, session, mount, key):
    answer, log = _scripted(session)
    code, text = _run(tmp_path, ['--mount', mount, '--extrinsics-error', '0.004', '0.0',
                                 '--all', '--apply'], answer)
    assert code == cv.EXIT_OK, text
    s = session['s']
    T_true = np.asarray(s.cfg.get(key), dtype=float)
    T_start = np.asarray(s.cfg.get('camera.extrinsics_inline.matrix'), dtype=float)
    assert np.linalg.norm(T_start[:3, 3] - T_true[:3, 3]) > 0.0039
    path = tmp_path / 'sim_site' / 'extrinsics.yaml'
    ext = load_extrinsics(path)
    assert ext.mount == mount
    assert np.linalg.norm(ext.T[:3, 3] - T_true[:3, 3]) < 0.001, text
    assert rotation_angle(ext.T[:3, :3].T @ T_true[:3, :3]) < 1e-6
    assert sum('offset "dx dy [h]"' in p for p in log['prompts']) == 3
    assert 'keys written:' in text and '    matrix' in text and '    translation' in text
    assert 'base frame' in text and "+y points to the robot's left" in text
    # Back at the observe pose with the gripper open before the file is written.
    q, width = log['at_confirm']
    assert np.allclose(q, s.cfg.get('robot.poses.observe_joints'), atol=1e-3)
    assert width == pytest.approx(float(s.cfg.get('robot.gripper.max_width')), abs=1e-3)


def test_sim_interactive_without_apply(tmp_path, session):
    # #0: hover, accept the simulated measurement; #1: skip; then finish.
    answer, log = _scripted(session, replies=['', '', 's', 'q'])
    code, text = _run(tmp_path, ['--extrinsics-error', '0.004', '0.0', '--speed', '90'],
                      answer)
    assert code == cv.EXIT_OK, text
    assert 'is above robot.speed_percent' in text
    assert '[sim operator] measured dx dy h' in text
    assert sum(p.startswith('next: ') for p in log['prompts']) == 3
    assert 'over 1 measurement ' in text and 'add --apply' in text
    assert not (tmp_path / 'sim_site' / 'extrinsics.yaml').exists()


def test_sim_point_check(tmp_path, session):
    answer, log = _scripted(session, replies=[''])     # Enter = simulated measurement
    code, text = _run(tmp_path, ['--point', '0.30', '0.05', '0.0'], answer)
    assert code == cv.EXIT_OK, text
    assert 'fingertip error (real - reported position) = [' in text
    assert 'the TCP is accurate' in text and 'add --apply' not in text
    assert 'only 1 measurement' not in text
    assert not (tmp_path / 'sim_site').exists()


def test_sim_declined_write_and_index(tmp_path, session):
    answer, log = _scripted(session, confirm='n')
    code, text = _run(tmp_path, ['--extrinsics-error', '0.004', '0.0', '--index', '1', '9',
                                 '--apply'], answer)
    assert code == cv.EXIT_CANCELLED, text
    assert '--index 9: no such detection' in text and 'nothing written' in text
    assert sum('offset "dx dy [h]"' in p for p in log['prompts']) == 1
    assert not (tmp_path / 'sim_site' / 'extrinsics.yaml').exists()


def test_sim_stop_during_measurement_returns_to_observe(tmp_path, session):
    answer, log = _scripted(session, replies=['q'])
    code, text = _run(tmp_path, ['--all', '--apply'], answer)
    assert code == cv.EXIT_CANCELLED, text
    assert sum('offset "dx dy [h]"' in p for p in log['prompts']) == 1
    lines = [ln for ln in text.splitlines() if ln.startswith(('MOTION:', 'GRIPPER:'))]
    assert lines[-2].startswith('MOTION: move_joints to the observe pose')
    assert lines[-1].startswith('GRIPPER: open')
    assert 'no offsets measured, nothing written' in text

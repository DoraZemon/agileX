"""teach_classes: label commands, sample bookkeeping, evaluation, --from-dir and sim main()."""
from __future__ import annotations

from collections import Counter
import io
import json
import math
from pathlib import Path

from cube_task.camera.recorder import save_frame
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.types import Detection
from cube_task.perception.classify_knn import KnnClassifier
from cube_task.perception.features import FEATURE_NAMES
from cube_task.perception.pipeline import Perception
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from cube_task.sim.world import SimWorld
from cube_task.tools import teach_classes as tc
from cube_task.tools.onsite import ground_truth_label
from cube_task.tools.ui import Ui
import numpy as np
import pytest
from test_task_fakes import still_wrist_camera

D = len(FEATURE_NAMES)


def scripted(commands=(), answers=None):
    """Return an input function: commands for command prompts, answers for (y/n) questions."""
    cmds = list(commands)
    answers = dict(answers or {})

    def fn(prompt):
        fn.asked.append(prompt)
        for key, ans in answers.items():
            if key in prompt:
                return ans
        if '(y/n)' in prompt:
            return 'y'
        if not cmds:
            raise EOFError
        return cmds.pop(0)

    fn.asked = []
    return fn


def make_ui(tmp_path, commands=(), answers=None):
    out = io.StringIO()
    fn = scripted(commands, answers)
    ui = Ui(input_fn=fn, headless=True, preview_dir=str(tmp_path / 'preview'), output=out)
    return ui, fn, out


def rows(n, value=0.0):
    return np.full((n, D), float(value))


# ------------------------------------------------------------------ label commands
def test_command_kind():
    assert tc.command_kind(None) == 'none' and tc.command_kind('  ') == 'none'
    assert tc.command_kind('Q') == 'quit' and tc.command_kind('c') == 'capture'
    assert tc.command_kind('u') == 'undo' and tc.command_kind('e') == 'evaluate'
    assert tc.command_kind('w') == 'write' and tc.command_kind('gt') == 'ground_truth'
    assert tc.command_kind('0 apple') == 'label' and tc.command_kind('all apple') == 'label'
    assert tc.command_kind('Single pear') == 'single' and tc.command_kind('single off') == 'single'
    assert tc.command_kind('y') == 'accept' and tc.command_kind('OK') == 'accept'


def test_parse_label_command_all_numbers_and_lists():
    assert tc.parse_label_command('all apple', 3) == ([(None, 'apple')], '')
    parts, err = tc.parse_label_command('0 apple; 1,2 banana ;', 4)
    assert err == '' and parts == [([0], 'apple'), ([1, 2], 'banana')]
    parts, err = tc.parse_label_command('3 lemon; ALL apple', 4)
    assert err == '' and parts == [([3], 'lemon'), (None, 'apple')]


@pytest.mark.parametrize('text, n, msg', [
    ('', 3, 'empty'),
    ('0 apple', 0, 'no valid instances'),
    ('5 apple', 3, '0..2'),
    ('x apple', 3, 'instance number'),
    ('0 green pepper', 3, 'no spaces'),
    ('0 unknown', 3, 'letters'),
    ('all a; all b', 3, 'only once'),
    ('0 a; 0,1 b', 3, 'named twice'),
])
def test_parse_label_command_errors(text, n, msg):
    parts, err = tc.parse_label_command(text, n)
    assert parts is None and msg in err


def test_plan_labeling_explicit_numbers_win_over_all():
    parts, _ = tc.parse_label_command('1 lemon; all apple', 4)
    assign, notes, err = tc.plan_labeling(parts, {0: 'banana'}, [True, True, True, False])
    assert err == '' and assign == {1: 'lemon', 2: 'apple'}
    assert any('#0 banana' in n for n in notes) and any('#3' in n for n in notes)
    parts, _ = tc.parse_label_command('3 apple', 4)
    assign, notes, err = tc.plan_labeling(parts, {}, [True, True, True, False])
    assert assign == {} and 'nothing left' in err and 'incomplete' in notes[0]
    parts, _ = tc.parse_label_command('0 apple; 1 apple', 4)
    assign, notes, err = tc.plan_labeling(parts, {0: 'banana'}, [True] * 4)
    assert assign == {} and 'already labeled' in err
    parts, _ = tc.parse_label_command('all apple', 2)
    assert 'nothing left' in tc.plan_labeling(parts, {0: 'a', 1: 'b'}, [True, True])[2]


def test_check_label_against_the_task_labels():
    known = ['apple', 'banana', 'green_pepper']
    assert tc.check_label('Apple', known) == ('apple', 'ok', [])
    assert tc.check_label('banan', known) == ('banan', 'unlisted', ['banana'])
    assert tc.check_label('kiwi', []) == ('kiwi', 'unchecked', [])


def test_confirm_labels_warns_asks_once_and_can_decline(tmp_path):
    ui, fn, out = make_ui(tmp_path, answers={'anyway': 'n', 'spelled': 'y'})
    accepted = set()
    assert tc.confirm_labels(ui, ['APPLE'], ['apple'], 'L2', accepted) == {'APPLE': 'apple'}
    assert fn.asked == []
    assert tc.confirm_labels(ui, ['banan'], ['apple', 'banana'], 'L2', accepted) is None
    assert 'not in tasks.L2.labels' in out.getvalue() and 'did you mean banana' in out.getvalue()
    assert tc.confirm_labels(ui, ['kiwi'], [], 'L2', accepted) == {'kiwi': 'kiwi'}
    assert tc.confirm_labels(ui, ['kiwi'], [], 'L2', accepted) == {'kiwi': 'kiwi'}
    assert sum('spelled' in p for p in fn.asked) == 1 and accepted == {'kiwi'}
    book = tc.SampleBook()
    book.add(tc.Batch('x', ['kiwi'], rows(1)))
    assert tc.accepted_labels(book, []) == {'kiwi'} and tc.accepted_labels(book, ['a']) == set()


# ------------------------------------------------------------------ bookkeeping
def test_sample_book_add_undo_and_views():
    book = tc.SampleBook()
    assert not book.dirty and book.counts() == {}
    book.add(tc.Batch('0 apple; 1 banana', ['apple', 'banana'],
                      np.vstack([rows(1, 1), rows(1, 2)]), view=1, indices=[0, 1]))
    book.add(tc.Batch('g', ['lemon'], rows(1, 3), view=2, indices=[0], source='sim_gt'))
    assert book.labeled_in_view(1) == {0: 'apple', 1: 'banana'}
    assert book.labeled_in_view(2) == {0: 'lemon'} and book.labeled_in_view(-1) == {}
    assert book.counts() == {'apple': 1, 'banana': 1, 'lemon': 1} and book.dirty
    with pytest.raises(ValueError, match='labeled already'):
        book.add(tc.Batch('1 kiwi', ['kiwi'], rows(1, 4), view=1, indices=[1]))
    for bad in (tc.Batch('nan', ['apple'], np.full((1, D), np.nan)),
                tc.Batch('unknown', ['unknown'], rows(1)),
                tc.Batch('two labels', ['a', 'b'], rows(1)),
                tc.Batch('no index', ['a'], rows(1), view=3),
                tc.Batch('features', ['a'], np.zeros((1, 3)))):
        with pytest.raises(ValueError):
            book.add(bad)
    book.mark_saved()
    assert not book.dirty and book.n_session == 3
    undone = book.undo()
    assert undone.text == 'g' and book.labeled_in_view(2) == {} and book.dirty
    assert book.n_session == 2 and book.source_counts() == {'live': 2}
    assert book.undo().view == 1 and book.undo() is None and book.counts() == {}


def test_sample_book_base_samples_and_duplicates():
    book = tc.SampleBook()
    book.set_base(np.vstack([rows(1, 1), rows(1, 2)]), ['apple', 'banana'])
    assert not book.dirty and book.counts() == {'apple': 1, 'banana': 1}
    book.add(tc.Batch('again', ['apple', 'apple'], np.vstack([rows(1, 1), rows(1, 5)]),
                      source='from_dir'))
    X, y = book.arrays()
    assert X.shape == (3, D) and y == ['apple', 'banana', 'apple']
    assert book.source_counts() == {'appended': 2, 'from_dir': 2}
    assert book.undo() is not None and book.counts() == {'apple': 1, 'banana': 1}
    with pytest.raises(ValueError):
        book.set_base(rows(2), ['a'])


def test_merge_samples_drops_exact_duplicates_only():
    X, y, dup = tc.merge_samples(rows(2, 1), ['a', 'a'],
                                 np.vstack([rows(1, 1), rows(1, 2)]), ['a', 'b'])
    assert dup == 2 and y == ['a', 'b'] and X.shape == (2, D)
    X, y, dup = tc.merge_samples(rows(1, 1), ['a'], rows(1, 1), ['b'])
    assert dup == 0 and y == ['a', 'b']
    X, y, dup = tc.merge_samples([], [], rows(1), ['a'])
    assert y == ['a'] and dup == 0 and X.shape == (1, D)
    with pytest.raises(ValueError, match='different features'):
        tc.merge_samples(rows(1), ['a'], rows(1), ['b'], names_a=FEATURE_NAMES,
                         names_b=FEATURE_NAMES[:-1])
    with pytest.raises(ValueError):
        tc.merge_samples(rows(2), ['a'], rows(1), ['b'])
    with pytest.raises(ValueError):
        tc.merge_samples(np.zeros((1, 3)), ['a'], rows(1), ['b'])


# ------------------------------------------------------------------ evaluation
CONFUSION = {'apple': {'apple': 8, 'tomato': 2}, 'tomato': {'tomato': 9, 'apple': 1},
             'lemon': {'lemon': 9, 'unknown': 1}}


def test_confused_pairs_weak_classes_and_confusion_table():
    pairs = tc.confused_pairs(CONFUSION, 0.1)
    assert [(p['a'], p['b'], p['a_as_b'], p['b_as_a']) for p in pairs] == [
        ('apple', 'tomato', 2, 1)]
    assert abs(pairs[0]['rate'] - 0.15) < 1e-9 and tc.confused_pairs(CONFUSION, 0.2) == []
    text = tc.confusion_table(CONFUSION)
    lines = text.splitlines()
    assert lines[0].split()[-3:] == ['2', 'unk', 'accuracy']
    assert lines[2].split() == ['0', 'apple', '8', '.', '2', '.', '80', '%']
    assert tc.confusion_table({}).startswith('(no')
    assert tc.weak_classes({'a': 8, 'b': 7, 'c': 0}) == [('b', 7), ('c', 0)]
    assert tc.counts_text({'apple': 9, 'kiwi': 2}, ['lemon', 'apple']) == (
        'lemon 0 (<8), apple 9, kiwi 2 (<8)')
    assert tc.counts_text({}) == 'no samples yet'


def clusters(spec, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, center, n in spec:
        X.append(rng.normal(center, 0.3, (n, D)))
        y += [label] * n
    return np.vstack(X), y


def test_evaluate_samples_warns_about_weak_confused_and_missing_classes():
    X, y = clusters([('a', 0.0, 10), ('b', 5.0, 10), ('c', 5.0, 10), ('d', -5.0, 3)])
    rep = tc.evaluate_samples(X, y, k=3, max_distance=10.0, known=['a', 'b', 'c', 'd', 'e'],
                              task='L2')
    assert rep['n'] == 33 and rep['classes'] == ['a', 'b', 'c', 'd']
    assert 0.3 < rep['accuracy'] < 0.95 and rep['per_class']['a']['accuracy'] == 1.0
    assert [(p['a'], p['b']) for p in rep['confused']] == [('b', 'c')]
    assert rep['weak'] == [('d', 3)] and rep['missing_labels'] == ['e']
    text = rep['text']
    assert 'leave-one-out' in text and 'WARNING: fewer than 8 samples: d 3' in text
    assert 'b and c are confused' in text and 'tasks.L2.labels without samples: e' in text
    report = tc.build_report(rep, 'now', 'test', Path('m.npz'), task='L2')
    assert json.loads(json.dumps(report, allow_nan=False))['weak_classes'] == {'d': 3}
    assert report['feature_names'] == list(FEATURE_NAMES) and report['task'] == 'L2'


def test_evaluate_samples_one_class_unknown_and_empty():
    X, y = clusters([('a', 0.0, 4)])
    rep = tc.evaluate_samples(X, y)
    assert rep['accuracy'] is None and 'only one class' in rep['text']
    assert tc.build_report(rep, 'now', 'test', Path('m.npz'))['accuracy'] is None
    X, y = clusters([('a', 0.0, 9), ('b', 9.0, 9)])
    rep = tc.evaluate_samples(X, y, max_distance=1e-3, known=['a', 'b'])
    assert rep['n_unknown'] == 18 and 'farther than max_distance' in rep['text']
    assert tc.evaluate_samples(np.zeros((0, D)), [])['warnings'] == ['no samples yet']


# ------------------------------------------------------------------ sessions / output
@pytest.mark.parametrize('session, dirname, known, explicit, label', [
    ('apple_1', '20261112_apple_1', ['apple'], None, 'apple'),
    ('apple_1', 'x', [], None, 'apple'),
    ('capture', '20261112_banana_2', [], None, 'banana'),
    ('', 'carrot', ['carrot'], None, 'carrot'),
    (None, 'block_2', ['block_2'], None, 'block_2'),
    ('Apple', 'x', ['apple'], None, 'apple'),
    ('capture', 'weird name!', [], None, 'weird_name'),
    ('apple', 'x', [], 'lemon', 'lemon'),
    ('capture', '', [], None, ''),
])
def test_session_label(session, dirname, known, explicit, label):
    assert tc.session_label(session, dirname, known, explicit)[0] == label


def test_parse_dir_spec(tmp_path):
    assert tc.parse_dir_spec('apple=%s' % tmp_path) == ('apple', tmp_path)
    assert tc.parse_dir_spec(str(tmp_path)) == (None, tmp_path)
    assert tc.parse_dir_spec('/data/a=b/c') == (None, Path('/data/a=b/c'))


def test_output_paths_and_frames_per_view(tmp_path):
    cfg = load_config(use_overlays=False)
    cfg.set('logging.dir', str(tmp_path))
    assert tc.model_out_path(cfg, 'real') == cfg.path(tc.MODEL_KEY)
    assert tc.model_out_path(cfg, 'sim') == tmp_path / 'sim_site' / 'knn_model.npz'
    assert tc.model_out_path(cfg, 'sim', str(tmp_path / 'm.npz')) == tmp_path / 'm.npz'
    assert tc.report_path(tmp_path / 'm.npz') == tmp_path / 'm_report.json'
    assert tc.frames_per_view(cfg, 'real') == 3 and tc.frames_per_view(cfg, 'replay') == 1
    assert tc.knn_params(cfg) == (3, 3.0)


def test_config_hints(tmp_path):
    cfg = load_config(use_overlays=False)
    path = tmp_path / 'm.npz'
    text = '\n'.join(tc.config_hints(cfg, 'L1', 'real', path, ['red', 'pink'],
                                     ['red', 'green', 'blue']))
    assert '--set perception.classifier.knn.model_file=%s' % path in text
    assert 'tasks.L1.classifier' in text and 'class_to_zone has no zone for red, pink' in text
    hints = tc.config_hints(cfg, 'L2', 'real', cfg.path(tc.MODEL_KEY), ['apple'], [])
    assert 'the tools read it from' in hints[0]
    assert 'tasks.L2.labels: [apple]' in '\n'.join(hints)


def test_layout_unchanged():
    a = [(0.3, 0.0, 0.0), (0.2, 0.1, 1.0)]
    assert tc.layout_unchanged(a, [(0.302, 0.001, math.pi - 0.02), (0.2, 0.1, 1.03)])
    assert not tc.layout_unchanged(a, [(0.32, 0.0, 0.0), (0.2, 0.1, 1.0)])
    assert not tc.layout_unchanged(a, [(0.3, 0.0, 0.5), (0.2, 0.1, 1.0)])
    assert not tc.layout_unchanged([], []) and not tc.layout_unchanged(a, a[:1])


def test_instance_rows_rejected_summary_and_drawing():
    a = Detection(label='unknown', score=0.0, center_px=(10.0, 30.0),
                  center_base=np.array([0.3, -0.05, 0.06]), length=0.07, width=0.06,
                  height=0.06, yaw=0.2)
    b = Detection(label='unknown', score=0.0, valid=False, reject_reason='too_small')
    rows_ = tc.instance_rows([a, a], [True, False], {0: 'apple'})
    assert rows_[0] == ['0', '70x60x60', '[300, -50]', 'apple'] and 'incomplete' in rows_[1][3]
    assert tc.rejected_summary([a, b, b]) == 'too_small x2' and tc.rejected_summary([a]) == ''
    assert tc.layout_of([a, b]) == [(0.3, -0.05, 0.2)]
    img = np.zeros((40, 80, 3), np.uint8)
    assert tc.draw_instances(img, [a], {0: 'apple'}, [True]) is img and img.any()
    edge = Detection(label='unknown', score=0.0, center_px=(79.0, 39.0))
    img = np.zeros((40, 200, 3), np.uint8)
    tc.draw_instances(img, [edge], {0: 'green_pepper'}, [True])
    assert img[:, 120:].any() and not img[:, -1].any()     # moved left, not clipped
    pred = {0: ('pear', 0.4), 1: ('pear', 0.9)}
    rows_ = tc.instance_rows([a, a], [True, True], {0: 'apple'}, pred)
    assert rows_[0][3] == 'apple' and rows_[1][3] == 'pear? (predicted, score 0.90)'


class _View:
    """Teacher state of one view, without camera or arm (for the helpers that label it)."""

    def __new__(cls, tmp_path, n_dets, single='pear'):
        ui, _fn, out = make_ui(tmp_path)
        t = tc._Teacher.__new__(tc._Teacher)
        t.ui, t.book, t.known, t.task, t.accepted = ui, tc.SampleBook(), [], 'L2', set()
        t.single, t.pred, t.checked, t.view, t.frame = single, {}, [0, 0], 1, None
        t.source = 'live'
        t.dets = [Detection(label='unknown', score=0.0) for _ in range(n_dets)]
        t.usable, t.X = [True] * n_dets, rows(n_dets, 1.0)
        t.out_text = out
        return t


def test_single_object_mode_labels_only_a_view_with_one_instance(tmp_path):
    t = _View(tmp_path, 1)
    t._label_single()
    assert t.book.counts() == {'pear': 1} and t.book.batches[-1].source == 'single'
    t = _View(tmp_path, 2)
    t._label_single()
    assert t.book.counts() == {} and '2 instances in this view' in t.out_text.getvalue()


def test_accept_takes_the_predictions_and_counts_the_right_ones(tmp_path):
    t = _View(tmp_path, 3, single=None)
    t.pred = {0: ('pear', 0.9), 1: ('banana', 0.8), 2: ('unknown', 0.0)}
    assert t.label('1 potato')                         # a wrong prediction, fixed by number
    assert t.accept()
    assert t.book.labeled_in_view(1) == {0: 'pear', 1: 'potato'}
    assert t.checked == [1, 2] and '#2 predicted unknown' in t.out_text.getvalue()
    assert not t.accept()                              # nothing left to accept


def test_auto_capture_waits_for_a_still_arm_at_a_new_pose(tmp_path, monkeypatch):
    clock = SimClock()
    poses = []
    monkeypatch.setattr(tc, 'read_pose', lambda arm: type('P', (), {
        'T_base_flange': poses.pop(0)})())
    t = _View(tmp_path, 0)
    t.s = type('S', (), {'arm': object(), 'clock': clock})()
    t.view_pose, t._last_pose, t._still_since = np.eye(4), None, None
    shots = []
    t.new_view = lambda: shots.append(clock.now()) or setattr(t, 'view_pose', t._last_pose)

    def at(x):
        T = np.eye(4)
        T[0, 3] = x
        return T
    for x in [0.0, 0.01, 0.02, 0.05] + [0.05] * 6 + [0.051] * 3 + [0.05] * 8:
        poses.append(at(x))
        t.auto_tick()
        clock.sleep(0.1)
    assert len(shots) == 1                      # still for 0.6 s, 50 mm from the last view
    for _ in range(10):                         # still at the view just taken: no new shot
        poses.append(at(0.05))
        t.auto_tick()
        clock.sleep(0.1)
    assert len(shots) == 1


def test_wrap_lines_fits_the_panel_width():
    lines = tc.wrap_lines(['samples: ' + ', '.join('class%02d 3 (<8)' % i for i in range(11)),
                           'short'], 360)
    assert len(lines) > 2 and all(len(s) <= 40 for s in lines) and lines[-1] == 'short'
    assert lines[1].startswith('  ')


# ------------------------------------------------------------------ sim helpers
def sim_cfg(info):
    base = load_config(use_overlays=False)
    return Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, base.sources)


def render(world, info, seed):
    cam = still_wrist_camera(world, SimClock(), info['intrinsics'],
                             info['T_base_camera_observe'], seed=seed)
    cam.start()
    frame = cam.get_frame()
    cam.stop()
    return frame


def record_sessions(root, specs, n_frames=3):
    """Save n_frames sim frames per (directory, label, extra) showing only objects of label."""
    counts = [0] * len(specs)
    for seed in range(200):
        world, info = make_scenario('L2', seed)
        for k, (dirname, label, extra) in enumerate(specs):
            objs = [o for o in world.objects if o.label == label]
            if counts[k] >= n_frames or not objs:
                continue
            one = SimWorld(table_z=0.0, objects=objs, zones=list(world.zones), seed=seed)
            save_frame(root / dirname, counts[k], render(one, info, seed), extra=extra)
            counts[k] += 1
        if min(counts) >= n_frames:
            return
    raise AssertionError('not enough sim scenes for %s' % (specs,))


# ------------------------------------------------------------------ main()
def test_real_mode_without_extrinsics_stops_before_any_hardware(tmp_path, monkeypatch):
    def no_session(*args, **kwargs):
        raise AssertionError('must not open the camera or the arm')

    monkeypatch.setattr(tc, 'open_session', no_session)
    ui, fn, out = make_ui(tmp_path, ['q'])
    rc = tc.main(['--task', 'L2', '--headless', '--log-dir', str(tmp_path),
                  '--out', str(tmp_path / 'm.npz'), '--set', 'camera.extrinsics_inline=null',
                  '--set', 'camera.extrinsics_file=%s' % (tmp_path / 'none.yaml')], ui=ui)
    assert rc == 2 and 'needs the camera extrinsics' in out.getvalue()
    assert not (tmp_path / 'm.npz').exists()


def test_from_dir_learns_one_class_per_session_and_appends(tmp_path):
    cap = tmp_path / 'cap'
    record_sessions(cap, [('20261112_apple_1', 'apple', {'session': 'apple_1'}),
                          ('carrot', 'carrot', {'session': 'capture'}),
                          ('misc', 'eggplant', {}),
                          ('lemon', 'lemon', {'session': 'lemon'})])
    model = tmp_path / 'models' / 'knn.npz'
    common = ['--mode', 'sim', '--task', 'L2', '--log-dir', str(tmp_path), '--headless',
              '--set', 'tasks.L2.labels=[apple, carrot, eggplant, lemon]', '--out', str(model)]
    ui, fn, out = make_ui(tmp_path)
    rc = tc.main(common + ['--from-dir', str(cap / '20261112_apple_1'), str(cap / 'carrot'),
                           'eggplant=%s' % (cap / 'misc')], ui=ui)
    text = out.getvalue()
    assert rc == 0, text
    assert fn.asked == []                      # all labels are task labels: no questions
    assert "label 'apple' (from session 'apple_1')" in text
    assert "label 'carrot' (from directory 'carrot')" in text
    assert "label 'eggplant' (from LABEL=DIR)" in text
    assert 'tasks.L2.labels without samples: lemon' in text
    knn = KnnClassifier.load(model)
    first = Counter(knn.y)
    assert knn.labels == ['apple', 'carrot', 'eggplant'] and min(first.values()) >= 3
    rep = json.loads(tc.report_path(model).read_text())
    assert rep['source'] == 'from_dir' and rep['feature_names'] == list(FEATURE_NAMES)
    assert rep['samples_per_class'] == dict(first) and rep['created'] == knn.created
    assert rep['accuracy'] is not None and set(rep['confusion']) == set(first)
    assert [d['frames'] for d in rep['dirs']] == [3, 3, 3]

    # --append keeps the old samples; re-learning the apple session adds no duplicates
    ui, fn, out = make_ui(tmp_path)
    rc = tc.main(common + ['--append', '--from-dir', str(cap / 'lemon'),
                           str(cap / '20261112_apple_1')], ui=ui)
    assert rc == 0, out.getvalue()
    assert '--append: keeping the' in out.getvalue()
    second = Counter(KnnClassifier.load(model).y)
    assert second['lemon'] >= 3 and all(second[k] == v for k, v in first.items())
    assert len(list(model.parent.glob('knn.bak_*.npz'))) == 1
    assert len(list(model.parent.glob('knn_report.bak_*.json'))) == 1

    ui, fn, out = make_ui(tmp_path)
    assert tc.main(common + ['--from-dir', str(tmp_path / 'missing')], ui=ui) == 1
    assert 'not a directory' in out.getvalue() and 'nothing written' in out.getvalue()


def test_sim_session_writes_a_model_that_classifies_a_new_scene(tmp_path):
    cmds = ['h', 'all banana', 'u', '0 banan', 'g', 's', 'c', 'g', 'e', 'w', 'c', 'g', 'q']
    ui, fn, out = make_ui(tmp_path, cmds, answers={'anyway': 'n'})
    rc = tc.main(['--mode', 'sim', '--task', 'L2', '--seed', '3', '--sim-scenes', '10',
                  '--log-dir', str(tmp_path), '--headless'], ui=ui)
    text = out.getvalue()
    assert rc == 0, text
    assert "undid 'all banana' of view 1" in text and 'did you mean banana' in text
    assert 'not labeled (nothing changed)' in text and 'leave-one-out:' in text
    assert any('before quitting' in p for p in fn.asked)       # unsaved view 3 labels
    site = tmp_path / 'sim_site'
    model = site / 'knn_model.npz'
    rep = json.loads(tc.report_path(model).read_text())
    assert rep['source'] == 'sim' and rep['views'] == 3 and set(rep['sources']) == {'sim_gt'}
    assert rep['mode'] == 'sim' and rep['task'] == 'L2'
    assert len(list(site.glob('knn_model.bak_*.npz'))) == 1    # the second write kept the first
    assert (tmp_path / 'preview' / 'teach_classes.jpg').is_file()

    knn = KnnClassifier.load(model)
    world, info = make_scenario('L2', 7)
    res = Perception(sim_cfg(info), classifier=knn).process(render(world, info, 7),
                                                            info['T_base_camera_observe'])
    pairs = [(d.label, ground_truth_label(world, d)) for d in res.valid_detections()]
    pairs = [p for p in pairs if p[1] is not None]
    assert len(pairs) >= 5, pairs
    assert sum(a == b for a, b in pairs) >= 0.9 * len(pairs), pairs


def test_sim_views_are_predicted_and_checked_against_the_ground_truth(tmp_path):
    cmds = ['g', 's', 'c', 'y', 'single banana', 'c', 'single off', 'q']
    ui, fn, out = make_ui(tmp_path, cmds, answers={'before quitting': 'n'})
    rc = tc.main(['--mode', 'sim', '--task', 'L2', '--seed', '4', '--sim-scenes', '12',
                  '--log-dir', str(tmp_path), '--headless'], ui=ui)
    text = out.getvalue()
    assert rc == 0, text
    assert 'predicted (label?)' in text and 'predictions right so far' in text
    m = [ln for ln in text.splitlines() if ln.startswith('predictions right so far')][0]
    right, n = (int(v) for v in m.split(':')[1].split('(')[0].split('/'))
    assert n >= 4 and right >= 0.8 * n, m
    assert 'single-object mode: keep one banana' in text
    assert 'single-object mode (banana):' in text                 # the sim scene has several
    assert 'single-object mode off' in text

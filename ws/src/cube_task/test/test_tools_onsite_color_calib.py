"""color_calib: command parsing, HSV range overlap analysis, colors.yaml merge, sim runs."""
from __future__ import annotations

import argparse
import io
import re
from types import SimpleNamespace

from cube_task.core.config import load_config, read_yaml
from cube_task.core.types import ArmHealth, Detection, MotionResult
from cube_task.perception.classify_color import ColorClassifier
from cube_task.tools import color_calib as cc
from cube_task.tools import common, onsite
from cube_task.tools.onsite import write_site_yaml
from cube_task.tools.ui import Ui
import numpy as np
import pytest

LABELS = ['red', 'green', 'blue']
KEY = 'perception.classifier.color.classes'


def hsv_samples(hue, n=400, spread=3.0, s=(170, 240), v=(80, 230), seed=0):
    """Return n synthetic OpenCV HSV pixels around hue (wrapping at 180)."""
    rng = np.random.default_rng(seed)
    h = np.rint(rng.normal(hue, spread, n)) % 180
    sat = rng.integers(s[0], s[1] + 1, n)
    val = rng.integers(v[0], v[1] + 1, n)
    return np.stack([h, sat, val], axis=1).astype(np.uint8)


# ------------------------------------------------------------------ command parsing
def test_parse_command_assignments():
    c = cc.parse_command('0 red; 1 green', 3, LABELS)
    assert c.kind == 'assign' and c.pairs == [(0, 'red'), (1, 'green')]
    assert cc.parse_command('all blue', 3, LABELS).pairs == [(0, 'blue'), (1, 'blue'), (2, 'blue')]
    assert cc.parse_command('2,0 RED', 3, LABELS).pairs == [(0, 'red'), (2, 'red')]
    # full-width separators from a Chinese input method
    c = cc.parse_command('0 red；1,2 Blue', 3, LABELS)
    assert c.pairs == [(0, 'red'), (1, 'blue'), (2, 'blue')]
    assert cc.parse_command('0 red;', 3, LABELS).pairs == [(0, 'red')]


def test_parse_command_rejects_the_whole_line():
    for text, needle in (('0 red; 3 blue', '0..2'), ('0 purple', 'tasks.L1.labels'),
                         ('0 red; 0 green', 'labeled twice'), ('0 red green', 'expected'),
                         ('0 unknown', 'label'), (';', 'nothing to label'),
                         ('x', 'unknown command')):
        c = cc.parse_command(text, 3, LABELS)
        assert c.kind == 'error' and needle in c.error, (text, c.error)
        assert c.pairs == []
    assert 'no instances' in cc.parse_command('all red', 0, LABELS).error


def test_parse_command_words_and_margins():
    kinds = {'c': 'capture', 'U': 'undo', 'p': 'preview', 's': 'summary', 'w': 'write',
             'q': 'quit', 'esc': 'quit', 'h': 'help', '?': 'help', '': 'refresh',
             '  ': 'refresh'}
    for text, kind in kinds.items():
        assert cc.parse_command(text, 3, LABELS).kind == kind, text
    assert cc.parse_command('m 3', 3).values == [3.0]
    assert cc.parse_command('m 2.5 20', 3).values == [2.5, 20.0]
    for bad in ('m', 'm -1', 'm a', 'm 1 2 3'):
        assert cc.parse_command(bad, 3).kind == 'error', bad


def test_match_label_without_task_labels_accepts_any_label():
    assert cc.match_label('white', []) == ('white', '')
    assert cc.match_label('Red', ['red']) == ('red', '')
    lab, err = cc.match_label('pink', ['red'], 'tasks.L1.labels')
    assert lab is None and 'tasks.L1.labels' in err
    assert cc.parse_command('0 white', 1, []).pairs == [(0, 'white')]


# ------------------------------------------------------------------ samples
def test_sample_book_relabel_and_undo():
    b = cc.SampleBook()
    red, green = hsv_samples(0, 50), hsv_samples(60, 30)
    assert b.assign(0, [(0, 'red', red), (1, 'green', green)]) == []
    assert b.assign(1, [(0, 'red', red)]) == []
    assert b.views_by_label() == {'green': [0], 'red': [0, 1]}
    assert b.pixels_by_label()['red'].shape == (100, 3)
    notes = b.assign(0, [(1, 'red', green)])            # relabel #1 of view 0
    assert notes == ['#1 was green, now red'] and b.labels() == ['red']
    assert b.undo() == ['view 0 #1 red (back to green)']
    assert b.label_of(0, 1) == 'green'
    state = b.state()
    assert b.undo() == ['view 1 #0 red'] and b.label_of(1, 0) is None
    assert b.state() != state
    assert b.undo() is not None and b.samples == {}
    assert b.undo() is None


# ------------------------------------------------------------------ ranges / overlap
def test_red_wrapping_around_hue_0_180():
    red = np.concatenate([hsv_samples(178, seed=1), hsv_samples(1, seed=2)])
    assert red[:, 0].min() == 0 and red[:, 0].max() >= 175     # really wraps
    ranges, errors = cc.compute_ranges({'red': red}, cc.RangeOptions())
    assert errors == {}
    rr = ranges['red']
    assert len(rr) == 2 and rr[0][1][0] == 180 and rr[1][0][0] == 0
    assert cc.inside_fraction(red, rr) > 0.95
    st = cc.hue_stats(red)
    assert cc.hue_distance(st['hue'], 0.0) < 3.0 and st['spread'] < 15.0
    assert '|0-' in cc.fmt_ranges(rr) and cc.fmt_ranges(rr).startswith('H 1')
    # the classifier accepts the ranges and labels the samples red
    clf = ColorClassifier({'red': rr, 'green': [[[50, 60, 40], [70, 255, 255]]]})
    fr = clf.fractions(red)
    assert fr['red'] > 0.95 and fr['green'] == 0.0


def test_overlapping_orange_red_pair_is_reported():
    samples = {'red': hsv_samples(4, spread=3.0, seed=3),
               'orange': hsv_samples(10, spread=3.0, seed=4),
               'green': hsv_samples(60, seed=5)}
    ranges, _ = cc.compute_ranges(samples, cc.RangeOptions())
    fr = cc.overlap_fractions(samples, ranges)
    assert set(fr['red']) == {'orange', 'green'}
    assert fr['red']['orange'] > 0.2 and fr['orange']['red'] > 0.2
    assert fr['red']['green'] == 0.0 and fr['green']['red'] == 0.0
    pairs = cc.overlap_pairs(fr, threshold=0.05)
    assert [(a, b) for a, b, _f, _g in pairs] == [('orange', 'red')]
    w = cc.overlap_warnings(pairs)
    assert len(w) == 1 and w[0].startswith('orange / red overlap') and 'capture more' in w[0]
    # a narrower hue margin reduces the overlap
    narrow, _ = cc.compute_ranges(samples, cc.RangeOptions(hue_margin=0.0, lo_pct=10,
                                                           hi_pct=90))
    fr2 = cc.overlap_fractions(samples, narrow)
    assert fr2['red']['orange'] < fr['red']['orange']
    # well separated classes give no warning
    two = ('red', 'green')
    assert cc.overlap_pairs(cc.overlap_fractions({k: samples[k] for k in two},
                                                 {k: ranges[k] for k in two})) == []


def test_overlap_with_a_kept_class_without_samples():
    samples = {'red': hsv_samples(10, spread=2.0, seed=6)}
    ranges, _ = cc.compute_ranges(samples, cc.RangeOptions())
    kept = {'orange': [[[9, 110, 80], [20, 255, 255]]]}         # default.yaml orange
    eff = cc.effective_classes(kept, ranges)
    pairs = cc.overlap_pairs(cc.overlap_fractions(samples, eff))
    assert len(pairs) == 1 and pairs[0][:2] == ('red', 'orange') and pairs[0][3] is None
    w = cc.overlap_warnings(pairs)[0]
    assert 'orange has no samples' in w and 'label orange blocks too' in w


def test_view_weak_and_missing_label_warnings():
    w = cc.view_warnings({'red': [0, 1], 'green': [2, 2]}, min_views=2)
    assert len(w) == 1 and w[0].startswith('green: samples from 1 view only')
    assert cc.view_warnings({'red': [0]}, min_views=1) == []
    good = cc.InstanceSample(0, 0, 'red', hsv_samples(0, seed=7))
    bad = cc.InstanceSample(1, 2, 'red', hsv_samples(60, seed=8))      # mislabeled green
    ranges = {'red': [[[170, 100, 50], [180, 255, 255]], [[0, 100, 50], [8, 255, 255]]]}
    w = cc.weak_instance_warnings([good, bad], ranges, 0.55)
    assert len(w) == 1 and w[0].startswith('view 1 #2 (red): only 0 %')
    assert cc.missing_label_warnings(LABELS, {'red': 1, 'green': 1}) == [
        'labels without HSV ranges: blue; the color classifier refuses to start until they '
        'are calibrated']
    assert cc.missing_label_warnings([], {}) == []


def test_effective_classes_follow_the_task_labels():
    base = {'red': 'old', 'orange': 'o', 'green': 'g'}
    eff = cc.effective_classes(base, {'red': 'new', 'blue': 'b'}, ['red', 'green', 'blue'])
    assert eff == {'red': 'new', 'green': 'g', 'blue': 'b'}
    assert set(cc.effective_classes(base, {'red': 'new'})) == {'red', 'orange', 'green'}


def test_hue_stats_flags_mixed_colors_and_achromatic_samples():
    mixed = np.concatenate([hsv_samples(0, 200, seed=9), hsv_samples(60, 200, seed=10)])
    assert cc.hue_stats(mixed)['spread'] > cc.MIXED_HUE_SPREAD
    gray = hsv_samples(30, 100, s=(0, 15), seed=11)
    st = cc.hue_stats(gray)
    assert st['hue'] is None and st['spread'] is None and st['s'] < 16
    empty = {'n': 0, 'hue': None, 'spread': None, 's': None, 'v': None}
    assert cc.hue_stats(np.zeros((0, 3), np.uint8)) == empty
    # an achromatic class gets the full hue range, like hsv_ranges_from_samples documents
    ranges, _ = cc.compute_ranges({'white': gray}, cc.RangeOptions())
    assert ranges['white'][0][0][0] == 0 and ranges['white'][0][1][0] == 180
    _r, errors = cc.compute_ranges({'red': hsv_samples(0, 5)}, cc.RangeOptions())
    assert 'at least 10' in errors['red']


# ------------------------------------------------------------------ colors.yaml
EXISTING = {'perception': {'classifier': {'color': {
    'classes': {'red': [[[0, 1, 2], [3, 4, 5]]], 'yellow': [[[21, 90, 80], [34, 255, 255]]]},
    'min_fraction': 0.4}}}, 'note': 'kept'}
NEW = {'red': [[[175, 160, 70], [180, 255, 250]], [[0, 160, 70], [6, 255, 250]]],
       'blue': [[[np.int64(104), 150, 60], [116, 255, 255]]]}


def test_build_colors_yaml_keeps_other_classes_unless_replace():
    data, keys, kept, removed = cc.build_colors_yaml(EXISTING, NEW)
    classes = data['perception']['classifier']['color']['classes']
    assert list(classes) == ['red', 'yellow', 'blue']
    assert classes['red'] == NEW['red'] and classes['yellow'] == [[[21, 90, 80], [34, 255, 255]]]
    assert type(classes['blue'][0][0][0]) is int
    assert data['perception']['classifier']['color']['min_fraction'] == 0.4
    assert data['note'] == 'kept'
    assert keys == ['%s.red' % KEY, '%s.blue' % KEY] and kept == ['yellow'] and removed == []
    assert EXISTING['perception']['classifier']['color']['classes']['red'] == [[[0, 1, 2],
                                                                                [3, 4, 5]]]
    data, keys, kept, removed = cc.build_colors_yaml(EXISTING, NEW, replace=True)
    assert list(data['perception']['classifier']['color']['classes']) == ['red', 'blue']
    assert kept == [] and removed == ['yellow']
    data, keys, _k, _r = cc.build_colors_yaml({}, {'green': [[[50, 60, 40], [70, 255, 255]]]})
    assert data == {'perception': {'classifier': {'color': {'classes': {
        'green': [[[50, 60, 40], [70, 255, 255]]]}}}}}
    assert cc.build_colors_yaml({'perception': 'junk'}, NEW)[0]['perception']['classifier']


def test_written_colors_yaml_round_trips_with_backup(tmp_path):
    path = tmp_path / 'site' / 'colors.yaml'
    data, _keys, _k, _r = cc.build_colors_yaml({}, NEW)
    assert write_site_yaml(path, data, header='first') is None
    text = path.read_text(encoding='utf-8')
    assert text.startswith('# first\n')
    assert 'red: [[[175, 160, 70], [180, 255, 250]], [[0, 160, 70], [6, 255, 250]]]' in text
    back = read_yaml(path)
    assert back['perception']['classifier']['color']['classes'] == {
        k: cc.plain_ranges(v) for k, v in NEW.items()}
    green = {'green': [[[50, 60, 40], [70, 255, 255]]]}
    data2, _keys, kept, _r = cc.build_colors_yaml(back, green)
    backup = write_site_yaml(path, data2, header='second')
    assert backup is not None and backup.is_file() and kept == ['red', 'blue']
    assert read_yaml(backup) == back
    assert cc.overridden_classes(read_yaml(path)['perception']['classifier']['color'][
        'classes'], NEW) == []


def test_overridden_classes_and_same_layout():
    assert cc.overridden_classes({'red': NEW['red']}, NEW) == ['blue']
    assert cc.overridden_classes(None, {'red': NEW['red']}) == ['red']
    assert cc.overridden_classes({'red': 'junk'}, {'red': NEW['red']}) == ['red']
    prev = [[0.2, 0.0], [0.3, 0.1]]
    assert cc.same_layout(prev, [[0.305, 0.1], [0.2, 0.004]])
    assert not cc.same_layout(prev, [[0.2, 0.0], [0.35, 0.1]])
    assert not cc.same_layout(prev, [[0.2, 0.0]])
    assert not cc.same_layout(None, prev) and not cc.same_layout([], [])


def test_instance_hsv_erodes_the_mask_border_and_caps_pixels():
    color = np.zeros((40, 40, 3), np.uint8)
    color[10:30, 10:30] = (0, 200, 0)            # 2 px green rim (a neighbor / the table)
    color[12:28, 12:28] = (0, 0, 200)            # red top face
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    det = Detection(label='unknown', score=0.0, bbox=(8, 8, 32, 32), mask=mask)
    frame = SimpleNamespace(color=color)
    raw = cc.instance_hsv(frame, det, erode_px=0, max_pixels=4000)
    assert raw.shape == (400, 3) and np.any(raw[:, 0] == 60)
    er = cc.instance_hsv(frame, det, erode_px=2, max_pixels=4000)
    assert er.shape == (256, 3) and np.all(er[:, 0] == 0)
    assert cc.instance_hsv(frame, det, erode_px=2, max_pixels=100).shape == (100, 3)
    # too small to erode: the uneroded mask is used
    small = np.zeros((40, 40), bool)
    small[12:16, 12:17] = True
    det2 = Detection(label='unknown', score=0.0, bbox=(12, 12, 17, 16), mask=small)
    assert cc.instance_hsv(frame, det2, erode_px=2).shape == (20, 3)
    assert cc.instance_hsv(SimpleNamespace(color=None), det).shape == (0, 3)


def test_preview_rows_apply_min_score_and_report_mismatches():
    dets = [SimpleNamespace(label='red', score=0.9, extra={'color_fractions': {
                'red': 0.9, 'blue': 0.05}}),
            SimpleNamespace(label='green', score=0.5, extra={'color_fractions': {
                'green': 0.5, 'red': 0.0}}),
            SimpleNamespace(label='unknown', score=0.0, extra={})]
    rows, problems, checked = cc.preview_rows(dets, ['red', 'green', None], 0.55)
    assert [r[2] for r in rows] == ['red', 'unknown', 'unknown']
    assert rows[0][4] == 'blue 0.05' and rows[2][4] == '-'
    assert checked == 2 and problems == ['#1: expected green, classified unknown (fraction 0.50)']
    rows, problems, checked = cc.preview_rows(dets, [None, None, None], 0.55,
                                              truth=['red', 'green', None])
    assert checked == 2 and len(problems) == 1 and rows[0][-1] == 'red' and rows[2][-1] == '?'


def test_bad_options_are_refused(tmp_path):
    p = cc.build_parser()
    for extra in (['--lo-pct', '60', '--hi-pct', '40'], ['--hue-margin', '-1'],
                  ['--erode-px', '-1'], ['--min-views', '0'], ['--overlap-warn', '0'],
                  ['--average', '0'], ['--max-pixels', '10']):
        with pytest.raises(ValueError):
            cc.options_from_args(p.parse_args(extra))
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=lambda _p: 'q', output=out)
    assert cc.main(['--mode', 'sim', '--log-dir', str(tmp_path), '--hue-margin', '99'],
                   ui=ui) == 2
    assert 'hue margin' in out.getvalue()


# ------------------------------------------------------------------ real-arm safety (fake arm)
class FakeRealArm:
    """Records commands; never moves anything."""

    def __init__(self, health):
        self._health = health
        self.calls = []

    def health(self):
        return self._health

    def set_speed_percent(self, percent):
        self.calls.append(('speed', percent))

    def move_joints(self, q, **_kw):
        self.calls.append(('move_joints', [round(v, 3) for v in q]))
        return MotionResult(ok=True)


def _real_session(arm):
    cfg = load_config(use_overlays=False)
    cfg.set('robot.speed_percent', 30)
    cfg.set('robot.poses.observe_joints', [0.1, 0.2, -0.2, 0.0, 0.0, 0.0])
    return SimpleNamespace(cfg=cfg, arm=arm)


@pytest.mark.parametrize('answer, moved', [('n', False), ('y', True)])
def test_observe_motion_asks_first_and_clamps_the_speed(answer, moved):
    arm = FakeRealArm(ArmHealth(ok=True, enabled=True, ctrl_mode=1))
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=lambda _p: answer, output=out)
    args = SimpleNamespace(mode='real', observe=True, speed=50)
    assert cc.prepare_pose(args, _real_session(arm), ui) is moved
    text = out.getvalue()
    assert 'MOTION: move_joints to the observe pose' in text and 'at 30 % speed' in text
    assert '--speed 50 is above robot.speed_percent 30' in text
    assert arm.calls[0] == ('speed', 30)
    assert [c[0] for c in arm.calls].count('move_joints') == (1 if moved else 0)


def test_observe_motion_waits_while_in_teach_mode_and_real_mode_reads_only():
    arm = FakeRealArm(ArmHealth(ok=False, reason='teach', ctrl_mode=onsite.CTRL_TEACHING))
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=lambda _p: 'q', output=out)
    args = SimpleNamespace(mode='real', observe=True, speed=None)
    assert cc.prepare_pose(args, _real_session(arm), ui) is False
    assert arm.calls == [] and 'drag-teach mode' in out.getvalue()
    # without --observe the real arm is never commanded
    arm2 = FakeRealArm(ArmHealth(ok=True, enabled=True))
    assert cc.prepare_pose(SimpleNamespace(mode='real', observe=False, speed=None),
                           _real_session(arm2), ui) is True
    assert arm2.calls == []


# ------------------------------------------------------------------ scripted sim runs
class ScriptedOperator:
    """Answer prompts in order; a callable answer gets the output printed so far."""

    def __init__(self, script):
        self.out = io.StringIO()
        self.script = list(script)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.script:
            raise EOFError
        a = self.script.pop(0)
        return a(self.out.getvalue()) if callable(a) else a


def label_by_truth(text):
    """Return '0 red; 1 green; ...' from the last '[sim] ground truth' line."""
    line = re.findall(r'\[sim\] ground truth: (.*)', text)[-1]
    return '; '.join(part.lstrip('#') for part in line.split(', '))


def run_sim(tmp_path, script, *extra):
    op = ScriptedOperator(script)
    ui = Ui(headless=True, preview_dir=str(tmp_path / 'preview'), input_fn=op, output=op.out)
    code = cc.main(['--mode', 'sim', '--seed', '3', '--log-dir', str(tmp_path / 'log'),
                    '--headless'] + list(extra), ui=ui)
    return code, op.out.getvalue()


def classify_new_scene(colors_file, seed):
    """Run L1 perception on a new sim scene with colors_file; return dets, truth, clf, classes."""
    p = argparse.ArgumentParser()
    common.add_common_args(p)
    args = p.parse_args(['--mode', 'sim', '--seed', str(seed), '--config', str(colors_file)])
    ui = Ui(headless=True, output=io.StringIO())
    with common.open_session(args, arm=True, camera=True, perception=True) as s:
        ok, why = onsite.go_observe(s.arm, s.cfg, onsite.MotionGuard(ui, 'sim', 100))
        assert ok, why
        frame = onsite.capture_view(s.camera, s.cfg)
        T_bc, why = onsite.camera_pose_for_frame(onsite.load_extrinsics_or_none(s.cfg), frame,
                                                 s.arm, onsite.flange_to_tcp(s.cfg))
        assert T_bc is not None, why
        clf = s.perception.classifier
        res = s.perception.process(frame, T_bc)
        truth = [onsite.ground_truth_label(s.world, d) for d in res.valid_detections()]
        return res.valid_detections(), truth, clf, s.cfg.get(KEY)


def test_sim_run_labels_three_blocks_writes_and_classifies_a_new_scene(tmp_path):
    code, text = run_sim(tmp_path, [label_by_truth, 'c', label_by_truth, 'p', 'w', 'q'])
    assert code == 0, text
    path = tmp_path / 'log' / 'sim_site' / 'colors.yaml'
    assert 'wrote %s' % path in text
    for lab in LABELS:
        assert '    %s.%s\n' % (KEY, lab) in text
    assert '[sim operator] rearrange the blocks' in text
    after = text.split('labeled in view 1')[1]
    assert 'no warnings' in after and 'preview OK: 3 of 3 instances' in after
    assert (tmp_path / 'preview' / 'color_calib.jpg').is_file()
    assert (tmp_path / 'preview' / 'color_calib_preview.jpg').is_file()
    assert 'done: %s written 1 time' % path in text
    written = read_yaml(path)['perception']['classifier']['color']['classes']
    assert set(written) == set(LABELS)
    assert len(written['red']) == 2                 # the red range wraps around hue 0 / 180
    dets, truth, clf, cfg_classes = classify_new_scene(path, seed=7)
    assert isinstance(clf, ColorClassifier) and set(clf.ranges) == set(LABELS)
    assert {k: cfg_classes[k] for k in LABELS} == written
    assert len(dets) == 3 and sorted(truth) == sorted(LABELS)
    assert [d.label for d in dets] == truth
    assert min(d.score for d in dets) > 0.8


def test_sim_run_rejects_bad_lines_undoes_and_never_writes_unasked(tmp_path):
    script = ['5 red', '0 purple', label_by_truth, 'u', label_by_truth,
              'w', 'n',              # warnings (one view only): do not write
              'q', 'n',              # unsaved: do not quit yet
              'm 3 20']              # then the input ends (EOF)
    code, text = run_sim(tmp_path, script, '--mount', 'eye_in_hand')
    assert code == 0, text
    assert "error: '5 red': instance numbers must be in 0..2" in text
    assert "error: label 'purple' is not in tasks.L1.labels (red, green, blue)" in text
    assert 'undone: view 0 #0 ' in text
    assert 'samples from 1 view only' in text and 'not written' in text
    assert 'margins: hue 3, S/V 20' in text
    assert 'quit: the ranges of this session were NOT written' in text
    assert not (tmp_path / 'log' / 'sim_site' / 'colors.yaml').exists()


@pytest.mark.parametrize('replace', [False, True])
def test_sim_run_merges_into_the_existing_file_with_backup(tmp_path, replace):
    path = tmp_path / 'log' / 'sim_site' / 'colors.yaml'
    write_site_yaml(path, EXISTING)
    extra = ['--replace'] if replace else []
    code, text = run_sim(tmp_path, [label_by_truth, 'w', 'y', 'q'], *extra)
    assert code == 0, text
    data = read_yaml(path)
    classes = data['perception']['classifier']['color']['classes']
    assert data['note'] == 'kept' and data['perception']['classifier']['color'][
        'min_fraction'] == 0.4
    assert classes['red'] != EXISTING['perception']['classifier']['color']['classes']['red']
    assert set(classes) == set(LABELS) | (set() if replace else {'yellow'})
    if replace:
        assert 'classes removed (--replace): yellow' in text
    else:
        assert 'classes kept from the previous file: yellow' in text
    backups = list(path.parent.glob('colors.bak_*.yaml'))
    assert len(backups) == 1 and read_yaml(backups[0]) == EXISTING
    assert 'previous file kept as %s' % backups[0] in text

"""teach_record: recording teach_classes views and reviewing the labeled samples."""
from __future__ import annotations

import io
import json

from cube_task.core.types import Detection
from cube_task.tools import teach_classes as tc
from cube_task.tools import teach_record as tr
from cube_task.tools.ui import Ui
import numpy as np


def _ui(tmp_path, commands=(), answers=None):
    cmds = list(commands)
    answers = dict(answers or {})

    def fn(prompt):
        for key, ans in answers.items():
            if key in prompt:
                return ans
        if '(y/n)' in prompt:
            return 'y'
        if not cmds:
            raise EOFError
        return cmds.pop(0)
    out = io.StringIO()
    return Ui(input_fn=fn, headless=True, preview_dir=str(tmp_path / 'preview'),
              output=out), out


def test_sample_ids_and_review_commands():
    assert tr.sample_id(0, 3, 1, 1) == '3#1' and tr.sample_id(1, 3, 1, 2) == '2:3#1'
    assert tr.parse_sample_id('3#1', 1) == (0, 3, 1)
    assert tr.parse_sample_id('2:3#1', 2) == (1, 3, 1)
    assert tr.parse_sample_id('3#1', 2) is None and tr.parse_sample_id('3:3#1', 2) is None
    assert tr.parse_sample_id('3-1', 1) is None
    parts, err = tr.parse_review_command('3#1 pear; 4#0 x', 1)
    assert err == '' and parts == [((0, 3, 1), 'pear'), ((0, 4, 0), '')]
    assert tr.parse_review_command('3#1', 1)[0] is None
    assert tr.parse_review_command('3:1 pear', 1)[0] is None


def test_labels_file_round_trip_and_instance_matching(tmp_path):
    assert tr.load_labels(tmp_path) == {}
    tr.save_labels(tmp_path, {2: {0: 'pear', 1: 'banana'}, 3: {}})
    assert tr.load_labels(tmp_path) == {2: {0: 'pear', 1: 'banana'}}
    assert json.loads((tmp_path / tr.LABELS_FILE).read_text())['views'] == {
        '2': {'0': 'pear', '1': 'banana'}}
    rec = [{'n': 0, 'center_px': [100.0, 100.0]}, {'n': 1, 'center_px': [300.0, 200.0]}]
    dets = [Detection(label='unknown', score=0.0, center_px=(305.0, 198.0)),
            Detection(label='unknown', score=0.0, center_px=(400.0, 400.0)),
            Detection(label='unknown', score=0.0, center_px=(104.0, 97.0))]
    assert tr.match_instances(rec, dets) == {0: 2, 1: 0}
    assert tr.match_instances(rec, dets[1:2]) == {}
    crop = tr.crop_of(np.zeros((50, 60, 3), np.uint8), (55, 45, 70, 60), pad=4)
    assert crop.shape == (9, 9, 3)


def test_live_session_is_recorded_then_reviewed_and_written(tmp_path):
    ui, out = _ui(tmp_path, ['g', 'c', '0 pear', 'q'], answers={'before quitting': 'n'})
    rc = tc.main(['--mode', 'sim', '--task', 'L2', '--seed', '4', '--sim-scenes', '0',
                  '--log-dir', str(tmp_path), '--headless'], ui=ui)
    text = out.getvalue()
    assert rc == 0, text
    rec = next((tmp_path / 'teach_classes').iterdir())
    assert 'recording every view to %s' % rec in text
    labels = tr.load_labels(rec)
    assert sorted(labels) == [1, 2] and labels[2] == {0: 'pear'} and len(labels[1]) >= 4
    assert (rec / '000002_meta.json').is_file() and (rec / '000002_instances.json').is_file()
    meta = json.loads((rec / '000001_meta.json').read_text())
    assert 'T_base_flange' in meta['extra']                       # eye_in_hand pose kept

    ui2, out2 = _ui(tmp_path, ['l', '2#0 x', 'e', 'w', 'q'])
    rc = tc.main(['--mode', 'sim', '--task', 'L2', '--seed', '4', '--log-dir', str(tmp_path),
                  '--headless', '--review', str(rec)], ui=ui2)
    text2 = out2.getvalue()
    assert rc == 0, text2
    n = sum(len(m) for m in labels.values())
    assert 'loaded %d labeled samples' % n in text2
    assert '2#0 (pear) dropped' in text2 and 'leave-one-out' in text2
    assert (tmp_path / 'sim_site' / 'knn_model.npz').is_file()
    assert 0 not in tr.load_labels(rec).get(2, {})
    assert list(rec.glob('labels.bak_*.json'))                    # the old labels kept
    assert (rec / 'review').is_dir() and list((rec / 'review').glob('*.jpg'))

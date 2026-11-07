"""Tests for core/logger.py (RunLogger, to_jsonable)."""
from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from cube_task.core.clock import SimClock
from cube_task.core.logger import RunLogger, to_jsonable
from cube_task.core.types import Detection


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture(name):
    # caplog does not work when the ROS launch_testing plugin replaces the logger class.
    lg = logging.getLogger(name)
    h = _Records()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def _events(log):
    with open(log.events_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def test_directory_name_and_events(tmp_path):
    clock = SimClock(start=500.0)
    log = RunLogger(tmp_path / 'runs', 'L1', clock)
    assert log.dir.parent == tmp_path / 'runs'
    assert re.fullmatch(r'\d{8}_\d{6}_L1', log.dir.name)
    clock.advance(1.25)
    log.event('observe', n=np.int64(3), z=np.float32(0.5), pose=np.eye(4),
              bad=float('nan'), ok=np.bool_(True), t=7, path=tmp_path)
    clock.advance(0.75)
    log.event('pick', label='red')
    ev = _events(log)
    assert [e['kind'] for e in ev] == ['observe', 'pick']
    assert ev[0]['t'] == 1.25 and ev[1]['t'] == 2.0
    assert ev[0]['n'] == 3 and ev[0]['z'] == 0.5 and ev[0]['ok'] is True
    assert ev[0]['pose'] == np.eye(4).tolist()
    assert ev[0]['bad'] is None              # strict JSON has no NaN
    assert ev[0]['t_'] == 7                  # reserved key renamed
    assert ev[0]['path'] == str(tmp_path)
    assert re.match(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}', ev[0]['wall'])
    assert log.n_events == 2 and log.n_failures == 0


def test_event_is_mirrored_to_python_logger(tmp_path):
    log = RunLogger(tmp_path, 'L2', SimClock())
    with _capture('cube_task.events') as records:
        log.event('grasp', width=0.0312, label='apple')
    msgs = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert len(msgs) == 1
    assert 'grasp' in msgs[0] and 'width=0.0312' in msgs[0] and 'label="apple"' in msgs[0]
    assert logging.getLogger('cube_task.events').parent.name in ('cube_task', 'root')


def test_name_override_and_no_reuse(tmp_path):
    a = RunLogger(tmp_path, 'L1', SimClock(), name='bench seed/1')
    b = RunLogger(tmp_path, 'L1', SimClock(), name='bench seed/1')
    assert a.dir.name == 'bench_seed_1'
    assert b.dir.name == 'bench_seed_1_1'


def test_save_image_json_text_and_summary(tmp_path):
    clock = SimClock()
    log = RunLogger(tmp_path, 'L1', clock)
    img = np.zeros((40, 60, 3), np.uint8)
    img[10:20, 10:30] = (0, 0, 255)
    p = log.save_image('obs_001', img)
    assert p.name == 'obs_001.jpg' and p.is_file()
    assert cv2.imread(str(p)).shape == (40, 60, 3)
    p = log.save_image('mask.png', np.eye(8, dtype=bool))
    assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).max() == 255
    depth_mm = (np.arange(100, dtype=np.uint16) * 600).reshape(10, 10)
    p = log.save_image('depth.png', depth_mm)
    assert np.array_equal(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), depth_mm)
    assert log.save_image('float', np.zeros((4, 4), np.float32)) is None   # no raise
    assert log.save_image('none', None) is None
    assert log.n_failures == 2

    det = Detection(label='red', score=0.9, mask=np.ones((480, 640), bool),
                    center_base=np.array([0.3, 0.0, 0.02]))
    p = log.save_json('detections', {'dets': [det], 'big': np.zeros((100, 100))})
    data = json.loads(p.read_text())
    assert p.name == 'detections.json'
    assert data['dets'][0]['label'] == 'red' and 'mask' not in data['dets'][0]
    assert data['big'] == {'ndarray': [100, 100], 'dtype': 'float64'}
    p = log.save_text('notes', 'hello 你好')
    assert p.name == 'notes.txt' and p.read_text(encoding='utf-8') == 'hello 你好'

    clock.advance(3.5)
    p = log.close({'placed': 3, 'ok': True})
    s = json.loads(p.read_text())
    assert s['placed'] == 3 and s['ok'] is True
    assert s['_logger']['elapsed_s'] == 3.5
    assert _events(log)[-1]['kind'] == 'close'
    log.event('late')                        # after close: reopens, no raise
    assert _events(log)[-1]['kind'] == 'late'


def test_save_images_disabled(tmp_path):
    log = RunLogger(tmp_path, 'L1', SimClock(), save_images=False)
    assert log.save_image('x', np.zeros((4, 4, 3), np.uint8)) is None
    assert not list(log.dir.glob('*.jpg'))


def test_thread_safety(tmp_path):
    log = RunLogger(tmp_path, 'L1', SimClock())

    def worker(k):
        for i in range(200):
            log.event('tick', worker=k, i=i, payload='x' * 50)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ev = _events(log)
    assert len(ev) == 1600
    assert sorted((e['worker'], e['i']) for e in ev) == [(k, i) for k in range(8)
                                                         for i in range(200)]


def test_never_raises_on_io_failure(tmp_path):
    log = RunLogger(tmp_path, 'L1', SimClock())
    log.events_path.mkdir()                  # open() for append now fails
    (log.dir / 'summary.json').mkdir()
    with _capture('cube_task.logger') as records:
        log.event('a', x=1)
        log.event('b', x=2)
        assert log.save_json('summary.json', {'a': 1}) is None
        assert log.close({'done': True}) is None

    class Unprintable:
        def __str__(self):
            raise RuntimeError('no')

    log.event('c', obj=Unprintable())
    assert log.n_failures >= 4
    assert any('failed' in r.getMessage() for r in records if r.levelno >= logging.WARNING)


def test_unusable_base_dir_falls_back_to_temp(tmp_path):
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    log = RunLogger(blocker / 'runs', 'L1', SimClock())
    assert log.enabled and log.dir.is_dir()
    log.event('x')
    assert _events(log)[0]['kind'] == 'x'


def test_to_jsonable_misc():
    @dataclass
    class P:
        a: int
        b: np.ndarray

    out = to_jsonable({1: (np.float64(1.5), {np.int32(2)}), 'p': P(1, np.arange(3)),
                       'inf': float('inf'), 'b': b'abc'})
    assert out == {'1': [1.5, [2]], 'p': {'a': 1, 'b': [0, 1, 2]}, 'inf': None,
                   'b': {'bytes': 3}}
    json.dumps(out, allow_nan=False)

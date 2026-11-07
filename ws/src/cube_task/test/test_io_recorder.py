"""Tests for camera/recorder.py and camera/file_camera.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cube_task.camera.file_camera import FileCamera
from cube_task.camera.recorder import list_frames, load_frame, next_index, save_frame
from cube_task.core.clock import SimClock
from cube_task.core.interfaces import CameraError
from cube_task.core.types import CameraIntrinsics, Frame

INTR = CameraIntrinsics(fx=605.0, fy=604.5, cx=319.2, cy=241.7, width=64, height=48,
                        dist=np.array([0.01, -0.02, 0.0, 0.0, 0.001]))


def _frame(seed=0, stamp=12.5):
    rng = np.random.default_rng(seed)
    color = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    depth = rng.uniform(0.2, 1.5, (48, 64)).astype(np.float32)
    depth[0, :5] = 0.0
    depth[1, :5] = np.nan
    depth[2, :5] = np.inf
    depth[3, 0] = -0.3
    depth[4, 0] = 70.0                      # beyond 65.535 m: clipped
    return Frame(color=color, depth=depth, intrinsics=INTR, stamp=stamp,
                 frame_id='camera_color_optical_frame', source='ros',
                 meta={'seq': seed, 'arr': np.arange(3)})


def test_round_trip(tmp_path):
    f = _frame()
    T = np.eye(4)
    T[:3, 3] = [0.3, 0.1, 0.4]
    meta_path = save_frame(tmp_path, 7, f, extra={'T_base_flange': T, 'joints': [0.1] * 6})
    assert meta_path.name == '000007_meta.json'
    assert {p.name for p in tmp_path.iterdir()} == {
        '000007_color.png', '000007_depth.png', '000007_meta.json'}
    meta = json.loads(meta_path.read_text())
    assert meta['depth_unit'] == 'mm' and meta['source'] == 'ros'
    g, extra = load_frame(tmp_path, 7)
    assert np.array_equal(g.color, f.color)
    valid = np.isfinite(f.depth) & (f.depth > 0) & (f.depth < 65.0)
    assert np.max(np.abs(g.depth[valid] - f.depth[valid])) <= 0.0005 + 1e-6
    assert np.all(g.depth[~(np.isfinite(f.depth) & (f.depth > 0))] == 0.0)
    assert g.depth[4, 0] == pytest.approx(65.535, abs=1e-3)
    assert g.depth.dtype == np.float32
    assert g.intrinsics.to_dict() == INTR.to_dict()
    assert (g.stamp, g.frame_id, g.source) == (12.5, f.frame_id, 'ros')
    assert g.meta == {'seq': 0, 'arr': [0, 1, 2]}
    assert np.allclose(np.array(extra['T_base_flange']), T)
    assert extra['joints'] == [0.1] * 6


def test_npy_depth_is_exact(tmp_path):
    f = _frame(1)
    save_frame(tmp_path, 0, f, save_depth_npy=True)
    g, _ = load_frame(tmp_path, 0)
    valid = np.isfinite(f.depth) & (f.depth > 0)
    assert np.array_equal(g.depth[valid], f.depth[valid])
    assert np.all(g.depth[~valid] == 0.0)


def test_frame_without_depth_and_errors(tmp_path):
    f = _frame()
    f.depth = None
    save_frame(tmp_path, 3, f)
    g, extra = load_frame(tmp_path, 3)
    assert g.depth is None and extra == {}
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path, 4)
    with pytest.raises(ValueError):
        save_frame(tmp_path, -1, f)
    bad = _frame()
    bad.color = bad.color[:, :, 0]
    with pytest.raises(ValueError, match='HxWx3'):
        save_frame(tmp_path, 5, bad)
    bad = _frame()
    bad.depth = bad.depth[:10]
    with pytest.raises(ValueError, match='does not match'):
        save_frame(tmp_path, 5, bad)
    (tmp_path / '000003_color.png').unlink()
    with pytest.raises(FileNotFoundError, match='color'):
        load_frame(tmp_path, 3)


def test_list_frames(tmp_path):
    assert list_frames(tmp_path / 'missing') == []
    assert next_index(tmp_path) == 0
    for i in (5, 1, 12):
        save_frame(tmp_path, i, _frame(i))
    (tmp_path / '000099_color.png').write_bytes(b'')   # incomplete: no meta
    assert list_frames(tmp_path) == [1, 5, 12]
    assert next_index(tmp_path) == 13


def _record(tmp_path, n=3):
    for i in range(n):
        save_frame(tmp_path, 10 + i, _frame(i, stamp=100.0 + i), extra={'k': i})


def test_file_camera_order_loop_and_stamps(tmp_path):
    _record(tmp_path)
    clock = SimClock(start=50.0)
    cam = FileCamera(tmp_path, clock, loop=True)
    cam.start()
    assert len(cam) == 3 and cam.intrinsics().fx == INTR.fx
    got = [cam.get_frame() for _ in range(7)]
    assert [f.meta['file_index'] for f in got] == [10, 11, 12, 10, 11, 12, 10]
    assert [f.meta['extra']['k'] for f in got[:3]] == [0, 1, 2]
    assert all(f.source == 'file' and f.meta['orig_source'] == 'ros' for f in got)
    assert got[1].meta['orig_stamp'] == 101.0
    stamps = [f.stamp for f in got]
    assert stamps[0] == 50.0
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert stamps[1] - stamps[0] == pytest.approx(1.0 / 30.0)
    # newer_than in the future: the clock is advanced until the stamp is newer.
    f = cam.get_frame(timeout=1.0, newer_than=clock.now() + 0.5)
    assert f is not None and f.stamp > stamps[-1] + 0.5
    # ...but never beyond the timeout.
    t0 = clock.now()
    assert cam.get_frame(timeout=0.2, newer_than=t0 + 5.0) is None
    assert clock.now() - t0 <= 0.2 + 1e-9
    cam.stop()
    cam.stop()


def test_file_camera_no_loop_and_repeat(tmp_path):
    _record(tmp_path, 2)
    cam = FileCamera(tmp_path, SimClock(), loop=False, repeat=2)
    cam.start()
    idx = []
    while True:
        f = cam.get_frame()
        if f is None:
            break
        idx.append(f.meta['file_index'])
    assert idx == [10, 10, 11, 11]


def test_file_camera_errors(tmp_path):
    with pytest.raises(CameraError, match='does not exist'):
        FileCamera(tmp_path / 'nope', SimClock()).start()
    with pytest.raises(CameraError, match='no frames'):
        FileCamera(tmp_path, SimClock()).start()
    with pytest.raises(CameraError, match='before start'):
        FileCamera(tmp_path, SimClock()).get_frame()
    with pytest.raises(CameraError):
        FileCamera('', SimClock())

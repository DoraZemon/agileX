"""Tests for frame and bag recording (no camera, no ROS)."""
import json

import numpy as np

from piper_vision.camera import Intrinsics
from piper_vision.frames import list_frames, load_frame
from piper_vision.record import (depth_to_mm, next_index, safe_name, save_frame,
                                 session_dir)


def _intr() -> Intrinsics:
    return Intrinsics(380.0, 380.0, 319.5, 239.5, 640, 480)


def _color() -> np.ndarray:
    img = np.zeros((8, 10, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    return img


def test_save_frame_round_trips_through_the_package_reader(tmp_path):
    depth = np.full((8, 10), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    T = np.eye(4)
    T[:3, 3] = (0.3, 0.1, 0.2)
    save_frame(str(tmp_path), 0, _color(), depth, _intr(), 'banana', 'eye_to_hand', T)

    assert list_frames(str(tmp_path)) == [0]
    fr = load_frame(str(tmp_path), 0)
    assert fr.color.shape == (8, 10, 3) and fr.color[4, 4, 1] == 200
    assert fr.depth.shape == (8, 10)
    assert abs(fr.depth[4, 4] - 0.5) < 1e-3
    assert fr.depth[0, 0] == 0.0 and fr.depth[0, 1] == 0.0
    assert np.allclose(fr.T_base_flange, T)
    assert abs(fr.intrinsics.fx - 380.0 * 10 / 640) < 1e-9
    assert fr.meta['extra']['session'] == 'banana'
    assert fr.meta['extra']['mount'] == 'eye_to_hand'
    assert next_index(str(tmp_path)) == 1


def test_saved_frames_load_with_cube_task_recorder(tmp_path):
    """The training tools read frames through cube_task.camera.recorder."""
    import sys
    from pathlib import Path

    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'cube_task'))
    recorder = pytest.importorskip('cube_task.camera.recorder')
    ct_load = recorder.load_frame

    depth = np.linspace(0.3, 1.2, 80, dtype=np.float32).reshape(8, 10)
    save_frame(str(tmp_path), 3, _color(), depth, _intr(), 'pear', 'eye_in_hand')
    frame, extra = ct_load(tmp_path, 3)
    assert frame.color.dtype == np.uint8
    assert abs(float(frame.depth[4, 5]) - float(depth[4, 5])) < 1.5e-3
    assert extra['session'] == 'pear' and extra['mount'] == 'eye_in_hand'
    # 640x480 intrinsics are scaled to the 10x8 image.
    assert frame.intrinsics.width == 10 and frame.intrinsics.height == 8
    assert abs(frame.intrinsics.fx - 380.0 * 10 / 640) < 1e-9


def test_color_only_frame_has_no_depth_files(tmp_path):
    save_frame(str(tmp_path), 0, _color(), None, _intr(), 'bell_pepper', 'eye_to_hand')
    fr = load_frame(str(tmp_path), 0)
    assert fr.depth is None
    meta = json.loads((tmp_path / '000000_meta.json').read_text())
    assert meta['depth_file'] is None and meta['depth_npy_file'] is None


def test_depth_to_mm_clips_and_drops_invalid():
    d = np.array([0.0, -1.0, np.nan, np.inf, 0.0004, 1.5, 80.0])
    mm = depth_to_mm(d)
    assert mm.tolist() == [0, 0, 0, 0, 1, 1500, 65535]


def test_names_and_paths(tmp_path):
    assert safe_name('  bell pepper! ', 'x') == 'bell_pepper'
    assert safe_name('***', 'capture') == 'capture'
    d = session_dir('banana', root=str(tmp_path), date='20261112')
    assert d.name == '20261112_banana'
    assert next_index(str(tmp_path / 'missing')) == 0


def test_save_frame_rejects_bad_input(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        save_frame(str(tmp_path), -1, _color(), None, _intr(), 'a', 'eye_to_hand')
    with pytest.raises(ValueError):
        save_frame(str(tmp_path), 0, _color(), np.zeros((2, 2)), _intr(), 'a', 'eye_to_hand')

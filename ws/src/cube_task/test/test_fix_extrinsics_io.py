"""Review fixes in core/extrinsics.py: .json saves, tcp parent, rpy guard, optical child frame."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math

import numpy as np
import pytest
import yaml

from cube_task.core.config import Config
from cube_task.core.extrinsics import (camera_pose_in_base, Extrinsics, extrinsics_from_cfg,
                                       extrinsics_from_dict, load_extrinsics, save_extrinsics)
from cube_task.core.transforms import inv_T, pose6_to_T, rot_to_quat, rot_to_rpy
from cube_task.tools import calib_points as cp
from cube_task.tools.onsite import write_extrinsics_site
from cube_task.tools.ui import Ui

T_BC = pose6_to_T([0.32, -0.05, 0.62, math.pi - 0.05, 0.03, -1.5])
T_FC = pose6_to_T([0.06, 0.0, 0.03, 0.0, -0.2, math.pi / 2])
# REP-103 camera_link -> camera_color_optical_frame (x right, y down, z forward).
T_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0, -0.015],
                           [0.0, -1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture(name='cube_task.extrinsics'):
    lg = logging.getLogger(name)
    h = _Records()
    lg.addHandler(h)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)


def _warnings(records):
    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return path


def _handeye_json(T):
    return json.dumps({'position': T[:3, 3].tolist(),
                       'orientation': rot_to_quat(T[:3, :3]).tolist(),
                       'rpy': [list(rot_to_rpy(T[:3, :3]))]}, indent=4)


# -------------------------------------------------------------- [onsite] arm-3, [geometry] arm-2
def test_save_to_a_json_path_writes_json_that_round_trips(tmp_path):
    ext = Extrinsics(T=T_BC, mount='eye_in_hand', source='official', rms=0.002, stamp='s')
    p = save_extrinsics(tmp_path / 'extrinsics.json', ext)
    data = json.loads(p.read_text(encoding='utf-8'))
    assert data['mount'] == 'eye_in_hand' and data['parent_frame'] == 'flange'
    assert any('cube_task camera extrinsics' in line for line in data['_comment'])
    with _capture() as records:
        back = load_extrinsics(p)
    assert not _warnings(records)
    assert np.array_equal(back.T, ext.T)
    assert (back.mount, back.parent_frame, back.source, back.rms, back.stamp) == (
        'eye_in_hand', 'flange', 'official', 0.002, 's')
    # YAML paths keep the commented YAML format.
    q = save_extrinsics(tmp_path / 'extrinsics.yaml', ext)
    assert q.read_text(encoding='utf-8').startswith('# cube_task camera extrinsics')
    assert np.array_equal(load_extrinsics(q).T, ext.T)


def test_documented_json_flow_survives_a_site_write(tmp_path):
    """Store the handeye JSON as config/site/extrinsics.json, then write it like --apply."""
    site = tmp_path / 'config' / 'site'
    site.mkdir(parents=True)
    _write(site / 'extrinsics.json', _handeye_json(T_BC))
    cfg = Config({'camera': {'mount': 'eye_in_hand',
                             'extrinsics_file': 'config/site/extrinsics.json'}}, tmp_path, [])
    ext = extrinsics_from_cfg(cfg)
    corrected = Extrinsics(T=pose6_to_T([0.005, 0, 0, 0, 0, 0]) @ ext.T, mount=ext.mount,
                           source='calib_verify')
    path, backup, keys = write_extrinsics_site(cfg.path('camera.extrinsics_file'), corrected)
    assert backup is not None and 'matrix' in keys
    again = extrinsics_from_cfg(cfg)
    assert np.allclose(again.T, corrected.T, atol=1e-12)
    assert json.loads(path.read_text(encoding='utf-8'))['source'] == 'calib_verify'


def test_a_json_file_already_overwritten_with_yaml_still_loads(tmp_path):
    ext = Extrinsics(T=T_BC, mount='eye_in_hand', source='calib_verify')
    yaml_text = save_extrinsics(tmp_path / 'x.yaml', ext).read_text(encoding='utf-8')
    p = _write(tmp_path / 'extrinsics.json', yaml_text)      # what the old code wrote
    with _capture() as records:
        back = load_extrinsics(p)
    assert np.allclose(back.T, T_BC, atol=1e-12)
    assert any('YAML, not JSON' in m for m in _warnings(records))
    # Neither JSON nor YAML: still the JSON error.
    with pytest.raises(ValueError, match='invalid JSON'):
        load_extrinsics(_write(tmp_path / 'bad.json', '{"matrix": [1, 2'))
    with pytest.raises(ValueError, match='invalid JSON'):
        load_extrinsics(_write(tmp_path / 'text.json', 'just some words\n'))


def test_npy_targets_are_refused_and_left_untouched(tmp_path):
    p = tmp_path / 'extrinsics.npy'
    np.save(str(p), T_BC)
    before = p.read_bytes()
    with pytest.raises(ValueError, match=r'\.npy.*\.yaml'):
        save_extrinsics(p, Extrinsics(T=T_FC, mount='eye_in_hand'))
    assert p.read_bytes() == before
    assert np.allclose(load_extrinsics(p).T, T_BC)


def _scripted(answers):
    queues = {k: list(v) for k, v in answers.items()}

    def fn(prompt):
        for key, q in queues.items():
            if key in prompt:
                return q.pop(0)
        raise AssertionError('unexpected prompt %r' % prompt)
    return fn


@pytest.mark.parametrize('name, written', [('ext.json', 'ext.json'), ('ext.npy', 'ext.yaml')])
def test_calib_points_writes_a_loadable_file_for_json_and_npy(tmp_path, name, written):
    out = io.StringIO()
    ui = Ui(headless=True, input_fn=_scripted({'w = write': ['w']}), output=out,
            preview_dir=str(tmp_path / 'preview'))
    code = cp.main(['--mode', 'sim', '--headless', '--log-dir',
                    str(tmp_path), '--set', 'camera.extrinsics_file=config/site/%s' % name],
                   ui=ui)
    text = out.getvalue()
    assert code == 0, text
    path = tmp_path / 'sim_site' / written
    ext = load_extrinsics(path)
    assert ext.mount == 'eye_in_hand' and ext.source == 'calib_points'
    assert not (tmp_path / 'sim_site' / 'ext.npy').exists()
    if written.endswith('.json'):
        assert json.loads(path.read_text(encoding='utf-8'))['mount'] == 'eye_in_hand'
    assert 'to use it: --set camera.extrinsics_file=%s' % path in text


# ------------------------------------------------------------------ [geometry] arm-4
def test_tcp_parent_is_converted_to_flange_with_the_file_tcp_offset(tmp_path):
    off = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]             # driver offset used when calibrating
    T_tc = inv_T(pose6_to_T(off)) @ T_FC              # what the calibration returned
    d = {'mount': 'eye_in_hand', 'parent_frame': 'tcp', 'matrix': T_tc.tolist(),
         'tcp_offset': off}
    ext = load_extrinsics(_write(tmp_path / 'e.yaml', yaml.safe_dump(d)))
    assert ext.parent_frame == 'flange' and np.allclose(ext.T, T_FC, atol=1e-12)
    assert 'tcp_offset' in ext.source
    # The team's robot.tcp_offset (0.14) no longer moves the camera estimate.
    T_bf = pose6_to_T([0.25, 0.05, 0.30, math.pi, 0.0, 0.3])
    assert np.allclose(camera_pose_in_base(ext, T_bf, pose6_to_T([0, 0, 0.14, 0, 0, 0])),
                       T_bf @ T_FC, atol=1e-12)
    # Saved and loaded again it stays T_flange_camera.
    back = load_extrinsics(save_extrinsics(tmp_path / 'saved.json', ext))
    assert back.parent_frame == 'flange' and np.allclose(back.T, T_FC, atol=1e-12)
    # Named key and inverted key work the same way; rotation in tcp_offset is applied.
    off2 = [0.0, 0.01, 0.1425, 0.0, 0.0, 0.3]
    T_tc2 = inv_T(pose6_to_T(off2)) @ T_FC
    for key, M in (('T_tcp_camera', T_tc2), ('T_camera_tcp', inv_T(T_tc2))):
        e = extrinsics_from_dict({key: M.tolist(), 'tcp_offset': off2})
        assert (e.mount, e.parent_frame) == ('eye_in_hand', 'flange')
        assert np.allclose(e.T, T_FC, atol=1e-12)


def test_tcp_parent_without_tcp_offset_is_refused():
    for d in ({'mount': 'eye_in_hand', 'parent_frame': 'tcp', 'matrix': T_FC.tolist()},
              {'T_tcp_camera': T_FC.tolist()},
              {'T_tcp_camera': T_FC.tolist(), 'tcp_offset': None}):
        with pytest.raises(ValueError, match='needs "tcp_offset.*parent_frame: flange'):
            extrinsics_from_dict(d)
    cfg = Config({'camera': {'mount': 'eye_in_hand', 'extrinsics_inline': {
        'parent_frame': 'tcp', 'matrix': T_FC.tolist()}}}, '.', [])
    with pytest.raises(ValueError, match='tcp_offset'):
        extrinsics_from_cfg(cfg)
    cfg.set('camera.extrinsics_inline.tcp_offset', [0, 0, 0.1, 0, 0, 0])
    assert extrinsics_from_cfg(cfg).parent_frame == 'flange'
    with pytest.raises(ValueError, match='look like degrees'):
        extrinsics_from_dict({'T_tcp_camera': T_FC.tolist(),
                              'tcp_offset': [0, 0, 0.1, 0, 0, 90]})
    with pytest.raises(ValueError, match='6 numbers'):
        extrinsics_from_dict({'T_tcp_camera': T_FC.tolist(), 'tcp_offset': [0, 0, 0.1]})
    with pytest.raises(ValueError, match='millimeters'):
        extrinsics_from_dict({'T_tcp_camera': T_FC.tolist(),
                              'tcp_offset': [0, 0, 140, 0, 0, 0]})
    with pytest.raises(ValueError, match='wrist camera'):
        extrinsics_from_dict({'mount': 'eye_to_hand', 'parent_frame': 'tcp',
                              'matrix': T_BC.tolist(), 'tcp_offset': [0] * 6})


def test_handeye_json_without_parent_still_loads_as_flange_with_a_warning(tmp_path):
    p = _write(tmp_path / 'cal.json', _handeye_json(T_FC))
    with _capture() as records:
        ext = load_extrinsics(p, default_mount='eye_in_hand')
    assert ext.parent_frame == 'flange' and np.allclose(ext.T, T_FC)
    msgs = _warnings(records)
    assert any('"parent_frame": "tcp"' in m and '"tcp_offset"' in m for m in msgs)
    # tcp_offset with parent_frame flange is ignored with a warning.
    d = {'T_flange_camera': T_FC.tolist(), 'tcp_offset': [0, 0, 0.1, 0, 0, 0]}
    with _capture() as records:
        ext = extrinsics_from_dict(d)
    assert np.allclose(ext.T, T_FC) and any('ignored' in m for m in _warnings(records))


# ------------------------------------------------------------------ [geometry] arm-5
def test_degrees_under_the_radian_rpy_key_are_refused():
    d = {'mount': 'eye_in_hand', 'xyz': [0.7, 0.0, 0.62], 'rpy': [-147.171, 0.0, 90.0]}
    with pytest.raises(ValueError, match=r'rpy \[-147.171, 0.0, 90.0\]: \|angle\| > 2\*pi.*'
                                         r'rpy_deg'):
        extrinsics_from_dict(d)
    d['rpy_deg'] = d.pop('rpy')
    ext = extrinsics_from_dict(d)
    assert np.allclose(ext.T[:3, :3], pose6_to_T(
        [0, 0, 0] + [math.radians(v) for v in d['rpy_deg']])[:3, :3])
    # Up to 2*pi is accepted (wrapped angles), nested handeye rpy too.
    ext = extrinsics_from_dict({'mount': 'eye_in_hand', 'xyz': [0.7, 0, 0.6],
                                'rpy': [2 * math.pi, 0.0, -2 * math.pi]})
    assert np.allclose(ext.T[:3, :3], np.eye(3), atol=1e-9)
    with pytest.raises(ValueError, match='look like degrees'):
        extrinsics_from_dict({'mount': 'eye_in_hand', 'translation': [0.7, 0, 0.6],
                              'rpy': [[0.0, 7.0, 0.0]]})


# ------------------------------------------------------------------ [sensing] cam-5
@pytest.mark.parametrize('child', ['camera_link', 'camera_color_frame', 'camera', 'base_link'])
def test_non_optical_child_frames_are_refused(child):
    d = {'mount': 'eye_in_hand', 'matrix': T_BC.tolist(), 'child_frame': child}
    with pytest.raises(ValueError) as e:
        extrinsics_from_dict(d, what='ext.yaml')
    msg = str(e.value)
    assert msg.startswith('ext.yaml: ') and 'REP-103' in msg and 'T_link_optical' in msg
    assert 'ros2 run tf2_ros tf2_echo %s camera_color_optical_frame' % child in msg
    with pytest.raises(ValueError, match='REP-103'):
        Extrinsics(T=T_BC, mount='eye_in_hand', child_frame=child)


def test_optical_child_frames_are_accepted_in_any_case():
    for child in ('camera_color_optical_frame', 'CAMERA_COLOR_OPTICAL_FRAME',
                  '/camera_color_optical_frame', 'cam_depth_optical_frame', '', None):
        d = {'mount': 'eye_in_hand', 'matrix': T_BC.tolist(), 'child_frame': child}
        ext = extrinsics_from_dict(d)
        assert ext.child_frame.lower().endswith('_optical_frame')


def test_T_link_optical_is_composed(tmp_path):
    T_b_link = T_BC @ inv_T(T_LINK_OPTICAL)           # camera_link pose given by organizers
    t, q = T_b_link[:3, 3].tolist(), rot_to_quat(T_b_link[:3, :3]).tolist()
    d = {'mount': 'eye_in_hand', 'child_frame': 'camera_link', 'translation': t,
         'quaternion': q, 'T_link_optical': T_LINK_OPTICAL.tolist()}
    ext = load_extrinsics(_write(tmp_path / 'link.yaml', yaml.safe_dump(d)))
    assert np.allclose(ext.T, T_BC, atol=1e-9)
    assert ext.child_frame == 'camera_color_optical_frame'
    assert 'camera_link composed with T_link_optical' in ext.source
    back = load_extrinsics(save_extrinsics(tmp_path / 'saved.yaml', ext))
    assert np.allclose(back.T, T_BC, atol=1e-9)
    # tf2_echo translation + quaternion as a mapping works too.
    d['T_link_optical'] = {'translation': T_LINK_OPTICAL[:3, 3].tolist(),
                           'quaternion': rot_to_quat(T_LINK_OPTICAL[:3, :3]).tolist()}
    assert np.allclose(extrinsics_from_dict(d).T, T_BC, atol=1e-9)
    # Wrong uses are refused.
    d['T_link_optical'] = T_LINK_OPTICAL.tolist()
    with pytest.raises(ValueError, match='already an optical frame'):
        extrinsics_from_dict(dict(d, child_frame='camera_color_optical_frame'))
    with pytest.raises(ValueError, match='already an optical frame'):
        extrinsics_from_dict({k: v for k, v in d.items() if k != 'child_frame'})
    mm = T_LINK_OPTICAL.copy()
    mm[1, 3] = -15.0
    with pytest.raises(ValueError, match='T_link_optical.*millimeters'):
        extrinsics_from_dict(dict(d, T_link_optical=mm.tolist()))

"""Tests for core/extrinsics.py: formats, validation, config lookup, pose composition."""
from __future__ import annotations

import contextlib
import json
import logging
import math

import numpy as np
import pytest
import yaml

from cube_task.core.config import Config
from cube_task.core.extrinsics import (camera_pose_in_base, Extrinsics, extrinsics_from_cfg,
                                       load_extrinsics, save_extrinsics, validate_transform)
from cube_task.core.transforms import inv_T, make_T, pose6_to_T, rot_to_quat, rot_to_rpy

# A generic rigid transform used as T_flange_camera in the format tests.
T_BC = pose6_to_T([0.32, -0.05, 0.62, math.pi - 0.05, 0.03, -1.5])
# A wrist camera (T_flange_camera).
T_FC = pose6_to_T([0.06, 0.0, 0.03, 0.0, -0.2, math.pi / 2])


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture(name='cube_task.extrinsics'):
    # caplog does not work when the ROS launch_testing plugin replaces the logger class.
    lg = logging.getLogger(name)
    h = _Records()
    lg.addHandler(h)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)


def _messages(records, level=logging.WARNING):
    return [r.getMessage() for r in records if r.levelno >= level]


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return path


def _assert_T(a, b, tol=1e-9):
    assert np.allclose(np.asarray(a), np.asarray(b), atol=tol), (a, b)


def test_round_trip_our_format(tmp_path):
    ext = Extrinsics(T=T_BC, mount='eye_in_hand', parent_frame='flange',
                     child_frame='camera_color_optical_frame', source='official',
                     rms=0.0021, stamp='2026-11-05T10:00:00')
    p = save_extrinsics(tmp_path / 'sub' / 'extrinsics.yaml', ext)
    text = p.read_text(encoding='utf-8')
    assert text.startswith('# cube_task camera extrinsics')
    assert 'T_flange_camera' in text
    back = load_extrinsics(p)
    _assert_T(back.T, T_BC, 1e-12)
    assert (back.mount, back.parent_frame, back.child_frame) == (
        'eye_in_hand', 'flange', 'camera_color_optical_frame')
    assert back.source == 'official'
    assert back.rms == pytest.approx(0.0021)
    assert back.stamp == '2026-11-05T10:00:00'
    assert not list(tmp_path.glob('sub/*.tmp'))


def test_round_trip_eye_in_hand_flange_and_tcp_is_not_saved(tmp_path):
    ext = Extrinsics(T=T_FC, mount='eye_in_hand', parent_frame='flange')
    back = load_extrinsics(save_extrinsics(tmp_path / 'e.yaml', ext))
    assert back.mount == 'eye_in_hand' and back.parent_frame == 'flange'
    assert back.name == 'T_flange_camera'
    _assert_T(back.T, T_FC, 1e-12)
    assert 'eye_in_hand' in back.describe()
    # A T_tcp_camera file could not record its TCP offset: refused.
    tcp = Extrinsics(T=T_FC, mount='eye_in_hand', parent_frame='tcp')
    with pytest.raises(ValueError, match='T_flange_camera'):
        save_extrinsics(tmp_path / 't.yaml', tcp)
    assert not (tmp_path / 't.yaml').exists()


def test_default_parent_frame():
    assert Extrinsics(T=T_FC, mount='eye-in-hand').parent_frame == 'flange'


@pytest.mark.parametrize('data', [
    {'mount': 'eye_to_hand', 'matrix': T_BC.tolist()},
    {'T_base_camera': T_BC.tolist()},
    {'T_camera_base': inv_T(T_BC).tolist()},
    {'parent_frame': 'base', 'matrix': T_BC.tolist()},
    {'parent_frame': 'base_link', 'mount': 'eye_in_hand', 'matrix': T_BC.tolist()},
])
def test_fixed_camera_calibrations_are_refused(tmp_path, data):
    p = _write(tmp_path / 'fixed.yaml', yaml.safe_dump(data))
    with pytest.raises(ValueError, match='wrist camera'):
        load_extrinsics(p)
    with pytest.raises(ValueError, match='wrist camera'):
        Extrinsics(T=T_BC, mount='eye_to_hand')
    with pytest.raises(ValueError, match='wrist camera'):
        Extrinsics(T=T_BC, mount='eye_in_hand', parent_frame='base')
    with pytest.raises(ValueError, match='wrist camera'):
        load_extrinsics(_write(tmp_path / 'bare.yaml', yaml.safe_dump(T_BC.tolist())),
                        default_mount='eye_to_hand')


def test_plain_matrix_and_T_keys(tmp_path):
    p = _write(tmp_path / 'm.yaml', yaml.safe_dump({'mount': 'eye_in_hand',
                                                    'matrix': T_BC.tolist()}))
    _assert_T(load_extrinsics(p).T, T_BC)
    p = _write(tmp_path / 'm.json', json.dumps({'T': T_BC.tolist()}))
    ext = load_extrinsics(p)
    _assert_T(ext.T, T_BC)
    assert (ext.mount, ext.parent_frame) == ('eye_in_hand', 'flange')
    # Flat 16 numbers.
    p = _write(tmp_path / 'flat.yaml', yaml.safe_dump({'mount': 'eye_in_hand',
                                                       'T': T_BC.ravel().tolist()}))
    _assert_T(load_extrinsics(p).T, T_BC)


def test_named_keys_set_parent_and_invert(tmp_path):
    p = _write(tmp_path / 'a.yaml', yaml.safe_dump({'T_flange_camera': T_FC.tolist()}))
    ext = load_extrinsics(p)
    assert (ext.mount, ext.parent_frame) == ('eye_in_hand', 'flange')
    _assert_T(ext.T, T_FC)
    p = _write(tmp_path / 'b.yaml', yaml.safe_dump({'T_camera_flange': inv_T(T_FC).tolist()}))
    ext = load_extrinsics(p)
    assert (ext.mount, ext.parent_frame) == ('eye_in_hand', 'flange')
    _assert_T(ext.T, T_FC)


def test_bare_list_text_and_npy(tmp_path):
    p = _write(tmp_path / 'bare.yaml', yaml.safe_dump(T_BC.tolist()))
    _assert_T(load_extrinsics(p).T, T_BC)
    rows = '\n'.join(' '.join('%.17g' % v for v in row) for row in T_BC)
    p = _write(tmp_path / 'm.txt', rows + '\n')
    _assert_T(load_extrinsics(p).T, T_BC)
    rows = '\n'.join('[%s]' % ', '.join('%.17g' % v for v in row) for row in T_BC)
    p = _write(tmp_path / 'm2.txt', rows + '\n')
    _assert_T(load_extrinsics(p).T, T_BC)
    p = tmp_path / 'm.npy'
    np.save(str(p), T_BC)
    _assert_T(load_extrinsics(p).T, T_BC)


@pytest.mark.parametrize('rot_key', ['quaternion', 'rotation_q', 'rotation_m', 'rpy',
                                     'rpy_deg'])
def test_translation_plus_rotation(tmp_path, rot_key):
    t = T_BC[:3, 3].tolist()
    q = rot_to_quat(T_BC[:3, :3]).tolist()
    rpy = list(rot_to_rpy(T_BC[:3, :3]))
    d = {'mount': 'eye_in_hand', 'translation': t}
    if rot_key == 'quaternion':
        d['quaternion'] = q
    elif rot_key == 'rotation_q':
        d['rotation'] = dict(zip('xyzw', q))
    elif rot_key == 'rotation_m':
        d['rotation'] = T_BC[:3, :3].tolist()
    elif rot_key == 'rpy':
        d['rpy'] = rpy
    else:
        d['rpy_deg'] = [math.degrees(v) for v in rpy]
    p = _write(tmp_path / 'x.yaml', yaml.safe_dump(d))
    _assert_T(load_extrinsics(p).T, T_BC)


def test_xyz_rpy_and_dict_vectors(tmp_path):
    rpy = list(rot_to_rpy(T_FC[:3, :3]))
    d = {'mount': 'eye_in_hand', 'parent_frame': 'flange',
         'xyz': T_FC[:3, 3].tolist(), 'rpy': rpy}
    _assert_T(load_extrinsics(_write(tmp_path / 'a.yaml', yaml.safe_dump(d))).T, T_FC)
    q = rot_to_quat(T_FC[:3, :3])
    d = {'mount': 'eye_in_hand', 'parent_frame': 'link6',
         'translation': dict(zip('xyz', T_FC[:3, 3].tolist())),
         'rotation': dict(zip('xyzw', q.tolist()))}
    ext = load_extrinsics(_write(tmp_path / 'b.yaml', yaml.safe_dump(d)))
    assert ext.parent_frame == 'flange'
    _assert_T(ext.T, T_FC)


def _handeye_json(T):
    """Build a file as written by handeye_calibration_ros (rpy nested in a list)."""
    return json.dumps({'position': T[:3, 3].tolist(),
                       'orientation': rot_to_quat(T[:3, :3]).tolist(),
                       'rpy': [list(rot_to_rpy(T[:3, :3]))]}, indent=4)


def test_handeye_json_eye_in_hand_warns_and_defaults_to_flange(tmp_path):
    p = _write(tmp_path / 'cal.json', _handeye_json(T_FC))
    with _capture() as records:
        ext = load_extrinsics(p, default_mount='eye_in_hand')
    assert ext.parent_frame == 'flange'
    _assert_T(ext.T, T_FC)
    assert any('tcp_offset' in m for m in _messages(records))
    # parent_frame tcp added by hand needs the driver tcp_offset of the calibration.
    d = json.loads(p.read_text())
    d['parent_frame'] = 'tcp'
    with pytest.raises(ValueError, match='tcp_offset'):
        load_extrinsics(_write(tmp_path / 'cal2.json', json.dumps(d)),
                        default_mount='eye_in_hand')
    d['tcp_offset'] = [0.0, 0.0, 0.1425, 0.0, 0.0, 0.0]
    with _capture() as records:
        ext = load_extrinsics(_write(tmp_path / 'cal3.json', json.dumps(d)),
                              default_mount='eye_in_hand')
    assert ext.parent_frame == 'flange'
    _assert_T(ext.T, pose6_to_T(d['tcp_offset']) @ T_FC)
    assert not _messages(records)


def test_handeye_without_mount_is_the_wrist_camera(tmp_path):
    p = _write(tmp_path / 'cal.json', _handeye_json(T_FC))
    ext = load_extrinsics(p)
    assert (ext.mount, ext.parent_frame) == ('eye_in_hand', 'flange')
    _assert_T(ext.T, T_FC)
    assert 'handeye_calibration_ros' in ext.source


def test_handeye_rpy_mismatch_is_rejected(tmp_path):
    d = json.loads(_handeye_json(T_BC))
    d['rpy'][0][2] += 0.1
    p = _write(tmp_path / 'cal.json', json.dumps(d))
    with pytest.raises(ValueError, match='disagree'):
        load_extrinsics(p, default_mount='eye_in_hand')


def test_inconsistent_matrix_and_quaternion(tmp_path):
    ext = Extrinsics(T=T_BC, mount='eye_in_hand')
    p = save_extrinsics(tmp_path / 'e.yaml', ext)
    d = yaml.safe_load(p.read_text())
    d['matrix'][0][3] += 0.01
    _write(p, yaml.safe_dump(d))
    with pytest.raises(ValueError, match='disagree by 10.00 mm'):
        load_extrinsics(p)


def test_tiny_rotation_error_is_reorthonormalized():
    T = T_BC.copy()
    T[:3, :3] += 1e-5 * np.array([[1, 2, 0], [0, -1, 3], [2, 0, 1]])
    with _capture() as records:
        clean = validate_transform(T, 'test')
    R = clean[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    assert np.allclose(clean, T_BC, atol=1e-4)
    assert any('re-orthonormalized' in m for m in _messages(records))


@pytest.mark.parametrize('case, match', [
    ('scaled', 'not orthonormal'),
    ('reflection', 'reflection'),
    ('nan', 'NaN'),
    ('mm', 'millimeters'),
    ('bottom', 'bottom row'),
    ('shape', '4x4'),
])
def test_validation_errors(case, match):
    T = T_BC.copy()
    if case == 'scaled':
        T[:3, :3] *= 1.01
    elif case == 'reflection':
        T[:3, 0] *= -1
    elif case == 'nan':
        T[0, 3] = float('nan')
    elif case == 'mm':
        T[:3, 3] *= 1000.0
    elif case == 'bottom':
        T[3, 0] = 0.1
    elif case == 'shape':
        T = T[:2]
    with pytest.raises(ValueError, match=match):
        Extrinsics(T=T, mount='eye_in_hand')


def test_mount_and_parent_errors(tmp_path):
    with pytest.raises(ValueError, match='unknown camera mount'):
        Extrinsics(T=T_BC, mount='on_table')
    with pytest.raises(ValueError, match='unknown parent_frame'):
        Extrinsics(T=T_FC, mount='eye_in_hand', parent_frame='camera_link')
    with pytest.raises(ValueError, match='rms'):
        Extrinsics(T=T_FC, mount='eye_in_hand', rms=-1.0)
    with pytest.raises(ValueError, match='implies parent_frame'):
        load_extrinsics(_write(tmp_path / 'x.yaml', yaml.safe_dump(
            {'T_flange_camera': T_FC.tolist(), 'parent_frame': 'tcp'})))


def test_file_level_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_extrinsics(tmp_path / 'missing.yaml')
    with pytest.raises(ValueError, match='empty'):
        load_extrinsics(_write(tmp_path / 'empty.yaml', '  \n'))
    with pytest.raises(ValueError, match='invalid JSON'):
        load_extrinsics(_write(tmp_path / 'bad.json', '{"matrix": [1, 2'))
    with pytest.raises(ValueError, match='no transform found'):
        load_extrinsics(_write(tmp_path / 'none.yaml', 'mount: eye_in_hand\nfoo: 1\n'))
    with pytest.raises(ValueError, match='without rotation'):
        load_extrinsics(_write(tmp_path / 't.yaml', 'mount: eye_in_hand\n'
                                                    'translation: [0, 0, 0.5]\n'))
    with pytest.raises(ValueError, match='quaternion norm'):
        load_extrinsics(_write(tmp_path / 'q.yaml', 'mount: eye_in_hand\n'
                                                    'translation: [0, 0, 0.5]\n'
                                                    'quaternion: [0, 0, 0, 2]\n'))


def _cfg(tmp_path, **camera):
    data = {'camera': {'mount': 'eye_in_hand', 'extrinsics_file': 'config/site/ext.yaml',
                       'extrinsics_inline': None}}
    data['camera'].update(camera)
    return Config(data, tmp_path, [])


def test_extrinsics_from_cfg_file_inline_and_missing(tmp_path):
    cfg = _cfg(tmp_path)
    assert extrinsics_from_cfg(cfg) is None          # file does not exist
    save_extrinsics(tmp_path / 'config' / 'site' / 'ext.yaml',
                    Extrinsics(T=T_BC, mount='eye_in_hand', source='official'))
    ext = extrinsics_from_cfg(cfg)
    assert ext.source == 'official'
    _assert_T(ext.T, T_BC)
    T2 = make_T(np.eye(3), [0.3, 0.0, 0.5]) @ pose6_to_T([0, 0, 0, math.pi, 0, 0])
    cfg = _cfg(tmp_path, extrinsics_inline={'mount': 'eye_in_hand', 'matrix': T2.tolist()})
    ext = extrinsics_from_cfg(cfg)
    _assert_T(ext.T, T2)
    assert 'inline' in ext.source
    cfg = _cfg(tmp_path, extrinsics_file='')
    assert extrinsics_from_cfg(cfg) is None


def test_extrinsics_from_cfg_refuses_a_fixed_camera(tmp_path):
    save_extrinsics(tmp_path / 'config' / 'site' / 'ext.yaml',
                    Extrinsics(T=T_FC, mount='eye_in_hand'))
    with pytest.raises(ValueError, match='wrist camera'):
        extrinsics_from_cfg(_cfg(tmp_path, mount='eye_to_hand'))
    cfg = _cfg(tmp_path, extrinsics_inline={'mount': 'eye_to_hand', 'matrix': T_BC.tolist()})
    with pytest.raises(ValueError, match='wrist camera'):
        extrinsics_from_cfg(cfg)
    with pytest.raises(ValueError, match='mapping'):
        extrinsics_from_cfg(_cfg(tmp_path, extrinsics_inline=[1, 2, 3]))


def test_extrinsics_from_cfg_inline_uses_cfg_mount(tmp_path):
    cfg = _cfg(tmp_path, mount='eye_in_hand', extrinsics_inline={'matrix': T_FC.tolist()})
    ext = extrinsics_from_cfg(cfg)
    assert (ext.mount, ext.parent_frame) == ('eye_in_hand', 'flange')


def test_camera_pose_in_base():
    T_bf = pose6_to_T([0.25, 0.05, 0.30, math.pi, 0.0, 0.3])
    T_ft = pose6_to_T([0.0, 0.0, 0.14, 0.0, 0.0, 0.0])
    eih = Extrinsics(T=T_FC, mount='eye_in_hand', parent_frame='flange')
    _assert_T(camera_pose_in_base(eih, T_base_flange=T_bf), T_bf @ T_FC)
    with pytest.raises(ValueError, match='T_base_flange'):
        camera_pose_in_base(eih)
    tcp = Extrinsics(T=T_FC, mount='eye_in_hand', parent_frame='tcp')
    _assert_T(camera_pose_in_base(tcp, T_base_flange=T_bf, T_flange_tcp=T_ft),
              T_bf @ T_ft @ T_FC)
    with pytest.raises(ValueError, match='T_flange_tcp'):
        camera_pose_in_base(tcp, T_base_flange=T_bf)
    bad = T_bf.copy()
    bad[0, 0] = float('nan')
    with pytest.raises(ValueError, match='T_base_flange'):
        camera_pose_in_base(eih, T_base_flange=bad)
    # A camera point maps consistently through the chain.
    p_cam = np.array([0.01, -0.02, 0.35, 1.0])
    p_base = camera_pose_in_base(tcp, T_bf, T_ft) @ p_cam
    assert np.allclose(p_base, T_bf @ (T_ft @ (T_FC @ p_cam)))

"""Tests for camera/factory.py (make_camera)."""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from cube_task.camera.factory import make_camera
from cube_task.camera.file_camera import FileCamera
from cube_task.camera.recorder import save_frame
from cube_task.camera.ros_camera import RosCamera
from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.core.types import CameraIntrinsics, Frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTR = {'fx': 500.0, 'fy': 500.0, 'cx': 319.5, 'cy': 239.5, 'width': 640, 'height': 480}
T_FC = [[0, -1, 0, 0.06], [1, 0, 0, 0.0], [0, 0, 1, 0.03], [0, 0, 0, 1]]


def _cfg(**over):
    cfg = load_config(use_overlays=False, root=ROOT)
    for k, v in over.items():
        cfg.set(k.replace('__', '.'), v)
    return cfg


@pytest.fixture
def fake_sim(monkeypatch):
    """Install a fake cube_task.sim.sim_camera recording the SimCamera arguments."""
    calls = []

    class SimCamera:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    mod = types.ModuleType('cube_task.sim.sim_camera')
    mod.SimCamera = SimCamera
    import cube_task.sim as sim_pkg
    monkeypatch.setitem(sys.modules, 'cube_task.sim.sim_camera', mod)
    monkeypatch.setattr(sim_pkg, 'sim_camera', mod, raising=False)
    return calls


def test_ros_backend():
    cam = make_camera(_cfg(), SimClock())
    assert isinstance(cam, RosCamera)
    assert cam.color_topic == '/camera/color/image_raw' and cam.depth_scale == 0.001


def test_file_backend(tmp_path):
    f = Frame(color=np.zeros((4, 6, 3), np.uint8), depth=np.ones((4, 6), np.float32),
              intrinsics=CameraIntrinsics(5.0, 5.0, 2.5, 1.5, 6, 4), stamp=1.0)
    save_frame(tmp_path, 0, f)
    cam = make_camera(_cfg(camera__backend='file', camera__file__dir=str(tmp_path),
                           camera__file__loop=False), SimClock())
    assert isinstance(cam, FileCamera) and cam.loop is False and cam.repeat == 1
    cam.start()
    assert cam.get_frame().depth.shape == (4, 6)
    with pytest.raises(ValueError, match='camera.file.dir'):
        make_camera(_cfg(camera__backend='file'), SimClock())


def test_unknown_backend():
    with pytest.raises(ValueError, match='unknown camera.backend'):
        make_camera(_cfg(camera__backend='usb'), SimClock())
    with pytest.raises(ValueError, match='unknown camera.backend'):
        make_camera(_cfg(camera__backend=None), SimClock())


def test_sim_eye_in_hand(fake_sim):
    arm = object()
    cfg = _cfg(camera__backend='sim', camera__mount='eye_in_hand', sim__intrinsics=INTR,
               sim__T_flange_camera=T_FC)
    clock, world = SimClock(), object()
    make_camera(cfg, clock, sim_world=world, arm=arm)
    kw = fake_sim[-1]
    assert kw['world'] is world and kw['clock'] is clock and kw['arm'] is arm
    assert isinstance(kw['intrinsics'], CameraIntrinsics) and kw['intrinsics'].width == 640
    assert np.allclose(kw['T_flange_camera'], T_FC) and 'T_base_camera' not in kw
    with pytest.raises(ValueError, match='needs the arm'):
        make_camera(cfg, SimClock(), sim_world=object())
    with pytest.raises(ValueError, match='eye_to_hand'):
        make_camera(_cfg(camera__backend='sim', camera__mount='eye_to_hand', sim__intrinsics=INTR,
                         sim__T_flange_camera=T_FC), SimClock(), sim_world=object(), arm=arm)


@pytest.mark.parametrize('over, match', [
    ({}, 'sim_world'),
    ({'sim__intrinsics': None}, 'sim.intrinsics'),
    ({'sim__intrinsics': {'fx': 1.0}}, 'sim.intrinsics is invalid'),
    ({'sim__T_flange_camera': None}, 'sim.T_flange_camera'),
    ({'sim__T_flange_camera': [[1, 0], [0, 1]]}, '4x4'),
])
def test_sim_missing_params(fake_sim, over, match):
    kw = {'camera__backend': 'sim', 'camera__mount': 'eye_in_hand', 'sim__intrinsics': INTR,
          'sim__T_flange_camera': T_FC}
    kw.update(over)
    world = None if match == 'sim_world' else object()
    with pytest.raises(ValueError, match=match):
        make_camera(_cfg(**kw), SimClock(), sim_world=world, arm=object())
    assert not fake_sim

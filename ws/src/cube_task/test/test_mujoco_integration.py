"""
MuJoCo camera, video, physics monitor and the --mode mujoco / --backend mujoco plumbing.

Run with the MuJoCo venv: MUJOCO_GL=egl env -u PYTHONPATH
~/.venvs/piper-mujoco/bin/python -m pytest -q test/test_mujoco_integration.py.
No import-time pytest.importorskip / pytest.skip (see test_mujoco_scene.py: under
the ROS launch_testing pytest plugin it aborts the collection of every test).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cube_task.core.clock import SimClock
from cube_task.core.transforms import inv_T
from cube_task.perception.factory import make_perception
from cube_task.sim.mujoco.arm import MujocoArm
from cube_task.sim.mujoco.camera import MujocoCamera
from cube_task.sim.mujoco.run import (failure_causes, final_truth, match_objects,
                                      PhysicsMonitor, scene_config)
from cube_task.sim.mujoco.scene import DEFAULT_ASSETS_DIR
from cube_task.sim.scenarios import make_scenario
from cube_task.sim.sim_camera import FRAME_PERIOD_S

try:
    import mujoco
except ImportError:  # system Python: the tests below are skipped
    mujoco = None

pytestmark = [
    pytest.mark.skipif(mujoco is None, reason='MuJoCo not installed (use ~/.venvs/piper-mujoco)'),
    pytest.mark.skipif(not (Path(DEFAULT_ASSETS_DIR).expanduser() / 'piper.xml').is_file(),
                       reason='PiPER MJCF not found in %s' % DEFAULT_ASSETS_DIR),
]


def _rig(task='L1', seed=0, mount='eye_to_hand'):
    """Scene + config + clock with the arm at the observe pose, settled."""
    world, info = make_scenario(task, seed, mount, 'normal')
    cfg, mw = scene_config(info, sim_world=world)
    clock = SimClock()
    sim = mw.attach(clock)
    arm = MujocoArm(cfg, clock, sim)
    arm.set_joints(cfg.get('robot.poses.observe_joints'))
    arm.connect()
    mw.settle(0.3)
    return world, info, cfg, mw, clock, arm


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_camera_geometry_matches_the_scene(mount):
    """Intrinsics, extrinsics and depth agree with the physics state (no noise)."""
    world, info, cfg, mw, clock, arm = _rig('L1', 0, mount)
    cam = MujocoCamera(mw, clock, noise_std=0.0, dropout=0.0, color_noise=0.0)
    cam.start()
    try:
        t0 = clock.now()
        frame = cam.get_frame(newer_than=t0)
        assert frame is not None and frame.source == 'mujoco'
        assert frame.stamp == pytest.approx(t0 + FRAME_PERIOD_S)
        assert mw.sim.time == pytest.approx(frame.stamp, abs=mw.model.opt.timestep)
        assert frame.color.shape == (480, 640, 3) and frame.depth.dtype == np.float32
        T_bc = frame.meta['T_base_camera_true']
        if mount == 'eye_in_hand':
            T_bf = arm.get_flange_pose().value
            assert np.allclose(T_bf @ mw.T_flange_camera, T_bc, atol=1e-9)
        else:
            assert np.allclose(mw.T_base_camera, T_bc, atol=1e-9)
        # The top center of every cube projects to a pixel whose depth is its z.
        K, T_cb = frame.intrinsics.K(), inv_T(T_bc)
        sizes = {o.name: o.size for o in world.objects}
        for name in sizes:
            p = mw.object_com(name)
            p[2] = mw.object_pose(name)[2, 3] + sizes[name][2]
            q = T_cb[:3, :3] @ p + T_cb[:3, 3]
            u, v = (int(round(c)) for c in (K @ (q / q[2]))[:2])
            assert float(frame.depth[v, u]) == pytest.approx(q[2], abs=0.001)
        # Perception localizes the cubes in the base frame within 3 mm.
        res = make_perception(cfg, 'L1').process(frame, T_bc)
        matched = match_objects(mw, res.valid_detections())
        assert sorted(n for _d, n in matched) == sorted(sizes)
        for d, name in matched:
            assert d.label == mw.object(name).label
            err = np.hypot(*(np.asarray(d.center_base[:2]) - mw.object_com(name)[:2]))
            assert err < 0.003, (name, err)
            top = mw.object_pose(name)[2, 3] + sizes[name][2]
            assert abs(d.top_z - top) < 0.003
    finally:
        cam.stop()


def test_camera_noise_is_seeded_and_faults_work():
    _world, _info, cfg, mw, clock, _arm = _rig('L1', 0, 'eye_to_hand')
    frames = []
    for _ in range(2):
        cam = MujocoCamera(mw, clock, cfg, seed=5)
        cam.start()
        frames.append(cam.get_frame())
        cam.stop()
    a, b = frames
    assert np.array_equal(a.color, b.color) and np.array_equal(a.depth, b.depth)
    clean = MujocoCamera(mw, clock, noise_std=0.0, dropout=0.0, color_noise=0.0)
    clean.start()
    c = clean.get_frame()
    valid = c.depth > 0
    assert 0.0 < np.mean(a.depth[valid] == 0) < 0.05          # dropout, mostly at edges
    both = valid & (a.depth > 0)
    z = c.depth[both].astype(float)
    k = (a.depth[both] - z) / (cfg.get('sim.noise_std') * (z / 0.5) ** 2)
    assert 0.9 < float(np.std(k)) < 1.1                       # noise_std * (z / 0.5)^2
    clean.inject('none_frames')
    t = clock.now()
    assert clean.get_frame(timeout=0.2) is None and clock.now() == pytest.approx(t + 0.2)
    clean.inject('no_depth')
    assert clean.get_frame().depth is None
    clean.stop()
    assert clean.get_frame() is None                          # stopped


def test_physics_monitor_explains_misplaced_objects():
    """A cube moved into a wrong zone and one knocked off the table get causes."""
    world, info, cfg, mw, clock, _arm = _rig('L1', 0, 'eye_to_hand')
    mon = PhysicsMonitor(mw).start()
    names = [o.name for o in world.objects]
    wrong = next(z for z in mw.zones if z != info['class_to_zone'][mw.object(names[0]).label])
    T = mw.object_pose(names[0])
    T[:2, 3] = mw.zones[wrong].center[:2]
    T[2, 3] += 0.02
    mw.set_object_pose(names[0], T)
    T = mw.object_pose(names[1])
    T[:3, 3] = [0.6, 0.69, 0.05]                              # over the table edge
    mw.set_object_pose(names[1], T)
    mw.sim.d.qvel[mw.object(names[1]).dof_adr + 1] = 1.0     # rolling off sideways
    truth = final_truth(mw, info, settle_s=2.0)
    assert truth == {'correct': 0, 'wrong': 1, 'remaining': 1, 'dropped': 1}
    fails = {f['object']: f for f in failure_causes(mw, info, mon, 'timeout')}
    assert fails[names[0]]['kind'] == 'wrong' and 'into a zone' in fails[names[0]]['cause']
    assert fails[names[1]]['kind'] == 'dropped'
    assert 'off the table' in fails[names[1]]['cause']
    assert fails[names[2]]['kind'] == 'remaining'
    assert fails[names[2]]['cause'] == 'not attempted (timeout)'
    assert [e['what'] for e in mon.events if e['object'] == names[1]][:1] == ['off_table']


def test_factories_need_the_mujoco_world():
    from cube_task.camera.factory import make_camera
    from cube_task.robot.factory import make_arm
    _world, _info, cfg, mw, clock, _arm = _rig('L1', 0, 'eye_to_hand')
    with pytest.raises(ValueError, match='MujocoWorld'):
        make_camera(cfg, clock, sim_world=None)
    with pytest.raises(ValueError, match='MujocoWorld'):
        make_arm(cfg, clock, sim_world=None)
    assert type(make_camera(cfg, clock, sim_world=mw)).__name__ == 'MujocoCamera'
    assert type(make_arm(cfg, clock, sim_world=mw).inner).__name__ == 'MujocoArm'


def test_run_task_mujoco_l1_with_video(tmp_path, capsys):
    """run_task --mode mujoco: seed 0 places 3/3 by physics and records a valid H.264 MP4."""
    import imageio_ffmpeg
    from cube_task.tools import run_task
    video = tmp_path / 'l1.mp4'
    code = run_task.main(['--task', 'L1', '--mode', 'mujoco', '--seed', '0',
                          '--log-dir', str(tmp_path / 'runs'), '--video', str(video),
                          '--video-speed', '4'])
    out = capsys.readouterr().out
    assert code == 0, out[-2000:]
    assert 'L1 COMPLETE: placed 3/3' in out
    assert 'ground truth (physics): correct 3, wrong 0, remaining 0, dropped 0' in out
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = reader.__next__()
    reader.close()
    assert meta['codec'].startswith('h264') and meta['pix_fmt'].startswith('yuv420p')
    assert tuple(meta['size']) == (1280, 720) and meta['fps'] == 30
    assert 10.0 < meta['duration'] < 25.0            # ~60 s of physics at 4x
    assert video.with_suffix('.jpg').stat().st_size > 10000
    logs = list((tmp_path / 'runs').glob('*_L1'))
    assert len(logs) == 1
    phys = json.loads((logs[0] / 'physics.json').read_text())
    assert phys['truth']['correct'] == 3 and phys['failures'] == []
    assert sum(e['what'] == 'grasped' for e in phys['events']) == 3
    events = [json.loads(ln) for ln in (logs[0] / 'events.jsonl').read_text().splitlines()]
    assert sum(e['kind'] == 'physics' and e['what'] == 'landed' for e in events) == 3


def test_run_task_refuses_video_outside_mujoco_mode(capsys):
    from cube_task.tools import run_task
    code = run_task.main(['--task', 'L1', '--mode', 'sim', '--video', '/tmp/x.mp4'])
    assert code == run_task.EXIT_REFUSED
    assert '--mode mujoco' in capsys.readouterr().out


def test_wrapper_script_runs_the_venv_python(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / 'py'
    fake.write_text('#!/usr/bin/env bash\necho "$MUJOCO_GL|${PYTHONPATH-unset}|$PWD|$*"\n')
    fake.chmod(0o755)
    env = dict(os.environ, MUJOCO_PYTHON=str(fake), PYTHONPATH='/opt/ros/x')
    env.pop('MUJOCO_GL', None)
    res = subprocess.run(['bash', str(root / 'scripts' / 'mujoco_run.sh'), 'run_task', '-h'],
                         env=env, capture_output=True, text=True, timeout=10)
    assert res.returncode == 0
    assert res.stdout.strip() == 'egl|unset|%s|-m cube_task run_task -h' % root


class _FakeWall:
    """Wall clock stand-in: sleeping only advances its own time."""

    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        self.slept += dt


def test_viewer_syncs_at_real_speed_and_is_optional(monkeypatch):
    import mujoco.viewer
    from cube_task.sim.mujoco.run import Viewer
    _world, _info, _cfg, mw, _clock, _arm = _rig('L1', 0, 'eye_to_hand')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    assert Viewer(mw).open() is False                     # no display: run without it

    class Handle:
        syncs = 0
        closed = False

        def is_running(self):
            return True

        def sync(self):
            Handle.syncs += 1

        def close(self):
            Handle.closed = True

    monkeypatch.setenv('DISPLAY', ':99')
    monkeypatch.setattr(mujoco.viewer, 'launch_passive', lambda m, d, **kw: Handle())
    wall = _FakeWall()
    v = Viewer(mw, fps=20.0, wall_clock=wall)
    assert v.open() is True
    mw.settle(1.0)
    assert 19 <= Handle.syncs <= 21
    assert wall.t == pytest.approx(1.0, abs=0.06)          # paced to the virtual time
    v.close()
    mw.settle(0.5)
    assert Handle.closed and Handle.syncs <= 21

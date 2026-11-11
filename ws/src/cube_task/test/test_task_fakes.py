"""Local fakes for the task-layer tests: world, arm, camera, perception, logger."""
from __future__ import annotations

import copy
import math
import types

from cube_task.core.clock import SimClock
from cube_task.core.config import load_config
from cube_task.core.interfaces import ArmInterface, CameraSource
from cube_task.core.transforms import closing_angle_of, top_down_T
from cube_task.core.types import (ArmHealth, CameraIntrinsics, Detection, Frame, GripperResult,
                                  MotionResult, PerceptionResult, Stamped)
import numpy as np

ZONES = {
    'zone_red': {'center': [0.20, 0.30, 0.0], 'size': [0.10, 0.10], 'label': 'red tray'},
    'zone_green': {'center': [0.32, 0.30, 0.0], 'size': [0.10, 0.10], 'label': 'green tray'},
    'zone_blue': {'center': [0.20, -0.30, 0.0], 'size': [0.10, 0.10], 'label': 'blue tray'},
    'zone_yellow': {'center': [0.32, -0.30, 0.0], 'size': [0.10, 0.10], 'wall_height': 0.05,
                    'label': 'yellow bin'},
}


def make_cfg(task='L1', **overrides):
    """Return a Config for tests: default.yaml + test zones + sim backend."""
    cfg = load_config(use_overlays=False)
    cfg.set('zones', copy.deepcopy(ZONES))
    cfg.set('robot.backend', 'sim')
    cfg.set('robot.poses.observe_joints', [0.0, 0.3, -0.5, 0.0, 0.6, 0.0])
    cfg.set('camera.mount', 'eye_to_hand')
    cfg.set('robot.gripper.max_width', 0.07)      # FakeArm's stroke
    cfg.set('tasks.L1.labels', ['red', 'green', 'blue'])
    cfg.set('tasks.L1.class_to_zone', {'red': 'zone_red', 'green': 'zone_green',
                                       'blue': 'zone_blue'})
    cfg.set('tasks.L2.labels', ['apple', 'banana', 'carrot', 'eggplant', 'orange', 'corn'])
    cfg.set('tasks.L2.class_to_zone', {'apple': 'zone_red', 'orange': 'zone_red',
                                       'banana': 'zone_yellow', 'corn': 'zone_yellow',
                                       'carrot': 'zone_green', 'eggplant': 'zone_blue'})
    cfg.set('tasks.L3.labels', ['apple', 'banana', 'orange', 'carrot'])
    for k, v in overrides.items():
        cfg.set(k.replace('__', '.'), v)
    return cfg


def make_det(label, x, y, size=0.03, height=0.03, yaw=0.0, table_z=0.0, length=None,
             score=0.9):
    """Return a valid Detection of a box at (x, y) on a table at table_z."""
    length = size if length is None else length
    top = table_z + height
    return Detection(label=label, score=score, score_kind='color', bbox=(10, 10, 20, 20),
                     center_base=np.array([x, y, top]), top_z=top, height=height, yaw=yaw,
                     length=max(length, size), width=min(length, size), n_points=200,
                     extra={'table_z': table_z})


class FakeWorld:
    """Objects on a table; the fake arm attaches / releases them."""

    def __init__(self, zones=None):
        self.objects = []                  # list of Detection (true state)
        self.held = None
        self.placed = {}                   # zone name -> [labels]
        self.dropped = []                  # labels released outside any zone
        self.miss_next = 0                 # next N closes grab nothing
        self.always_miss = set()           # labels that can never be grasped
        self.extra_detections = []         # e.g. unknown objects, never graspable
        self.zones = zones or {}
        self.on_place = None               # callback(zone, label)

    def add(self, det):
        self.objects.append(det)
        return det

    def try_grasp(self, T):
        if self.miss_next > 0:
            self.miss_next -= 1
            return None
        x, y, z = T[:3, 3]
        for d in self.objects:
            if math.hypot(d.center_base[0] - x, d.center_base[1] - y) < 0.01 \
                    and z < d.top_z and d.label not in self.always_miss:
                self.objects = [o for o in self.objects if o is not d]
                self.held = d
                return float(d.width)
        return None

    def release(self, T):
        d, self.held = self.held, None
        if d is None:
            return None
        x, y = float(T[0, 3]), float(T[1, 3])
        for name, z in self.zones.items():
            if z.contains_xy(x, y):
                self.placed.setdefault(name, []).append(d.label)
                if self.on_place:
                    self.on_place(name, d.label)
                return name
        d.center_base = np.array([x, y, d.center_base[2]])
        self.objects.append(d)
        self.dropped.append(d.label)
        return None


class FakeArm(ArmInterface):
    """
    Scripted ArmInterface. Records every command in ``commands``.

    fail_fn(kind, target, linear, index) -> MotionResult or None scripts motion
    failures; index counts all motion calls from 0.
    """

    def __init__(self, clock, world=None, move_time=1.0, grip_time=0.5, max_width=0.07):
        self.clock = clock
        self.world = world
        self.move_time = move_time
        self.grip_time = grip_time
        self.max_width = max_width
        self.commands = []
        self.tcp = top_down_T(0.25, 0.0, 0.30, 0.0)
        self.width = 0.0
        self.healthy = True
        self.fail_fn = None
        self.close_width_fn = None         # () -> width reported after closing
        self.width_fn = None               # (stage index) -> width override for reads
        self.no_width = False
        self.n_motion = 0
        self.estopped = False

    def connect(self):
        pass

    def enable(self, timeout=5.0):
        return True

    def health(self):
        return ArmHealth(ok=self.healthy, reason='' if self.healthy else 'fake fault')

    def get_tcp_pose(self):
        return Stamped(self.tcp.copy(), self.clock.now())

    def get_flange_pose(self):
        return Stamped(self.tcp.copy(), self.clock.now())

    def get_joints(self):
        return Stamped(np.zeros(6), self.clock.now())

    def _motion(self, kind, target, linear):
        idx = self.n_motion
        self.n_motion += 1
        self.commands.append((kind, np.array(target, dtype=float), bool(linear)))
        if self.abort_event.is_set():
            return MotionResult(ok=False, error='aborted')
        if self.fail_fn is not None:
            r = self.fail_fn(kind, target, linear, idx)
            if r is not None:
                return r
        self.clock.sleep(self.move_time)
        if kind == 'move_tcp':
            self.tcp = np.array(target, dtype=float)
            if self.world is not None and self.world.held is not None:
                self.world.held.center_base = self.tcp[:3, 3].copy()
        else:
            self.tcp = top_down_T(0.05, 0.25, 0.30, 0.0)
        return MotionResult(ok=True, duration=self.move_time, final_tcp=self.tcp.copy(),
                            pos_err=0.0)

    def move_tcp(self, T_base_tcp, linear=False, timeout=None):
        return self._motion('move_tcp', T_base_tcp, linear)

    def move_joints(self, q, timeout=None):
        return self._motion('move_joints', q, False)

    def gripper(self, width, force=None, wait=True):
        self.commands.append(('gripper', float(width), force))
        if self.abort_event.is_set():
            return GripperResult(ok=False, width=self.width, error='aborted')
        self.clock.sleep(self.grip_time)
        if width < 0.01:
            w = None
            if self.world is not None:
                w = self.world.try_grasp(self.tcp)
            if self.close_width_fn is not None:
                w = self.close_width_fn()
            self.width = 0.0 if w is None else float(w)
        else:
            if self.world is not None and self.world.held is not None:
                self.world.release(self.tcp)
            self.width = min(float(width), self.max_width)
        return GripperResult(ok=True, width=None if self.no_width else self.width)

    def gripper_width(self):
        if self.no_width:
            return None
        w = self.width if self.width_fn is None else self.width_fn(self)
        return Stamped(w, self.clock.now())

    def set_speed_percent(self, percent):
        self.commands.append(('speed', percent, None))

    def stop(self):
        self.commands.append(('stop', None, None))

    def emergency_stop(self):
        self.estopped = True
        self.commands.append(('estop', None, None))

    def close(self):
        pass

    # helpers for assertions
    def motions(self):
        return [c for c in self.commands if c[0] in ('move_tcp', 'move_joints')]

    def gripper_cmds(self):
        return [c[1] for c in self.commands if c[0] == 'gripper']


def closing_of(cmd):
    return closing_angle_of(cmd[1])


class FakeSafeArm(FakeArm):
    """FakeArm with the SafeArm extras check_reachable / set_table_z."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reach_fn = None
        self.table_z = None
        self.reach_checks = 0

    def check_reachable(self, T_base_tcp, q_seed=None):
        self.reach_checks += 1
        if self.reach_fn is not None:
            return self.reach_fn(T_base_tcp)
        return True, ''

    def set_table_z(self, z):
        self.table_z = z


INTR = CameraIntrinsics(fx=60.0, fy=60.0, cx=32.0, cy=24.0, width=64, height=48)


class FakeCamera(CameraSource):
    """Frames stamped with the SimClock; advances time by one frame period."""

    def __init__(self, clock, period=1.0 / 30.0, depth_value=0.8):
        self.clock = clock
        self.period = period
        self.depth_value = depth_value
        self.none_frames = 0               # next N calls return None
        self.stale = False                 # return frames 5 s old
        self.calls = 0

    def start(self):
        pass

    def stop(self):
        pass

    def intrinsics(self):
        return INTR

    def get_frame(self, timeout=1.0, newer_than=None):
        self.calls += 1
        if self.none_frames > 0:
            self.none_frames -= 1
            self.clock.sleep(timeout)
            return None
        if newer_than is not None and newer_than >= self.clock.now():
            self.clock.sleep(newer_than - self.clock.now())
        self.clock.sleep(self.period)
        stamp = self.clock.now()
        if self.stale:
            stamp = (newer_than or stamp) + 1e-3
            self.clock.sleep(5.0)
        color = np.zeros((48, 64, 3), np.uint8)
        depth = np.full((48, 64), self.depth_value, np.float32)
        return Frame(color=color, depth=depth, intrinsics=INTR, stamp=stamp, source='fake')


class FakePerception:
    """Return the world's objects (plus extra detections) as a PerceptionResult."""

    def __init__(self, clock, world, process_time=0.2):
        self.clock = clock
        self.world = world
        self.process_time = process_time
        self.calls = []

    def process(self, frame, T_base_camera, zones=None):
        self.calls.append(np.array(T_base_camera))
        self.clock.sleep(self.process_time)
        dets = [copy.deepcopy(d) for d in self.world.objects + self.world.extra_detections]
        return PerceptionResult(detections=dets, frame_stamp=frame.stamp)


class FakeLogger:
    """In-memory RunLogger stand-in."""

    def __init__(self):
        self.dir = ''
        self.events = []
        self.images = []
        self.json = {}

    def event(self, kind, **fields):
        self.events.append(dict(fields, kind=kind))

    def save_image(self, name, bgr):
        self.images.append(name)

    def save_json(self, name, obj):
        self.json[name] = obj

    def kinds(self, kind):
        return [e for e in self.events if e['kind'] == kind]


class FakeExt:
    """Minimal core.extrinsics.Extrinsics stand-in."""

    def __init__(self, T=None, mount='eye_to_hand'):
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float)
        self.mount = mount
        self.parent_frame = 'base' if mount == 'eye_to_hand' else 'flange'


def default_camera_T():
    """Camera 0.8 m above the pick area looking down (optical z = base -Z)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.30, 0.0, 0.80]
    return T


def _fake_camera_pose_in_base(ext, T_base_flange=None, T_flange_tcp=None):
    if ext.mount == 'eye_to_hand':
        return ext.T.copy()
    if T_base_flange is None:
        raise ValueError('need flange pose')
    return np.asarray(T_base_flange) @ ext.T


def _fake_capture_averaged(camera, n, newer_than=None, timeout=2.0):
    last = newer_than
    frame = None
    for _ in range(max(1, int(n))):
        f = camera.get_frame(timeout=timeout, newer_than=last)
        if f is None:
            break
        frame, last = f, f.stamp
    return frame


def install_fake_modules(monkeypatch, ext_from_cfg=None):
    """Replace core.extrinsics and camera.utils with local fakes for one test."""
    ext_mod = types.ModuleType('cube_task.core.extrinsics')
    ext_mod.camera_pose_in_base = _fake_camera_pose_in_base
    ext_mod.extrinsics_from_cfg = ext_from_cfg or (lambda cfg: None)
    cam_mod = types.ModuleType('cube_task.camera.utils')
    cam_mod.capture_averaged = _fake_capture_averaged
    monkeypatch.setitem(__import__('sys').modules, 'cube_task.core.extrinsics', ext_mod)
    monkeypatch.setitem(__import__('sys').modules, 'cube_task.camera.utils', cam_mod)
    return ext_mod, cam_mod


class Rig:
    """Everything a runner test needs, wired together."""

    def __init__(self, task='L1', safe=True, **cfg_overrides):
        from cube_task.core.zones import parse_zones
        self.cfg = make_cfg(task, **cfg_overrides)
        self.clock = SimClock()
        self.world = FakeWorld(parse_zones(self.cfg))
        cls = FakeSafeArm if safe else FakeArm
        self.arm = cls(self.clock, self.world)
        self.camera = FakeCamera(self.clock)
        self.perception = FakePerception(self.clock, self.world)
        self.logger = FakeLogger()
        self.ext = FakeExt(default_camera_T(), self.cfg.get('camera.mount'))
        self.task = task

    def runner(self, **kw):
        from cube_task.task.runner import TaskRunner
        kw.setdefault('extrinsics', self.ext)
        return TaskRunner(self.cfg, self.task, self.arm, self.camera, self.perception,
                          self.clock, self.logger, **kw)


def test_fake_world_grasp_and_release():
    from cube_task.core.zones import parse_zones
    w = FakeWorld(parse_zones(make_cfg()))
    d = w.add(make_det('red', 0.3, 0.0))
    assert w.try_grasp(top_down_T(0.3, 0.0, 0.015, 0.0)) == d.width
    assert w.release(top_down_T(0.20, 0.30, 0.1, 0.0)) == 'zone_red'
    assert w.placed == {'zone_red': ['red']}


def test_fake_camera_respects_newer_than():
    clock = SimClock()
    cam = FakeCamera(clock)
    f = cam.get_frame(newer_than=clock.now() + 1.0)
    assert f.stamp > 1001.0

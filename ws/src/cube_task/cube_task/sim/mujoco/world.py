"""
State of a built MuJoCo scene (see scene.build_scene).

MujocoWorld holds the MjModel / MjData, the scenario info, the zones with their
real floor heights and one ObjectRecord per object. It answers what the runner,
tests and benchmarks ask: object poses, which objects are held, what each zone
holds, the score, and config_overlay(), which makes the robot code use the
scene (robot.backend / camera.backend 'mujoco', the MuJoCo TCP, the rendered
camera pose and intrinsics). attach(clock) creates the engine.MujocoSim that
steps the physics with the clock.

Object frames are the SimObject frames: origin at the bottom center, z up.
Object positions for zones / scoring use the center of mass.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.transforms import rot_to_quat
from ...core.types import CameraIntrinsics, Zone
from ..scenarios import perturb_pose
from .engine import MujocoSim, require_mujoco

REST_LIN_SPEED = 0.02        # m/s, an object slower than this (and REST_ANG_SPEED) is at rest
REST_ANG_SPEED = 0.5         # rad/s
ZONE_TOP = 0.08              # m, center of mass at most this far above a zone floor counts
TABLE_TOL = 0.01             # m, center of mass at most this far below the table top = on it
STATUSES = ('zone', 'held', 'table', 'dropped')


@dataclass(frozen=True)
class ObjectRecord:
    """One scenario object in the MuJoCo model (ids index MjModel arrays)."""

    name: str
    label: str
    shape: str
    body: str
    body_id: int
    geom_ids: Tuple[int, ...]
    qpos_adr: int                # freejoint qpos: x y z qw qx qy qz
    dof_adr: int                 # freejoint qvel: v (world) w (body frame)
    mass: float                  # kg
    color_bgr: Tuple[int, ...]


class MujocoWorld:
    """
    MuJoCo model + data of one scenario, with scoring helpers.

    Attributes: model, data (mujoco.MjModel / MjData), sim (engine.MujocoSim
    or None before attach), info (scenario info), meta (names / ids for the
    arm, camera and recorder, see scene._make_meta), zones (name -> core Zone
    with the real floor z), objects (ObjectRecord list), table_z, tcp_z
    (fingertip end along link6 z = robot.tcp_offset z), max_width (gripper
    stroke, m), materials ('legacy' | 'official'), xml_path.
    Reads take sim.lock when attached, so a camera thread sees a consistent state.
    """

    def __init__(self, model: Any, data: Any, info: Dict[str, Any], meta: Dict[str, Any],
                 zones: Dict[str, Zone], objects: Sequence[ObjectRecord], clock: Any = None):
        self._mj = require_mujoco()
        self.model = model
        self.data = data
        self.info = info
        self.meta = meta
        self.zones: Dict[str, Zone] = dict(zones)
        self.objects: List[ObjectRecord] = list(objects)
        self._by_name = {o.name: o for o in self.objects}
        self._geom_obj = {g: o.name for o in self.objects for g in o.geom_ids}
        body = self._mj.mjtObj.mjOBJ_BODY
        self._finger_ids = tuple(int(self._mj.mj_name2id(model, body, b))
                                 for b in meta['finger_bodies'])
        self.table_z = float(meta['table_z'])
        self.tcp_z = float(meta['tcp_z'])
        self.max_width = float(meta['max_width'])
        self.xml_path = str(meta.get('xml_path', ''))
        self.materials = str(meta.get('materials', 'legacy'))
        self.sim: Optional[MujocoSim] = None
        self._lock = threading.RLock()
        if clock is not None:
            self.attach(clock)

    # ------------------------------------------------------------- engine
    def attach(self, clock: Any) -> MujocoSim:
        """Create the MujocoSim stepping this scene with clock (once); return it."""
        if self.sim is not None:
            if self.sim.clock is clock:
                return self.sim
            raise RuntimeError('MujocoWorld is already attached to another clock')
        self.sim = MujocoSim(self.model, self.data, clock, self.meta)
        return self.sim

    def _locked(self):
        return self.sim.lock if self.sim is not None else self._lock

    def forward(self) -> None:
        """Recompute positions / contacts from qpos (mj_forward)."""
        with self._locked():
            self._mj.mj_forward(self.model, self.data)

    def settle(self, seconds: float) -> None:
        """
        Advance physics by seconds.

        Attached: through the clock (controllers and frame hooks run). Not
        attached: plain mj_step calls with the current ctrl.
        """
        if self.sim is not None:
            self.sim.settle(seconds)
            return
        with self._lock:
            for _ in range(int(round(float(seconds) / float(self.model.opt.timestep)))):
                self._mj.mj_step(self.model, self.data)

    # ------------------------------------------------------------- objects
    def object(self, name: str) -> ObjectRecord:
        """Return the record of an object; KeyError if unknown."""
        return self._by_name[name]

    def geom_object(self, geom_id: int) -> Optional[str]:
        """Return the name of the object owning a geom id, or None."""
        return self._geom_obj.get(int(geom_id))

    def object_pose(self, name: str) -> np.ndarray:
        """Return T_base_obj (bottom-center frame, 4x4) of an object."""
        b = self._by_name[name].body_id
        with self._locked():
            T = np.eye(4)
            T[:3, :3] = np.asarray(self.data.xmat[b], dtype=float).reshape(3, 3)
            T[:3, 3] = np.asarray(self.data.xpos[b], dtype=float)
        return T

    def object_poses(self) -> Dict[str, np.ndarray]:
        """Return name -> T_base_obj of every object."""
        return {o.name: self.object_pose(o.name) for o in self.objects}

    def object_com(self, name: str) -> np.ndarray:
        """Return the center of mass (base frame, m) of an object."""
        with self._locked():
            return np.asarray(self.data.xipos[self._by_name[name].body_id], dtype=float).copy()

    def object_speed(self, name: str) -> Tuple[float, float]:
        """Return (linear m/s, angular rad/s) speed of an object."""
        a = self._by_name[name].dof_adr
        with self._locked():
            v = np.asarray(self.data.qvel[a:a + 6], dtype=float)
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def max_object_speed(self) -> Tuple[float, float]:
        """Return the largest (linear, angular) object speed."""
        sp = [self.object_speed(o.name) for o in self.objects] or [(0.0, 0.0)]
        return max(s[0] for s in sp), max(s[1] for s in sp)

    def set_object_pose(self, name: str, T_base_obj: Any) -> None:
        """Teleport an object to T_base_obj (bottom-center frame) at rest (tests, setup)."""
        rec = self._by_name[name]
        T = np.asarray(T_base_obj, dtype=float)
        x, y, z, w = rot_to_quat(T[:3, :3])
        with self._locked():
            self.data.qpos[rec.qpos_adr:rec.qpos_adr + 7] = [T[0, 3], T[1, 3], T[2, 3],
                                                             w, x, y, z]
            self.data.qvel[rec.dof_adr:rec.dof_adr + 6] = 0.0
            self._mj.mj_forward(self.model, self.data)

    def held_objects(self) -> List[str]:
        """Return the objects touching both fingers (in scene order)."""
        m, d = self.model, self.data
        touch: Dict[str, set] = {}
        with self._locked():
            n = int(d.ncon)
            if n == 0:
                return []
            geoms = np.asarray(d.contact.geom[:n], dtype=int)
            excluded = np.asarray(d.contact.exclude[:n], dtype=int)
            for (g1, g2), ex in zip(geoms, excluded):
                if ex:
                    continue
                for ga, gb in ((g1, g2), (g2, g1)):
                    obj = self._geom_obj.get(int(ga))
                    fb = int(m.geom_bodyid[gb])
                    if obj is not None and fb in self._finger_ids:
                        touch.setdefault(obj, set()).add(fb)
        return [o.name for o in self.objects if len(touch.get(o.name, ())) == 2]

    # ------------------------------------------------------------- scoring
    def zone_at(self, com: Sequence[float]) -> Optional[str]:
        """
        Return the zone holding a center of mass, or None.

        The xy must be inside the zone footprint and z below floor + ZONE_TOP
        (nearest zone center on overlap). Speed is not checked here.
        """
        x, y, z = (float(v) for v in com[:3])
        hits = [zn for zn in self.zones.values()
                if zn.contains_xy(x, y) and z < float(zn.center[2]) + ZONE_TOP]
        if not hits:
            return None
        return min(hits, key=lambda zn: math.hypot(x - zn.center[0], y - zn.center[1])).name

    def on_table(self, com: Sequence[float]) -> bool:
        """Check that a point is over the table top and not below it (TABLE_TOL)."""
        x0, x1, y0, y1 = self.meta['table_extent']
        return (x0 <= float(com[0]) <= x1 and y0 <= float(com[1]) <= y1
                and float(com[2]) >= self.table_z - TABLE_TOL)

    def object_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Return name -> {label, status, zone, com, speed, at_rest} for every object.

        status: 'held' (touching both fingers), 'zone' (at rest in a zone, see
        zone_at), 'table' (anything else over the table: on it, moving, in
        flight) or 'dropped' (off the table, e.g. on the floor).
        """
        with self._locked():
            return self._status(set(self.held_objects()))

    def _status(self, held: set) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for rec in self.objects:
            com = self.object_com(rec.name)
            lin, ang = self.object_speed(rec.name)
            rest = lin < REST_LIN_SPEED and ang < REST_ANG_SPEED
            zone = self.zone_at(com) if rest and rec.name not in held else None
            if rec.name in held:
                status = 'held'
            elif zone is not None:
                status = 'zone'
            elif self.on_table(com):
                status = 'table'
            else:
                status = 'dropped'
            out[rec.name] = {'label': rec.label, 'status': status, 'zone': zone,
                             'com': [round(float(v), 4) for v in com],
                             'speed': round(lin, 4), 'at_rest': bool(rest)}
        return out

    def placements(self) -> Dict[str, List[str]]:
        """Return zone name -> labels of the objects at rest in it (every zone listed)."""
        out: Dict[str, List[str]] = {name: [] for name in self.zones}
        for st in self.object_status().values():
            if st['status'] == 'zone':
                out[st['zone']].append(st['label'])
        return out

    def score(self, class_to_zone: Dict[str, str]) -> Dict[str, int]:
        """
        Count correct / wrong placements, remaining targets and dropped objects.

        correct: at rest in zone class_to_zone[label]. wrong: at rest in any
        other zone (labels without a zone included). remaining: target objects
        (label in class_to_zone) still over the table or held. dropped: objects
        of any label that left the table (floor).
        """
        res = {'correct': 0, 'wrong': 0, 'remaining': 0, 'dropped': 0}
        for st in self.object_status().values():
            if st['status'] == 'zone':
                key = 'correct' if class_to_zone.get(st['label']) == st['zone'] else 'wrong'
                res[key] += 1
            elif st['status'] == 'dropped':
                res['dropped'] += 1
            elif st['label'] in class_to_zone:
                res['remaining'] += 1
        return res

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary for logs."""
        return {'task': self.info.get('task'), 'seed': self.info.get('seed'),
                'table_z': self.table_z, 'objects': self.object_status(),
                'placements': self.placements(), 'held': self.held_objects()}

    # -------------------------------------------------------------- camera
    @property
    def task_camera(self) -> str:
        """Return the MuJoCo name of the task camera ('task_cam' or 'wrist_cam')."""
        return str(self.meta['camera'])

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return the intrinsics of the task camera as rendered (MuJoCo pinhole)."""
        return CameraIntrinsics.from_dict(self.meta['intrinsics'])

    @property
    def T_base_camera(self) -> Optional[np.ndarray]:
        """Return the fixed task camera pose (eye_to_hand), optical frame, or None."""
        T = self.meta.get('T_base_camera')
        return None if T is None else np.array(T, dtype=float)

    @property
    def T_flange_camera(self) -> Optional[np.ndarray]:
        """Return the wrist camera pose in link6 (eye_in_hand), optical frame, or None."""
        T = self.meta.get('T_flange_camera')
        return None if T is None else np.array(T, dtype=float)

    def camera_pose(self, name: Optional[str] = None) -> np.ndarray:
        """Return the current T_base_camera (optical frame) of a MuJoCo camera."""
        mj = self._mj
        cid = int(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, name or self.task_camera))
        if cid < 0:
            raise KeyError('no camera named %r' % name)
        with self._locked():
            R = np.asarray(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)
            p = np.asarray(self.data.cam_xpos[cid], dtype=float)
        T = np.eye(4)
        T[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = p
        return T

    # -------------------------------------------------------------- config
    def config_overlay(self, extrinsics_error: Sequence[float] = (0.0, 0.0)) -> Dict[str, Any]:
        """
        Return a config overlay to deep_merge over the scenario overlay of info.

        Sets robot.backend / camera.backend 'mujoco', robot.tcp_offset (MuJoCo
        fingertip), robot.gripper.max_width (the scene's stroke, see
        sim.mujoco.gripper_max_width), perception.table.expected_z, the
        zones with their real floors, sim.intrinsics of the rendered camera,
        sim.T_base_camera / sim.T_flange_camera as rendered and
        camera.extrinsics_inline: the rendered pose with extrinsics_error
        ([translation m, rotation deg]) applied in the direction
        scenario_config_overlay uses for the same seed.
        """
        if len(extrinsics_error) != 2:
            raise ValueError('extrinsics_error must be [translation m, rotation deg]')
        mount = self.info['mount']
        T_true = self.T_base_camera if mount == 'eye_to_hand' else self.T_flange_camera
        rng = np.random.default_rng([int(self.info['seed']), 7919])
        T_est = perturb_pose(T_true, extrinsics_error[0], extrinsics_error[1], rng)
        zones = {}
        for name, z in self.zones.items():
            zones[name] = {'center': [float(v) for v in z.center],
                           'size': [float(v) for v in z.size], 'yaw': float(z.yaw),
                           'wall_height': float(z.wall_height), 'label': str(z.label)}
            if z.release_height is not None:
                zones[name]['release_height'] = float(z.release_height)
        err = [float(extrinsics_error[0]), float(extrinsics_error[1])]
        return {
            'zones': zones,
            'robot': {'backend': 'mujoco',
                      'tcp_offset': [float(v) for v in self.meta['tcp_offset']],
                      'gripper': {'max_width': self.max_width}},
            'camera': {'backend': 'mujoco', 'mount': mount,
                       'extrinsics_inline': {'mount': mount, 'matrix': _as_list(T_est)}},
            'perception': {'table': {'expected_z': self.table_z}},
            'sim': {'intrinsics': dict(self.meta['intrinsics']),
                    'T_base_camera': _as_list(self.T_base_camera),
                    'T_flange_camera': _as_list(self.T_flange_camera),
                    'extrinsics_error': err,
                    'mujoco': {'camera': self.task_camera, 'table_z': self.table_z,
                               'tcp_z': self.tcp_z, 'gripper_max_width': self.max_width}},
        }


def _as_list(T: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    return None if T is None else [[float(v) for v in row] for row in np.asarray(T, dtype=float)]

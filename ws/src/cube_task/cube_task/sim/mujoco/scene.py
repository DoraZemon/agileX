"""
MuJoCo scene of a competition scenario (sim.scenarios.make_scenario).

build_scene(info) turns a scenario into one MJCF model: the menagerie PiPER
(piper.xml, unchanged except named finger pads with grasp friction, the
'tcp' / 'flange' sites and the wrist camera on link6), a table whose top is at
table_z (the lowest point of the robot base, so the base stands on it), the
place zones (flat mats: visual boxes without collision; walled bins and trays:
floor + 4 walls with collision), the scenario objects as free bodies with
realistic masses and friction, and the cameras: 'task_cam' (eye_to_hand) or
'wrist_cam' on link6 (eye_in_hand), plus 'overview_cam' and 'top_cam' for
videos. World frame == robot base frame. Inside MuJoCo the MuJoCo model is the
robot: its own kinematics define the flange (link6) and the TCP site.

The XML is written next to piper.xml (``_scene_<task>_<mount>_<difficulty>_s<seed>_<hash>.xml``)
so its mesh paths resolve. It opens in the MuJoCo viewer:
``python -m mujoco.viewer --mjcf=<MujocoWorld.xml_path>``.

Camera frames: the optical frame has x right, y down, z forward; a MuJoCo
camera looks along its -z with +y up, so R_mujoco = R_optical @ diag(1, -1, -1)
(the same matrix converts back). fovy = 2 atan((H / 2) / fy); the rendered
image then has fx = fy = (H / 2) / tan(fovy / 2), cx = (W - 1) / 2,
cy = (H - 1) / 2.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import logging
import math
import os
import re
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.transforms import make_T, rot_to_quat, rot_z
from ...core.types import CameraIntrinsics, Zone
from ..world import SimObject, SimWorld, WALL_THICKNESS
from .engine import require_mujoco
from .world import MujocoWorld, ObjectRecord

_LOG = logging.getLogger('cube_task.sim.mujoco.scene')

DEFAULT_ASSETS_DIR = '~/.cache/cube_task/mujoco/agilex_piper'
PIPER_XML = 'piper.xml'

# Names in the PiPER model (mujoco_menagerie agilex_piper) and added by the scene.
ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
GRIPPER_ACTUATOR = 'gripper'
FINGER_JOINT = 'joint7'              # slide 0..0.035 m; gripper width = 2 * joint7
FINGER_MIMIC_JOINT = 'joint8'        # = -joint7 (equality constraint)
FINGER_BODIES = ('link7', 'link8')
FLANGE_BODY = 'link6'
BASE_BODY = 'base_link'
TCP_SITE = 'tcp'
FLANGE_SITE = 'flange'
TASK_CAM = 'task_cam'
WRIST_CAM = 'wrist_cam'
OVERVIEW_CAM = 'overview_cam'
TOP_CAM = 'top_cam'
TABLE_GEOM = 'table'
FLOOR_GEOM = 'floor'

# Physics. Elliptic cones with impratio 10 make friction stiff; Newton converges
# in a few iterations, the limits only cap bad steps. The noslip pass removes
# the creep of soft contacts: without it a lemon (ellipsoid on its side) rolls
# ~2 mm/s from rest and held fruit slide out of the pads faster. It costs about
# +40 % per step (60 us per step with 6 L2 objects).
TIMESTEP = 0.002
SOLVER_ITERATIONS = 50
LS_ITERATIONS = 20
NOSLIP_ITERATIONS = 5

# Table (box, top at table_z) and the floor below it, base frame (m).
TABLE_X = (-0.45, 0.95)
TABLE_Y = (-0.70, 0.70)
TABLE_THICKNESS = 0.03
TABLE_HEIGHT = 0.75
TABLE_FRICTION = (0.6, 0.005, 0.0001)
CEILING_Z = 1.6                      # the eye_to_hand camera hangs from a pole up to here
BIN_FLOOR = 0.003                    # bin / tray floor thickness; objects rest on it
MAT_THICKNESS = 0.0005               # flat mats: visual only, objects rest on the table

# Contact materials: friction (slide, spin m, roll m). MuJoCo uses the larger
# value of the two geoms, so the pads set the grasp friction and the objects
# set object-table friction. Spin friction at the pads stops a held object
# from swinging about the closing axis; roll friction stops balls rolling away.
PAD_FRICTION = (1.5, 0.02, 0.0005)
PAD_CONDIM = 4
PAD_SOLIMP = (0.95, 0.99, 0.001)
# Wooden blocks: 700 kg/m^3 (35 mm cube ~30 g). Plastic fruit / vegetable
# models are hollow: effective 250 kg/m^3, clipped to 15..60 g.
WOOD = {'friction': (0.9, 0.005, 0.0005), 'density': 700.0, 'mass': (0.010, 0.080),
        'condim': 4}
PLASTIC = {'friction': (0.8, 0.005, 0.0005), 'density': 250.0, 'mass': (0.015, 0.060),
           'condim': 6}

# Official competition materials (scene materials 'official', or SimObject.material).
# Object geoms get priority 1, so their friction / solref / solimp / condim decide
# every contact (finger pads, table, cards): smooth plastic on the rubber pads
# ~0.55, textured EVA foam ~1.0. Foam is slightly soft (solref time constant
# 0.01 s, lower solimp: a 1.5 N grip squeezes a cube ~1 mm per side); its 50 mm
# cube weighs ~10 g (80 kg/m^3). Plastic models
# are hollow shells (shell inertia) with the organizers' masses (OFFICIAL_MASS).
FOAM = {'friction': (1.0, 0.01, 0.0005), 'density': 80.0, 'mass': (0.002, 0.050),
        'condim': 4, 'solref': (0.01, 1.0), 'solimp': (0.9, 0.95, 0.001), 'priority': 1,
        'shell': False}
PLASTIC_MODEL = {'friction': (0.55, 0.005, 0.0005), 'density': 250.0, 'mass': (0.015, 0.150),
                 'condim': 6, 'solimp': PAD_SOLIMP, 'priority': 1, 'shell': True}
MATERIALS = {'wood': WOOD, 'legacy_plastic': PLASTIC, 'foam': FOAM, 'plastic': PLASTIC_MODEL}
SCENE_MATERIALS = ('legacy', 'official')
# kg, official L2 models (organizers' table); L1 foam cubes from FOAM density.
OFFICIAL_MASS = {'cucumber': 0.043, 'pear': 0.100, 'banana': 0.050, 'potato': 0.057,
                 'persimmon': 0.100, 'bell_pepper': 0.082}
# Official zones: flat 230 g/m^2 hard-card sheets, 0.3 mm thick, no walls. With
# sim.mujoco.card_collision (default true) the card collides, its top 0.3 mm above
# the table, so objects can slide / roll off it.
CARD_THICKNESS = 0.0003
CARD_FRICTION = (0.45, 0.005, 0.0001)

# Gripper stroke (sim.mujoco.gripper_max_width): the menagerie model opens 0.070 m;
# 0.100 (new AGX gripper) widens the finger joint ranges, finger geometry unchanged.
GRIPPER_WIDTH_RANGE = (0.04, 0.12)

# Offscreen multisampling. 0: depth (and color) are sampled at pixel centers,
# exact to ~0.01 mm. With MSAA the depth resolve takes an off-center sample:
# -0.5 mm at 0.74 m on the tilted table of the eye_to_hand view (4 samples).
# A video renderer may set model.vis.quality.offsamples = 4 before creating its
# own mujoco.Renderer for smoother frames.
OFFSCREEN_SAMPLES = 0

# Video cameras (look_at eye, target, image-down hint) and their fovy (deg).
OVERVIEW_VIEW = ((0.95, -0.72, 0.62), (0.21, 0.02, 0.07), (0.0, 0.0, -1.0), 45.0)
TOP_VIEW = ((0.24, 0.0, 1.15), (0.24, 0.0, 0.0), (-1.0, 0.0, 0.0), 42.0)

# Compiled models by scene file (content-addressed). Compiling takes ~0.85 s
# (large robot meshes), a deepcopy ~30 ms; each model holds ~20 MB of meshes.
MODEL_CACHE_SIZE = 4
_MODEL_CACHE: 'OrderedDict[str, Any]' = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()

_FLIP = np.diag([1.0, -1.0, -1.0])


# ------------------------------------------------------------------ helpers
def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Read a dotted key from a core Config, a nested dict or None."""
    if cfg is None:
        return default
    if hasattr(cfg, 'data') and hasattr(cfg, 'get'):
        v = cfg.get(key, default)
        return default if v is None else v
    cur = cfg
    for part in key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def _f(v: Any) -> str:
    """Format a number or a sequence of numbers for MJCF."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return ' '.join(_f(x) for x in np.asarray(v, dtype=float).ravel())
    s = '%.12g' % float(v)
    return '0' if s == '-0' else s


def _el(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    """Append a child element; numbers / sequences are formatted with _f."""
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is None:
            continue
        e.set(k.rstrip('_'), v if isinstance(v, str) else _f(v))
    return e


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Return the MuJoCo (w, x, y, z) quaternion of a rotation matrix."""
    x, y, z, w = rot_to_quat(np.asarray(R, dtype=float))
    return np.array([w, x, y, z])


def _rgba(color_bgr: Sequence[int], alpha: float = 1.0, scale: float = 1.0) -> List[float]:
    b, g, r = [min(255.0, float(c) * scale) / 255.0 for c in color_bgr]
    return [r, g, b, float(alpha)]


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))


# ------------------------------------------------------------------- camera
def optical_to_mujoco(R: np.ndarray) -> np.ndarray:
    """Return the MuJoCo camera rotation (-z forward, +y up) of an optical-frame rotation."""
    return np.asarray(R, dtype=float) @ _FLIP


def mujoco_to_optical(R: np.ndarray) -> np.ndarray:
    """Return the optical-frame rotation (x right, y down, z forward) of a MuJoCo camera."""
    return np.asarray(R, dtype=float) @ _FLIP


def fovy_deg(intr: CameraIntrinsics) -> float:
    """Return the MuJoCo fovy (deg) that reproduces fy at the image height."""
    return math.degrees(2.0 * math.atan((float(intr.height) / 2.0) / float(intr.fy)))


def intrinsics_for_fovy(fovy: float, width: int, height: int) -> CameraIntrinsics:
    """Return the pinhole intrinsics of a MuJoCo camera rendered at width x height."""
    f = (float(height) / 2.0) / math.tan(math.radians(float(fovy)) / 2.0)
    return CameraIntrinsics(f, f, (width - 1) / 2.0, (height - 1) / 2.0, int(width), int(height))


def _camera(parent: ET.Element, name: str, T_parent_opt: np.ndarray, fovy: float) -> None:
    R = optical_to_mujoco(T_parent_opt[:3, :3])
    _el(parent, 'camera', name=name, pos=T_parent_opt[:3, 3], quat=_quat_wxyz(R), fovy=fovy,
        mode='fixed')


def _look_at(eye, target, down_hint) -> np.ndarray:
    from ..scenarios import look_at
    return look_at(eye, target, down_hint=down_hint)


# --------------------------------------------------------------- PiPER model
def assets_dir_for(cfg: Any = None, assets_dir: Optional[str] = None) -> Path:
    """Return the directory holding piper.xml (argument, sim.mujoco.assets_dir, default)."""
    d = assets_dir or _cfg_get(cfg, 'sim.mujoco.assets_dir', None) or DEFAULT_ASSETS_DIR
    path = Path(str(d)).expanduser().resolve()
    if not (path / PIPER_XML).is_file():
        raise FileNotFoundError(
            'PiPER MuJoCo model not found: %s. Copy mujoco_menagerie/agilex_piper '
            '(piper.xml + assets/) there or set sim.mujoco.assets_dir.' % (path / PIPER_XML))
    return path


def _geom_min_z(mj, m, d, g: int) -> float:
    """Return the lowest world z of geom g (exact for meshes, capsules, spheres, boxes)."""
    R = np.asarray(d.geom_xmat[g], dtype=float).reshape(3, 3)
    p = np.asarray(d.geom_xpos[g], dtype=float)
    s = np.asarray(m.geom_size[g], dtype=float)
    t = int(m.geom_type[g])
    if t == mj.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[a:a + n], dtype=float)
        return float(np.min(v @ R[2]) + p[2])
    if t == mj.mjtGeom.mjGEOM_CAPSULE:
        return float(p[2] - abs(R[2, 2]) * s[1] - s[0])
    if t == mj.mjtGeom.mjGEOM_SPHERE:
        return float(p[2] - s[0])
    return float(p[2] - np.abs(R[2]) @ s)


@functools.lru_cache(maxsize=4)
def piper_geometry(assets_dir: str) -> Dict[str, Any]:
    """
    Measure the PiPER model: table_z, fingertip z in link6 and the gripper range.

    Returns {'table_z': lowest point of the base_link geoms (m, 0.1 mm),
    'tcp_z': fingertip end of the finger pads along link6 z (m, 0.1 mm),
    'max_width': 2 * the joint7 upper limit (m)}. Compiles piper.xml once
    per directory (about 1 s, the meshes are large).
    """
    mj = require_mujoco()
    m = mj.MjModel.from_xml_path(str(Path(assets_dir) / PIPER_XML))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    body = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    base = body[BASE_BODY]
    table_z = min(_geom_min_z(mj, m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] == base)
    i6 = body[FLANGE_BODY]
    R6 = np.asarray(d.xmat[i6], dtype=float).reshape(3, 3)
    p6 = np.asarray(d.xpos[i6], dtype=float)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    tips = []
    for g in range(m.ngeom):
        if (m.geom_bodyid[g] in (body[FINGER_BODIES[0]], body[FINGER_BODIES[1]])
                and int(m.geom_type[g]) == mj.mjtGeom.mjGEOM_BOX):
            Rg = np.asarray(d.geom_xmat[g], dtype=float).reshape(3, 3)
            w = (corners * m.geom_size[g]) @ Rg.T + d.geom_xpos[g]
            tips.append(float(np.max((w - p6) @ R6[:, 2])))
    if len(tips) != 4:
        raise RuntimeError('expected 4 finger pad boxes in %s, found %d' % (PIPER_XML, len(tips)))
    j7 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, FINGER_JOINT)
    return {'table_z': round(table_z, 4) + 0.0, 'tcp_z': round(max(tips), 4),
            'max_width': round(2.0 * float(m.jnt_range[j7][1]), 6)}


# ------------------------------------------------------------------ objects
COMPOUND_PARTS = {'sphere': 1, 'ellipsoid': 3, 'cylinder': 2, 'box': 3, 'capsule': 2}


def scene_materials(info: Optional[Dict[str, Any]] = None, objects: Sequence[Any] = (),
                    cfg: Any = None) -> str:
    """
    Return the material set of a scene: 'legacy' or 'official'.

    info['materials'] wins, then config sim.mujoco.materials; without either,
    'official' when an object is 'compound' or carries a material / mass,
    else 'legacy' (wooden blocks, light plastic models, visual mats).
    """
    m = (info or {}).get('materials') or _cfg_get(cfg, 'sim.mujoco.materials', None)
    if m is None:
        m = 'official' if any(o.shape == 'compound' or getattr(o, 'material', None)
                              or getattr(o, 'mass', None) is not None
                              for o in objects) else 'legacy'
    if m not in SCENE_MATERIALS:
        raise ValueError('scene materials must be one of %s, got %r' % (SCENE_MATERIALS, m))
    return str(m)


def object_material_name(obj: SimObject, materials: str = 'legacy') -> str:
    """Return the MATERIALS key of an object (its own material, else by shape)."""
    name = getattr(obj, 'material', None)
    if name:
        if name not in MATERIALS:
            raise ValueError('object %s: unknown material %r (expected one of %s)'
                             % (obj.name, name, sorted(MATERIALS)))
        return str(name)
    if materials == 'official':
        return 'foam' if obj.shape == 'box' else 'plastic'
    return 'wood' if obj.shape == 'box' else 'legacy_plastic'


def object_material(obj: SimObject, materials: str = 'legacy') -> Dict[str, Any]:
    """Return the contact / mass material of an object (see MATERIALS)."""
    return MATERIALS[object_material_name(obj, materials)]


def _compound_part(obj_name: str, k: int, p: Dict[str, Any]
                   ) -> Tuple[str, Dict[str, Any], float]:
    """Return (geom type, MJCF attributes, volume) of one part of a 'compound' object."""
    what = 'object %s part %d' % (obj_name, k)
    kind = str(p.get('shape'))
    if kind not in COMPOUND_PARTS:
        raise ValueError('%s: unknown shape %r (expected one of %s)'
                         % (what, kind, sorted(COMPOUND_PARTS)))
    s = [float(v) for v in p.get('size', ())]
    if len(s) != COMPOUND_PARTS[kind] or not all(math.isfinite(v) and v > 0 for v in s):
        raise ValueError('%s: %s needs %d positive sizes, got %r'
                         % (what, kind, COMPOUND_PARTS[kind], p.get('size')))
    pos = np.asarray(p.get('pos', (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    yaw = float(p.get('yaw', 0.0) or 0.0)
    if not (np.all(np.isfinite(pos)) and math.isfinite(yaw)):
        raise ValueError('%s: non-finite pos / yaw' % what)
    attrs: Dict[str, Any] = {'pos': pos}
    if kind == 'capsule':
        r, hl = s
        u = hl * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        attrs = {'size': [r], 'fromto': np.concatenate([pos - u, pos + u])}
        vol = math.pi * r * r * 2.0 * hl + 4.0 / 3.0 * math.pi * r ** 3
    else:
        if yaw != 0.0:
            attrs['quat'] = _quat_wxyz(rot_z(yaw))
        if kind == 'sphere':
            attrs['size'] = [s[0]]
            vol = 4.0 / 3.0 * math.pi * s[0] ** 3
        elif kind == 'ellipsoid':
            attrs['size'] = s
            vol = 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
        elif kind == 'cylinder':
            attrs['size'] = [s[0], s[1] / 2.0]
            vol = math.pi * s[0] ** 2 * s[1]
        else:
            attrs['size'] = [v / 2.0 for v in s]
            vol = s[0] * s[1] * s[2]
    color = p.get('color_bgr')
    if color is not None:
        c = [int(v) for v in color]
        if len(c) != 3 or not all(0 <= v <= 255 for v in c):
            raise ValueError('%s: color_bgr must be 3 values in 0..255, got %r' % (what, color))
        attrs['_color_bgr'] = c
    return kind, attrs, vol


def _object_parts(obj: SimObject) -> List[Tuple[str, Dict[str, Any], float]]:
    """Return [(geom type, MJCF attributes, volume m^3)] in the object frame (bottom center)."""
    s = obj.size
    if obj.shape == 'compound':
        if not len(s):
            raise ValueError('object %s: compound needs at least one part' % obj.name)
        return [_compound_part(obj.name, k, p) for k, p in enumerate(s)]
    if obj.shape == 'box':
        return [('box', {'size': [v / 2.0 for v in s], 'pos': [0.0, 0.0, s[2] / 2.0]},
                 s[0] * s[1] * s[2])]
    if obj.shape == 'sphere':
        return [('sphere', {'size': [s[0]], 'pos': [0.0, 0.0, s[0]]},
                 4.0 / 3.0 * math.pi * s[0] ** 3)]
    if obj.shape == 'ellipsoid':
        return [('ellipsoid', {'size': list(s), 'pos': [0.0, 0.0, s[2]]},
                 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2])]
    if obj.shape == 'cylinder':
        return [('cylinder', {'size': [s[0], s[1] / 2.0], 'pos': [0.0, 0.0, s[1] / 2.0]},
                 math.pi * s[0] ** 2 * s[1])]
    r, pts = s[0], np.asarray(s[1], dtype=float)
    ball = 4.0 / 3.0 * math.pi * r ** 3
    segs = [(a, b) for a, b in zip(pts[:-1], pts[1:]) if np.linalg.norm(b - a) > 1e-6]
    if obj.shape == 'sphere_cluster' or not segs:
        return [('sphere', {'size': [r], 'pos': p}, ball) for p in pts]
    return [('capsule', {'size': [r], 'fromto': np.concatenate([a, b])},
             math.pi * r * r * float(np.linalg.norm(b - a)) + ball) for a, b in segs]


def _part_area(gtype: str, attrs: Dict[str, Any]) -> float:
    """Return the surface area (m^2) of a part, to spread a shell's mass over its parts."""
    s = [float(v) for v in attrs['size']]
    if gtype == 'sphere':
        return 4.0 * math.pi * s[0] ** 2
    if gtype == 'ellipsoid':
        a, b, c, p = s[0], s[1], s[2], 1.6075          # Knud Thomsen, error < 1.1 %
        return 4.0 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1.0 / p)
    if gtype == 'cylinder':
        return 2.0 * math.pi * s[0] * (2.0 * s[1]) + 2.0 * math.pi * s[0] ** 2
    if gtype == 'capsule':
        ft = np.asarray(attrs['fromto'], dtype=float)
        return (2.0 * math.pi * s[0] * float(np.linalg.norm(ft[3:] - ft[:3]))
                + 4.0 * math.pi * s[0] ** 2)
    return 8.0 * (s[0] * s[1] + s[1] * s[2] + s[2] * s[0])


def object_mass(obj: SimObject, materials: str = 'legacy') -> float:
    """
    Return the simulated mass (kg).

    SimObject.mass when set; official plastic models: OFFICIAL_MASS of the
    label; else material density x volume, clipped to the material's range.
    """
    given = getattr(obj, 'mass', None)
    if given is not None:
        m = float(given)
        if not (math.isfinite(m) and 0.0 < m <= 5.0):
            raise ValueError('object %s: mass must be in (0, 5] kg, got %r' % (obj.name, given))
        return m
    name = object_material_name(obj, materials)
    if name == 'plastic' and obj.label in OFFICIAL_MASS:
        return OFFICIAL_MASS[obj.label]
    mat = MATERIALS[name]
    parts = _object_parts(obj)
    if obj.shape == 'capsule_chain' and parts and parts[0][0] == 'capsule':
        r = obj.size[0]
        ball = 4.0 / 3.0 * math.pi * r ** 3
        vol = sum(v - ball for _t, _a, v in parts) + ball      # union of the chain
    else:
        vol = sum(v for _t, _a, v in parts)
    lo, hi = mat['mass']
    return float(min(max(vol * mat['density'], lo), hi))


def _add_object(wb: ET.Element, asset: ET.Element, obj: SimObject, T: np.ndarray,
                materials: str = 'legacy') -> str:
    """Add obj as a free body at pose T (bottom-center frame); return the body name."""
    name = 'obj_%s' % _safe(obj.name)
    mat = object_material(obj, materials)
    _el(asset, 'material', name='ct_' + name, rgba=_rgba(obj.color_bgr), specular=0.15,
        shininess=0.3)
    body = _el(wb, 'body', name=name, pos=T[:3, 3], quat=_quat_wxyz(T[:3, :3]))
    _el(body, 'freejoint', name=name + '_free')
    parts = _object_parts(obj)
    shell = bool(mat.get('shell', False))
    weights = [_part_area(t, a) if shell else v for t, a, v in parts]
    total = sum(weights)
    mass = object_mass(obj, materials)
    for k, ((gtype, attrs, _vol), w) in enumerate(zip(parts, weights)):
        attrs = dict(attrs)
        color = attrs.pop('_color_bgr', None)
        material = 'ct_' + name
        if color is not None:
            material = 'ct_%s_p%d' % (name, k)
            _el(asset, 'material', name=material, rgba=_rgba(color), specular=0.15,
                shininess=0.3)
        _el(body, 'geom', name='%s_g%d' % (name, k), type=gtype, material=material,
            friction=mat['friction'], condim=mat['condim'], mass=mass * w / total,
            solref=mat.get('solref'), solimp=mat.get('solimp'), priority=mat.get('priority'),
            shellinertia='true' if shell else None, **attrs)
    return name


# -------------------------------------------------------------------- zones
FLAT_KINDS = ('mat', 'card', 'card_visual')


def zone_floor_z(zone: Zone, table_z: float, flat: str = 'mat') -> float:
    """
    Return the z an object rests at in the zone.

    Walled bins / trays: the bin floor top. Flat zones: table_z for legacy
    mats and visual-only cards, the card top for colliding cards (flat 'card').
    """
    if zone.wall_height > 0.0:
        return float(table_z) + BIN_FLOOR
    return float(table_z) + (CARD_THICKNESS if flat == 'card' else 0.0)


def _add_zone(wb: ET.Element, asset: ET.Element, zone: Zone, color_bgr: Sequence[int],
              table_z: float, flat: str = 'mat') -> str:
    """
    Add a flat zone or a walled bin / tray; return the body name.

    flat (zones without walls): 'mat' visual 0.5 mm box ('<body>_mat', legacy);
    'card' 0.3 mm hard card that collides ('<body>_card', top flush 0.3 mm above
    the table); 'card_visual' the same card without collision.
    """
    if flat not in FLAT_KINDS:
        raise ValueError('flat zone kind must be one of %s, got %r' % (FLAT_KINDS, flat))
    name = 'zn_%s' % _safe(zone.name)
    _el(asset, 'material', name='ct_' + name, rgba=_rgba(color_bgr), specular=0.05,
        shininess=0.1)
    _el(asset, 'material', name='ct_%s_wall' % name, rgba=_rgba(color_bgr, scale=0.85),
        specular=0.1, shininess=0.2)
    c = np.asarray(zone.center, dtype=float)
    body = _el(wb, 'body', name=name, pos=[c[0], c[1], table_z],
               quat=_quat_wxyz(rot_z(float(zone.yaw))))
    hx, hy = zone.size[0] / 2.0, zone.size[1] / 2.0
    if zone.wall_height <= 0.0 and flat == 'mat':
        _el(body, 'geom', name=name + '_mat', type='box', size=[hx, hy, MAT_THICKNESS / 2.0],
            pos=[0.0, 0.0, MAT_THICKNESS / 2.0], material='ct_' + name, contype=0,
            conaffinity=0, group=1)
        return name
    if zone.wall_height <= 0.0:
        off = {} if flat == 'card' else {'contype': 0, 'conaffinity': 0, 'group': 1}
        _el(body, 'geom', name=name + '_card', type='box', size=[hx, hy, CARD_THICKNESS / 2.0],
            pos=[0.0, 0.0, CARD_THICKNESS / 2.0], material='ct_' + name,
            friction=CARD_FRICTION, **off)
        return name
    t = WALL_THICKNESS
    _el(body, 'geom', name=name + '_floor', type='box', size=[hx, hy, BIN_FLOOR / 2.0],
        pos=[0.0, 0.0, BIN_FLOOR / 2.0], material='ct_' + name, friction=TABLE_FRICTION)
    hz = (BIN_FLOOR + float(zone.wall_height)) / 2.0
    walls = (('px', [hx + t / 2.0, 0.0], [t / 2.0, hy + t]),
             ('nx', [-hx - t / 2.0, 0.0], [t / 2.0, hy + t]),
             ('py', [0.0, hy + t / 2.0], [hx, t / 2.0]),
             ('ny', [0.0, -hy - t / 2.0], [hx, t / 2.0]))
    for tag, xy, half in walls:
        _el(body, 'geom', name='%s_wall_%s' % (name, tag), type='box', size=half + [hz],
            pos=xy + [hz], material='ct_%s_wall' % name, friction=TABLE_FRICTION)
    return name


# -------------------------------------------------------------------- robot
def _find_body(root: ET.Element, name: str) -> ET.Element:
    b = root.find(".//body[@name='%s']" % name)
    if b is None:
        raise RuntimeError('%s has no body %r' % (PIPER_XML, name))
    return b


def _set_gripper_width(root: ET.Element, max_width: float) -> None:
    """Set the finger joint ranges and the gripper actuator ctrlrange to a stroke of max_width."""
    half = float(max_width) / 2.0
    for name, rng in ((FINGER_JOINT, [0.0, half]), (FINGER_MIMIC_JOINT, [-half, 0.0])):
        j = root.find(".//joint[@name='%s']" % name)
        if j is None:
            raise RuntimeError('%s has no joint %r' % (PIPER_XML, name))
        j.set('range', _f(rng))
    act = root.find(".//actuator/position[@name='%s']" % GRIPPER_ACTUATOR)
    if act is None:
        raise RuntimeError('%s has no position actuator %r' % (PIPER_XML, GRIPPER_ACTUATOR))
    act.set('inheritrange', '0')
    act.set('ctrllimited', 'true')
    act.set('ctrlrange', _f([0.0, half]))


def _setup_robot(root: ET.Element, tcp_z: float, timestep: float, noslip: int,
                 max_width: Optional[float] = None) -> List[str]:
    """
    Set physics options, name the finger pads, add the tcp / flange sites. Return pad names.

    max_width: gripper stroke (m); None keeps the model's finger joint ranges.
    """
    if max_width is not None:
        _set_gripper_width(root, max_width)
    opt = root.find('option')
    if opt is None:
        opt = _el(root, 'option')
    for k, v in (('timestep', timestep), ('iterations', SOLVER_ITERATIONS),
                 ('ls_iterations', LS_ITERATIONS), ('noslip_iterations', int(noslip)),
                 ('impratio', 10), ('cone', 'elliptic'), ('integrator', 'implicitfast')):
        opt.set(k, v if isinstance(v, str) else _f(v))
    for key in root.findall('keyframe'):      # qpos size changes with the objects
        root.remove(key)
    link6 = _find_body(root, FLANGE_BODY)
    _el(link6, 'site', name=TCP_SITE, pos=[0.0, 0.0, tcp_z], size=[0.004],
        rgba=[1.0, 0.2, 0.2, 0.8], group=4)
    _el(link6, 'site', name=FLANGE_SITE, pos=[0.0, 0.0, 0.0], size=[0.004],
        rgba=[0.2, 0.2, 1.0, 0.8], group=4)
    pads = []
    for bname in FINGER_BODIES:
        boxes = [g for g in _find_body(root, bname).findall('geom') if g.get('type') == 'box']
        for k, g in enumerate(boxes):
            g.set('name', '%s_pad%d' % (bname, k))
            g.set('friction', _f(PAD_FRICTION))
            g.set('condim', str(PAD_CONDIM))
            g.set('solimp', _f(PAD_SOLIMP))
            pads.append(g.get('name'))
    return pads


def _camera_housing(parent: ET.Element, name: str, T_parent_opt: np.ndarray) -> None:
    """Add a visual camera body behind the lens (outside the view: z_optical < 0)."""
    R, p = T_parent_opt[:3, :3], T_parent_opt[:3, 3]
    _el(parent, 'geom', name=name + '_housing', type='box', size=[0.045, 0.013, 0.012],
        pos=p - 0.0135 * R[:, 2], quat=_quat_wxyz(R), rgba=[0.12, 0.12, 0.13, 1.0],
        contype=0, conaffinity=0, group=1, density=0)


def _wrist_camera(root: ET.Element, T_flange_camera: np.ndarray, fovy: float) -> None:
    link6 = _find_body(root, FLANGE_BODY)
    _camera(link6, WRIST_CAM, T_flange_camera, fovy)
    _camera_housing(link6, WRIST_CAM, T_flange_camera)
    p = T_flange_camera[:3, 3]
    # Bracket from the housing to the flange body (cosmetic).
    _el(link6, 'geom', name=WRIST_CAM + '_bracket', type='box', size=[0.014, 0.012, 0.004],
        pos=[p[0] + 0.024, p[1], max(0.006, p[2] - 0.022)], rgba=[0.2, 0.2, 0.22, 1.0],
        contype=0, conaffinity=0, group=1, density=0)


# -------------------------------------------------------------------- world
def _add_environment(root: ET.Element, table_z: float, table_color_bgr: Sequence[int]
                     ) -> Tuple[ET.Element, ET.Element]:
    """Add rendering options, lights, floor and table. Return (asset, worldbody)."""
    vis = _el(root, 'visual')
    _el(vis, 'global', offwidth=1920, offheight=1080)
    _el(vis, 'quality', shadowsize=4096, offsamples=OFFSCREEN_SAMPLES)
    _el(vis, 'headlight', ambient=[0.30] * 3, diffuse=[0.10] * 3, specular=[0.0] * 3)
    _el(vis, 'map', znear=0.01, zfar=10.0)       # x extent 1 -> meters
    _el(root, 'statistic', extent=1.0, center=[0.2, 0.0, 0.15])
    asset = _el(root, 'asset')
    _el(asset, 'texture', name='ct_sky', type='skybox', builtin='gradient',
        rgb1=[0.55, 0.62, 0.70], rgb2=[0.12, 0.13, 0.15], width=512, height=3072)
    _el(asset, 'texture', name='ct_grid', type='2d', builtin='checker', mark='edge',
        rgb1=[0.26, 0.27, 0.29], rgb2=[0.31, 0.32, 0.34], markrgb=[0.4, 0.4, 0.42],
        width=300, height=300)
    _el(asset, 'material', name='ct_floor', texture='ct_grid', texuniform='true',
        texrepeat=[4, 4], reflectance=0.05)
    _el(asset, 'material', name='ct_table', rgba=_rgba(table_color_bgr), specular=0.05,
        shininess=0.1)
    _el(asset, 'material', name='ct_leg', rgba=[0.35, 0.33, 0.30, 1.0])
    wb = _el(root, 'worldbody')
    # Key light: a spot (a directional light's shadow map gives acne on the
    # table); cutoff 35 deg from 2.6 m covers the whole table evenly.
    key_pos = np.array([0.5, -0.5, table_z + 2.5])
    key_dir = np.array([0.25, 0.0, table_z]) - key_pos
    _el(wb, 'light', name='key', directional='false', pos=key_pos,
        dir=key_dir / np.linalg.norm(key_dir), cutoff=35.0, exponent=1.0,
        diffuse=[0.45] * 3, specular=[0.1] * 3, castshadow='true')
    _el(wb, 'light', name='fill', directional='true', pos=[-0.5, 0.6, 2.0],
        dir=[0.4, -0.4, -1.0], diffuse=[0.20] * 3, specular=[0.0] * 3, castshadow='false')
    floor_z = table_z - TABLE_HEIGHT
    _el(wb, 'geom', name=FLOOR_GEOM, type='plane', size=[3.0, 3.0, 0.05],
        pos=[0.25, 0.0, floor_z], material='ct_floor')
    cx, cy = sum(TABLE_X) / 2.0, sum(TABLE_Y) / 2.0
    hx, hy = (TABLE_X[1] - TABLE_X[0]) / 2.0, (TABLE_Y[1] - TABLE_Y[0]) / 2.0
    _el(wb, 'geom', name=TABLE_GEOM, type='box', size=[hx, hy, TABLE_THICKNESS / 2.0],
        pos=[cx, cy, table_z - TABLE_THICKNESS / 2.0], material='ct_table',
        friction=TABLE_FRICTION)
    leg_h = (TABLE_HEIGHT - TABLE_THICKNESS) / 2.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _el(wb, 'geom', name='table_leg_%d%d' % (sx > 0, sy > 0), type='box',
                size=[0.025, 0.025, leg_h], material='ct_leg', contype=0, conaffinity=0,
                pos=[cx + sx * (hx - 0.06), cy + sy * (hy - 0.06), floor_z + leg_h], group=1)
    return asset, wb


def _eye_to_hand_camera(wb: ET.Element, T_base_camera: np.ndarray, fovy: float,
                        table_z: float) -> None:
    """Add task_cam with a housing and a pole from it up to the ceiling (outside its view)."""
    _camera(wb, TASK_CAM, T_base_camera, fovy)
    _camera_housing(wb, TASK_CAM, T_base_camera)
    back = T_base_camera[:3, 3] - 0.015 * T_base_camera[:3, 2]
    top = table_z + CEILING_Z
    if top > float(back[2]) + 0.02:
        _el(wb, 'geom', name=TASK_CAM + '_pole', type='cylinder',
            size=[0.010, (top - back[2]) / 2.0], pos=[back[0], back[1], (top + back[2]) / 2.0],
            rgba=[0.25, 0.25, 0.27, 1.0], contype=0, conaffinity=0, group=1)


def _video_cameras(wb: ET.Element) -> None:
    for name, (eye, target, hint, fovy) in ((OVERVIEW_CAM, OVERVIEW_VIEW), (TOP_CAM, TOP_VIEW)):
        _camera(wb, name, _look_at(eye, target, hint), fovy)


def _scenario_world(info: Dict[str, Any], world: Optional[SimWorld]) -> SimWorld:
    """Return the SimWorld of info (rebuilt with make_scenario when not given)."""
    if world is None:
        from ..scenarios import make_scenario
        world, _again = make_scenario(info['task'], int(info['seed']), info['mount'],
                                      info.get('difficulty', 'normal'))
    if [o.name for o in world.objects] != [o['name'] for o in info['objects']]:
        raise ValueError('the scenario world does not match info (task %s seed %s): pass the '
                         'world make_scenario returned with this info'
                         % (info['task'], info['seed']))
    return world


def _write_xml(assets: Path, root: ET.Element, key: str) -> Path:
    """Write the scene next to piper.xml (atomic, content-addressed); return its path."""
    text = ET.tostring(root, encoding='unicode')
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
    path = assets / ('_scene_%s_%s.xml' % (key, digest))
    if not path.is_file():
        fd, tmp = tempfile.mkstemp(prefix='.tmp_scene_', suffix='.xml', dir=str(assets))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(text)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return path


def _compile(mj, path: Path) -> Any:
    """Return a fresh MjModel of a scene file (a copy of a cached compile when possible)."""
    key = str(path)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            _MODEL_CACHE.move_to_end(key)
            return copy.deepcopy(model)
    model = mj.MjModel.from_xml_path(key)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > MODEL_CACHE_SIZE:
            _MODEL_CACHE.popitem(last=False)
    return copy.deepcopy(model)


def build_scene(info: Dict[str, Any], cfg: Any = None, assets_dir: Optional[str] = None,
                clock: Any = None, world: Optional[SimWorld] = None) -> MujocoWorld:
    """
    Build the MuJoCo scene of a scenario. Return a MujocoWorld.

    info: from sim.scenarios.make_scenario(task, seed, mount, difficulty).
    cfg: optional core Config or nested dict; reads robot.poses.home_joints
    (initial arm pose, default zeros), sim.mujoco.assets_dir,
    sim.mujoco.timestep (0.002 s), sim.mujoco.noslip_iterations (5),
    sim.mujoco.gripper_max_width (null = the model's 0.070 m stroke; 0.100 widens
    the finger joint ranges), sim.mujoco.materials ('legacy' | 'official', see
    scene_materials; info['materials'] wins) and sim.mujoco.card_collision
    (true: official flat zones are 0.3 mm cards that collide).
    assets_dir: directory of piper.xml (default DEFAULT_ASSETS_DIR).
    clock: core.clock.SimClock; when given, the physics engine is created now,
    else call MujocoWorld.attach(clock) later. world: the scenario SimWorld
    (rebuilt from info with make_scenario when None). The arm starts at
    home_joints with the gripper closed; objects rest on the table, not settled.
    """
    mj = require_mujoco()
    task, mount = str(info['task']), str(info['mount'])
    if mount not in ('eye_to_hand', 'eye_in_hand'):
        raise ValueError('unknown camera mount %r' % mount)
    seed, difficulty = int(info['seed']), str(info.get('difficulty', 'normal'))
    sim_world = _scenario_world(info, world)
    assets = assets_dir_for(cfg, assets_dir)
    geo = piper_geometry(str(assets))
    table_z, tcp_z = float(geo['table_z']), float(geo['tcp_z'])
    timestep = float(_cfg_get(cfg, 'sim.mujoco.timestep', TIMESTEP))
    if not 1e-4 <= timestep <= 0.01:
        raise ValueError('sim.mujoco.timestep must be in [0.0001, 0.01] s, got %r' % timestep)
    noslip = int(_cfg_get(cfg, 'sim.mujoco.noslip_iterations', NOSLIP_ITERATIONS))
    width = _cfg_get(cfg, 'sim.mujoco.gripper_max_width', None)
    if width is not None:
        width = float(width)
        if not (math.isfinite(width)
                and GRIPPER_WIDTH_RANGE[0] <= width <= GRIPPER_WIDTH_RANGE[1]):
            raise ValueError('sim.mujoco.gripper_max_width must be in [%g, %g] m, got %r'
                             % (GRIPPER_WIDTH_RANGE + (width,)))
        geo = dict(geo, max_width=round(width, 6))
    materials = scene_materials(info, sim_world.objects, cfg)
    flat = 'mat'
    if materials == 'official':
        flat = 'card' if bool(_cfg_get(cfg, 'sim.mujoco.card_collision', True)) else 'card_visual'
    home = [float(v) for v in _cfg_get(cfg, 'robot.poses.home_joints', [0.0] * 6)]
    if len(home) != 6 or not all(math.isfinite(v) for v in home):
        raise ValueError('robot.poses.home_joints must be 6 finite angles, got %r' % (home,))
    intr = info['intrinsics']
    fovy = fovy_deg(intr)
    rendered = intrinsics_for_fovy(fovy, intr.width, intr.height)
    if (abs(rendered.fx - intr.fx) > 0.5 or abs(rendered.cx - intr.cx) > 0.5
            or abs(rendered.cy - intr.cy) > 0.5):
        _LOG.warning('MuJoCo renders fx = fy = %.1f, cx %.1f, cy %.1f (scenario fx %.1f, '
                     'cx %.1f, cy %.1f); use MujocoWorld.config_overlay() intrinsics',
                     rendered.fx, rendered.cx, rendered.cy, intr.fx, intr.cx, intr.cy)

    root = ET.parse(str(assets / PIPER_XML)).getroot()
    root.set('model', 'cube_task_%s_%s_s%d' % (task, difficulty, seed))
    pads = _setup_robot(root, tcp_z, timestep, noslip, max_width=width)
    if mount == 'eye_in_hand':
        _wrist_camera(root, np.asarray(info['T_flange_camera'], dtype=float), fovy)
    asset, wb = _add_environment(root, table_z, sim_world.table_color_bgr)
    colors = {z.name: z.color_bgr for z in sim_world.zones}
    zones: Dict[str, Zone] = {}
    zone_bodies: Dict[str, str] = {}
    for name, z in info['zones'].items():
        zone_bodies[name] = _add_zone(wb, asset, z, colors.get(name, (110, 110, 110)), table_z,
                                      flat)
        c = np.asarray(z.center, dtype=float)
        floor = zone_floor_z(z, table_z, flat)
        zones[name] = Zone(name=str(name), center=np.array([c[0], c[1], floor]),
                           size=(float(z.size[0]), float(z.size[1])), yaw=float(z.yaw),
                           wall_height=float(z.wall_height), release_height=z.release_height,
                           label=str(z.label))
    bodies: Dict[str, str] = {}
    for obj in sim_world.objects:
        T = obj.T_base_obj
        T[2, 3] += table_z - sim_world.table_z
        c = obj.center_xy()
        inside = [z for z in zones.values() if z.contains_xy(c[0], c[1])]
        if inside:
            T[2, 3] += max(float(z.center[2]) for z in inside) - table_z
        bodies[obj.name] = _add_object(wb, asset, obj, T, materials)
    if mount == 'eye_to_hand':
        _eye_to_hand_camera(wb, np.asarray(info['T_base_camera'], dtype=float), fovy, table_z)
    _video_cameras(wb)
    _el(_el(root, 'contact'), 'exclude', body1=FINGER_BODIES[0], body2=FINGER_BODIES[1])
    path = _write_xml(assets, root, '%s_%s_%s_s%d' % (task, mount, difficulty, seed))
    model = _compile(mj, path)
    data = mj.MjData(model)
    _init_arm(mj, model, data, home)
    meta = _make_meta(mj, model, info, geo, home, pads, zone_bodies, bodies, sim_world, path)
    meta['fovy'] = fovy
    meta['materials'] = materials
    meta['flat_zones'] = flat
    meta['object_materials'] = {o.name: object_material_name(o, materials)
                                for o in sim_world.objects}
    objects = _object_records(mj, model, sim_world, bodies)
    _LOG.debug('scene %s: %d objects, %d zones, table_z %.4f, tcp_z %.4f',
               path.name, len(objects), len(zones), table_z, tcp_z)
    return MujocoWorld(model, data, info, meta, zones, objects, clock=clock)


def _init_arm(mj, m, d, home: Sequence[float]) -> None:
    """Put the arm at home (clipped to the joint ranges), gripper closed, and hold it there."""
    for name, q in zip(ARM_JOINTS, home):
        j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
        lo, hi = (float(v) for v in m.jnt_range[j])
        qc = min(max(float(q), lo), hi)
        d.qpos[m.jnt_qposadr[j]] = qc
        d.ctrl[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)] = qc
    d.ctrl[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)] = 0.0
    mj.mj_forward(m, d)


def camera_table(m) -> Dict[str, Dict[str, Any]]:
    """Return {camera: {'parent', 'T_parent_camera' (optical 4x4), 'fovy' (deg)}} of a model."""
    mj = require_mujoco()
    out = {}
    for i in range(m.ncam):
        R = np.zeros(9)
        mj.mju_quat2Mat(R, np.asarray(m.cam_quat[i], dtype=float))
        T = make_T(mujoco_to_optical(R.reshape(3, 3)), np.asarray(m.cam_pos[i], dtype=float))
        out[mj.mj_id2name(m, mj.mjtObj.mjOBJ_CAMERA, i)] = {
            'parent': mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[i])) or 'world',
            'T_parent_camera': T, 'fovy': float(m.cam_fovy[i])}
    return out


def _as_list(T: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    return None if T is None else [[float(v) for v in row] for row in np.asarray(T)]


def _make_meta(mj, m, info: Dict[str, Any], geo: Dict[str, Any], home: Sequence[float],
               pads: List[str], zone_bodies: Dict[str, str], bodies: Dict[str, str],
               sim_world: SimWorld, path: Path) -> Dict[str, Any]:
    """Return the names / ids / camera data the arm, camera and recorder need."""
    def gid(name: str) -> int:
        return int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name))

    mount = info['mount']
    cams = camera_table(m)
    task_cam = WRIST_CAM if mount == 'eye_in_hand' else TASK_CAM
    intr = info['intrinsics']
    T_cam = cams[task_cam]['T_parent_camera']
    return {
        'task': info['task'], 'seed': int(info['seed']), 'mount': mount,
        'difficulty': info.get('difficulty', 'normal'), 'xml_path': str(path),
        'arm_joints': list(ARM_JOINTS), 'arm_actuators': list(ARM_JOINTS),
        'gripper_actuator': GRIPPER_ACTUATOR, 'finger_joint': FINGER_JOINT,
        'finger_mimic_joint': FINGER_MIMIC_JOINT, 'finger_bodies': list(FINGER_BODIES),
        'finger_pads': list(pads), 'finger_geoms': [gid(p) for p in pads],
        'flange_body': FLANGE_BODY, 'flange_site': FLANGE_SITE, 'tcp_site': TCP_SITE,
        'tcp_z': float(geo['tcp_z']), 'tcp_offset': [0.0, 0.0, float(geo['tcp_z']), 0.0, 0.0, 0.0],
        'max_width': float(geo['max_width']), 'home_joints': [float(v) for v in home],
        'base_body': BASE_BODY, 'table_z': float(geo['table_z']), 'table_geom': gid(TABLE_GEOM),
        'floor_geom': gid(FLOOR_GEOM), 'floor_z': float(geo['table_z']) - TABLE_HEIGHT,
        'table_extent': [TABLE_X[0], TABLE_X[1], TABLE_Y[0], TABLE_Y[1]],
        'camera': task_cam, 'camera_parent': cams[task_cam]['parent'],
        'cameras': sorted(cams), 'video_cameras': [OVERVIEW_CAM, TOP_CAM],
        'camera_poses': {k: {'parent': v['parent'], 'fovy': v['fovy'],
                             'T_parent_camera': _as_list(v['T_parent_camera'])}
                         for k, v in cams.items()},
        'intrinsics': intrinsics_for_fovy(cams[task_cam]['fovy'], intr.width,
                                          intr.height).to_dict(),
        'T_base_camera': _as_list(T_cam) if mount == 'eye_to_hand' else None,
        'T_flange_camera': _as_list(T_cam) if mount == 'eye_in_hand' else None,
        'zone_bodies': dict(zone_bodies), 'object_bodies': dict(bodies),
        'objects': [{'name': o.name, 'label': o.label, 'body': bodies[o.name]}
                    for o in sim_world.objects],
    }


def _object_records(mj, m, sim_world: SimWorld, bodies: Dict[str, str]) -> List[ObjectRecord]:
    out = []
    for obj in sim_world.objects:
        b = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, bodies[obj.name]))
        j = int(m.body_jntadr[b])
        out.append(ObjectRecord(
            name=obj.name, label=obj.label, shape=obj.shape, body=bodies[obj.name], body_id=b,
            geom_ids=tuple(int(g) for g in range(m.ngeom) if m.geom_bodyid[g] == b),
            qpos_adr=int(m.jnt_qposadr[j]), dof_adr=int(m.jnt_dofadr[j]),
            mass=float(m.body_mass[b]), color_bgr=tuple(obj.color_bgr)))
    return out

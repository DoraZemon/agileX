"""
Tests of the MuJoCo scene with the official competition materials.

Foam 50 mm cubes, the six plastic L2 models as 'compound' objects (shared shape
spec: parts in the object frame, origin at the bottom center, z up), flat 200 mm
card zones and the 0.070 / 0.100 m gripper strokes. Run with the MuJoCo venv:
MUJOCO_GL=egl env -u PYTHONPATH ~/.venvs/piper-mujoco/bin/python -m pytest -q
test/test_mujoco_official.py. No import-time pytest.importorskip / pytest.skip
(see test_mujoco_scene.py).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.transforms import inv_T, make_T, rot_z, top_down_T
from cube_task.sim.mujoco import scene as sc
from cube_task.sim.scenarios import make_scenario
from cube_task.sim.world import SimObject, SimWorld, SimZone

try:
    import mujoco
except ImportError:  # system Python: the tests below are skipped
    mujoco = None

ASSETS = os.path.expanduser(sc.DEFAULT_ASSETS_DIR)
pytestmark = [
    pytest.mark.skipif(mujoco is None, reason='MuJoCo not installed (use ~/.venvs/piper-mujoco)'),
    pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, sc.PIPER_XML)),
                       reason='PiPER MJCF not found in %s' % ASSETS),
]

# ------------------------------------------------------------ official objects
FOAM_CUBE = 0.050
GREEN_STEM = (40, 110, 60)
# Reference shapes from the organizers' facts (m). Lying models: long axis = object x.
OFFICIAL_PARTS = {
    # 260 mm x 40 mm, slightly curved: three capsules bent by 0.08 rad.
    'cucumber': [
        {'shape': 'capsule', 'size': [0.020, 0.0367], 'pos': [0.0, 0.0, 0.020]},
        {'shape': 'capsule', 'size': [0.020, 0.0367], 'pos': [-0.0731, 0.0029, 0.020],
         'yaw': 0.08},
        {'shape': 'capsule', 'size': [0.020, 0.0367], 'pos': [0.0731, 0.0029, 0.020],
         'yaw': -0.08}],
    # 180 mm x 40 mm, curved: three capsules bent by 0.35 rad.
    'banana': [
        {'shape': 'capsule', 'size': [0.020, 0.0233], 'pos': [0.0, 0.0, 0.020]},
        {'shape': 'capsule', 'size': [0.020, 0.0233], 'pos': [-0.0453, 0.0080, 0.020],
         'yaw': 0.35},
        {'shape': 'capsule', 'size': [0.020, 0.0233], 'pos': [0.0453, 0.0080, 0.020],
         'yaw': -0.35}],
    # Upright, 125 mm with the stem, 85 mm bulb, a neck tapering ~15 deg from 52 mm
    # to the top (45 mm at the default grasp height), small flat base.
    'pear': [
        {'shape': 'cylinder', 'size': [0.020, 0.003], 'pos': [0.0, 0.0, 0.0015]},
        {'shape': 'ellipsoid', 'size': [0.0425, 0.0425, 0.042], 'pos': [0.0, 0.0, 0.042]},
        {'shape': 'ellipsoid', 'size': [0.026, 0.026, 0.050], 'pos': [0.0, 0.0, 0.070]},
        {'shape': 'cylinder', 'size': [0.003, 0.012], 'pos': [0.0, 0.0, 0.119],
         'color_bgr': GREEN_STEM}],
    # Oval 100 x 70 mm footprint, 55 mm thick (thickness assumed).
    'potato': [
        {'shape': 'ellipsoid', 'size': [0.050, 0.035, 0.0275], 'pos': [0.0, 0.0, 0.0275]}],
    # Oblate 70 mm x 50 mm with a flat green calyx on top.
    'persimmon': [
        {'shape': 'ellipsoid', 'size': [0.035, 0.035, 0.025], 'pos': [0.0, 0.0, 0.025]},
        {'shape': 'cylinder', 'size': [0.018, 0.003], 'pos': [0.0, 0.0, 0.0495],
         'color_bgr': GREEN_STEM}],
    # 90 mm with the green stem, 70 mm wide, blocky body on rounded lobes.
    'bell_pepper': [
        {'shape': 'ellipsoid', 'size': [0.035, 0.035, 0.008], 'pos': [0.0, 0.0, 0.008]},
        {'shape': 'cylinder', 'size': [0.035, 0.060], 'pos': [0.0, 0.0, 0.035]},
        {'shape': 'ellipsoid', 'size': [0.035, 0.035, 0.015], 'pos': [0.0, 0.0, 0.065]},
        {'shape': 'cylinder', 'size': [0.006, 0.012], 'pos': [0.0, 0.0, 0.084],
         'color_bgr': GREEN_STEM}],
}
OFFICIAL_COLORS = {'cucumber': (40, 100, 30), 'banana': (40, 210, 235),
                   'pear': (60, 200, 190), 'potato': (90, 190, 220),
                   'persimmon': (30, 120, 235), 'bell_pepper': (40, 40, 200)}
# Height (m) and the narrowest top-down closing width (m) at the default grasp depth.
OFFICIAL_HEIGHT = {'cucumber': 0.040, 'banana': 0.040, 'pear': 0.125, 'potato': 0.055,
                   'persimmon': 0.051, 'bell_pepper': 0.090, 'cube': FOAM_CUBE}
CLOSING_WIDTH = {'cucumber': 0.040, 'banana': 0.040, 'pear': 0.045, 'potato': 0.070,
                 'persimmon': 0.070, 'bell_pepper': 0.070, 'cube': FOAM_CUBE}
CUBE_COLORS = {'red': (40, 40, 200), 'green': (60, 170, 60), 'yellow': (40, 200, 220)}
CARDS = (('zone_green', (0.20, 0.30), (70, 170, 70), 'green'),
         ('zone_yellow', (0.20, -0.30), (50, 205, 225), 'yellow'),
         ('zone_red', (0.42, 0.26), (60, 60, 200), 'red'))
CARD_SIZE = 0.200
OPEN_MARGIN = 0.020            # robot.gripper.open_margin
WIDTH_MARGIN = 0.005           # grasp.max_object_width = max_width - this


class CompoundStandIn:
    """A 'compound' SimObject stand-in while sim/world.py does not accept the shape."""

    shape = 'compound'

    def __init__(self, name, label, parts, color_bgr, T, mass=None, material=None):
        self.name, self.label = str(name), str(label)
        self.size = tuple(dict(p) for p in parts)
        self.color_bgr = tuple(int(c) for c in color_bgr)
        self._T = np.array(T, dtype=float)
        self.mass, self.material = mass, material

    @property
    def T_base_obj(self):
        return self._T.copy()

    def center_xy(self):
        return self._T[:2, 3].copy()


def make_object(name, label, x=0.30, y=0.0, yaw=0.0, z=0.0, mass=None, material=None):
    """Return an official object resting at (x, y, z) with yaw; label 'cube_<color>' = foam."""
    T = make_T(rot_z(yaw), (x, y, z))
    if label.startswith('cube_'):
        color = label[5:]
        obj = SimObject(name, color, 'box', (FOAM_CUBE,) * 3, CUBE_COLORS[color], T)
    else:
        try:
            obj = SimObject(name, label, 'compound', OFFICIAL_PARTS[label],
                            OFFICIAL_COLORS[label], T)
        except ValueError:
            obj = CompoundStandIn(name, label, OFFICIAL_PARTS[label], OFFICIAL_COLORS[label],
                                  T, mass, material)
    for key, v in (('mass', mass), ('material', material)):
        if v is not None and getattr(obj, key, None) != v:
            setattr(obj, key, v)
    return obj


def official_scene(objects, mount='eye_to_hand', cfg=None, clock=None):
    """Build a scene with the official cards and the given objects (materials 'official')."""
    _w, info = make_scenario('L1', 1, mount)
    zones = [SimZone(n, (x, y, 0.0), (CARD_SIZE, CARD_SIZE), 0.0, c, 0.0, label)
             for n, (x, y), c, label in CARDS]
    world = SimWorld(table_z=0.0, objects=list(objects), zones=zones, seed=1)
    info = dict(info, materials='official', zones={z.name: z.to_zone() for z in zones},
                objects=[{'name': o.name, 'label': o.label} for o in objects],
                class_to_zone={})
    return sc.build_scene(info, cfg=cfg, world=world, clock=clock)


def arm_cfg(mw, force=1.5, speed=50):
    """Return the default config with the scene overlay, grip force (N) and speed."""
    base = load_config(use_overlays=False)
    cfg = Config(deep_merge(base.data, mw.config_overlay()), base.root, list(base.sources))
    cfg.set('robot.gripper.force', float(force))
    cfg.set('robot.speed_percent', int(speed))
    return cfg


def _rel(mw, name):
    """Return the object position in the TCP frame (m)."""
    T = inv_T(mw.sim.site_pose(sc.TCP_SITE)) @ mw.object_pose(name)
    return T[:3, 3].copy()


def grasp_trial(label, max_width, force, lift=0.15, carry_s=3.0, xy=(0.30, 0.0), speed=50,
                depth=None):
    """
    Grasp one object top-down like GraspPlanner, lift it, carry it, report the outcome.

    Returns {'result': 'held' | 'slipped' | 'dropped' | 'not_graspable' | 'error',
    'slip_mm' (object motion in the TCP frame from the close to the end),
    'width_mm' (measured closed width), 'detail'}. Slipped = moved > 10 mm in the
    fingers but still held; dropped = not held at the end. depth: TCP below the
    object top (m); None = the GraspPlanner default (half the height, 8..30 mm).
    """
    from cube_task.sim.mujoco.arm import MujocoArm
    width = CLOSING_WIDTH[label]
    if width > max_width - WIDTH_MARGIN:
        return {'result': 'not_graspable', 'slip_mm': None, 'width_mm': None,
                'detail': 'too_wide: %.0f mm > %.0f mm' % (width * 1e3,
                                                           (max_width - WIDTH_MARGIN) * 1e3)}
    obj = make_object('o1', 'cube_red' if label == 'cube' else label, xy[0], xy[1])
    clock = SimClock()
    mw = official_scene([obj], cfg={'sim': {'mujoco': {'gripper_max_width': max_width}}},
                        clock=clock)
    arm = MujocoArm(arm_cfg(mw, force, speed), clock, mw.sim)
    h = OFFICIAL_HEIGHT[label]
    zg = mw.table_z + h - (min(max(0.5 * h, 0.008), 0.030) if depth is None else depth)
    angle = math.pi / 2
    Tg = top_down_T(xy[0], xy[1], zg, angle)
    pre = arm.ik.solve_with_tilt(top_down_T(xy[0], xy[1], zg + 0.06, angle))
    lift_t = arm.ik.solve_with_tilt(top_down_T(xy[0], xy[1], zg + lift, angle))
    carry_t = arm.ik.solve_with_tilt(top_down_T(xy[0] - 0.05, xy[1] + 0.15, zg + lift, angle))
    if pre is None or lift_t is None or carry_t is None:
        return {'result': 'error', 'slip_mm': None, 'width_mm': None, 'detail': 'no IK'}
    arm.set_joints(pre[1])
    arm.connect()
    arm.enable()
    mw.sim.settle(0.3)
    steps = [('open', lambda: arm.gripper(min(max_width, width + OPEN_MARGIN))),
             ('descend', lambda: arm.move_tcp(Tg, linear=True)),
             ('close', lambda: arm.gripper(0.0, force=force))]
    for what, fn in steps:
        r = fn()
        if not r.ok:
            return {'result': 'error', 'slip_mm': None, 'width_mm': None,
                    'detail': '%s: %s %s' % (what, r.error, r.detail)}
    closed = float(arm.gripper_width().value)
    p0 = _rel(mw, obj.name)
    t0 = clock.now()
    arm.move_tcp(lift_t[0], linear=True)
    arm.move_tcp(carry_t[0])
    arm.move_tcp(lift_t[0])
    if clock.now() - t0 < carry_s:
        mw.sim.settle(carry_s - (clock.now() - t0))
    held = obj.name in mw.held_objects()
    com_z = float(mw.object_com(obj.name)[2])
    slip = float(np.linalg.norm(_rel(mw, obj.name) - p0))
    if not held or com_z < mw.table_z + 0.05:
        result = 'dropped'
    else:
        result = 'slipped' if slip > 0.010 else 'held'
    return {'result': result, 'slip_mm': round(slip * 1e3, 1), 'width_mm': round(closed * 1e3, 1),
            'detail': 'carried %.1f s' % (clock.now() - t0)}


def drop_trial(label, height, offset=0.0, push=0.0, tilt_deg=0.0, settle_s=8.0):
    """
    Release an object at rest `height` above the red card and report where it ends.

    offset: start distance (m) of the object center from the card center toward
    the card's +x edge; push: initial speed (m/s) toward that edge; tilt_deg:
    initial tilt about the object y axis. Returns {'on_card', 'moved_mm', 'tipped'}.
    """
    obj = make_object('o1', 'cube_red' if label == 'cube' else label)
    mw = official_scene([obj], clock=SimClock())
    zone = mw.zones['zone_red']
    c = np.array([zone.center[0] + offset, zone.center[1], zone.center[2] + height])
    ct, st = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    R = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    mw.set_object_pose(obj.name, make_T(R, c))
    rec = mw.object(obj.name)
    mw.data.qvel[rec.dof_adr] = push
    mw.sim.settle(settle_s)
    T = mw.object_pose(obj.name)
    st_ = mw.object_status()[obj.name]
    return {'on_card': st_['zone'] == 'zone_red',
            'moved_mm': round(float(np.linalg.norm(T[:2, 3] - c[:2])) * 1e3, 1),
            'tipped': bool(T[2, 2] < math.cos(math.radians(30.0)))}


# ------------------------------------------------------------------- tests
def _l2_row():
    labels = sorted(OFFICIAL_PARTS)
    xs = {'cucumber': (0.30, -0.16), 'banana': (0.30, -0.07), 'pear': (0.20, 0.05),
          'potato': (0.30, 0.08), 'persimmon': (0.42, -0.05), 'bell_pepper': (0.42, 0.07)}
    objs = [make_object('m_' + k, k, *xs[k]) for k in labels]
    objs.append(make_object('c_red', 'cube_red', 0.17, -0.105))
    return objs


def test_compound_bodies_compile_rest_and_carry_materials():
    objs = _l2_row()
    mw = official_scene(objs, clock=SimClock())
    m = mw.model
    assert mw.materials == 'official' and mw.meta['flat_zones'] == 'card'
    start = {o.name: mw.object_com(o.name) for o in mw.objects}
    mw.sim.settle(1.5)
    lin, ang = mw.max_object_speed()
    assert lin < 0.003 and ang < 0.1, (lin, ang)
    for rec in mw.objects:
        moved = np.linalg.norm(mw.object_com(rec.name) - start[rec.name])
        assert moved < 0.003, (rec.name, moved)
        assert mw.object_status()[rec.name]['status'] == 'table'
        if rec.label in sc.OFFICIAL_MASS:
            assert rec.mass == pytest.approx(sc.OFFICIAL_MASS[rec.label], abs=1e-9)
            assert len(rec.geom_ids) == len(OFFICIAL_PARTS[rec.label])
            mat = sc.PLASTIC_MODEL
        else:
            assert rec.mass == pytest.approx(0.010, abs=1e-9)        # 50 mm foam cube
            mat = sc.FOAM
            g = rec.geom_ids[0]
            assert np.allclose(m.geom_solref[g], sc.FOAM['solref'])
        for g in rec.geom_ids:
            assert np.allclose(m.geom_friction[g], mat['friction'])
            assert m.geom_priority[g] == 1 and m.geom_condim[g] == mat['condim']
        top = max(float(mw.data.geom_xpos[g][2]) for g in rec.geom_ids)
        assert top < mw.table_z + OFFICIAL_HEIGHT.get(rec.label, FOAM_CUBE)
    # Per-part colors: the pear stem and the persimmon calyx are green.
    for name, k in (('m_pear', 3), ('m_persimmon', 1)):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'obj_%s_g%d' % (name, k))
        assert np.allclose(m.mat_rgba[m.geom_matid[g]][:3] * 255.0, GREEN_STEM[::-1], atol=0.6)
    # Hollow plastic shells: more inertia than a solid body of the same mass.
    b = mw.object('m_persimmon').body_id
    solid = 0.2 * 0.100 * (0.035 ** 2 + 0.025 ** 2)
    assert max(m.body_inertia[b]) > 1.3 * 0.4 * 0.100 * 0.035 ** 2 > solid
    for name, z in mw.zones.items():
        body = mw.meta['zone_bodies'][name]
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, body + '_card')
        assert m.geom_contype[g] != 0 and m.geom_size[g][2] == pytest.approx(0.00015)
        assert z.center[2] == pytest.approx(mw.table_z + sc.CARD_THICKNESS)
        assert z.wall_height == 0.0


def test_objects_slide_off_a_card_and_count_on_it():
    potato = make_object('p1', 'potato')
    mw = official_scene([potato], clock=SimClock())
    card = mw.zones['zone_green']
    start = (card.center[0], card.center[1], card.center[2] + 0.002)
    mw.set_object_pose('p1', make_T(np.eye(3), start))
    mw.sim.settle(0.8)
    st = mw.object_status()['p1']
    assert st['zone'] == 'zone_green'
    assert mw.object_pose('p1')[2, 3] == pytest.approx(card.center[2], abs=0.0008)
    rec = mw.object('p1')
    mw.data.qvel[rec.dof_adr] = 1.6                  # slides across the 0.3 mm card edge
    mw.sim.settle(7.0)                               # the oval rocks for ~6.5 s
    com = mw.object_com('p1')
    assert com[0] > card.center[0] + CARD_SIZE / 2 + 0.035
    st = mw.object_status()['p1']
    assert st['status'] == 'table' and st['at_rest']
    assert com[2] == pytest.approx(mw.table_z + 0.0275, abs=0.002)


@pytest.mark.parametrize('width', [0.07, 0.10])
def test_gripper_width_variant_opens_fully(width):
    from cube_task.sim.mujoco.arm import MujocoArm
    clock = SimClock()
    mw = official_scene([make_object('c1', 'cube_red', 0.42, -0.15)],
                        cfg={'sim': {'mujoco': {'gripper_max_width': width}}}, clock=clock)
    m = mw.model
    j7 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sc.FINGER_JOINT)
    j8 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sc.FINGER_MIMIC_JOINT)
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, sc.GRIPPER_ACTUATOR)
    assert np.allclose(m.jnt_range[j7], [0.0, width / 2])
    assert np.allclose(m.jnt_range[j8], [-width / 2, 0.0])
    assert np.allclose(m.actuator_ctrlrange[act], [0.0, width / 2])
    assert mw.max_width == pytest.approx(width)
    over = mw.config_overlay()
    assert over['robot']['gripper']['max_width'] == pytest.approx(width)
    assert over['sim']['mujoco']['gripper_max_width'] == pytest.approx(width)
    cfg = arm_cfg(mw, force=2.0)
    cfg.set('robot.gripper.max_width', 0.10)
    arm = MujocoArm(cfg, clock, mw.sim)
    arm.set_joints(arm.ik.solve(top_down_T(0.30, 0.0, 0.10, math.pi / 2))[0])
    arm.connect()
    assert arm.enable()
    r = arm.gripper(0.10)
    assert r.ok and r.width == pytest.approx(width, abs=0.001), r
    assert arm.gripper(0.0).width == pytest.approx(0.0, abs=0.001)
    assert arm.command_log[-1]['force'] == pytest.approx(2.0)
    f = float(m.actuator_forcerange[act][1])
    assert f == pytest.approx(2.0 * 2.0)            # robot.gripper.force x grip_force_scale


def test_bad_official_inputs():
    with pytest.raises(ValueError):
        official_scene([make_object('c1', 'cube_red')],
                       cfg={'sim': {'mujoco': {'gripper_max_width': 0.2}}})
    bad = CompoundStandIn('b', 'pear', [{'shape': 'cone', 'size': [0.01]}], (1, 2, 3),
                          np.eye(4))
    with pytest.raises(ValueError):
        official_scene([bad])
    bad = CompoundStandIn('b', 'pear', [{'shape': 'box', 'size': [0.01, 0.02]}], (1, 2, 3),
                          np.eye(4))
    with pytest.raises(ValueError):
        official_scene([bad])
    heavy = make_object('c1', 'cube_red', mass=0.5, material='plastic')
    assert sc.object_mass(heavy, 'official') == pytest.approx(0.5)
    assert sc.object_material_name(heavy, 'official') == 'plastic'
    with pytest.raises(ValueError):
        sc.object_material_name(make_object('c2', 'cube_red', material='steel'), 'official')
    with pytest.raises(ValueError):
        sc.scene_materials({'materials': 'shiny'})
    w, info = make_scenario('L1', 1)
    assert sc.scene_materials(info, w.objects) == 'legacy'
    assert sc.scene_materials(None, [make_object('p', 'pear')]) == 'official'
    visual = official_scene([make_object('c1', 'cube_red')],
                            cfg={'sim': {'mujoco': {'card_collision': False}}})
    g = mujoco.mj_name2id(visual.model, mujoco.mjtObj.mjOBJ_GEOM, 'zn_zone_red_card')
    assert visual.model.geom_contype[g] == 0
    assert visual.zones['zone_red'].center[2] == pytest.approx(visual.table_z)


# Graspability: only the clear cases of the measured table (see grasp_trial).
@pytest.mark.parametrize('label,width,force,depth,expected', [
    ('cube', 0.07, 1.5, None, 'held'),
    ('cucumber', 0.07, 1.5, None, 'held'),
    ('persimmon', 0.07, 3.0, None, 'not_graspable'),
    ('bell_pepper', 0.10, 3.0, None, 'held'),
    ('pear', 0.07, 3.0, None, 'dropped'),        # 30 mm below the stem: above the neck
    ('pear', 0.07, 3.0, 0.050, 'held'),          # at the neck
])
def test_graspability_clear_cases(label, width, force, depth, expected):
    res = grasp_trial(label, width, force, depth=depth)
    assert res['result'] == expected, res

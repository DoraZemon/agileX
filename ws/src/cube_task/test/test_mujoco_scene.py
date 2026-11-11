"""
Tests for sim/mujoco/scene.py and sim/mujoco/world.py (MuJoCo scene of the scenarios).

Run with the MuJoCo venv: MUJOCO_GL=egl env -u PYTHONPATH
~/.venvs/piper-mujoco/bin/python -m pytest -q test/test_mujoco_scene.py.
No import-time pytest.importorskip / pytest.skip here: the ROS launch_testing
pytest plugin imports every test module inside ``except Exception``, and the
Skipped exception (a BaseException) then aborts the whole session's collection
(``collected 0 items / 1 skipped`` for all ~980 tests under /usr/bin/python3).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.extrinsics import extrinsics_from_cfg
from cube_task.core.transforms import inv_T, make_T, pose_error
from cube_task.core.zones import parse_zones
from cube_task.sim.mujoco import scene as sc
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from cube_task.sim.world import SimObject

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

W, H = 640, 480
_CACHE = {}


def _build(task, mount='eye_to_hand', difficulty='normal', seed=1, cached=False):
    """Build a scene (fresh unless cached=True; cached worlds must not be changed)."""
    key = (task, mount, difficulty, seed)
    if cached and key in _CACHE:
        return _CACHE[key]
    world, info = make_scenario(task, seed, mount, difficulty)
    mw = sc.build_scene(info, world=world)
    if cached:
        _CACHE[key] = mw
    return mw


def _set_arm(mw, q):
    m, d = mw.model, mw.data
    for name, v in zip(mw.meta['arm_joints'], q):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[j]] = v
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = v
    mw.forward()


def _render(mw, camera, width=W, height=H):
    """Return (rgb, depth m, geom id per pixel or -1) of a camera (mujoco.Renderer)."""
    try:
        r = mujoco.Renderer(mw.model, height, width)
    except Exception as e:  # noqa: B902 - no GL context on this machine
        pytest.skip('MuJoCo offscreen rendering unavailable: %s' % e)
    try:
        r.update_scene(mw.data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(mw.data, camera=camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(mw.data, camera=camera)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
    finally:
        r.close()
    geom = np.where(seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[:, :, 0], -1)
    return rgb, depth, geom


def _camera_K(mw, camera, width=W, height=H):
    cid = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return sc.intrinsics_for_fovy(float(mw.model.cam_fovy[cid]), width, height)


def _project(intr, T_base_camera, p):
    q = inv_T(T_base_camera) @ np.append(np.asarray(p, dtype=float), 1.0)
    assert q[2] > 0.05
    return (int(round(intr.fx * q[0] / q[2] + intr.cx)),
            int(round(intr.fy * q[1] / q[2] + intr.cy)))


def _table_depth(intr, T_base_camera, table_z, width=W, height=H):
    """Return the z-depth (m) of the table plane z = table_z for every pixel center."""
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], -1)
    R, c = T_base_camera[:3, :3], T_base_camera[:3, 3]
    dz = rays @ R[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (table_z - c[2]) / dz


def _geom_id(mw, name):
    return mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_GEOM, name)


# --------------------------------------------------------------- geometry
def test_piper_geometry_and_meta():
    geo = sc.piper_geometry(os.path.realpath(ASSETS))
    assert abs(geo['table_z']) < 0.001                   # base mesh bottom at base_link z = 0
    assert geo['tcp_z'] == pytest.approx(0.135, abs=0.0005)
    assert geo['max_width'] == pytest.approx(0.07)
    mw = _build('L1', cached=True)
    m, meta = mw.model, mw.meta
    assert mw.table_z == geo['table_z'] and mw.tcp_z == geo['tcp_z']
    for kind, names in ((mujoco.mjtObj.mjOBJ_JOINT, meta['arm_joints'] + [meta['finger_joint']]),
                        (mujoco.mjtObj.mjOBJ_ACTUATOR,
                         meta['arm_actuators'] + [meta['gripper_actuator']]),
                        (mujoco.mjtObj.mjOBJ_BODY, [meta['flange_body'], meta['base_body']]
                         + meta['finger_bodies']),
                        (mujoco.mjtObj.mjOBJ_SITE, [meta['tcp_site'], meta['flange_site']])):
        for n in names:
            assert mujoco.mj_name2id(m, kind, n) >= 0, n
    b6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'link6')
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    assert m.site_bodyid[tcp] == b6
    assert np.allclose(m.site_pos[tcp], [0.0, 0.0, mw.tcp_z])
    assert len(meta['finger_geoms']) == 4
    for g in meta['finger_geoms']:
        assert m.geom_bodyid[g] in [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                    for b in meta['finger_bodies']]
        assert np.allclose(m.geom_friction[g], sc.PAD_FRICTION)
        assert m.geom_condim[g] == sc.PAD_CONDIM
    assert os.path.dirname(mw.xml_path) == os.path.realpath(ASSETS)


COMBOS = ([(t, mo, d, 1) for t in ('L1', 'L2') for mo in ('eye_to_hand', 'eye_in_hand')
           for d in ('normal', 'hard')]
          + [('L1', 'eye_to_hand', 'normal', 2), ('L2', 'eye_to_hand', 'normal', 2),
             ('L2', 'eye_to_hand', 'hard', 3)])


@pytest.mark.parametrize('task,mount,difficulty,seed', COMBOS)
def test_scene_compiles_and_objects_settle(task, mount, difficulty, seed):
    world, info = make_scenario(task, seed, mount, difficulty)
    mw = sc.build_scene(info, world=world, clock=SimClock())
    m, d = mw.model, mw.data
    assert [o.name for o in mw.objects] == [o['name'] for o in info['objects']]
    assert set(mw.zones) == set(info['zones'])
    cams = mw.meta['cameras']
    assert sc.OVERVIEW_CAM in cams and sc.TOP_CAM in cams
    assert mw.task_camera == (sc.WRIST_CAM if mount == 'eye_in_hand' else sc.TASK_CAM)
    assert (sc.TASK_CAM in cams) == (mount == 'eye_to_hand')
    start = {o.name: mw.object_com(o.name) for o in mw.objects}
    for o in world.objects:                              # resting on the table at start
        lo = mw.object_pose(o.name)[2, 3]
        assert lo == pytest.approx(mw.table_z, abs=1e-9)
        assert mw.object_pose(o.name)[:2, 3] == pytest.approx(o.T_base_obj[:2, 3], abs=1e-9)
    mw.sim.settle(1.0)
    assert mw.sim.n_steps == 500
    lin, ang = mw.max_object_speed()
    assert lin < 0.003 and ang < 0.1, (lin, ang)
    table = _geom_id(mw, sc.TABLE_GEOM)
    touching = set()
    for i in range(d.ncon):
        g1, g2 = (int(g) for g in d.contact.geom[i])
        assert d.contact.dist[i] > -0.0015               # no deep penetration
        for a, b in ((g1, g2), (g2, g1)):
            if b == table and mw.geom_object(a) is not None:
                touching.add(mw.geom_object(a))
    for o in mw.objects:
        moved = mw.object_com(o.name) - start[o.name]
        assert np.linalg.norm(moved) < 0.002, (o.name, moved)
        assert o.name in touching                        # nothing fell through or flew off
    status = mw.object_status()
    assert all(s['status'] == 'table' and s['at_rest'] for s in status.values())
    n_targets = sum(1 for o in mw.objects if o.label in info['class_to_zone'])
    assert mw.score(info['class_to_zone']) == {'correct': 0, 'wrong': 0,
                                               'remaining': n_targets, 'dropped': 0}
    assert mw.held_objects() == []
    assert np.isfinite(m.opt.timestep) and m.opt.timestep == pytest.approx(sc.TIMESTEP)
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == pytest.approx(10)


def test_object_masses_and_materials():
    mw = _build('L1', difficulty='hard', cached=True)
    world, _info = make_scenario('L1', 1, 'eye_to_hand', 'hard')
    for o, rec in zip(world.objects, mw.objects):
        assert rec.mass == pytest.approx(700.0 * o.size[0] * o.size[1] * o.size[2], rel=1e-6)
        assert 0.015 < rec.mass < 0.05
        g = rec.geom_ids[0]
        assert mw.model.geom_condim[g] == sc.WOOD['condim']
    mw2 = _build('L2', difficulty='hard', cached=True)
    for rec in mw2.objects:
        assert 0.015 - 1e-9 <= rec.mass <= 0.06 + 1e-9, (rec.name, rec.mass)
        for g in rec.geom_ids:
            assert np.allclose(mw2.model.geom_friction[g], sc.PLASTIC['friction'])
            assert mw2.model.geom_condim[g] == 6
        rgb = mw2.model.mat_rgba[mw2.model.geom_matid[rec.geom_ids[0]]][:3] * 255.0
        assert np.allclose(rgb, rec.color_bgr[::-1], atol=0.6)     # BGR -> RGB
    cube = SimObject('c', 'red', 'box', (0.035,) * 3, (40, 40, 200), np.eye(4))
    assert sc.object_mass(cube) == pytest.approx(0.030, abs=0.001)     # 35 mm wooden cube
    apple = SimObject('a', 'apple', 'sphere', (0.031,), (45, 45, 185), np.eye(4))
    assert 0.015 < sc.object_mass(apple) < 0.06


def test_build_is_deterministic():
    a, b = _build('L2', seed=4), _build('L2', seed=4)
    assert a.xml_path == b.xml_path
    a.settle(0.3)
    b.settle(0.3)
    assert np.array_equal(a.data.qpos, b.data.qpos)


# ---------------------------------------------------------------- cameras
def test_camera_frame_conversion_roundtrip():
    R = make_scenario('L1', 1)[1]['T_base_camera'][:3, :3]
    Rm = sc.optical_to_mujoco(R)
    assert np.allclose(Rm[:, 2], -R[:, 2]) and np.allclose(Rm[:, 1], -R[:, 1])
    assert np.allclose(Rm[:, 0], R[:, 0])
    assert np.allclose(sc.mujoco_to_optical(Rm), R)
    intr = make_scenario('L1', 1)[1]['intrinsics']
    k = sc.intrinsics_for_fovy(sc.fovy_deg(intr), intr.width, intr.height)
    assert (k.fx, k.fy, k.cx, k.cy) == pytest.approx((intr.fy, intr.fy, 319.5, 239.5))


def _check_table_depth(mw, camera, T_base_camera):
    intr = _camera_K(mw, camera)
    _rgb, depth, geom = _render(mw, camera)
    table = geom == _geom_id(mw, sc.TABLE_GEOM)
    assert table.sum() > 0.25 * W * H
    expect = _table_depth(intr, T_base_camera, mw.table_z)
    err = np.abs(depth - expect)[table]
    # Pixels next to an object / robot edge may mix depths; the table interior must match.
    assert np.percentile(err, 99) < 0.001, np.percentile(err, 99)
    v, u = H * 3 // 4, W // 2                            # one known pixel, in the open
    if table[v, u]:
        assert abs(depth[v, u] - expect[v, u]) < 0.001


def test_task_cam_depth_matches_table_geometry():
    mw = _build('L1', cached=True)
    T = mw.camera_pose(sc.TASK_CAM)
    assert pose_error(T, mw.T_base_camera)[0] < 1e-9
    assert pose_error(T, np.asarray(mw.info['T_base_camera']))[0] < 1e-9
    assert np.allclose(T, mw.info['T_base_camera'], atol=1e-9)
    assert mw.intrinsics.fx == pytest.approx(mw.info['intrinsics'].fy)
    _check_table_depth(mw, sc.TASK_CAM, T)


def test_wrist_cam_follows_link6_and_depth_matches():
    mw = _build('L2', mount='eye_in_hand')
    _set_arm(mw, mw.info['observe_joints'])
    b6 = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
    T6 = make_T(mw.data.xmat[b6].reshape(3, 3), mw.data.xpos[b6])
    T = mw.camera_pose(sc.WRIST_CAM)
    assert np.allclose(T, T6 @ np.asarray(mw.info['T_flange_camera']), atol=1e-9)
    assert np.allclose(mw.T_flange_camera, mw.info['T_flange_camera'], atol=1e-9)
    # MuJoCo and SDK kinematics differ by a few mm: the camera is near the scenario view.
    assert np.linalg.norm(T[:3, 3] - mw.info['T_base_camera_observe'][:3, 3]) < 0.02
    _check_table_depth(mw, sc.WRIST_CAM, T)


def _chroma(rgb):
    c = np.asarray(rgb, dtype=float)
    return c / max(float(np.linalg.norm(c)), 1e-9)


@pytest.mark.parametrize('task,mount', [('L1', 'eye_to_hand'), ('L2', 'eye_in_hand')])
def test_projection_with_optical_intrinsics_hits_object(task, mount):
    """Project object tops with the overlay's optical-frame camera model; check the pixel."""
    mw = _build(task, mount=mount, cached=(mount == 'eye_to_hand'))
    world, info = make_scenario(task, 1, mount)
    if mount == 'eye_in_hand':
        _set_arm(mw, info['observe_joints'])
        b6 = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        T = make_T(mw.data.xmat[b6].reshape(3, 3), mw.data.xpos[b6]) @ mw.T_flange_camera
    else:
        T = mw.T_base_camera
    intr = mw.intrinsics
    assert (intr.width, intr.height) == (W, H)
    rgb, _depth, geom = _render(mw, mw.task_camera)
    colors = {o.name: _chroma(o.color_bgr[::-1]) for o in world.objects}
    for obj in world.objects:
        c = obj.center_xy()
        u, v = _project(intr, T, (c[0], c[1], obj.top_z() - 0.002))
        assert 0 <= u < W and 0 <= v < H
        rec = mw.object(obj.name)
        assert geom[v, u] in rec.geom_ids, (obj.name, u, v, geom[v, u])
        seen = _chroma(rgb[v, u])
        best = max(colors, key=lambda n: float(seen @ colors[n]))
        if task == 'L1':                                 # distinct block colors
            assert best == obj.name, (obj.name, rgb[v, u])
        assert float(seen @ colors[obj.name]) > 0.97, (obj.name, rgb[v, u])
        # A mirrored camera model would look elsewhere.
        um = W - 1 - u
        if abs(um - u) > 40:
            assert geom[v, um] not in rec.geom_ids


@pytest.mark.parametrize('task', ['L1', 'L2'])
def test_zones_render_and_collide_as_built(task):
    mw = _build(task, cached=True)
    world, _info = make_scenario(task, 1)
    T = mw.camera_pose(sc.TOP_CAM)
    intr = _camera_K(mw, sc.TOP_CAM)
    rgb, _depth, geom = _render(mw, sc.TOP_CAM)
    table_c = _chroma(np.array(world.table_color_bgr[::-1], dtype=float))
    for z in world.zones:
        zone = mw.zones[z.name]
        body = mw.meta['zone_bodies'][z.name]
        walled = z.wall_height > 0
        floor_geom = _geom_id(mw, body + ('_floor' if walled else '_mat'))
        assert floor_geom >= 0
        assert zone.center[2] == pytest.approx(mw.table_z + (sc.BIN_FLOOR if walled else 0.0))
        assert (mw.model.geom_contype[floor_geom] != 0) == walled
        u, v = _project(intr, T, (zone.center[0], zone.center[1], zone.center[2]))
        assert geom[v, u] == floor_geom, (z.name, geom[v, u])
        seen = _chroma(rgb[v, u])
        assert float(seen @ _chroma(z.color_bgr[::-1])) > float(seen @ table_c) - 1e-3
        assert float(seen @ _chroma(z.color_bgr[::-1])) > 0.98
        if walled:
            wall = _geom_id(mw, body + '_wall_px')
            assert mw.model.geom_contype[wall] != 0
            top = (zone.center[0] + z.size[0] / 2.0 + 0.002, zone.center[1],
                   zone.center[2] + z.wall_height)
            u, v = _project(intr, T, top)
            assert geom[v, u] == wall, (z.name, geom[v, u])


# ---------------------------------------------------------------- scoring
def _upright(x, y, z, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return make_T(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), (x, y, z))


def _small(world, category, cats):
    objs = [o for o in world.objects if cats.get(o.label) == category
            and float(np.max(o.local_aabb()[1][:2] - o.local_aabb()[0][:2])) < 0.08]
    return objs[0] if objs else None


def _seed_with_small_objects():
    for seed in range(1, 40):
        world, info = make_scenario('L2', seed)
        if (_small(world, 'fruit', info['categories']) is not None
                and _small(world, 'vegetable', info['categories']) is not None):
            return seed
    raise AssertionError('no L2 seed with small objects')


def test_placements_and_score_after_moving_objects():
    seed = _seed_with_small_objects()
    world, info = make_scenario('L2', seed)
    mw = sc.build_scene(info, world=world)
    cats = info['categories']
    fruit_bin = mw.zones['fruit_bin']
    fruit = _small(world, 'fruit', cats)
    veg = _small(world, 'vegetable', cats)
    other = [o for o in world.objects if o is not fruit and o is not veg]
    flyer = [o for o in other if cats[o.label] == 'vegetable'][0]      # goes to the empty bin
    dropped = [o for o in other if cats[o.label] == 'fruit'][0]
    rest = [o for o in other if o is not flyer and o is not dropped]
    # Bottom-center frames: rest the fruit on the fruit bin floor, put the vegetable
    # in the wrong bin, hold one object in the air over its bin, drop one on the floor.
    mw.set_object_pose(fruit.name, _upright(fruit_bin.center[0] - 0.03, fruit_bin.center[1],
                                            fruit_bin.center[2] + 1e-4))
    mw.set_object_pose(veg.name, _upright(fruit_bin.center[0] + 0.035, fruit_bin.center[1],
                                          fruit_bin.center[2] + 1e-4, math.pi / 2))
    target_zone = info['class_to_zone'][flyer.label]
    assert target_zone == 'veg_bin'
    tz = mw.zones[target_zone]
    mw.set_object_pose(flyer.name, _upright(tz.center[0], tz.center[1], tz.center[2] + 0.12,
                                            math.pi / 4))
    mw.set_object_pose(dropped.name, _upright(0.2, -0.9, mw.meta['floor_z'] + 1e-4))
    st = mw.object_status()
    assert st[flyer.name]['status'] == 'table'           # above the zone, not in it
    assert st[dropped.name]['status'] == 'dropped'
    assert st[fruit.name] == dict(st[fruit.name], status='zone', zone='fruit_bin')
    mw.settle(1.5)
    st = mw.object_status()
    assert st[fruit.name]['zone'] == 'fruit_bin' and st[veg.name]['zone'] == 'fruit_bin'
    assert st[flyer.name]['zone'] == target_zone, st[flyer.name]
    assert st[dropped.name]['status'] == 'dropped'
    assert st[fruit.name]['com'][2] > fruit_bin.center[2]   # rests on the bin floor
    place = mw.placements()
    assert sorted(place['fruit_bin']) == sorted([fruit.label, veg.label])
    assert place['veg_bin'] == [flyer.label]
    left = sum(1 for o in rest if o.label in info['class_to_zone'])
    assert mw.score(info['class_to_zone']) == {'correct': 2, 'wrong': 1, 'remaining': left,
                                               'dropped': 1}


def test_l1_mat_placement_counts_without_walls():
    mw = _build('L1')
    world, info = make_scenario('L1', 1)
    red = mw.zones[info['class_to_zone']['red']]
    blue = mw.zones[info['class_to_zone']['blue']]
    by_label = {o.label: o for o in world.objects}
    mw.set_object_pose(by_label['red'].name, _upright(red.center[0], red.center[1], mw.table_z))
    mw.set_object_pose(by_label['green'].name, _upright(blue.center[0], blue.center[1],
                                                        mw.table_z))
    mw.settle(0.5)
    assert mw.placements()[red.name] == ['red']
    assert mw.score(info['class_to_zone']) == {'correct': 1, 'wrong': 1, 'remaining': 1,
                                               'dropped': 0}


def test_held_objects_needs_both_fingers():
    mw = _build('L1')
    m, d = mw.model, mw.data
    rec = mw.objects[0]
    e = make_scenario('L1', 1)[0].objects[0].size[0]
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    R = d.site_xmat[tcp].reshape(3, 3).copy()
    center = d.site_xpos[tcp] - 0.015 * R[:, 2]          # 15 mm up the pads from the tip
    mw.set_object_pose(rec.name, make_T(R, center - R[:, 2] * (e / 2.0)))
    j7 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'joint7')]
    j8 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'joint8')]
    for q7, held in ((e / 2.0 - 0.0005, [rec.name]), (0.035, [])):
        d.qpos[j7], d.qpos[j8] = q7, -q7
        mw.forward()
        assert mw.held_objects() == held
        assert (mw.object_status()[rec.name]['status'] == 'held') == bool(held)
    # One finger only: shift the cube toward link8 (+y of link6) and close a little.
    mw.set_object_pose(rec.name, make_T(R, center - R[:, 2] * (e / 2.0) + R[:, 1] * 0.01))
    d.qpos[j7], d.qpos[j8] = e / 2.0 + 0.0095, -(e / 2.0 + 0.0095)
    mw.forward()
    assert mw.held_objects() == []


# ----------------------------------------------------------------- config
@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_config_overlay_merges_over_default(mount):
    task = 'L2'
    mw = _build(task, mount=mount, cached=(mount == 'eye_to_hand'))
    info = mw.info
    base = load_config(use_overlays=False)
    data = deep_merge(base.data, scenario_config_overlay(info))
    data = deep_merge(data, mw.config_overlay())
    cfg = Config(data, base.root, base.sources)
    assert cfg.get('robot.backend') == 'mujoco' and cfg.get('camera.backend') == 'mujoco'
    assert cfg.get('camera.mount') == mount
    assert cfg.get('robot.tcp_offset') == pytest.approx([0, 0, mw.tcp_z, 0, 0, 0])
    assert cfg.get('robot.gripper.max_width') == pytest.approx(0.07)
    assert cfg.get('perception.table.expected_z') == pytest.approx(mw.table_z)
    assert cfg.get('tasks.L2.class_to_zone') == info['class_to_zone']   # scenario keys kept
    zones = parse_zones(cfg)
    assert set(zones) == set(mw.zones)
    for name, z in zones.items():
        assert np.allclose(z.center, mw.zones[name].center)
        assert z.center[2] == pytest.approx(mw.table_z + sc.BIN_FLOOR)     # L2: walled bins
        assert z.wall_height == pytest.approx(info['zones'][name].wall_height)
    k = cfg.get('sim.intrinsics')
    cid = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_CAMERA, mw.task_camera)
    f = (k['height'] / 2.0) / math.tan(math.radians(mw.model.cam_fovy[cid]) / 2.0)
    assert (k['fx'], k['fy'], k['cx'], k['cy']) == pytest.approx(
        (f, f, (k['width'] - 1) / 2.0, (k['height'] - 1) / 2.0))
    ext = extrinsics_from_cfg(cfg)
    true = mw.T_base_camera if mount == 'eye_to_hand' else mw.T_flange_camera
    assert ext.mount == mount and np.allclose(ext.T, true, atol=1e-9)
    key = 'sim.T_base_camera' if mount == 'eye_to_hand' else 'sim.T_flange_camera'
    assert np.allclose(cfg.get(key), true, atol=1e-12)
    other = 'sim.T_flange_camera' if mount == 'eye_to_hand' else 'sim.T_base_camera'
    assert cfg.get(other) is None
    # Perturbed estimate: same direction as the kinematic sim overlay, true pose unchanged.
    over = mw.config_overlay(extrinsics_error=(0.01, 1.0))
    ref = scenario_config_overlay(info, extrinsics_error=(0.01, 1.0))
    est = np.array(over['camera']['extrinsics_inline']['matrix'])
    assert np.allclose(est, ref['camera']['extrinsics_inline']['matrix'], atol=1e-9)
    dt, dr = pose_error(est, true)
    assert dt == pytest.approx(0.01, abs=1e-9) and math.degrees(dr) == pytest.approx(1.0)
    assert np.allclose(over['sim'][key.split('.')[1]], true)
    with pytest.raises(ValueError):
        mw.config_overlay(extrinsics_error=(0.01,))


def test_attach_once_per_clock():
    mw = _build('L1')
    assert mw.sim is None
    t0 = float(mw.data.time)
    mw.settle(0.1)                                       # unattached: plain mj_step
    assert float(mw.data.time) == pytest.approx(t0 + 0.1)
    clock = SimClock()
    sim = mw.attach(clock)
    assert mw.attach(clock) is sim and sim.meta['tcp_site'] == 'tcp'
    with pytest.raises(RuntimeError):
        mw.attach(SimClock())
    clock.sleep(0.2)
    assert sim.n_steps == 100
    with pytest.raises(FileNotFoundError):
        sc.build_scene(mw.info, assets_dir='/nonexistent/piper')


def test_cfg_options_and_bad_input():
    world, info = make_scenario('L1', 1)
    home = [0.3, 0.5, -0.4, 0.1, 0.2, -0.3]
    mw = sc.build_scene(info, world=world, cfg={'robot': {'poses': {'home_joints': home}}})
    m, d = mw.model, mw.data
    for name, q in zip(mw.meta['arm_joints'], home):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert d.qpos[m.jnt_qposadr[j]] == pytest.approx(q) and d.ctrl[a] == pytest.approx(q)
    assert mw.meta['home_joints'] == home
    assert m.opt.noslip_iterations == sc.NOSLIP_ITERATIONS
    with pytest.raises(ValueError):
        sc.build_scene(info, world=world, cfg={'sim': {'mujoco': {'timestep': 0.05}}})
    with pytest.raises(ValueError):
        sc.build_scene(info, world=world, cfg={'robot': {'poses': {'home_joints': [0.0] * 5}}})
    with pytest.raises(ValueError):
        sc.build_scene(info, world=make_scenario('L2', 1)[0])     # world of another task

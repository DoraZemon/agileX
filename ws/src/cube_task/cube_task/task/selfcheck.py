"""
Pre-run self check: configuration, calibration, assets and hardware.

``self_check`` lists every problem at once, so one on-site run shows everything
that must be fixed. Lines start with '[FAIL]', '[WARN]' or '[ OK ]'; the run
is refused when any FAIL is present (real mode). Hardware checks (arm health,
feedback, camera frame and depth) are skipped in sim mode.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import inv_T, is_rotation, pose6_to_T, top_down_T
from ..core.zones import parse_zones

TASKS = ('L1', 'L2')
#: typical TCP-to-object-bottom distance (m) used to check release poses.
CHECK_HELD_OFFSET = 0.03
#: minimum fraction of valid depth pixels inside the projected pick area.
MIN_VALID_DEPTH = 0.5
#: joints within this (rad, ~2 deg) of robot.poses.observe_joints count as at the observe pose.
OBSERVE_JOINT_TOL = 0.035


class _Report:
    def __init__(self) -> None:
        self.messages: List[str] = []
        self.failed = False

    def fail(self, msg: str) -> None:
        self.failed = True
        self.messages.append('[FAIL] ' + msg)

    def warn(self, msg: str) -> None:
        self.messages.append('[WARN] ' + msg)

    def good(self, msg: str) -> None:
        self.messages.append('[ OK ] ' + msg)


def _numbers(v, n: int) -> Optional[List[float]]:
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if len(vals) != n or not all(math.isfinite(x) for x in vals):
        return None
    return vals


def task_backend(cfg, task: str) -> str:
    """Classifier backend of a task (tasks.<T>.classifier, else perception.classifier)."""
    b = cfg.get('tasks.%s.classifier' % task, None)
    if not b:
        b = cfg.get('perception.classifier.backend', 'color')
    return str(b)


def observe_tcp_pose(cfg) -> Optional[np.ndarray]:
    """Return the fallback observe TCP pose (robot.poses.observe_tcp) or None."""
    v = _numbers(cfg.get('robot.poses.observe_tcp', None) or [], 4)
    return None if v is None else top_down_T(v[0], v[1], v[2], v[3])


def _point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _check_task_config(cfg, task: str, rep: _Report, zones) -> None:
    labels = cfg.get('tasks.%s.labels' % task, None) or []
    if not isinstance(labels, list) or not labels:
        rep.fail('tasks.%s.labels is empty' % task)
        labels = []
    else:
        rep.good('%s labels: %s' % (task, ', '.join(str(x) for x in labels)))
    c2z = cfg.get('tasks.%s.class_to_zone' % task, None) or {}
    if not isinstance(c2z, dict) or not c2z:
        rep.fail('tasks.%s.class_to_zone is empty: nothing would be picked' % task)
        return
    for label, zname in c2z.items():
        if zname not in zones:
            rep.fail('class_to_zone: label %s -> zone %r which is not in zones %s'
                     % (label, zname, sorted(zones)))
        if label not in labels:
            rep.warn('class_to_zone label %s is not in tasks.%s.labels' % (label, task))
    for label in labels:
        if label not in c2z:
            rep.warn('label %s has no zone in class_to_zone: it is never picked' % label)
    for label in cfg.get('tasks.%s.order' % task, None) or []:
        if label not in c2z:
            rep.warn('order label %s has no zone in class_to_zone' % label)


def _in_workspace(cfg, x: float, y: float, z: Optional[float] = None) -> Optional[str]:
    """Return why (x, y, z) is outside robot.workspace, or None when inside."""
    ws = cfg.get('robot.workspace', None) or {}
    for axis, val in (('x', x), ('y', y), ('z', z)):
        lim = ws.get(axis)
        if val is None or lim is None:
            continue
        if not float(lim[0]) <= val <= float(lim[1]):
            return '%s=%.3f outside [%.3f, %.3f]' % (axis, val, float(lim[0]), float(lim[1]))
    r = math.hypot(x, y)
    if ws.get('r_min') is not None and r < float(ws['r_min']):
        return 'radius %.3f < r_min %.3f' % (r, float(ws['r_min']))
    if ws.get('r_max') is not None and r > float(ws['r_max']):
        return 'radius %.3f > r_max %.3f' % (r, float(ws['r_max']))
    return None


def _zone_footprints(cfg, task: Optional[str]) -> dict:
    """
    Zone name -> (length, width) (m) of the largest object the task sends there.

    From tasks.<task>.class_to_zone and place.nominal_footprint; a zone that
    receives a label without a nominal footprint maps to None (center only).
    """
    if task is None:
        return {}
    c2z = cfg.get('tasks.%s.class_to_zone' % task, None) or {}
    nominal = cfg.get('place.nominal_footprint', {}) or {}
    out: dict = {}
    if not isinstance(c2z, dict):
        return out
    for label, zone in c2z.items():
        fp = _numbers(nominal.get(label) or [], 2)
        prev = out.get(str(zone), ())
        if fp is None or prev is None:
            out[str(zone)] = None
        elif prev == () or max(fp) > max(prev):
            out[str(zone)] = (max(fp), min(fp))
    return out


def _place_poses(cfg, zones, task: Optional[str] = None) -> dict:
    """
    Zone name -> [(T_pre, T_release), ...] for a typical held object, or an error string.

    The zone center first, then (with the task's footprint for the zone) the
    release positions task/runner tries after an unreachable one.
    """
    from ..perception.grasp import PlacePlanner, transit_height
    from .runner import RELEASE_RETRIES
    table_z = cfg.get('perception.table.expected_z', None)
    table_z = 0.0 if table_z is None else float(table_z)
    out = {}
    try:
        planner = PlacePlanner(cfg)
        tz = transit_height(cfg, list(zones.values()), table_z, CHECK_HELD_OFFSET)
    except ValueError as e:
        return {name: str(e) for name in zones}
    footprints = _zone_footprints(cfg, task)
    for name, zone in zones.items():
        p = planner.plan(zone, CHECK_HELD_OFFSET, tz)
        if not p.feasible:
            out[name] = p.reason
            continue
        cands = [(p.T_pre, p.T_release)]
        fp = footprints.get(name)
        if fp is not None:
            held = {'length': fp[0], 'width': fp[1], 'grip_width': fp[1]}
            avoid: List[Tuple[float, float]] = []
            while len(avoid) < RELEASE_RETRIES:
                avoid.append((float(cands[-1][1][0, 3]), float(cands[-1][1][1, 3])))
                q = planner.plan(zone, CHECK_HELD_OFFSET, tz, footprint=fp[0], held=held,
                                 avoid=avoid)
                if not q.feasible or any(
                        math.hypot(float(q.T_release[0, 3]) - a, float(q.T_release[1, 3]) - b)
                        < 1e-6 for a, b in avoid):
                    break
                cands.append((q.T_pre, q.T_release))
        out[name] = cands
    return out


def _pose_outside(cfg, T_pre, T_release) -> Optional[str]:
    """Return why the transit or release TCP pose is outside robot.workspace, or None."""
    for what, T in (('transit', T_pre), ('release', T_release)):
        why = _in_workspace(cfg, float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        if why:
            return '%s TCP pose %s' % (what, why)
    return None


def _xy_text(T) -> str:
    return '(%.3f, %.3f)' % (float(T[0, 3]), float(T[1, 3]))


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _polygons_overlap(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> bool:
    """Return True when two simple polygons (base XY) overlap or one contains the other."""
    a = [(float(p[0]), float(p[1])) for p in a]
    b = [(float(p[0]), float(p[1])) for p in b]
    if any(_point_in_polygon(x, y, b) for x, y in a):
        return True
    if any(_point_in_polygon(x, y, a) for x, y in b):
        return True
    for i in range(len(a)):
        for j in range(len(b)):
            if _segments_cross(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]):
                return True
    return False


def _referenced_zones(cfg) -> set:
    """Zone names used by tasks.*.class_to_zone."""
    used = set()
    for task in TASKS:
        c2z = cfg.get('tasks.%s.class_to_zone' % task, None) or {}
        if isinstance(c2z, dict):
            used.update(str(v) for v in c2z.values())
    return used


def _check_zones(cfg, rep: _Report, zones, task: Optional[str] = None) -> None:
    poly = cfg.get('perception.roi.pick_polygon', None) or []
    margin = float(cfg.get('perception.roi.zone_exclude_margin', 0.02) or 0.0)
    poses = _place_poses(cfg, zones, task)
    for name, zone in zones.items():
        x, y = float(zone.center[0]), float(zone.center[1])
        center_out = _in_workspace(cfg, x, y)
        p = poses.get(name)
        if isinstance(p, str):
            if center_out:
                rep.fail('zone %s center outside the workspace: %s' % (name, center_out))
            else:
                rep.fail('zone %s: cannot plan the release: %s' % (name, p))
            continue
        inside = [c for c in p if _pose_outside(cfg, *c) is None]
        if not inside:
            if center_out:
                rep.fail('zone %s center outside the workspace: %s' % (name, center_out))
            else:
                rep.fail('zone %s %s outside the workspace'
                         % (name, _pose_outside(cfg, *p[0])))
            continue
        if inside[0] is not p[0]:
            rep.warn('zone %s: releasing at the center is outside the workspace (%s); objects '
                     'go to the near part of the card, first at %s'
                     % (name, center_out or _pose_outside(cfg, *p[0]), _xy_text(inside[0][1])))
        else:
            for cx, cy in zone.corners_xy():
                why = _in_workspace(cfg, float(cx), float(cy))
                if why:
                    rep.warn('zone %s corner (%.3f, %.3f) outside the workspace: %s'
                             % (name, cx, cy, why))
                    break
        if len(poly) >= 3 and _polygons_overlap(zone.corners_xy(margin), poly):
            rep.warn('zone %s grown by perception.roi.zone_exclude_margin (%.3f m) overlaps the '
                     'pick polygon: objects there are ignored (touches_zone)' % (name, margin))
    used = _referenced_zones(cfg)
    for name in sorted(zones):
        if name not in used:
            rep.warn('zone %s is not used by any tasks.*.class_to_zone '
                     'but perception still ignores objects in it: remove it from poses.yaml if '
                     'it is left over from an earlier layout' % name)
    if zones:
        rep.good('zones: %s' % ', '.join(sorted(zones)))


def _check_robot_config(cfg, rep: _Report, mode: str) -> None:
    tcp = _numbers(cfg.get('robot.tcp_offset', None) or [], 6)
    if tcp is None:
        rep.fail('robot.tcp_offset must be 6 numbers [x y z roll pitch yaw]')
    elif not 0.05 < tcp[2] < 0.25:
        rep.fail('robot.tcp_offset z = %.3f m is not in (0.05, 0.25): measure flange -> '
                 'fingertip center' % tcp[2])
    else:
        rep.good('tcp_offset z = %.3f m' % tcp[2])
    mw = cfg.get('robot.gripper.max_width', None)
    ew = cfg.get('robot.gripper.empty_width', None)
    if mw is None or not 0.0 < float(mw) <= 0.12:
        rep.fail('robot.gripper.max_width must be in (0, 0.12] m, got %r' % (mw,))
    elif ew is None or not 0.0 <= float(ew) < float(mw):
        rep.fail('robot.gripper.empty_width must be in [0, max_width), got %r' % (ew,))
    force = cfg.get('robot.gripper.force', None)
    if force is not None and not 0.0 <= float(force) <= 3.0:
        rep.fail('robot.gripper.force %r outside [0, 3] N' % (force,))
    qj = cfg.get('robot.poses.observe_joints', None)
    if qj is not None:
        if _numbers(qj, 6) is None:
            rep.fail('robot.poses.observe_joints must be 6 joint angles (rad)')
        else:
            rep.good('observe pose: taught joints')
        return
    T = observe_tcp_pose(cfg)
    if T is None:
        rep.fail('no observe pose: set robot.poses.observe_joints (teach_poses) or observe_tcp')
        return
    if mode == 'real':
        rep.warn('robot.poses.observe_joints is not taught; using observe_tcp')
    why = _in_workspace(cfg, float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    if why:
        rep.fail('observe_tcp outside the workspace: %s' % why)


def _check_extrinsics(cfg, rep: _Report, extrinsics):
    ext = extrinsics
    if ext is None:
        try:
            from ..core.extrinsics import extrinsics_from_cfg
            ext = extrinsics_from_cfg(cfg)
        except ImportError as e:
            rep.fail('core.extrinsics is not available: %s' % e)
            return None
        except Exception as e:  # noqa: B902 - any parse error is a finding
            rep.fail('extrinsics could not be loaded: %s' % e)
            return None
    if ext is None:
        rep.fail('no extrinsics: set camera.extrinsics_file (%s) or camera.extrinsics_inline'
                 % cfg.get('camera.extrinsics_file', ''))
        return None
    mount = cfg.get('camera.mount', None)
    if getattr(ext, 'mount', None) != mount:
        rep.fail('extrinsics mount %r differs from camera.mount %r'
                 % (getattr(ext, 'mount', None), mount))
    T = np.asarray(getattr(ext, 'T', None), dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not is_rotation(T[:3, :3], 1e-3):
        rep.fail('extrinsics matrix is not a valid rigid transform')
        return None
    rep.good('extrinsics: mount %s, translation %s' % (
        getattr(ext, 'mount', '?'), [round(float(v), 3) for v in T[:3, 3]]))
    return ext


def _check_assets(cfg, task: str, rep: _Report) -> None:
    backend = task_backend(cfg, task)
    labels = cfg.get('tasks.%s.labels' % task, None) or []
    if 'color' in backend.split('+'):
        classes = cfg.get('perception.classifier.color.classes', None) or {}
        missing = [lab for lab in labels if lab not in classes]
        if missing:
            rep.fail('color classifier has no HSV ranges for: %s' % ', '.join(missing))
    if 'knn' in backend.split('+'):
        p = cfg.path('perception.classifier.knn.model_file')
        if p is None or not p.is_file():
            rep.fail('knn model file missing: %s (run teach_classes)' % p)
    if 'onnx' in backend.split('+'):
        for key in ('perception.classifier.onnx.model_file',
                    'perception.classifier.onnx.labels_file'):
            p = cfg.path(key)
            if p is None or not p.is_file():
                rep.fail('onnx asset missing: %s = %s' % (key, p))
    if backend.split('+')[0] not in ('color', 'knn', 'onnx'):
        rep.fail('unknown classifier backend %r' % backend)
    else:
        rep.good('classifier backend %s' % backend)


def _fresh(stamped, clock, max_age: float) -> bool:
    if clock is None:
        return True
    return clock.now() - float(stamped.stamp) <= max_age


def _check_arm(cfg, rep: _Report, arm, clock) -> None:
    max_age = float(cfg.get('robot.motion.feedback_max_age_s', 0.5))
    try:
        h = arm.health()
    except Exception as e:  # noqa: B902
        rep.fail('arm.health() raised: %s' % e)
        return
    if h is None or not h.ok:
        rep.fail('arm not healthy: %s' % (getattr(h, 'reason', 'no status')))
    else:
        rep.good('arm healthy')
    q = arm.get_joints()
    vals = None if q is None else _numbers(np.asarray(q.value).ravel(), 6)
    if vals is None:
        rep.fail('joint angles not readable')
    elif not _fresh(q, clock, max_age):
        rep.fail('joint feedback is stale')
    w = arm.gripper_width()
    mw = float(cfg.get('robot.gripper.max_width', 0.07) or 0.07)
    if w is None or w.value is None or not math.isfinite(float(w.value)):
        rep.fail('gripper width not readable')
    elif not -0.005 <= float(w.value) <= mw + 0.01:
        rep.fail('gripper width %.4f m outside [0, max_width %.3f]' % (float(w.value), mw))
    elif not _fresh(w, clock, max_age):
        rep.fail('gripper feedback is stale')


def arm_self_check(cfg, arm, mode: str = 'real', clock=None) -> Tuple[bool, List[str]]:
    """
    Check only what moving the arm by hand-given targets needs; return (ok, messages).

    robot.* config (tcp_offset, gripper, observe pose) and, in real mode, arm
    health and fresh joint / gripper feedback. Zones, class_to_zone, extrinsics
    and classifier assets are task items (self_check) and are not looked at.
    """
    rep = _Report()
    _check_robot_config(cfg, rep, mode)
    if mode == 'real':
        if arm is None:
            rep.fail('no arm given')
        else:
            _check_arm(cfg, rep, arm, clock)
    return not rep.failed, rep.messages


def camera_pose_for_check(cfg, ext, arm) -> Tuple[Optional[np.ndarray], str]:
    """Return (T_base_camera, '') or (None, why) using the current flange pose."""
    try:
        from ..core.extrinsics import camera_pose_in_base
    except ImportError as e:
        return None, 'core.extrinsics not available: %s' % e
    T_ft = pose6_to_T(cfg.get('robot.tcp_offset', [0, 0, 0, 0, 0, 0]))
    T_bf = None
    if getattr(ext, 'mount', '') == 'eye_in_hand':
        s = None if arm is None else arm.get_flange_pose()
        if s is None:
            return None, 'eye_in_hand needs the flange pose and none is available'
        T_bf = np.asarray(s.value, dtype=float)
    try:
        return np.asarray(camera_pose_in_base(ext, T_bf, T_ft), dtype=float), ''
    except Exception as e:  # noqa: B902
        return None, 'camera_pose_in_base failed: %s' % e


def pick_area_mask(poly, table_z: float, T_base_camera: np.ndarray, intr) -> np.ndarray:
    """
    Return the pick polygon (base XY at table_z) projected into the image as a mask.

    The mask is HxW bool. Raises ValueError when a corner is behind the camera.
    """
    import cv2
    pts = np.array([[float(p[0]), float(p[1]), float(table_z)] for p in poly])
    T_cb = inv_T(T_base_camera)
    pc = pts @ T_cb[:3, :3].T + T_cb[:3, 3]
    if np.any(pc[:, 2] <= 0.01):
        raise ValueError('pick polygon is behind the camera')
    u = intr.fx * pc[:, 0] / pc[:, 2] + intr.cx
    v = intr.fy * pc[:, 1] / pc[:, 2] + intr.cy
    mask = np.zeros((int(intr.height), int(intr.width)), np.uint8)
    poly_px = np.round(np.stack([u, v], axis=1)).astype(np.int32)
    cv2.fillPoly(mask, [poly_px], 1)
    return mask.astype(bool)


def _camera_frame(cfg, camera, newer_than: Optional[float] = None):
    """Return (frame, intrinsics, depth, '') or (None, None, None, why)."""
    timeout = float(cfg.get('camera.frame_timeout_s', 2.0))
    try:
        if newer_than is None:
            frame = camera.get_frame(timeout=timeout)
        else:
            frame = camera.get_frame(timeout=timeout, newer_than=newer_than)
    except Exception as e:  # noqa: B902
        return None, None, None, 'camera.get_frame raised: %s' % e
    if frame is None:
        return None, None, None, 'no camera frame within %.1f s' % timeout
    intr = frame.intrinsics if frame.intrinsics is not None else camera.intrinsics()
    if intr is None or not (intr.fx > 0 and intr.fy > 0 and intr.width > 0 and intr.height > 0):
        return None, None, None, 'camera intrinsics missing or invalid'
    if frame.depth is None:
        return None, None, None, 'camera frame has no depth'
    depth = np.asarray(frame.depth, dtype=np.float32)
    if depth.shape[:2] != (int(intr.height), int(intr.width)):
        return None, None, None, 'depth %s does not match intrinsics %dx%d' % (
            depth.shape, intr.width, intr.height)
    return frame, intr, depth, ''


def _pick_area_depth(cfg, depth, intr, ext, arm) -> Tuple[str, str]:
    """
    Check the valid depth inside the projected pick polygon.

    Returns (level, message), level 'ok', 'fail', 'skip' (no extrinsics or
    pick polygon) or 'no_pose' (camera pose unavailable).
    """
    poly = cfg.get('perception.roi.pick_polygon', None) or []
    if ext is None or len(poly) < 3:
        return 'skip', 'pick-area depth check skipped (no extrinsics or pick polygon)'
    T_bc, why = camera_pose_for_check(cfg, ext, arm)
    if T_bc is None:
        return 'no_pose', 'pick-area depth check skipped: %s' % why
    tz = cfg.get('perception.table.expected_z', None)
    try:
        mask = pick_area_mask(poly, 0.0 if tz is None else float(tz), T_bc, intr)
    except ValueError as e:
        return 'fail', str(e)
    n = int(mask.sum())
    if n == 0:
        return 'fail', 'pick polygon projects outside the image: check extrinsics / pick_polygon'
    d = depth[mask]
    frac = float(np.count_nonzero(np.isfinite(d) & (d > 0))) / n
    if frac < MIN_VALID_DEPTH:
        return 'fail', 'only %.0f%% valid depth in the pick area (need %.0f%%)' % (
            100 * frac, 100 * MIN_VALID_DEPTH)
    return 'ok', 'camera: %.0f%% valid depth in the pick area' % (100 * frac)


def at_observe_pose(cfg, arm, tol: float = OBSERVE_JOINT_TOL) -> bool:
    """Return True when the arm's joints are within tol (rad) of robot.poses.observe_joints."""
    qj = _numbers(cfg.get('robot.poses.observe_joints', None) or [], 6)
    if qj is None or arm is None:
        return False
    try:
        q = arm.get_joints()
    except Exception:  # noqa: B902 - unknown pose = not at the observe pose
        return False
    vals = None if q is None else _numbers(np.asarray(q.value).ravel(), 6)
    if vals is None:
        return False
    return max(abs(a - b) for a, b in zip(vals, qj)) <= tol


def check_pick_area(cfg, camera, ext, arm, newer_than: Optional[float] = None
                    ) -> Tuple[bool, str]:
    """
    Check that the camera sees the pick area now (fresh frame, current arm pose).

    For eye_in_hand the task runner calls this at the observe pose, where the
    self check's pick-area test is deferred to. Returns (ok, message); a
    missing frame or camera pose fails, a missing pick polygon passes.
    """
    _frame, intr, depth, err = _camera_frame(cfg, camera, newer_than)
    if err:
        return False, err
    level, msg = _pick_area_depth(cfg, depth, intr, ext, arm)
    return level in ('ok', 'skip'), msg


def _check_camera(cfg, rep: _Report, camera, ext, arm) -> None:
    _frame, intr, depth, err = _camera_frame(cfg, camera)
    if err:
        rep.fail(err)
        return
    level, msg = _pick_area_depth(cfg, depth, intr, ext, arm)
    if level == 'ok':
        rep.good(msg)
    elif level in ('skip', 'no_pose'):
        rep.warn(msg)
    elif getattr(ext, 'mount', '') == 'eye_in_hand' and not at_observe_pose(cfg, arm):
        # A wrist camera sees the pick area only from the observe pose; the task
        # runner repeats this check there (check_pick_area) before the start.
        rep.warn('%s (eye_in_hand: checked at the observe pose)' % msg)
    else:
        rep.fail(msg)


def _reach(arm, T, max_tilt_deg: Optional[float] = None) -> Tuple[bool, float, str]:
    """Return (reachable, tilt deg, why) of one TCP pose (task/reach.reach_adjust)."""
    from .reach import reach_adjust
    try:
        T2, tilt, why = reach_adjust(arm, T, max_tilt_deg=max_tilt_deg)
    except Exception as e:  # noqa: B902
        T2, tilt, why = None, 0.0, 'reachability check raised: %s' % e
    return T2 is not None, tilt, why


def _check_reachable(cfg, rep: _Report, arm, zones, task: Optional[str] = None) -> None:
    if cfg.get('robot.poses.observe_joints', None) is None:
        T = observe_tcp_pose(cfg)
        if T is not None:
            ok, tilt, why = _reach(arm, T)
            if not ok:
                rep.fail('observe_tcp not reachable (also with the tool leaned outward): %s'
                         % why)
            elif tilt > 0:
                rep.good('observe_tcp reachable with the tool leaned %.0f deg' % tilt)
    from .reach import free_tilt_limit
    # Like task/runner: the transit is a free-space pose and may lean up to max_tilt_free_deg.
    free = free_tilt_limit(arm, cfg)
    for name, p in _place_poses(cfg, zones, task).items():
        if isinstance(p, str):
            continue
        first = None
        for k, (T_pre, T_rel) in enumerate(p):
            if k > 0 and _pose_outside(cfg, T_pre, T_rel) is not None:
                continue
            res = [('transit',) + _reach(arm, T_pre, free), ('release',) + _reach(arm, T_rel)]
            if first is None:
                first = res
            if all(r[1] for r in res):
                where = '' if k == 0 else (' at %s (the center is out of reach)'
                                           % _xy_text(T_rel))
                tilt = max(r[2] for r in res)
                if where or tilt > 0:
                    rep.good('zone %s transit and release reachable%s with the tool leaned '
                             '%.0f deg' % (name, where, tilt))
                break
        else:
            for what, ok, _tilt, why in first or []:
                if not ok:
                    rep.fail('zone %s %s not reachable (also with the tool leaned outward): %s'
                             % (name, what, why))


def self_check(cfg, task: str, arm=None, camera=None, mode: str = 'real',
               extrinsics=None, clock=None) -> Tuple[bool, List[str]]:
    """
    Check everything needed for a run; return (ok, messages).

    mode: 'real' or 'sim' (sim skips arm health / feedback and camera checks).
    extrinsics: use this core.extrinsics.Extrinsics instead of loading from cfg.
    clock: optional Clock used to reject stale arm feedback.
    """
    rep = _Report()
    if task not in TASKS:
        rep.fail('unknown task %r (use %s)' % (task, ', '.join(TASKS)))
        return False, rep.messages
    if mode not in ('real', 'sim'):
        rep.fail('unknown mode %r' % mode)
    try:
        zones = parse_zones(cfg)
    except (ValueError, TypeError) as e:
        rep.fail('zones: %s' % e)
        zones = {}
    if not zones:
        rep.fail('no zones configured (teach_poses)')
    _check_task_config(cfg, task, rep, zones)
    _check_zones(cfg, rep, zones, task)
    _check_robot_config(cfg, rep, mode)
    if mode == 'real' and cfg.get('perception.table.expected_z', None) is None:
        rep.warn('perception.table.expected_z is not set: the grasp and safety floor follow the '
                 "camera's table estimate only (an extrinsics z offset puts the fingers into the "
                 "table); teach it with teach_poses 'p'")
    ext = _check_extrinsics(cfg, rep, extrinsics)
    _check_assets(cfg, task, rep)
    if cfg.get('camera.mount', None) != 'eye_in_hand':
        rep.fail('camera.mount is %r: the competition camera is the wrist camera, set '
                 'camera.mount: eye_in_hand' % (cfg.get('camera.mount', None),))
    if mode == 'real':
        if arm is None:
            rep.warn('no arm given: arm checks skipped')
        else:
            _check_arm(cfg, rep, arm, clock)
        if camera is None:
            rep.warn('no camera given: camera checks skipped')
        else:
            _check_camera(cfg, rep, camera, ext, arm)
    if arm is not None and hasattr(arm, 'check_reachable'):
        _check_reachable(cfg, rep, arm, zones, task)
    return not rep.failed, rep.messages

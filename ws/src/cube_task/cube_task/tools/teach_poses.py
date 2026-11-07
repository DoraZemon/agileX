"""
Teach site poses by hand: observe / home joints, place zones and the pick polygon.

The operator puts the arm in drag-teach mode (teach button on the arm) and
moves it by hand. This tool only reads the arm; it never moves it. Commands:

  o         observe pose: current joints -> robot.poses.observe_joints
            (eye_to_hand: warns when the camera sees the gripper in front of
            the pick polygon)
  h         home pose: current joints -> robot.poses.home_joints
  z [name]  zone: touch the zone floor center with the closed fingertips ->
            zones.<name>.center [x, y, floor_z]; then size, wall height and
            an optional yaw from a second touch on the zone's x axis
  p         pick polygon: touch 4 table corners in order ->
            perception.roi.pick_polygon and perception.table.expected_z
  d [name]  delete zone <name> from the poses file (undo with u, written by w;
            warns when another file still defines it: the YAML merge cannot
            delete keys of other files)
  l list, u undo, w write, q quit (asks to write unsaved changes)

Joint readings at most JOINT_LIMIT_TOL (0.01 rad) outside a joint limit (the
folded rest pose sits on the joint2 / joint3 limits) are clipped onto it.

w merges the changes into config/site/poses.yaml (the previous file is kept
as poses.bak_<time>.yaml; sim mode writes <logging.dir>/sim_site/poses.yaml),
prints the dotted keys written, checks every zone's transit / release pose on
a kinematic arm (OK / tilt / unreachable) and prints the self check FAIL items.
Headless: commands are terminal lines and the top-view map is saved to
--preview-dir. --mode sim: a simulated operator drags the sim arm to the
scenario's observe pose, zone centers and table corners.

Example commands
    teach_poses
    teach_poses --headless --task L2
    teach_poses --mode sim --task L1 --headless
"""
from __future__ import annotations

import argparse
import copy
import functools
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..core.config import Config, deep_merge, read_yaml
from ..core.transforms import neutral_closing_angle, top_down_T
from ..core.zones import parse_zone
from .common import add_common_args, load_cfg_for, open_session, setup_logging, TASKS
from .onsite import (add_onsite_args, connect_arm_readonly, dotted_keys, flange_to_tcp,
                     fmt_list, fmt_mm, format_table, is_teaching, kinematic_arm,
                     load_extrinsics_or_none, make_ui, next_command, operator_action,
                     output_path, overlay_file, parse_floats, read_pose, read_pose_averaged,
                     read_yaml_or_empty, sim_drag_joints, sim_drag_tcp, teach_hint, text_panel,
                     wall_stamp, write_report, write_site_yaml)

_LOG = logging.getLogger('cube_task.tools.teach_poses')

OUT_NAME = 'poses.yaml'
KEY_OBSERVE = 'robot.poses.observe_joints'
KEY_HOME = 'robot.poses.home_joints'
KEY_POLYGON = 'perception.roi.pick_polygon'
KEY_TABLE_Z = 'perception.table.expected_z'
ZONE_KEYS = ('center', 'size', 'yaw', 'wall_height')

HELP = ('o observe, h home, z [name] zone, p pick polygon, d [name] delete zone, l list, '
        'u undo, w write, q quit')
WINDOW_HELP = ('keys in the map window: o h z p d l u w q; Enter = type a command in the '
               'terminal (e.g. "z zone_red")')
ORDER_HINT = ('suggested order: p pick polygon (sets the table z), z for every zone, o observe '
              'pose, h home pose, w write')
COMMANDS = {'o': 'observe', 'h': 'home', 'z': 'zone', 'p': 'polygon', 'd': 'delete',
            'l': 'list', 'u': 'undo', 'w': 'write', 'q': 'quit'}

DEFAULT_ZONE_SIZE = (0.10, 0.10)
ZONE_SIZE_RANGE = (0.02, 0.60)     # m, accepted zone sizes (catches sizes typed in mm)
WALL_RANGE = (0.0, 0.30)           # m, accepted wall heights
MIN_YAW_DIST = 0.03                # m, the yaw touch must be this far from the zone center
SIM_YAW_DIST = 0.04                # m, where the simulated operator touches for the yaw
MIN_POLYGON_AREA = 0.01            # m^2
MIN_CORNER_DIST = 0.02             # m, closer neighbors = the same corner touched twice
MAX_CORNER_Z_SPREAD = 0.02         # m, table corners differing more -> warning
FLOOR_TABLE_TOL = 0.03             # m, zone floor this far from the table z -> warning
VIEW_MARGIN = 0.02                 # m, pick polygon grown by this for the gripper view check
READ_DT = 0.02                     # s between the averaged arm readings
JOINT_LIMIT_TOL = 0.01             # rad, readings this far outside a joint limit are clipped
# Fingertip box of the gripper in the TCP frame (m): pad width along tool x, finger
# thickness outside the opening along tool y, finger length from the fingertips (TCP)
# up along -tool z. Same values as the simulated gripper (sim/world.FINGER_SIZE).
FINGER_PAD = 0.020
FINGER_THICKNESS = 0.012
FINGER_LENGTH = 0.050
ZONE_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,39}$')
# Top-view map: base x up, base y to the left.
MAP_X = (-0.05, 0.65)              # m, base x range shown
MAP_Y = (-0.45, 0.45)              # m, base y range shown
MAP_SCALE = 700.0                  # px per m
_MISSING = object()


class _Delete:
    """Value in the changes mapping that removes its dotted key from the poses file."""

    def __repr__(self) -> str:
        """Return 'DELETE'."""
        return 'DELETE'

    def __copy__(self) -> '_Delete':
        """Keep the single marker instance."""
        return self

    def __deepcopy__(self, memo: Any) -> '_Delete':
        """Keep the single marker instance."""
        return self


#: changes[key] = DELETE removes key (e.g. 'zones.veg_bin') from the poses file.
DELETE = _Delete()


# ------------------------------------------------------------------ YAML mapping
def _r(v: Any, nd: int = 4) -> float:
    """Round v to nd decimals as a plain float (never -0.0)."""
    return float(round(float(v), nd)) + 0.0


def _finite_number(v: Any) -> bool:
    """Return True when v converts to a finite float."""
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def get_dotted(d: Any, key: str, default: Any = None) -> Any:
    """Return d['a']['b'] for key 'a.b', or default when a level is missing."""
    cur = d
    for p in key.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def set_dotted(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set d['a']['b'] = value for key 'a.b', creating missing (or non-mapping) levels."""
    parts = key.split('.')
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def pop_dotted(d: Dict[str, Any], key: str) -> bool:
    """Remove d['a']['b'] for key 'a.b' (and mappings left empty); return True if it existed."""
    parts = key.split('.')
    chain = [d]
    for p in parts[:-1]:
        nxt = chain[-1].get(p)
        if not isinstance(nxt, dict):
            return False
        chain.append(nxt)
    if parts[-1] not in chain[-1]:
        return False
    del chain[-1][parts[-1]]
    for parent, p, child in reversed(list(zip(chain[:-1], parts[:-1], chain[1:]))):
        if child:
            break
        del parent[p]
    return True


def apply_changes(existing: Optional[Dict[str, Any]], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the poses.yaml mapping to write: existing with every dotted change set.

    Changes apply in order; a DELETE value removes its key. Everything not
    changed (other zones, zone labels, other sections) is kept.
    """
    out = copy.deepcopy(existing or {})
    for k, v in changes.items():
        if v is DELETE:
            pop_dotted(out, k)
        else:
            set_dotted(out, k, copy.deepcopy(v))
    return out


def effective(view: Dict[str, Any], cfg, key: str, default: Any = None) -> Any:
    """Return key from the poses mapping view, else from cfg (the merged configuration)."""
    v = get_dotted(view, key, _MISSING)
    return cfg.get(key, default) if v is _MISSING else v


def file_header(stamp: str) -> str:
    """Return the comment written at the top of poses.yaml."""
    return ('Site poses taught with teach_poses (%s), merged over config/default.yaml.\n'
            'Units: m, rad. Re-run teach_poses to change them; previous versions are kept\n'
            'next to this file as poses.bak_<time>.yaml.' % stamp)


def _joint_limits(kin) -> Optional[np.ndarray]:
    """Return kin.limits as a (6, 2) array in rad, or None when kin has none."""
    lim = getattr(kin, 'limits', None)
    try:
        lim = None if lim is None else np.asarray(lim, dtype=float)
    except (TypeError, ValueError):
        return None
    return lim if lim is not None and lim.shape == (6, 2) else None


def joint_limit_violations(q: Sequence[float], kin, tol: float = 1e-9) -> List[str]:
    """Describe each joint more than tol (rad) outside its limit, in deg with 3 decimals."""
    lim = _joint_limits(kin)
    if lim is None:
        return [] if kin.within_limits(q) else list(kin.limit_violations(q))
    out = []
    for i, v in enumerate(q):
        lo, hi = lim[i]
        if v < lo - tol:
            out.append('joint%d = %.3f deg < %.3f deg' % (i + 1, math.degrees(v),
                                                          math.degrees(lo)))
        elif v > hi + tol:
            out.append('joint%d = %.3f deg > %.3f deg' % (i + 1, math.degrees(v),
                                                          math.degrees(hi)))
    return out


def clip_to_limits(q: Sequence[float], kin, tol: float = JOINT_LIMIT_TOL
                   ) -> Tuple[List[float], List[str]]:
    """
    Clip each joint at most tol (rad) outside its limit onto the limit.

    Returns (joints, notes about the clipped joints). Joints further out are
    left unchanged (joints_problem reports them).
    """
    out = [float(v) for v in q]
    lim = _joint_limits(kin)
    if lim is None or len(out) != 6 or not all(math.isfinite(v) for v in out):
        return out, []
    notes = []
    for i, v in enumerate(out):
        lo, hi = (float(x) for x in lim[i])
        edge = lo if v < lo else hi if v > hi else None
        if edge is None or abs(v - edge) > tol:
            continue
        out[i] = edge
        notes.append('joint%d = %.3f deg is %.3f deg outside its limit: recorded as %.3f deg '
                     '(readings up to %.3f deg outside are clipped)'
                     % (i + 1, math.degrees(v), math.degrees(abs(v - edge)),
                        math.degrees(edge), math.degrees(tol)))
    return out, notes


def joints_problem(q: Any, kin=None) -> str:
    """Return why q is not a usable joint pose ('' when it is): 6 finite rad within limits."""
    try:
        vals = [float(v) for v in q]
    except (TypeError, ValueError):
        return 'must be 6 joint angles (rad), got %r' % (q,)
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        return 'must be 6 finite joint angles (rad), got %r' % (q,)
    if kin is not None:
        bad = joint_limit_violations(vals, kin)
        if bad:
            # move_joints refuses any target outside the limits, so no tolerance here; a pose
            # read just outside a limit is clipped when it is taught (clip_to_limits).
            near = not joint_limit_violations(vals, kin, JOINT_LIMIT_TOL)
            return 'outside the joint limits: %s%s' % (', '.join(bad), (
                ' (re-teach it: o / h clip readings this close onto the limit)' if near
                else ''))
    return ''


def zone_sources(sources: Sequence[str], name: str, exclude: Optional[Path] = None
                 ) -> List[str]:
    """
    Return the config sources that define zones.<name>, except the file exclude.

    sources: Config.sources (YAML files and 'override:key=value' entries).
    """
    out: List[str] = []
    skip = None if exclude is None else Path(exclude).expanduser().resolve()
    for src in sources:
        s = str(src)
        if s.startswith('override:'):
            key, _sep, value = s[len('override:'):].partition('=')
            key = key.strip()
            hit = key == 'zones.' + name or key.startswith('zones.%s.' % name)
            if key == 'zones':
                try:
                    v = yaml.safe_load(value)
                except yaml.YAMLError:
                    v = None
                hit = isinstance(v, dict) and name in v
            if hit:
                out.append('--set %s' % key)
            continue
        p = Path(s).expanduser()
        if not p.is_file() or (skip is not None and p.resolve() == skip):
            continue
        try:
            zones = read_yaml(p).get('zones')
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if isinstance(zones, dict) and name in zones:
            out.append(str(p))
    return out


def zone_users(tasks_cfg: Any, name: str) -> List[str]:
    """Return 'tasks.<T>.class_to_zone (labels)' for each task that sends labels to name."""
    out: List[str] = []
    for task, d in (tasks_cfg.items() if isinstance(tasks_cfg, dict) else []):
        c2z = d.get('class_to_zone') if isinstance(d, dict) else None
        labels = [str(k) for k, z in (c2z.items() if isinstance(c2z, dict) else [])
                  if z is not None and str(z) == name]
        if labels:
            out.append('tasks.%s.class_to_zone (%s)' % (task, ', '.join(labels)))
    return out


# ------------------------------------------------------------------ zones
def zone_name_problem(name: str) -> str:
    """Return why name cannot be a zone name ('' when it can)."""
    if ZONE_NAME_RE.match(str(name or '')):
        return ''
    return ('zone name %r: start with a letter, then letters, digits, _ or - '
            '(at most 40 characters)' % (name,))


def task_zone_names(tasks_cfg: Any) -> List[str]:
    """Return the zone names used in tasks.*.class_to_zone (first-use order, no repeats)."""
    names: List[str] = []
    if not isinstance(tasks_cfg, dict):
        return names
    for task in tasks_cfg.values():
        c2z = task.get('class_to_zone') if isinstance(task, dict) else None
        if not isinstance(c2z, dict):
            continue
        for z in c2z.values():
            if z is not None and str(z) not in names:
                names.append(str(z))
    return names


def suggest_zone_names(tasks_cfg: Any, taught: Sequence[str]) -> List[str]:
    """Return the zone names of tasks.*.class_to_zone that are not in taught."""
    done = {str(t) for t in taught}
    return [n for n in task_zone_names(tasks_cfg) if n not in done]


def wrap_half_pi(a: float) -> float:
    """Wrap an angle (rad) into [-pi/2, pi/2): a rectangle's yaw is defined modulo pi."""
    return (float(a) + math.pi / 2.0) % math.pi - math.pi / 2.0


def yaw_from_points(center: Sequence[float], axis_point: Sequence[float],
                    min_dist: float = MIN_YAW_DIST) -> float:
    """
    Return the zone yaw (rad, in [-pi/2, pi/2)) from its center and a point on its x axis.

    Only x, y are used. Raises ValueError when the points are closer than min_dist (m).
    """
    d = np.asarray(axis_point, float).ravel()[:2] - np.asarray(center, float).ravel()[:2]
    dist = float(np.hypot(d[0], d[1]))
    if not (math.isfinite(dist) and dist >= min_dist):
        raise ValueError('the yaw point is %.0f mm from the zone center; touch a point at least '
                         '%.0f mm away on the zone x axis' % (dist * 1e3, min_dist * 1e3))
    return wrap_half_pi(math.atan2(float(d[1]), float(d[0])))


def zone_from_touches(center: Sequence[float], size: Sequence[float] = DEFAULT_ZONE_SIZE,
                      wall_height: float = 0.0, axis_point: Optional[Sequence[float]] = None,
                      yaw: float = 0.0) -> Dict[str, Any]:
    """
    Return the zone mapping {center, size, yaw, wall_height} from touched points (m, rad).

    center: TCP position with the fingertips on the zone floor center, giving
    center [x, y, floor_z]. size: extent along the zone's own x / y.
    axis_point: optional second touch on the zone x axis that sets the yaw;
    without it yaw is used. Raises ValueError for invalid values.
    """
    c = [float(v) for v in np.asarray(center, float).ravel()]
    if len(c) != 3 or not all(math.isfinite(v) for v in c):
        raise ValueError('zone center must be 3 finite numbers, got %r' % (c,))
    s = [float(v) for v in size]
    lo, hi = ZONE_SIZE_RANGE
    if len(s) != 2 or not all(math.isfinite(v) and lo <= v <= hi for v in s):
        raise ValueError('zone size must be two numbers in [%.2f, %.2f] m (meters, e.g. '
                         '0.10 0.10), got %r' % (lo, hi, s))
    h = float(wall_height)
    if not (math.isfinite(h) and WALL_RANGE[0] <= h <= WALL_RANGE[1]):
        raise ValueError('wall height must be in [%.2f, %.2f] m, got %r'
                         % (WALL_RANGE[0], WALL_RANGE[1], wall_height))
    a = float(yaw) if axis_point is None else yaw_from_points(c, axis_point)
    if not math.isfinite(a):
        raise ValueError('zone yaw must be finite, got %r' % (yaw,))
    return {'center': [_r(v) for v in c], 'size': [_r(v) for v in s],
            'yaw': _r(wrap_half_pi(a)), 'wall_height': _r(h)}


def zone_changes(name: str, zone: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dotted changes zones.<name>.center / size / yaw / wall_height of a zone."""
    return {'zones.%s.%s' % (name, k): copy.deepcopy(zone[k]) for k in ZONE_KEYS}


def workspace_xy_problem(workspace: Any, x: float, y: float) -> str:
    """Return why base (x, y) is outside robot.workspace (x / y, r_min / r_max); '' if inside."""
    ws = workspace if isinstance(workspace, dict) else {}
    x, y = float(x), float(y)
    for axis, v in (('x', x), ('y', y)):
        lim = ws.get(axis)
        if lim is not None and not float(lim[0]) <= v <= float(lim[1]):
            lo, hi = float(lim[0]) * 1e3, float(lim[1]) * 1e3
            return '%s %.0f mm outside [%.0f, %.0f] mm' % (axis, v * 1e3, lo, hi)
    r = math.hypot(x, y)
    if ws.get('r_min') is not None and r < float(ws['r_min']):
        return 'radius %.0f mm < r_min %.0f mm' % (r * 1e3, float(ws['r_min']) * 1e3)
    if ws.get('r_max') is not None and r > float(ws['r_max']):
        return 'radius %.0f mm > r_max %.0f mm' % (r * 1e3, float(ws['r_max']) * 1e3)
    return ''


def zone_warnings(name: str, zone: Dict[str, Any], workspace: Any = None, polygon: Any = None,
                  table_z: Optional[float] = None) -> List[str]:
    """Return warnings for a taught zone: outside the workspace, on the pick area, low floor."""
    z = parse_zone(name, zone)
    x, y, floor = (float(v) for v in z.center)
    out = []
    why = workspace_xy_problem(workspace, x, y)
    if why:
        out.append('zone %s center is outside the workspace: %s' % (name, why))
    try:
        poly = _xy(polygon) if polygon is not None else None
    except (TypeError, ValueError):
        poly = None
    if poly is not None and len(poly) >= 3 and np.all(np.isfinite(poly)):
        if point_in_polygon(x, y, poly):
            out.append('zone %s center lies inside the pick polygon' % name)
        elif any(point_in_polygon(cx, cy, poly) for cx, cy in z.corners_xy()):
            out.append('zone %s overlaps the pick polygon' % name)
    if _finite_number(table_z) and abs(floor - float(table_z)) > FLOOR_TABLE_TOL:
        out.append('zone %s floor z %.0f mm is %.0f mm from the table z %.0f mm: did the '
                   'fingertips touch the zone floor?' % (name, floor * 1e3,
                                                         abs(floor - float(table_z)) * 1e3,
                                                         float(table_z) * 1e3))
    return out


# ------------------------------------------------------------------ pick polygon
def _xy(points: Any) -> np.ndarray:
    """Return points as an (N, 2) float array of base XY (extra columns dropped)."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError('expected a list of [x, y] points, got shape %s' % (P.shape,))
    return P[:, :2]


def polygon_area(points: Any) -> float:
    """Return the signed shoelace area (m^2) of a base-XY polygon (> 0 counter-clockwise)."""
    p = _xy(points)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def is_convex_polygon(points: Any) -> bool:
    """Return True for a simple, strictly convex polygon in either orientation."""
    p = _xy(points)
    if len(p) < 3:
        return False
    e = np.roll(p, -1, axis=0) - p
    e2 = np.roll(e, -1, axis=0)
    cross = e[:, 0] * e2[:, 1] - e[:, 1] * e2[:, 0]
    if not (np.all(cross > 1e-9) or np.all(cross < -1e-9)):
        return False
    # A star polygon turns the same way at every corner too: require one full turn.
    turn = float(np.sum(np.arctan2(cross, np.sum(e * e2, axis=1))))
    return abs(abs(turn) - 2.0 * math.pi) < 1e-6


def ccw_polygon(points: Any) -> List[List[float]]:
    """Return the XY corners counter-clockwise (clockwise input reversed, same first corner)."""
    p = [[float(v[0]), float(v[1])] for v in _xy(points)]
    return [p[0]] + p[1:][::-1] if polygon_area(p) < 0 else p


def point_in_polygon(x: float, y: float, polygon: Any) -> bool:
    """Return True when base (x, y) is inside the polygon (even-odd rule)."""
    p = _xy(polygon)
    inside = False
    n = len(p)
    for i in range(n):
        x1, y1 = float(p[i][0]), float(p[i][1])
        x2, y2 = float(p[(i + 1) % n][0]), float(p[(i + 1) % n][1])
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def distance_to_polygon(x: float, y: float, polygon: Any) -> float:
    """Return the distance (m) from base (x, y) to the nearest polygon edge."""
    p = _xy(polygon)
    q = np.array([x, y], dtype=float)
    best = math.inf
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        ab = b - a
        t = float(np.clip(np.dot(q - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(q - (a + t * ab))))
    return best


def polygon_contains(polygon: Any, x: float, y: float, margin: float = 0.0) -> bool:
    """Return True when base (x, y) is inside the polygon or within margin (m) of its edges."""
    if point_in_polygon(x, y, polygon):
        return True
    return margin > 0.0 and distance_to_polygon(x, y, polygon) <= margin


def validate_pick_polygon(points: Any, workspace: Any = None, n_corners: int = 4,
                          min_area: float = MIN_POLYGON_AREA) -> List[str]:
    """
    Return the problems of a pick polygon (base XY, m); [] when it can be used.

    Checks: n_corners corners (0 = any number >= 3), finite values, neighbors at
    least MIN_CORNER_DIST apart, simple and convex (corners in order around the
    area), area > min_area, every corner inside robot.workspace (x / y ranges,
    r_min / r_max).
    """
    try:
        P = _xy(points)
    except (TypeError, ValueError) as e:
        return ['corners must be [x, y] numbers (%s)' % e]
    if n_corners and len(P) != n_corners:
        return ['need %d corners, got %d' % (n_corners, len(P))]
    if len(P) < 3:
        return ['need at least 3 corners, got %d' % len(P)]
    if not np.all(np.isfinite(P)):
        return ['corners must be finite numbers']
    problems = []
    n = len(P)
    for i in range(n):
        d = float(np.linalg.norm(P[(i + 1) % n] - P[i]))
        if d < MIN_CORNER_DIST:
            problems.append('corners %d and %d are only %.0f mm apart (the same corner touched '
                            'twice?)' % (i + 1, (i + 1) % n + 1, d * 1e3))
    if not problems and not is_convex_polygon(P):
        problems.append('the corners do not form a convex polygon: touch them in order around '
                        'the pick area (clockwise or counter-clockwise)')
    area = abs(polygon_area(P))
    if area <= min_area:
        problems.append('area %.4f m^2 is too small (need more than %.2f m^2)' % (area, min_area))
    for i, (x, y) in enumerate(P):
        why = workspace_xy_problem(workspace, float(x), float(y))
        if why:
            problems.append('corner %d %s mm is outside the workspace: %s'
                            % (i + 1, fmt_mm([x, y], 0), why))
    return problems


def pick_polygon_from_touches(touches: Any, workspace: Any = None
                              ) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """
    Return (changes or None, problems, notes) for touched table corners (TCP xyz, m).

    changes: perception.roi.pick_polygon (counter-clockwise XY) and
    perception.table.expected_z (mean corner z); None when there are
    problems. notes: orientation fixed, uneven corner heights.
    """
    T = np.asarray(touches, dtype=float).reshape(-1, 3)
    problems = validate_pick_polygon(T[:, :2], workspace)
    table_z = float(np.mean(T[:, 2])) if len(T) else math.nan
    zlim = workspace.get('z') if isinstance(workspace, dict) else None
    if len(T) and zlim is not None and not float(zlim[0]) <= table_z <= float(zlim[1]):
        problems.append('table z %.0f mm is outside the workspace z [%.0f, %.0f] mm'
                        % (table_z * 1e3, float(zlim[0]) * 1e3, float(zlim[1]) * 1e3))
    if problems:
        return None, problems, []
    notes = []
    spread = float(np.ptp(T[:, 2]))
    if spread > MAX_CORNER_Z_SPREAD:
        notes.append('WARNING: the corner heights differ by %.0f mm: is the table tilted, or was '
                     'a corner touched above the table?' % (spread * 1e3))
    if polygon_area(T) < 0:
        notes.append('the corners were touched clockwise; stored counter-clockwise')
    poly = [[_r(x), _r(y)] for x, y in ccw_polygon(T)]
    return {KEY_POLYGON: poly, KEY_TABLE_Z: _r(table_z)}, [], notes


# ------------------------------------------------------------------ gripper in view
def gripper_points(T_base_tcp: Any, open_width: float) -> np.ndarray:
    """
    Return the TCP and the 8 fingertip box corners in the base frame ((9, 3), m).

    The box spans tool x +-FINGER_PAD / 2, tool y +-(open_width / 2 +
    FINGER_THICKNESS) and from the fingertips (TCP) FINGER_LENGTH up along
    -tool z.
    """
    T = np.asarray(T_base_tcp, dtype=float)
    hx = FINGER_PAD / 2.0
    hy = max(0.0, float(open_width)) / 2.0 + FINGER_THICKNESS
    loc = [[0.0, 0.0, 0.0]]
    for z in (0.0, -FINGER_LENGTH):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                loc.append([sx * hx, sy * hy, z])
    return np.asarray(loc) @ T[:3, :3].T + T[:3, 3]


def table_hit_xy(point: Any, T_base_camera: Any, table_z: float) -> Optional[np.ndarray]:
    """
    Return where the camera's line of sight through point meets the table plane (base XY).

    None when the point is behind the camera or the ray does not reach the
    plane in front of the camera. A point projects into the image of a table
    region exactly when this hit lies inside the region (pinhole model, so the
    intrinsics do not matter).
    """
    T = np.asarray(T_base_camera, dtype=float)
    P = np.asarray(point, dtype=float).ravel()[:3]
    C = T[:3, 3]
    d = P - C
    if float(np.dot(T[:3, 2], d)) <= 1e-3:     # depth along the optical axis
        return None
    if d[2] >= -1e-9:
        return None
    t = (float(table_z) - float(C[2])) / float(d[2])
    if t <= 0.0:
        return None
    return (C + t * d)[:2]


def gripper_in_view(T_base_tcp: Any, T_base_camera: Any, polygon: Any, table_z: float,
                    open_width: float, margin: float = VIEW_MARGIN) -> List[int]:
    """
    Return the gripper points that the camera sees in front of the pick polygon.

    Indices into gripper_points (0 = TCP, 1-8 = fingertip box corners) whose
    line of sight meets the table inside the polygon grown by margin (m).
    """
    hits = []
    for i, P in enumerate(gripper_points(T_base_tcp, open_width)):
        h = table_hit_xy(P, T_base_camera, table_z)
        if h is not None and polygon_contains(polygon, float(h[0]), float(h[1]), margin):
            hits.append(i)
    return hits


def observe_view_check(T_base_tcp: Any, T_base_camera: Any, polygon: Any, table_z: float,
                       open_width: float, margin: float = VIEW_MARGIN) -> Tuple[bool, str]:
    """Return (ok, message) of the eye_to_hand gripper-in-view check of an observe pose."""
    hits = gripper_in_view(T_base_tcp, T_base_camera, polygon, table_z, open_width, margin)
    if not hits:
        return True, 'the gripper is outside the camera view of the pick area'
    n_box = len([i for i in hits if i > 0])
    parts = (['TCP'] if 0 in hits else []) + (
        ['%d of 8 fingertip box corners' % n_box] if n_box else [])
    return False, ('the camera sees the gripper (%s) in front of the pick area, so objects '
                   'behind it are hidden. Park the arm out of the line of sight of the camera '
                   'and teach the observe pose again.' % ', '.join(parts))


def observe_view_report(cfg, T_base_tcp: Any, polygon: Any = None,
                        table_z: Optional[float] = None) -> Tuple[Optional[bool], str]:
    """
    Run the gripper-in-view check of an eye_to_hand observe pose with the configured extrinsics.

    polygon / table_z default to the config (table z 0 when unknown). Returns
    (True = ok | False = gripper in view | None = skipped, message).
    """
    ext = load_extrinsics_or_none(cfg)
    if ext is None:
        return None, ('gripper-in-view check skipped: no usable extrinsics '
                      '(camera.extrinsics_file)')
    if ext.mount != 'eye_to_hand':
        return None, 'gripper-in-view check skipped: the camera is not eye_to_hand'
    poly = cfg.get(KEY_POLYGON, None) if polygon is None else polygon
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        return None, 'gripper-in-view check skipped: no pick polygon'
    tz = cfg.get(KEY_TABLE_Z, None) if table_z is None else table_z
    from ..core.extrinsics import camera_pose_in_base
    try:
        width = float(cfg.get('robot.gripper.max_width', 0.07) or 0.07)
        return observe_view_check(T_base_tcp, camera_pose_in_base(ext), poly,
                                  0.0 if tz is None else float(tz), width)
    except (TypeError, ValueError) as e:
        return None, 'gripper-in-view check skipped: %s' % e


# ------------------------------------------------------------------ validation / checks
def validate_poses(data: Dict[str, Any], kin=None, workspace: Any = None) -> List[str]:
    """Return the problems of a poses.yaml mapping (each names its dotted key); [] if fine."""
    problems = []
    for key in (KEY_OBSERVE, KEY_HOME):
        q = get_dotted(data, key, None)
        why = '' if q is None else joints_problem(q, kin)
        if why:
            problems.append('%s %s' % (key, why))
    zones = get_dotted(data, 'zones', None)
    if zones is not None and not isinstance(zones, dict):
        problems.append('zones must be a mapping zone name -> {center, size, ...}')
    for name, z in (zones.items() if isinstance(zones, dict) else []):
        try:
            parse_zone(str(name), z if isinstance(z, dict) else {})
        except (TypeError, ValueError) as e:
            problems.append('zones.%s: %s' % (name, e))
    poly = get_dotted(data, KEY_POLYGON, None)
    if poly is not None:
        problems.extend('%s: %s' % (KEY_POLYGON, p)
                        for p in validate_pick_polygon(poly, workspace, n_corners=0))
    tz = get_dotted(data, KEY_TABLE_Z, None)
    if tz is not None and not _finite_number(tz):
        problems.append('%s must be a number (m), got %r' % (KEY_TABLE_Z, tz))
    return problems


def reach_text(T2: Any, tilt_deg: float) -> Tuple[str, str]:
    """Return (kind, text) of a reach_adjust result: ok / tilt / unreachable."""
    if T2 is None:
        return 'unreachable', 'unreachable'
    if tilt_deg > 0.05:
        return 'tilt', 'tilt %.0f deg' % tilt_deg
    return 'ok', 'OK'


def zone_reach_rows(cfg, arm, held_offset: Optional[float] = None
                    ) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check every zone's transit and release pose the way the task runner plans them.

    PlacePlanner(cfg) with grasp.transit_height (table z =
    perception.table.expected_z, else 0; a typical held object held_offset
    below the TCP, default task.selfcheck.CHECK_HELD_OFFSET); each pose goes
    through task.reach.reach_adjust on arm (a kinematic SafeArm, so the tool
    may lean outward). Returns (transit z, rows {zone, transit, release,
    transit_z, release_z, result, why}); result is 'OK', 'tilt N deg' or
    'UNREACHABLE'. Raises ValueError for invalid zones.
    """
    from ..core.zones import parse_zones
    from ..perception.grasp import PlacePlanner, transit_height
    from ..task.reach import reach_adjust
    from ..task.selfcheck import CHECK_HELD_OFFSET
    held = CHECK_HELD_OFFSET if held_offset is None else float(held_offset)
    zones = parse_zones(cfg)
    tz = cfg.get(KEY_TABLE_Z, None)
    planner = PlacePlanner(cfg)
    transit = transit_height(cfg, list(zones.values()), 0.0 if tz is None else float(tz), held)
    rows = []
    for name, zone in zones.items():
        plan = planner.plan(zone, held, transit)
        row: Dict[str, Any] = {'zone': name, 'transit_z': float(plan.T_pre[2, 3]),
                               'release_z': float(plan.T_release[2, 3]), 'why': []}
        rows.append(row)
        if not plan.feasible:
            row.update(transit='-', release='-', result='UNREACHABLE')
            row['why'].append('no release plan: %s' % plan.reason)
            continue
        kinds, tilts = [], []
        for what, T in (('transit', plan.T_pre), ('release', plan.T_release)):
            try:
                T2, tilt, why = reach_adjust(arm, T)
            except (ValueError, np.linalg.LinAlgError) as e:
                T2, tilt, why = None, 0.0, 'check failed: %s' % e
            kind, row[what] = reach_text(T2, tilt)
            kinds.append(kind)
            tilts.append(tilt)
            if T2 is None:
                row['why'].append('%s: %s' % (what, why))
        if 'unreachable' in kinds:
            row['result'] = 'UNREACHABLE'
        elif 'tilt' in kinds:
            row['result'] = 'tilt %.0f deg' % max(tilts)
        else:
            row['result'] = 'OK'
    return transit, rows


def reach_report(cfg, arm, held_offset: Optional[float] = None) -> List[str]:
    """Return the per-zone reachability lines printed after a write."""
    try:
        transit, rows = zone_reach_rows(cfg, arm, held_offset)
    except (TypeError, ValueError) as e:
        return ['[FAIL] zone reachability not checked: %s' % e]
    if not rows:
        return ['zone reachability: no zones configured']

    def cell(text: str, z: float) -> str:
        return text if text == '-' else '%s (z %.0f mm)' % (text, z * 1e3)

    table = format_table(('zone', 'transit', 'release', 'result'), [
        [r['zone'], cell(r['transit'], r['transit_z']), cell(r['release'], r['release_z']),
         r['result']] for r in rows])
    lines = ['zone reachability (kinematic check of the place poses, transit TCP z %.0f mm):'
             % (transit * 1e3)]
    lines += ['  ' + s for s in table.splitlines()]
    for r in rows:
        lines += ['  %s %s' % (r['zone'], w) for w in r['why']]
    return lines


def pose_report(cfg, kin=None) -> List[str]:
    """Return the observe / home joint checks (limits, eye_to_hand gripper-in-view) as lines."""
    lines = []
    mount = str(cfg.get('camera.mount', '') or '')
    for key in (KEY_OBSERVE, KEY_HOME):
        q = cfg.get(key, None)
        if q is None:
            if key == KEY_OBSERVE:
                lines.append('[WARN] %s is not taught (command o)' % key)
            continue
        why = joints_problem(q, kin)
        if why:
            lines.append('[FAIL] %s %s' % (key, why))
            continue
        if key != KEY_OBSERVE or kin is None or mount != 'eye_to_hand':
            continue
        try:
            T_tcp = kin.fk([float(v) for v in q]) @ flange_to_tcp(cfg)
        except ValueError as e:
            lines.append('[FAIL] observe pose not checked: %s' % e)
            continue
        ok, msg = observe_view_report(cfg, T_tcp)
        level = {True: '[ OK ]', False: '[WARN]'}.get(ok, '[SKIP]')
        lines.append('%s observe pose: %s' % (level, msg))
    return lines


def check_tasks(cfg, task: Optional[str] = None) -> List[str]:
    """Return the tasks to self-check: task when given, else L1 / L2 with a class_to_zone."""
    if task:
        return [task]
    out = [t for t in TASKS if t != 'L3' and cfg.get('tasks.%s.class_to_zone' % t, None)]
    return out or ['L1']


def selfcheck_failures(cfg, tasks: Sequence[str], mode: str) -> List[str]:
    """Return the FAIL items of task.selfcheck.self_check (config only, no hardware)."""
    from ..task.selfcheck import self_check
    seen: Dict[str, List[str]] = {}
    for task in tasks:
        try:
            _ok, msgs = self_check(cfg, task, arm=None, camera=None,
                                   mode='sim' if mode == 'sim' else 'real')
        except Exception as e:  # noqa: B902 - a crashing check is a finding, not a crash
            msgs = ['[FAIL] self check crashed: %s' % e]
        for m in msgs:
            if m.startswith('[FAIL]'):
                seen.setdefault(m, []).append(task)
    if not seen:
        return ['self check (%s, config only): no FAIL items' % ' '.join(tasks)]
    lines = ['self check (%s, config only) FAIL items:' % ' '.join(tasks)]
    for m, ts in seen.items():
        lines.append('  %s%s' % ('%s: ' % ' '.join(ts) if len(tasks) > 1 else '', m))
    return lines


def post_write_report(cfg, mode: str, tasks: Sequence[str]) -> List[str]:
    """
    Return the checks printed after a write, for the configuration the tools will use.

    Zone reachability on a kinematic arm (onsite.kinematic_arm: nothing
    connects), the observe / home pose checks and the self check FAIL items.
    """
    lines: List[str] = []
    try:
        arm = kinematic_arm(cfg)
    except (TypeError, ValueError, KeyError) as e:
        arm = None
        lines.append('[FAIL] reachability not checked (robot config): %s' % e)
    if arm is not None:
        lines += reach_report(cfg, arm)
    lines += pose_report(cfg, getattr(arm, 'kin', None))
    lines += selfcheck_failures(cfg, tasks, mode)
    return lines


def use_text(cfg, mode: str, path: Path) -> str:
    """Return how the tools pick up the written poses file."""
    overlay = overlay_file(cfg, OUT_NAME)
    if Path(path).expanduser().resolve() == Path(overlay).expanduser().resolve():
        return 'the tools merge it over config/default.yaml (site overlay)'
    if mode == 'sim':
        return 'sim rehearsal file (config/site is untouched); to use it: --config %s' % path
    return 'this is not the site overlay %s; to use it: --config %s' % (overlay, path)


# ------------------------------------------------------------------ listing / map
def _fmt_joints(q: Any) -> str:
    """Format joint angles (rad) in degrees, or say they are not taught."""
    if q is None:
        return 'not taught'
    try:
        return '%s deg' % fmt_list(np.degrees(np.asarray(q, dtype=float)), 1)
    except (TypeError, ValueError):
        return 'invalid: %r' % (q,)


def _zone_row(mark: str, name: str, d: Any) -> List[str]:
    """Return a listing row [mark, name, center, size, yaw, wall] of a zone mapping."""
    try:
        z = parse_zone(str(name), d if isinstance(d, dict) else {})
    except (TypeError, ValueError) as e:
        return [mark, str(name), 'invalid: %s' % e, '', '', '']
    return [mark, str(name), fmt_mm(z.center, 1), fmt_mm(z.size, 0),
            '%.1f' % math.degrees(z.yaw), '%.0f' % (z.wall_height * 1e3)]


def format_state(view: Dict[str, Any], changes: Dict[str, Any], tasks_cfg: Any = None) -> str:
    """Return the 'l' listing of the poses mapping (values not written yet marked '*')."""
    def mark(prefix: str) -> str:
        return '*' if any(k == prefix or k.startswith(prefix + '.') for k in changes) else ' '

    lines = []
    for key, label in ((KEY_OBSERVE, 'observe joints'), (KEY_HOME, 'home joints')):
        lines.append('%s %-15s %s' % (mark(key), label, _fmt_joints(get_dotted(view, key))))
    poly = get_dotted(view, KEY_POLYGON)
    try:
        ptxt = 'not taught' if poly is None else '%s mm' % ', '.join(fmt_mm(p, 0) for p in poly)
    except (TypeError, ValueError):
        ptxt = 'invalid: %r' % (poly,)
    lines.append('%s %-15s %s' % (mark(KEY_POLYGON), 'pick polygon', ptxt))
    tz = get_dotted(view, KEY_TABLE_Z)
    ttxt = 'not taught' if tz is None else (
        '%.1f mm' % (float(tz) * 1e3) if _finite_number(tz) else 'invalid: %r' % (tz,))
    lines.append('%s %-15s %s' % (mark(KEY_TABLE_Z), 'table z', ttxt))
    zones = get_dotted(view, 'zones')
    zones = zones if isinstance(zones, dict) else {}
    rows = [_zone_row(mark('zones.%s' % n), n, d) for n, d in zones.items()]
    if rows:
        lines.append(format_table(('', 'zone', 'center mm', 'size mm', 'yaw deg', 'wall mm'),
                                  rows))
    else:
        lines.append('  no zones taught')
    todo = suggest_zone_names(tasks_cfg, list(zones))
    if todo:
        lines.append('  task zones not taught yet: %s' % ', '.join(todo))
    deleted = [k for k, v in changes.items() if v is DELETE]
    if deleted:
        lines.append('  deleted, not written yet: %s' % ', '.join(deleted))
    note = ' (* = unsaved; w writes them)' if changes else ''
    lines.append('  %d unsaved change(s)%s' % (len(changes), note))
    return '\n'.join(lines)


def map_px(x: float, y: float) -> Tuple[int, int]:
    """Return the top-view map pixel (u, v) of base (x, y): x points up, y to the left."""
    return (int(round((MAP_Y[1] - float(y)) * MAP_SCALE)),
            int(round((MAP_X[1] - float(x)) * MAP_SCALE)))


def draw_map(zones: Dict[str, Any], taught: Sequence[str], polygon: Any, workspace: Any,
             tcp: Optional[Sequence[float]] = None, lines: Sequence[str] = ()) -> np.ndarray:
    """
    Return a BGR top view of the base frame: workspace, pick polygon, zones, last TCP.

    zones: name -> core Zone; names in taught are drawn green, others gray.
    lines: text shown in a panel below the map.
    """
    import cv2
    w = int(round((MAP_Y[1] - MAP_Y[0]) * MAP_SCALE))
    h = int(round((MAP_X[1] - MAP_X[0]) * MAP_SCALE))
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    for gx in np.arange(0.0, MAP_X[1] + 1e-9, 0.1):
        v = map_px(gx, 0.0)[1]
        cv2.line(img, (0, v), (w - 1, v), (50, 50, 50), 1)
    for gy in np.arange(-0.4, 0.4 + 1e-9, 0.1):
        u = map_px(0.0, gy)[0]
        cv2.line(img, (u, 0), (u, h - 1), (50, 50, 50), 1)
    ws = workspace if isinstance(workspace, dict) else {}
    base = map_px(0.0, 0.0)
    for key in ('r_min', 'r_max'):
        if ws.get(key) is not None:
            cv2.circle(img, base, int(round(float(ws[key]) * MAP_SCALE)), (110, 110, 110), 1,
                       cv2.LINE_AA)
    if ws.get('x') is not None and ws.get('y') is not None:
        cv2.rectangle(img, map_px(float(ws['x'][1]), float(ws['y'][1])),
                      map_px(float(ws['x'][0]), float(ws['y'][0])), (110, 110, 110), 1)
    cv2.circle(img, base, 7, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.arrowedLine(img, base, map_px(0.08, 0.0), (60, 60, 230), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, base, map_px(0.0, 0.08), (60, 200, 60), 2, cv2.LINE_AA, tipLength=0.3)
    if polygon is not None and len(polygon) >= 3:
        pts = np.array([map_px(p[0], p[1]) for p in polygon], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 220, 220), 2, cv2.LINE_AA)
    done = set(taught)
    for name, z in zones.items():
        col = (80, 220, 80) if name in done else (140, 140, 140)
        pts = np.array([map_px(x, y) for x, y in z.corners_xy()], dtype=np.int32)
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        cx, cy = float(z.center[0]), float(z.center[1])
        tip = (cx + 0.5 * z.size[0] * math.cos(z.yaw), cy + 0.5 * z.size[0] * math.sin(z.yaw))
        cv2.line(img, map_px(cx, cy), map_px(tip[0], tip[1]), col, 1, cv2.LINE_AA)
        u, v = map_px(cx, cy)
        cv2.putText(img, str(name), (u - 30, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                    cv2.LINE_AA)
    if tcp is not None:
        cv2.drawMarker(img, map_px(tcp[0], tcp[1]), (60, 60, 255), cv2.MARKER_CROSS, 16, 2)
    return text_panel(img, list(lines) or [''])


# ------------------------------------------------------------------ operator session
def _sim_joints(arm, q: Sequence[float]) -> bool:
    """Drag the sim arm to joint angles q like a simulated operator; return True."""
    sim_drag_joints(arm, [float(v) for v in q])
    return True


def _floats_or(v: Any, n: int, default: Sequence[float]) -> List[float]:
    """Return v as n finite floats, else default."""
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        return [float(x) for x in default]
    ok = len(vals) == n and all(math.isfinite(x) for x in vals)
    return vals if ok else [float(x) for x in default]


class _Teacher:
    """State of one teach_poses run (thin operator / arm I/O around the pure helpers)."""

    def __init__(self, args, s, ui, path: Path, existing: Dict[str, Any]):
        self.args, self.s, self.ui, self.path = args, s, ui, Path(path)
        self.cfg = s.cfg
        self.mode = args.mode
        self.info = s.info or {}
        self.existing = existing
        self.changes: Dict[str, Any] = {}
        self.undo: List[Tuple[str, Dict[str, Any]]] = []
        self.last_tcp: Optional[np.ndarray] = None
        self.kin = getattr(s.arm, 'kin', None)
        ws = self.cfg.get('robot.workspace', None)
        self.workspace = ws if isinstance(ws, dict) else {}
        self.mount = str(self.cfg.get('camera.mount', 'eye_to_hand'))
        self.writes = 0
        ez = existing.get('zones') if isinstance(existing, dict) else None
        self.loaded_zones = set(ez) if isinstance(ez, dict) else set()
        self.deleted: set = set()      # zone names deleted in this session (written or not)

    # ------------------------------------------------------------ state
    def view(self) -> Dict[str, Any]:
        """Return the poses mapping with the unsaved changes applied."""
        return apply_changes(self.existing, self.changes)

    def get(self, key: str, default: Any = None) -> Any:
        """Return key from the poses mapping (with changes), else from the config."""
        return effective(self.view(), self.cfg, key, default)

    def record(self, what: str, changes: Dict[str, Any]) -> None:
        """Add changes (undoable with u) and refresh the map."""
        self.undo.append((what, dict(self.changes)))
        for k, v in changes.items():
            if v is DELETE:     # a deletion drops the pending changes below its key
                for sub in [c for c in self.changes if c.startswith(k + '.')]:
                    del self.changes[sub]
            # Re-insert at the end: changes apply in the order they were taught.
            self.changes.pop(k, None)
            self.changes[k] = v
        self.ui.say('recorded %s (not written yet, w writes): %s' % (what, ', '.join(changes)))
        self.show_map()

    def dump_unsaved(self) -> None:
        """Print the unsaved changes as YAML (so nothing taught is lost silently)."""
        deleted = [k for k, v in self.changes.items() if v is DELETE]
        self.ui.say('%d unsaved change(s) were NOT written:\n%s%s' % (
            len(self.changes), yaml.safe_dump(apply_changes({}, self.changes), sort_keys=False),
            'deleted: %s\n' % ', '.join(deleted) if deleted else ''))

    def zone_sources(self, name: str) -> List[str]:
        """Return where zones.<name> is defined other than the poses file (files, --set, sim)."""
        out = zone_sources(self.cfg.sources, name, self.path)
        if not out:
            cz = self.cfg.get('zones', None)
            merged = isinstance(cz, dict) and name in cz and name not in self.loaded_zones
            if merged or (self.mode == 'sim' and name in (self.info.get('zones') or {})):
                out.append('the merged configuration (sim scenario, --config or --set)')
        return out

    def base_zones(self) -> Dict[str, Any]:
        """Return the config zones without the deleted zones that only the poses file defines."""
        cz = self.cfg.get('zones', None)
        out = dict(cz) if isinstance(cz, dict) else {}
        for name in self.deleted:
            if name in out and not self.zone_sources(name):
                del out[name]
        return out

    def show_map(self) -> None:
        """Show (headless: save) the top-view map of the poses and zones."""
        view = self.view()
        vz = view.get('zones') if isinstance(view.get('zones'), dict) else {}
        raw = deep_merge({'z': self.base_zones()}, {'z': vz})['z']
        zones = {}
        for name, d in raw.items():
            try:
                zones[str(name)] = parse_zone(str(name), d if isinstance(d, dict) else {})
            except (TypeError, ValueError):
                continue
        lines = ['top view of the base frame: x up, y to the left, grid 10 cm',
                 'green: taught zones, gray: other config zones, yellow: pick polygon',
                 'red cross: last TCP; %d unsaved change(s) -> %s'
                 % (len(self.changes), self.path.name)]
        poly = self.get(KEY_POLYGON)
        try:
            img = draw_map(zones, list(vz), poly, self.workspace, self.last_tcp, lines)
        except (TypeError, ValueError, IndexError) as e:
            self.ui.say('map not drawn: %s' % e)
            return
        self.ui.show('teach_poses', img)
        if not self.ui.headless:
            self.ui.key(1)

    # ------------------------------------------------------------ arm / operator
    def read_arm(self, what: str):
        """Read the averaged arm pose while the operator holds still; None if unusable."""
        limit = float(self.args.max_jitter_mm)
        for _ in range(3):
            pose, dev = read_pose_averaged(self.s.arm, self.s.clock, max(1, self.args.samples),
                                           READ_DT)
            if pose is None:
                self.ui.say('no fresh arm feedback (arm on? CAN link up?): %s not recorded'
                            % what)
                return None
            if dev * 1e3 <= limit:
                self.last_tcp = np.asarray(pose.T_base_tcp, dtype=float)[:3, 3].copy()
                return pose
            self.ui.say('the arm moved %.1f mm while it was read (limit %.1f mm)'
                        % (dev * 1e3, limit))
            if not operator_action(self.ui, self.mode, 'Hold the arm still (or let go of it)'):
                break
        self.ui.say('%s not recorded' % what)
        return None

    def touch(self, text: str, target: Optional[Sequence[float]] = None
              ) -> Optional[np.ndarray]:
        """
        Ask the operator to touch a point with the fingertips; return the TCP position (m).

        Sim mode: the simulated operator drags the sim arm to target.

        Returns None when the operator cancels or the arm cannot be read.
        """
        action = None
        if self.mode == 'sim':
            if target is None:
                self.ui.say('[sim operator] nothing to touch for: %s' % text)
                return None
            x, y, z = (float(v) for v in target)
            T = top_down_T(x, y, z, neutral_closing_angle(x, y))
            action = functools.partial(sim_drag_tcp, self.s.arm, self.cfg, T)
        if not operator_action(self.ui, self.mode, text, action):
            return None
        pose = self.read_arm('the touch')
        if pose is None:
            return None
        T = np.asarray(pose.T_base_tcp, dtype=float)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(T[2, 2])))))
        self.ui.say('  TCP %s mm, tool %.0f deg from vertical' % (fmt_mm(T[:3, 3]), tilt))
        return T[:3, 3].copy()

    def joint_pose(self, what: str, text: str, sim_q: Optional[Sequence[float]]):
        """Ask the operator to drag the arm to a pose (sim: to sim_q); return (q, pose) or None."""
        action = None
        if self.mode == 'sim':
            if sim_q is None:
                self.ui.say('[sim operator] no simulated %s' % what)
                return None
            action = functools.partial(_sim_joints, self.s.arm, sim_q)
        if not operator_action(self.ui, self.mode, text, action):
            self.ui.say('%s not recorded' % what)
            return None
        pose = self.read_arm(what)
        if pose is None:
            return None
        q, notes = clip_to_limits([_r(v, 5) for v in pose.joints], self.kin)
        for n in notes:
            self.ui.say('note: %s' % n)
        why = joints_problem(q, self.kin)
        if why:
            self.ui.say('%s not recorded: the joints are %s' % (what, why))
            return None
        tcp = fmt_mm(pose.T_base_tcp[:3, 3])
        self.ui.say('%s: joints %s deg, TCP %s mm' % (what, fmt_list(np.degrees(q), 1), tcp))
        return q, pose

    def ask_floats(self, text: str, n: int, default: Sequence[float],
                   bounds: Tuple[float, float]) -> Optional[List[float]]:
        """Ask for n numbers within bounds (Enter = default); None when cancelled."""
        dtxt = ' '.join('%.3f' % float(v) for v in default)
        for _ in range(3):
            ans = self.ui.prompt(text, dtxt)
            if self.ui.eof or ans.strip().lower().startswith('q'):
                return None
            vals = parse_floats(ans, n)
            if vals is not None and all(bounds[0] <= v <= bounds[1] for v in vals):
                return vals
            self.ui.say('enter %d number(s) in [%.2f, %.2f] m separated by spaces, e.g. %s '
                        '(q = cancel)' % (n, bounds[0], bounds[1], dtxt))
        return None

    # ------------------------------------------------------------ commands
    def cmd_observe(self, _arg: str = '') -> None:
        """Record the current joints as the observe pose."""
        if self.mount == 'eye_to_hand':
            text = ('Drag the arm to the observe pose: parked where the camera sees the whole '
                    'pick area (eye_to_hand)')
        else:
            text = ('Drag the arm to the observe pose: wrist camera looking down at the whole '
                    'pick area (eye_in_hand)')
        got = self.joint_pose('observe pose', text, self.info.get('observe_joints'))
        if got is None:
            return
        q, pose = got
        if self.mount == 'eye_to_hand':
            ok, msg = observe_view_report(self.cfg, pose.T_base_tcp, self.get(KEY_POLYGON),
                                          self.get(KEY_TABLE_Z))
            self.ui.say('%s%s' % ({True: 'OK: ', False: 'WARNING: '}.get(ok, ''), msg))
        else:
            self.ui.say('eye_in_hand: check now (view, in another terminal) that the camera sees '
                        'the whole pick area from this pose')
        self.record('observe pose', {KEY_OBSERVE: q})

    def cmd_home(self, _arg: str = '') -> None:
        """Record the current joints as the home pose."""
        home = self.cfg.get(KEY_HOME, None)
        got = self.joint_pose('home pose', 'Drag the arm to the home pose (where it goes at the '
                              'end of a run, clear of the table, zones and camera view)',
                              [0.0] * 6 if home is None else home)
        if got is None:
            return
        q, pose = got
        tz = self.get(KEY_TABLE_Z)
        if _finite_number(tz):
            clear = float(self.workspace.get('min_table_clearance', 0.0) or 0.0)
            low = min(float(pose.T_base_tcp[2, 3]), float(pose.T_base_flange[2, 3]))
            if low < float(tz) + clear:
                self.ui.say('WARNING: the gripper is below the table z + clearance at this pose; '
                            'the safety layer will refuse to move there')
        self.record('home pose', {KEY_HOME: q})

    def cmd_zone(self, arg: str = '') -> None:
        """Teach one zone: center touch, size, wall height and an optional yaw touch."""
        view = self.view()
        zones = view.get('zones') if isinstance(view.get('zones'), dict) else {}
        tasks = self.cfg.get('tasks', None)
        names = task_zone_names(tasks)
        todo = suggest_zone_names(tasks, list(zones))
        if names:
            self.ui.say('zones in tasks.*.class_to_zone: %s' % ', '.join(
                n + ('' if n in todo else ' (taught)') for n in names))
        else:
            self.ui.say('no zone names in tasks.*.class_to_zone yet (config/site/tasks.yaml): '
                        'use the names you will map the classes to')
        name = arg or self.ui.prompt('zone name (q = cancel)', todo[0] if todo else '')
        if self.ui.eof or not name or name.lower() == 'q':
            self.ui.say('zone not recorded')
            return
        why = zone_name_problem(name)
        if why:
            self.ui.say(why)
            return
        if names and name not in names:
            self.ui.say('note: no task maps a class to %s (check the spelling)' % name)
        old = zones.get(name) if isinstance(zones.get(name), dict) else {}
        truth = None
        if self.mode == 'sim':
            truth = (self.info.get('zones') or {}).get(name)
            if truth is None:
                self.ui.say('[sim operator] the scenario has no zone %s (it has %s)' % (
                    name, ', '.join(sorted(self.info.get('zones') or {}))))
        c = self.touch('Touch the floor at the center of zone %s with the closed fingertips'
                       % name, None if truth is None else truth.center)
        if c is None:
            self.ui.say('zone %s not recorded' % name)
            return
        size = self.ask_floats('zone %s size along its x and y in m' % name, 2,
                               _floats_or(old.get('size'), 2, DEFAULT_ZONE_SIZE),
                               ZONE_SIZE_RANGE)
        wall = None if size is None else self.ask_floats(
            'wall height above the zone floor in m (0 = flat mat)', 1,
            _floats_or([old.get('wall_height', 0.0)], 1, [0.0]), WALL_RANGE)
        if wall is None:
            self.ui.say('zone %s not recorded' % name)
            return
        old_yaw = _floats_or([old.get('yaw', 0.0)], 1, [0.0])[0]
        axis = None
        if self.ui.confirm('Zone yaw: touch a second point on the zone x axis (the direction of '
                           'its first size value)? n = keep yaw %.1f deg'
                           % math.degrees(old_yaw), default=False):
            target = None
            if truth is not None:
                target = [float(truth.center[0]) + SIM_YAW_DIST * math.cos(truth.yaw),
                          float(truth.center[1]) + SIM_YAW_DIST * math.sin(truth.yaw),
                          float(truth.center[2])]
            axis = self.touch('Touch the zone %s floor at the middle of a side that the zone x '
                              'axis points to' % name, target)
            if axis is None:
                self.ui.say('zone %s not recorded' % name)
                return
        try:
            zone = zone_from_touches(c, size, wall[0], axis, old_yaw)
        except ValueError as e:
            self.ui.say('zone %s not recorded: %s' % (name, e))
            return
        self.ui.say('zone %s: center %s mm, size %s mm, yaw %.1f deg, wall %.0f mm' % (
            name, fmt_mm(zone['center']), fmt_mm(zone['size'], 0), math.degrees(zone['yaw']),
            zone['wall_height'] * 1e3))
        for w in zone_warnings(name, zone, self.workspace, self.get(KEY_POLYGON),
                               self.get(KEY_TABLE_Z)):
            self.ui.say('WARNING: ' + w)
        self.record('zone %s' % name, zone_changes(name, zone))

    def cmd_polygon(self, _arg: str = '') -> None:
        """Teach the pick polygon and the table z from 4 touched table corners."""
        self.ui.say('Touch the 4 corners of the pick area on the table with the closed '
                    'fingertips, in order around it (clockwise or counter-clockwise).')
        sim_poly = self.info.get('pick_polygon') if self.mode == 'sim' else None
        table_z = float(getattr(self.s.world, 'table_z', 0.0) or 0.0)
        touches = []
        for k in range(4):
            target = None
            if sim_poly and k < len(sim_poly):
                target = [float(sim_poly[k][0]), float(sim_poly[k][1]), table_z]
            p = self.touch('Corner %d of 4: touch the table at the corner' % (k + 1), target)
            if p is None:
                self.ui.say('pick polygon not recorded')
                return
            touches.append(p)
        changes, problems, notes = pick_polygon_from_touches(touches, self.workspace)
        for n in notes:
            self.ui.say(n)
        if changes is None:
            self.ui.say('pick polygon not recorded:')
            for p in problems:
                self.ui.say('  - %s' % p)
            return
        poly = changes[KEY_POLYGON]
        self.ui.say('pick polygon %s mm, area %.3f m^2, table z %.1f mm' % (
            ', '.join(fmt_mm(p, 0) for p in poly), abs(polygon_area(poly)),
            changes[KEY_TABLE_Z] * 1e3))
        self.record('pick polygon', changes)

    def cmd_delete(self, arg: str = '') -> None:
        """Delete a zone from the poses file (u undoes it, w writes it)."""
        view = self.view()
        zones = view.get('zones') if isinstance(view.get('zones'), dict) else {}
        if zones:
            self.ui.say('zones in %s: %s' % (self.path.name, ', '.join(sorted(zones))))
        name = arg or self.ui.prompt('zone to delete (q = cancel)', '')
        if self.ui.eof or not name or name.lower() == 'q':
            self.ui.say('nothing deleted')
            return
        others = self.zone_sources(name)
        if name not in zones:
            if others:
                self.ui.say('zone %s is not in %s; it is defined in %s: remove it there '
                            '(teach_poses only edits %s)'
                            % (name, self.path, ', '.join(others), self.path.name))
            else:
                self.ui.say('no zone %s in %s (zones: %s)'
                            % (name, self.path, ', '.join(sorted(zones)) or 'none'))
            return
        self.deleted.add(name)
        self.record('delete zone %s' % name, {'zones.%s' % name: DELETE})
        for u in zone_users(self.cfg.get('tasks', None), name):
            self.ui.say('WARNING: %s still sends objects to %s: change config/site/tasks.yaml '
                        'too (the self check fails until then)' % (u, name))
        if others:
            self.ui.say('WARNING: zone %s is also defined in %s. The YAML merge cannot delete '
                        'keys of other files, so after w the zone still exists from there: '
                        'remove it there too.' % (name, ', '.join(others)))

    def cmd_list(self, _arg: str = '') -> None:
        """Print the taught values and refresh the map."""
        self.ui.say('%s:' % self.path)
        self.ui.say(format_state(self.view(), self.changes, self.cfg.get('tasks', None)))
        self.show_map()

    def cmd_undo(self, _arg: str = '') -> None:
        """Undo the last recorded change (not yet written)."""
        if not self.undo:
            self.ui.say('nothing to undo%s' % (
                ' (already written; the previous file is kept as a backup)' if self.writes
                else ''))
            return
        what, snapshot = self.undo.pop()
        self.changes = snapshot
        self.ui.say('undone: %s' % what)
        self.show_map()

    def cmd_write(self, _arg: str = '') -> bool:
        """Merge the changes into the poses file (backup kept), then run the checks."""
        if not self.changes:
            self.ui.say('nothing to write: no unsaved changes')
            return True
        try:
            existing = read_yaml_or_empty(self.path)
        except (OSError, ValueError, yaml.YAMLError) as e:
            self.ui.say('NOT written: cannot read the current %s (%s); fix or move it, then w '
                        'again' % (self.path, e))
            return False
        data = apply_changes(existing, self.changes)
        for p in validate_poses(data, self.kin, self.workspace):
            self.ui.say('WARNING: %s' % p)
        keys = [k for k, v in self.changes.items() if v is not DELETE]
        deleted = [k for k, v in self.changes.items()
                   if v is DELETE and get_dotted(data, k, _MISSING) is _MISSING]
        try:
            backup = write_site_yaml(self.path, data, header=file_header(wall_stamp()))
        except OSError as e:
            self.ui.say('NOT written: %s' % e)
            return False
        self.ui.say(write_report(self.path, keys, backup))
        if deleted:
            self.ui.say('  keys deleted:\n%s' % '\n'.join('    %s' % k for k in deleted))
        kept = [k for k in dotted_keys(data) if k not in keys]
        if kept:
            self.ui.say('  kept from the previous file: %s' % ', '.join(kept))
        self.ui.say(use_text(self.cfg, self.mode, self.path))
        self.existing, self.changes, self.undo = data, {}, []
        self.writes += 1
        base = dict(self.cfg.data)
        if self.deleted:
            base['zones'] = self.base_zones()
        cfg2 = Config(deep_merge(base, data), self.cfg.root,
                      self.cfg.sources + [str(self.path)])
        for line in post_write_report(cfg2, self.mode, check_tasks(cfg2, self.args.task)):
            self.ui.say(line)
        self.show_map()
        return True

    def cmd_quit(self, _arg: str = '') -> bool:
        """Return True when the tool may exit (offers to write unsaved changes first)."""
        if not self.changes:
            return True
        self.ui.say('%d unsaved change(s): %s' % (len(self.changes), ', '.join(self.changes)))
        if self.ui.confirm('Write them to %s before quitting?' % self.path, default=True) and \
                self.cmd_write():
            return True
        if self.ui.eof:
            self.dump_unsaved()
            return True
        if self.ui.confirm('Quit WITHOUT writing (the unsaved changes are lost)?', default=False):
            self.dump_unsaved()
            return True
        return False

    def dispatch(self, line: str) -> bool:
        """Run one command line; return False when the tool should exit."""
        parts = str(line or '').strip().split(None, 1)
        if not parts:
            return True
        word, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else '')
        cmd = COMMANDS.get(word, word if word in COMMANDS.values() else '')
        if cmd == 'quit':
            return not self.cmd_quit()
        fn = {'observe': self.cmd_observe, 'home': self.cmd_home, 'zone': self.cmd_zone,
              'polygon': self.cmd_polygon, 'delete': self.cmd_delete, 'list': self.cmd_list,
              'undo': self.cmd_undo, 'write': self.cmd_write}.get(cmd)
        if fn is None:
            self.ui.say('unknown command %r; commands: %s' % (str(line).strip(), HELP))
            return True
        try:
            fn(arg)
        except Exception as e:  # noqa: B902 - keep the session and its unsaved changes alive
            _LOG.exception('command %s failed', cmd)
            self.ui.say('command %s failed: %s' % (cmd, e))
        return True

    def run(self) -> int:
        """Run the command loop; return the exit code."""
        ui = self.ui
        ui.say('teach_poses: mode %s, camera %s, writes %s' % (self.mode, self.mount, self.path))
        if self.existing:
            ui.say('the existing file is merged: its %d keys are kept unless taught again'
                   % len(dotted_keys(self.existing)))
        if self.mode == 'real':
            ui.say('This tool only reads the arm; it never moves it.')
            teach_hint(ui, self.s.arm)
        pose = read_pose(self.s.arm)
        if pose is None:
            ui.say('WARNING: no arm feedback yet (arm on? CAN link up?)')
        else:
            self.last_tcp = np.asarray(pose.T_base_tcp, dtype=float)[:3, 3].copy()
            ui.say('current TCP %s mm' % fmt_mm(self.last_tcp))
        self.show_map()
        if ui.headless:
            ui.say('map preview: %s' % ui.preview_path('teach_poses'))
        ui.say(ORDER_HINT)
        ui.say(HELP if ui.headless else WINDOW_HELP)
        try:
            while True:
                line = next_command(ui, HELP)
                if line is None:
                    continue
                if not self.dispatch(line):
                    break
        except KeyboardInterrupt:
            ui.say('interrupted')
            if self.changes:
                self.dump_unsaved()
            return 130
        if self.mode == 'real' and is_teaching(self.s.arm):
            ui.say('The arm is still in drag-teach mode: press the teach button on the arm to '
                   'leave it before a tool moves the arm.')
        return 0


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of teach_poses."""
    p = argparse.ArgumentParser(prog='teach_poses',
                                description=__doc__.split('\n\n')[0].strip())
    add_common_args(p, default_task=None)
    add_onsite_args(p, site=True)
    p.add_argument('--samples', type=int, default=8,
                   help='arm readings averaged per taught pose (default 8)')
    p.add_argument('--max-jitter-mm', type=float, default=3.0,
                   help='reject a reading when the TCP moved more than this while it was read')
    return p


def main(argv=None, ui=None) -> int:
    """Run teach_poses; return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.samples < 1 or not args.max_jitter_mm > 0:
        parser.error('--samples must be >= 1 and --max-jitter-mm > 0')
    if args.mode == 'replay':
        parser.error('teach_poses reads the arm: use --mode real (or --mode sim to rehearse)')
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    cfg = cfg_world_info[0]
    ui = make_ui(args, cfg, 'teach_poses', ui)
    path = output_path(cfg, args.mode, args.site_dir, overlay_file(cfg, OUT_NAME))
    try:
        existing = read_yaml_or_empty(path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        ui.say('cannot read %s (%s): fix or move the file, then run teach_poses again' % (path, e))
        ui.close()
        return 2
    rc = 2
    # Real mode: connect without enabling (poses are read in drag-teach mode); no camera.
    with open_session(args, arm=True, camera=False, perception=False,
                      connect=args.mode != 'real', cfg_world_info=cfg_world_info) as s:
        if connect_arm_readonly(s, ui, required=True, why='arm poses'):
            rc = _Teacher(args, s, ui, path, existing).run()
    ui.close()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

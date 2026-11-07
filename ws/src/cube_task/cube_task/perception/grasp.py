"""
Top-down grasp and place planning.

All poses are TCP poses (fingertip center) in the robot base frame, built with
``core.transforms.top_down_T`` (tool z = base -Z, tool y = finger closing axis at
``closing_angle`` from base +X). Units: meters and radians.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import choose_closing_angle, neutral_closing_angle, top_down_T
from ..core.types import Detection, GraspPlan, PlacePlan, Zone

_LOG = logging.getLogger('cube_task.perception.grasp')

#: length / width below this ratio counts as round (closing direction is free).
ROUND_RATIO = 1.06  # below this the minor axis is too uncertain to aim at
#: length / width below this ratio counts as a square block (L1, 90 deg symmetric).
SQUARE_RATIO = 1.15
#: TCP stays this far above table z + robot.workspace.min_table_clearance.
TABLE_MARGIN = 0.001
#: transit TCP height above the highest zone wall top.
WALL_TOP_CLEARANCE = 0.08
#: the bottom of a held object passes this far above every obstacle in transit.
HELD_CLEARANCE = 0.05
#: release over a walled bin with the object bottom this far above the rim.
WALL_RELEASE_CLEARANCE = 0.02
#: finger band for neighbor checks (m): half width across the closing axis (as
#: task.selection.FINGER_HALF_BAND), reach beyond the open finger face, and the
#: gap beyond the object faces that is ignored (noise of touching footprints).
FINGER_HALF_BAND = 0.015
FINGER_OUTER = 0.015
NEIGHBOR_CLEARANCE = 0.002
#: perceived table z farther than this from perception.table.expected_z is reported (m).
TABLE_Z_WARN_TOL = 0.01
#: a released object's bottom passes this far above the tallest object already in the zone.
STACK_CLEARANCE = 0.01
#: release slots keep the held object's footprint this far inside the zone edge (m).
SLOT_MARGIN = 0.015
#: a zone axis gets two release slots only when they are at least this far off center (m).
SLOT_MIN_OFFSET = 0.01


def _required_float(cfg, key: str) -> float:
    """Read a safety-relevant number; raise ValueError when missing or not finite."""
    v = cfg.get(key, None)
    if v is None:
        raise ValueError('config key %s is required' % key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError('config key %s must be a number, got %r' % (key, v))
    if not math.isfinite(f):
        raise ValueError('config key %s must be finite, got %r' % (key, v))
    return f


def _float(cfg, key: str, default: float) -> float:
    """Read a tuning number with a default equal to the one in default.yaml."""
    v = cfg.get(key, default)
    if v is None:
        v = default
    f = float(v)
    if not math.isfinite(f):
        raise ValueError('config key %s must be finite, got %r' % (key, v))
    return f


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def max_gripper_width(cfg) -> float:
    """Return robot.gripper.max_width (m); raise ValueError if missing or <= 0."""
    w = _required_float(cfg, 'robot.gripper.max_width')
    if w <= 0.0:
        raise ValueError('robot.gripper.max_width must be > 0, got %r' % w)
    return w


def expected_table_z(cfg) -> Optional[float]:
    """
    Return perception.table.expected_z (m, base z of the table taught by TCP touches).

    None when it is not set; ValueError when it is set but not a finite number.
    """
    v = cfg.get('perception.table.expected_z', None)
    if v is None:
        return None
    try:
        z = float(v)
    except (TypeError, ValueError):
        raise ValueError('perception.table.expected_z must be a number or null, got %r' % (v,))
    if not math.isfinite(z):
        raise ValueError('perception.table.expected_z must be finite, got %r' % (v,))
    return z


def table_floor_z(perceived_z: float, expected_z: Optional[float]) -> float:
    """
    Return the table z to keep the fingertips above (m, base frame).

    The taught table (expected_z) is a lower bound: a camera that sees the table
    too low (extrinsics z offset) must not lower the floor below the real table.
    """
    if expected_z is None:
        return float(perceived_z)
    return max(float(perceived_z), float(expected_z))


def table_z_warning(perceived_z: Optional[float], expected_z: Optional[float],
                    tol: float = TABLE_Z_WARN_TOL) -> str:
    """Return a warning when the perceived table z is off the taught one by > tol, else ''."""
    if perceived_z is None or expected_z is None or not _finite(perceived_z):
        return ''
    d = float(perceived_z) - float(expected_z)
    if abs(d) <= tol:
        return ''
    return ('table z differs from the taught table (perception.table.expected_z) by %+.0f mm: '
            'extrinsics z offset?' % (d * 1e3))


def detection_table_z(det: Detection) -> Optional[float]:
    """
    Table z (base frame, m) under a detection, or None when unknown.

    Uses ``det.extra['table_z']``; falls back to ``top_z - height``.
    """
    tz = (det.extra or {}).get('table_z', None)
    if _finite(tz):
        return float(tz)
    top = det.top_z
    if top is None and det.center_base is not None and len(det.center_base) >= 3:
        top = det.center_base[2]
    if _finite(top) and _finite(det.height):
        return float(top) - float(det.height)
    return None


def transit_height(cfg, zones: Iterable[Zone], table_z: float, held_offset: float = 0.0,
                   obstacles_top_z: Optional[float] = None) -> float:
    """
    Return the TCP z (base frame, m) for moves between the pick area and a zone.

    max of: grasp.transit_z_min; table_z + grasp.transit_clearance; highest zone
    wall top + 0.08; highest obstacle (table, wall tops, obstacles_top_z) +
    held_offset + 0.05, so the bottom of the held object (held_offset below the
    TCP) clears everything by 5 cm.
    """
    if not _finite(table_z):
        raise ValueError('transit_height: table_z must be finite, got %r' % (table_z,))
    if not _finite(held_offset) or held_offset < 0.0:
        raise ValueError('transit_height: held_offset must be >= 0, got %r' % (held_offset,))
    z = max(_float(cfg, 'grasp.transit_z_min', 0.15),
            float(table_z) + _float(cfg, 'grasp.transit_clearance', 0.12))
    highest = float(table_z)
    for zone in zones:
        top = float(zone.center[2]) + max(0.0, float(zone.wall_height))
        highest = max(highest, top)
        if zone.wall_height > 0.0:
            z = max(z, top + WALL_TOP_CLEARANCE)
    if _finite(obstacles_top_z):
        highest = max(highest, float(obstacles_top_z))
    return max(z, highest + float(held_offset) + HELD_CLEARANCE)


def _identity_plan(det: Detection, reason: str, close_width: float) -> GraspPlan:
    eye = np.eye(4)
    return GraspPlan(target=det, T_pre=eye.copy(), T_grasp=eye.copy(), T_lift=eye.copy(),
                     open_width=0.0, close_width=close_width, expected_width=0.0,
                     closing_angle=0.0, feasible=False, reason=reason)


class GraspPlanner:
    """
    Plan a vertical top-down grasp for one detection.

    Closing axis across the object's minor axis (``yaw + pi/2``), turned by the
    object's symmetry (pi, or pi/2 for square blocks: every detection in L1 and
    detections labeled with a tasks.L1 label in other tasks, length / width <
    1.15) to the equivalent angle closest to the neutral wrist angle. Other
    near-round objects (length / width < 1.2) use the neutral angle directly.
    Blocks with touching neighbors (det.extra['_neighbor_xy'], set by the
    perception pipeline) close along the other face pair when that keeps the
    fingers off the neighbors (finger_hits); the expected width is then the
    extent along that axis.
    """

    def __init__(self, cfg, task: str):
        self.task = str(task)
        self.max_width = max_gripper_width(cfg)
        mow = cfg.get('grasp.max_object_width', None)
        self.max_object_width = (self.max_width - 0.005 if mow is None else float(mow))
        if not (0.0 < self.max_object_width <= self.max_width):
            raise ValueError('grasp.max_object_width must be in (0, max_width], got %r'
                             % self.max_object_width)
        self.open_margin = _float(cfg, 'robot.gripper.open_margin', 0.02)
        self.close_width = _float(cfg, 'robot.gripper.close_width', 0.0)
        self.table_clearance = _required_float(cfg, 'robot.workspace.min_table_clearance')
        # The taught table height bounds the grasp floor from below (extrinsics z offset).
        self.expected_table_z = expected_table_z(cfg)
        self.approach_height = _float(cfg, 'grasp.approach_height', 0.08)
        self.lift_height = _float(cfg, 'grasp.lift_height', 0.10)
        self.depth_ratio = _float(cfg, 'grasp.grasp_depth_ratio', 0.5)
        self.min_depth = _float(cfg, 'grasp.min_grasp_depth', 0.008)
        self.max_depth = _float(cfg, 'grasp.max_grasp_depth', 0.030)
        if self.min_depth > self.max_depth:
            raise ValueError('grasp.min_grasp_depth > grasp.max_grasp_depth')
        self.square_blocks = bool(cfg.get('grasp.block_labels_square', True))
        self.use_cap_center = bool(cfg.get('grasp.use_cap_center', True))
        self.block_labels = set(str(x) for x in (cfg.get('tasks.L1.labels', []) or []))

    def plan(self, det: Detection) -> GraspPlan:
        """Return a GraspPlan; feasible=False with a reason when it cannot be done."""
        missing = [n for n in ('yaw', 'length', 'width', 'height') if not _finite(getattr(det, n))]
        c = det.center_base
        if c is None or len(c) < 2 or not (_finite(c[0]) and _finite(c[1])):
            missing.insert(0, 'center_base')
        top = det.top_z
        if not _finite(top) and c is not None and len(c) >= 3:
            top = c[2]
        if not _finite(top):
            missing.append('top_z')
        table_z = detection_table_z(det)
        if table_z is None:
            missing.append('table_z')
        if missing:
            return _identity_plan(det, 'missing_geometry: ' + ','.join(missing), self.close_width)
        x, y, top = float(c[0]), float(c[1]), float(top)
        length, width = float(det.length), float(det.width)
        height = float(det.height)
        if width <= 0.0 or length <= 0.0 or height <= 0.0:
            return _identity_plan(det, 'bad_geometry: length=%.4f width=%.4f height=%.4f'
                                  % (length, width, height), self.close_width)
        if length < width:
            length, width = width, length
        # Grasp at the center of the topmost points when perception provides it.
        # For round objects seen by an oblique camera the hidden far side biases the
        # footprint center toward the camera (about 0.15 r), while the top cap sits
        # straight above the true center.
        cap = det.extra.get('cap_center_xy') if det.extra else None
        if (self.use_cap_center and cap is not None and len(cap) >= 2
                and _finite(cap[0]) and _finite(cap[1])
                and math.hypot(float(cap[0]) - x, float(cap[1]) - y) <= 0.5 * width):
            x, y = float(cap[0]), float(cap[1])
        ratio = length / width
        neutral = neutral_closing_angle(x, y)
        is_block = self.task == 'L1' or det.label in self.block_labels
        if self.square_blocks and is_block and ratio < SQUARE_RATIO:
            # Square block: grip across two faces, any of the 4 face pairs works.
            closing = choose_closing_angle(float(det.yaw) + math.pi / 2.0, x, y, math.pi / 2.0)
            expected = width
            kind = 'square'
            closing, expected = self._avoid_neighbors(det, closing, x, y, expected, length)
        elif ratio < ROUND_RATIO:
            closing = neutral
            expected = length
            kind = 'round'
        else:
            closing = choose_closing_angle(float(det.yaw) + math.pi / 2.0, x, y, math.pi)
            # A partly hidden round cross-section (fruit lying on its side) makes the
            # visible width too small; the height is a second estimate of it.
            expected = width if is_block else max(width, min(length, height))
            kind = 'elongated'
            if is_block:
                # A block next to a taller one looks elongated (strip hidden behind it).
                closing, expected = self._avoid_neighbors(det, closing, x, y, expected, length)
        depth = min(max(self.depth_ratio * height, self.min_depth), self.max_depth)
        z_floor = (table_floor_z(table_z, self.expected_table_z) + self.table_clearance
                   + TABLE_MARGIN)
        z_grasp = max(top - depth, z_floor)
        plan = GraspPlan(
            target=det,
            T_pre=top_down_T(x, y, top + self.approach_height, closing),
            T_grasp=top_down_T(x, y, z_grasp, closing),
            T_lift=top_down_T(x, y, z_grasp + self.lift_height, closing),
            open_width=min(self.max_width, expected + self.open_margin),
            close_width=self.close_width,
            expected_width=expected,
            closing_angle=closing)
        if expected > self.max_object_width:
            plan.feasible = False
            plan.reason = 'too_wide: %.4f m > %.4f m' % (expected, self.max_object_width)
        elif z_grasp >= top:
            plan.feasible = False
            plan.reason = 'too_flat: top %.4f m is not above the lowest TCP z %.4f m' % (
                top, z_floor)
        _LOG.debug('grasp %s kind=%s closing=%.3f z=%.4f feasible=%s %s', det.label, kind,
                   closing, z_grasp, plan.feasible, plan.reason)
        return plan

    def finger_hits(self, det: Detection, closing: float, x: float, y: float,
                    expected: float) -> int:
        """
        Count neighbor footprint points in the band the open fingers sweep.

        Neighbor points: det.extra['_neighbor_xy'] (base XY, set by the
        perception pipeline for instances next to det). The band reaches
        NEIGHBOR_CLEARANCE beyond the object faces up to the outer finger
        side (open width / 2 + FINGER_OUTER) along the closing axis and
        FINGER_HALF_BAND across it, on both sides.
        """
        nb = det.extra.get('_neighbor_xy') if det.extra else None
        if nb is None:
            return 0
        nb = np.asarray(nb, dtype=np.float64).reshape(-1, 2)
        if nb.shape[0] == 0:
            return 0
        open_w = min(self.max_width, expected + self.open_margin)
        ax = np.array([math.cos(closing), math.sin(closing)])
        rel = nb - np.array([x, y])
        along = np.abs(rel @ ax)
        across = np.abs(rel @ np.array([-ax[1], ax[0]]))
        band = ((across <= FINGER_HALF_BAND) & (along >= expected / 2.0 + NEIGHBOR_CLEARANCE)
                & (along <= open_w / 2.0 + FINGER_OUTER))
        return int(np.count_nonzero(band))

    def _avoid_neighbors(self, det: Detection, closing: float, x: float, y: float,
                         expected: float, alt_expected: float) -> Tuple[float, float]:
        """
        Turn a block's closing axis by 90 deg when that frees the fingers.

        expected / alt_expected: object extent along the current / the turned
        closing axis. Returns (closing angle, expected width): the turned axis
        (closest to the neutral wrist angle) when it has fewer neighbor points
        in the finger band (finger_hits) and alt_expected fits the gripper,
        else the inputs unchanged.
        """
        hits = self.finger_hits(det, closing, x, y, expected)
        if hits == 0 or alt_expected > self.max_object_width:
            return closing, expected
        alt = choose_closing_angle(closing + math.pi / 2.0, x, y, math.pi)
        alt_hits = self.finger_hits(det, alt, x, y, alt_expected)
        if alt_hits < hits:
            _LOG.debug('grasp %s: closing %.3f -> %.3f rad avoids a neighbor (%d -> %d points)',
                       det.label, closing, alt, hits, alt_hits)
            return alt, alt_expected
        return closing, expected


def held_offset_of(plan: GraspPlan) -> float:
    """Distance (m) from the grasp TCP down to the object bottom (table z)."""
    tz = detection_table_z(plan.target)
    if tz is None:
        raise ValueError('held_offset_of: table z of the target is unknown')
    return max(0.0, float(plan.T_grasp[2, 3]) - tz)


def release_slots(zone: Zone, footprint: float, margin: float = SLOT_MARGIN
                  ) -> List[Tuple[float, float]]:
    """
    Return release xy positions (base frame, m) inside zone for one object.

    footprint: the object's largest horizontal extent (m). Up to 2 x 2
    positions along the zone axes, as far apart as the zone allows while the
    footprint stays margin inside the zone edge; diagonal pairs first. An
    axis too short for two positions SLOT_MIN_OFFSET or more off center keeps
    the zone center on that axis.
    """
    offs = []
    for size in zone.size[:2]:
        h = (float(size) - float(footprint)) / 2.0 - float(margin)
        offs.append(h if h >= SLOT_MIN_OFFSET else 0.0)
    hx, hy = offs
    if hx > 0.0 and hy > 0.0:
        loc = [(-hx, -hy), (hx, hy), (-hx, hy), (hx, -hy)]
    elif hx > 0.0:
        loc = [(-hx, 0.0), (hx, 0.0)]
    elif hy > 0.0:
        loc = [(0.0, -hy), (0.0, hy)]
    else:
        loc = [(0.0, 0.0)]
    c, s = math.cos(zone.yaw), math.sin(zone.yaw)
    cx, cy = float(zone.center[0]), float(zone.center[1])
    return [(cx + c * a - s * b, cy + s * a + c * b) for a, b in loc]


class PlacePlanner:
    """
    Plan the release above a place zone.

    The object bottom is released ``zone.release_height`` (else
    ``place.release_height``) above the zone floor, at least 2 cm above the
    wall rim of a walled bin and at least STACK_CLEARANCE above the tallest
    object already released into the zone. The TCP is ``held_offset`` above
    the object bottom. With the held object's footprint and the objects
    already in the zone (``occupied``), the release xy is the first free
    position of the zone (center first when ``center_first``, then
    release_slots), so later objects do not land on earlier ones
    (place.spread_releases, default true).
    """

    def __init__(self, cfg):
        self.max_width = max_gripper_width(cfg)
        self.release_height = _float(cfg, 'place.release_height', 0.06)
        self.retreat_height = _float(cfg, 'place.retreat_height', 0.10)
        ow = cfg.get('place.open_width', None)
        self.open_width = self.max_width if ow is None else min(float(ow), self.max_width)
        self.spread = bool(cfg.get('place.spread_releases', True))
        self.slot_margin = _float(cfg, 'place.slot_margin', SLOT_MARGIN)

    def release_xy(self, zone: Zone, footprint: Optional[float] = None,
                   occupied: Sequence[Sequence[float]] = (), center_first: bool = True
                   ) -> Tuple[float, float]:
        """
        Return the release xy (base frame, m) for an object of this footprint.

        occupied: (x, y, footprint, height) of the objects already released into
        zone. The first candidate (zone center when center_first, then
        release_slots) whose footprint does not overlap an occupied one wins;
        when all overlap, the one with the most clearance. Without a footprint
        (or with place.spread_releases false) the zone center.
        """
        center = (float(zone.center[0]), float(zone.center[1]))
        if not self.spread or not _finite(footprint) or float(footprint) <= 0.0:
            return center
        fp = float(footprint)
        cands = ([center] if center_first else []) + release_slots(zone, fp, self.slot_margin)
        occ = [o for o in occupied if len(o) >= 3 and all(_finite(v) for v in o[:3])]
        if not occ:
            return cands[0]
        best, best_clear = cands[0], -math.inf
        for x, y in cands:
            clear = min(math.hypot(x - float(o[0]), y - float(o[1]))
                        - (fp + float(o[2])) / 2.0 for o in occ)
            if clear >= 0.0:
                return x, y
            if clear > best_clear:
                best, best_clear = (x, y), clear
        return best

    def plan(self, zone: Optional[Zone], held_offset: float, transit_z: float,
             footprint: Optional[float] = None, occupied: Sequence[Sequence[float]] = (),
             center_first: bool = True) -> PlacePlan:
        """
        Return a PlacePlan for releasing into zone (poses in the base frame, m).

        footprint / occupied / center_first: see release_xy; the defaults give
        the zone center at the empty-zone height.
        """
        eye = np.eye(4)
        if zone is None:
            return PlacePlan('', eye, eye.copy(), eye.copy(), self.open_width, False, 'no_zone')
        if not _finite(held_offset) or held_offset < 0.0:
            return PlacePlan(zone.name, eye, eye.copy(), eye.copy(), self.open_width, False,
                             'bad_held_offset: %r' % (held_offset,))
        if not _finite(transit_z):
            return PlacePlan(zone.name, eye, eye.copy(), eye.copy(), self.open_width, False,
                             'bad_transit_z: %r' % (transit_z,))
        floor = float(zone.center[2])
        x, y = self.release_xy(zone, footprint, occupied, center_first)
        rh = self.release_height if zone.release_height is None else float(zone.release_height)
        z_rel = floor + rh + held_offset
        if zone.wall_height > 0.0:
            z_rel = max(z_rel, floor + zone.wall_height + WALL_RELEASE_CLEARANCE + held_offset)
        tallest = [float(o[3]) for o in occupied if len(o) >= 4 and _finite(o[3])]
        if tallest:
            # Objects already in the zone may have rolled anywhere: clear the tallest.
            z_rel = max(z_rel, floor + max(tallest) + STACK_CLEARANCE + held_offset)
        ang = neutral_closing_angle(x, y)
        z_pre = max(float(transit_z), z_rel)
        return PlacePlan(zone=zone.name,
                         T_pre=top_down_T(x, y, z_pre, ang),
                         T_release=top_down_T(x, y, z_rel, ang),
                         T_retreat=top_down_T(x, y, z_rel + self.retreat_height, ang),
                         open_width=self.open_width)


def zones_list(zones) -> list:
    """Return zones as a list whether given as a dict or an iterable."""
    if isinstance(zones, dict):
        return list(zones.values())
    return list(zones or [])

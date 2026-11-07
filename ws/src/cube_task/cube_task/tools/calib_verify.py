"""
Check the camera extrinsics and the TCP with the arm itself (hover test).

The arm moves to the observe pose like the task runner (robot.poses.observe_joints,
else observe_tcp leaned with task/reach.reach_adjust), the camera captures and the
task's perception (--task) lists the objects. For each chosen object (every valid
one, asked one by one; --index I [I ...]; --all without asking) the fingertip
center (TCP, fingers closed) moves above the grasp point of
perception/grasp.GraspPlanner (x, y, closing angle) to --hover m above the
detected top. The operator looks and types where the real object is relative to
the fingertip center in the base frame (dx dy in mm, optionally h = fingertip
height above the table in mm). The tool averages the offsets over the objects
(outliers are flagged), prints the mean translation correction d (real position
- camera estimate) and its spread, and with --apply (after confirmation) writes
corrected extrinsics (backup of the old file kept):

    eye_to_hand: T_base_camera <- Trans(d) @ T_base_camera
    eye_in_hand: T_flange_camera <- inv(T_base_flange_obs) @ Trans(d)
                                    @ T_base_flange_obs @ T_flange_camera

(the eye_in_hand rule is the equivalent correction at the observe pose). --point
X Y Z hovers above a base-frame point instead (TCP / arm check, nothing written).
The arm always returns to the observe pose at the end with the gripper open.

Real mode: every motion is printed and the first one needs a yes; --speed
(default 20) can only lower robot.speed_percent. --mode sim measures with the
simulated ground truth (Enter accepts it). Exit codes: 0 checked, 1 cancelled
or nothing checked / written, 2 setup problem (config, arm link, camera), 3 arm
fault, 130 Ctrl-C.

Example commands
    calib_verify --task L1
    calib_verify --task L2 --all --apply --headless
    calib_verify --point 0.30 0.00 0.00
    calib_verify --mode sim --task L1 --extrinsics-error 0.004 0 --all --apply --headless
"""
from __future__ import annotations

import argparse
import datetime
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_onsite_args, build_perception, capture_view, compose_view,
                     connect_arm_readonly, detection_table, draw_numbers, ensure_motion_ready,
                     extrinsics_out_path, flange_to_tcp, fmt_mm, format_table, go_observe,
                     load_extrinsics_or_none, make_ui, MotionGuard, nearest_sim_object,
                     parse_floats, recorded_flange_pose, resolve_speed, use_hint,
                     write_extrinsics_site, write_report)
from ..core.transforms import inv_T, neutral_closing_angle, top_down_T
from ..perception.grasp import detection_table_z, expected_table_z, table_floor_z, table_z_warning
from ..task.reach import reach_adjust

DEFAULT_SPEED = 20
DEFAULT_HOVER = 0.03
#: Allowed --hover (m): the fingertips must stay clear of the object top.
HOVER_RANGE = (0.01, 0.15)
#: Approach poses clear the highest detected object top by at least this (m).
SAFE_CLEARANCE = 0.05
#: Outlier: farther than max(OUTLIER_M, OUTLIER_SIGMA robust sigmas) from the median (m).
OUTLIER_M = 0.003
OUTLIER_SIGMA = 3.0
#: RMS spread (m) above which a single translation cannot explain the measurements.
SPREAD_WARN_M = 0.002
#: Corrections shorter than this (m) change little.
SMALL_CORRECTION_M = 0.001
#: Corrections longer than this (m) are refused (sign / unit mistake, or recalibrate).
MAX_CORRECTION_M = 0.03
MAX_OFFSET_MM = 100.0
MAX_HEIGHT_MM = 600.0
MAX_TRIES = 5
#: Motion errors after which the arm did not move: skip the object and go on.
SKIP_ERRORS = ('rejected', 'unreachable')
EXIT_OK, EXIT_CANCELLED, EXIT_SETUP, EXIT_FAULT, EXIT_INTERRUPTED = 0, 1, 2, 3, 130


# ------------------------------------------------------------------ texts
def measure_help(point_mode: bool = False) -> str:
    """Return the explanation of the base axes and of the answer format."""
    what = 'marked point' if point_mode else "object's top center"
    ground = 'the surface at the point' if point_mode else 'the table next to the object'
    return '\n'.join([
        'How to measure (robot base frame, millimeters):',
        '  +x points forward, away from the robot base (the direction the arm reaches out);',
        "  +y points to the robot's left (stand behind the robot facing +x: +y is on your",
        '  left); +z points up.',
        '  dx = where the real %s is along x, relative to the fingertip center:' % what,
        '       farther from the robot than the fingertips -> dx > 0, closer -> dx < 0.',
        "  dy = the same along y: more to the robot's left -> dy > 0, to its right -> dy < 0.",
        '  Look straight down along the fingers, or hold a ruler along x, then along y;',
        '  about 1 mm is good enough.',
        '  Optional third number h = height of the fingertips in mm, read on a ruler',
        '  standing on %s (the expected value is printed).' % ground,
        '  Answer for example "2 -1" or "2 -1 61"; "0 0" means centered.'])


def plan_text(point: Optional[Sequence[float]], hover: float, speed: int, mount: str) -> str:
    """Return what the tool is about to do (printed before the first motion)."""
    if point is None:
        what = ('capture and list the objects, then for each chosen object move above it and '
                'descend to %.0f mm above its detected top' % (hover * 1e3))
    else:
        what = 'move above the point %s mm and descend to %.0f mm above it' % (
            fmt_mm(point, 0), hover * 1e3)
    return ('calib_verify plan (%s camera, speed %d %%): move to the observe pose, open the '
            'gripper, %s with the fingers closed, wait while you look and measure, move back '
            'up. At the end: back to the observe pose, gripper open. If the fingers come '
            'close to touching anything, press the e-stop or Ctrl-C.' % (mount, speed, what))


def correction_formula(ext) -> str:
    """Return the update rule --apply uses for the mount / parent frame of ext."""
    if ext.parent_frame == 'base':
        return 'eye_to_hand: T_base_camera <- Trans(d) @ T_base_camera'
    P = 'T_base_flange_obs' if ext.parent_frame == 'flange' else \
        '(T_base_flange_obs @ T_flange_tcp)'
    return ('eye_in_hand: %s <- inv(%s) @ Trans(d) @ %s @ %s (the equivalent correction at '
            'the observe pose)' % (ext.name, P, P, ext.name))


def check_args(args: argparse.Namespace) -> str:
    """Return an error message for invalid option values ('' when fine)."""
    lo, hi = HOVER_RANGE
    if not (math.isfinite(args.hover) and lo <= args.hover <= hi):
        return '--hover must be in [%.2f, %.2f] m, got %r' % (lo, hi, args.hover)
    if args.point is not None:
        if not all(math.isfinite(v) for v in args.point):
            return '--point needs 3 finite numbers (m, base frame)'
        if args.apply:
            return '--apply needs detected objects; --point checks the TCP and writes nothing'
    if args.index is not None and any(i < 0 for i in args.index):
        return '--index values must be >= 0'
    return ''


def parse_measurement(text: str) -> Tuple[Optional[Tuple[float, float, Optional[float]]], str]:
    """
    Parse an offset answer 'dx dy' or 'dx dy h' (mm, spaces or commas).

    Returns ((dx, dy, h or None), '') or (None, error message).
    """
    vals = parse_floats(text, 2)
    if vals is None:
        vals = parse_floats(text, 3)
    if vals is None:
        return None, 'expected 2 or 3 numbers in mm, for example "2 -1" or "2 -1 61"'
    dx, dy = vals[0], vals[1]
    if max(abs(dx), abs(dy)) > MAX_OFFSET_MM:
        return None, ('offset above %.0f mm: answer in millimeters (a larger error needs a new '
                      'hand-eye calibration)' % MAX_OFFSET_MM)
    h = vals[2] if len(vals) == 3 else None
    if h is not None and not 0.0 < h <= MAX_HEIGHT_MM:
        return None, 'h = fingertip height in mm must be in (0, %.0f]' % MAX_HEIGHT_MM
    return (dx, dy, h), ''


# ------------------------------------------------------------------ correction math
def trans(d: Sequence[float]) -> np.ndarray:
    """Return the 4x4 pure translation by d (m)."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(d, dtype=float).reshape(3)
    return T


def parent_pose_in_base(ext, T_base_flange: Optional[np.ndarray] = None,
                        T_flange_tcp: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return T_base_<ext.parent_frame> at the observe pose.

    base -> identity; flange -> T_base_flange; tcp -> T_base_flange @
    T_flange_tcp. Raises ValueError when a needed pose is missing.
    """
    if ext.parent_frame == 'base':
        return np.eye(4)
    if T_base_flange is None:
        raise ValueError('the eye_in_hand correction needs the flange pose at the observe pose')
    T = np.asarray(T_base_flange, dtype=float)
    if ext.parent_frame == 'flange':
        return T.copy()
    if T_flange_tcp is None:
        raise ValueError('a T_tcp_camera correction needs T_flange_tcp (robot.tcp_offset)')
    return T @ np.asarray(T_flange_tcp, dtype=float)


def corrected_transform(T_parent_camera: np.ndarray, d: Sequence[float],
                        T_base_parent: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return T_parent_camera corrected by the base-frame translation d (m).

    T_new = inv(P) @ Trans(d) @ P @ T with P = T_base_parent, so the camera
    pose in the base frame at P moves by d and its rotation is unchanged.
    eye_to_hand (P = identity): T_base_camera <- Trans(d) @ T_base_camera.
    """
    P = np.eye(4) if T_base_parent is None else np.asarray(T_base_parent, dtype=float)
    return inv_T(P) @ trans(d) @ P @ np.asarray(T_parent_camera, dtype=float)


def corrected_extrinsics(ext, d: Sequence[float], T_base_flange: Optional[np.ndarray] = None,
                         T_flange_tcp: Optional[np.ndarray] = None, rms: Optional[float] = None,
                         n: int = 0, stamp: Optional[str] = None):
    """
    Return a new Extrinsics: ext corrected by the base-frame translation d (m).

    Keeps mount and frames. source notes the correction; rms (m) is the spread
    of the measurements after the correction (None keeps the old value).
    """
    from ..core.extrinsics import Extrinsics
    T = corrected_transform(ext.T, d, parent_pose_in_base(ext, T_base_flange, T_flange_tcp))
    prev = (ext.source or 'unknown source')[:160]
    src = 'calib_verify: d = %s mm from %d measurements, applied to %s' % (fmt_mm(d), n, prev)
    stamp = stamp or datetime.datetime.now().isoformat(timespec='seconds')
    return Extrinsics(T=T, mount=ext.mount, parent_frame=ext.parent_frame,
                      child_frame=ext.child_frame, source=src,
                      rms=ext.rms if rms is None else float(rms), stamp=stamp)


def sample_correction(target_xy: Sequence[float], table_z: Optional[float],
                      tcp_xyz: Sequence[float], dx_mm: float, dy_mm: float,
                      h_mm: Optional[float] = None) -> np.ndarray:
    """
    Return the base-frame correction d (m) of one measurement: real - camera estimate.

    xy: fingertip position + measured offset - detected target. z (NaN without
    h or table_z): fingertip z - h - estimated table z under the target.
    """
    tcp = np.asarray(tcp_xyz, dtype=float).reshape(3)
    t = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
    d = np.full(3, np.nan)
    d[0] = tcp[0] + float(dx_mm) / 1000.0 - t[0]
    d[1] = tcp[1] + float(dy_mm) / 1000.0 - t[1]
    if h_mm is not None and table_z is not None and math.isfinite(float(table_z)):
        d[2] = tcp[2] - float(h_mm) / 1000.0 - float(table_z)
    return d


def _mean3(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the mean of the masked rows; z over the rows with a finite z (0 when none)."""
    m = np.zeros(3)
    if mask.any():
        m[:2] = A[mask, :2].mean(axis=0)
        mz = mask & np.isfinite(A[:, 2])
        if mz.any():
            m[2] = float(A[mz, 2].mean())
    return m


def _deviations(A: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the distances (m) of the masked rows from ref (missing z counts as 0)."""
    R = A[mask] - ref
    R[~np.isfinite(R[:, 2]), 2] = 0.0
    return np.linalg.norm(R, axis=1)


def summarize_corrections(D: Any, names: Optional[Sequence[str]] = None,
                          outlier_m: float = OUTLIER_M, k_sigma: float = OUTLIER_SIGMA,
                          spread_warn_m: float = SPREAD_WARN_M, min_samples: int = 3
                          ) -> Dict[str, Any]:
    """
    Average per-measurement corrections and flag outliers.

    D: Nx3 base-frame corrections (m), z NaN where no height was measured (z
    averages only the measured ones, 0 when none). With 3 or more samples, one
    farther than max(outlier_m, k_sigma robust sigmas) from the median is an
    outlier; fewer than min_samples give a warning. Returns {'n', 'n_z',
    'mean' (all samples), 'mean_inliers', 'std' (per axis, inliers), 'rms' /
    'rms_inliers' (RMS distance from the mean), 'max_dev' (inliers),
    'residuals' (distance from the median), 'threshold', 'outliers' (sample
    positions), 'warnings'}.
    """
    A = np.asarray(D, dtype=float).reshape(-1, 3)
    n = int(A.shape[0])
    names = list(names) if names is not None else ['#%d' % k for k in range(n)]
    has_z = np.isfinite(A[:, 2])
    rep: Dict[str, Any] = {
        'n': n, 'n_z': int(has_z.sum()), 'mean': np.zeros(3), 'mean_inliers': np.zeros(3),
        'std': np.zeros(3), 'rms': 0.0, 'rms_inliers': 0.0, 'max_dev': 0.0,
        'residuals': np.zeros(n), 'threshold': float(outlier_m), 'outliers': [],
        'warnings': []}
    if n == 0:
        rep['warnings'].append('nothing measured')
        return rep
    med = np.zeros(3)
    med[:2] = np.median(A[:, :2], axis=0)
    if has_z.any():
        med[2] = float(np.median(A[has_z, 2]))
    every = np.ones(n, dtype=bool)
    r = _deviations(A, med, every)
    out = np.zeros(n, dtype=bool)
    if n >= 3:
        rep['threshold'] = max(float(outlier_m), float(k_sigma) * 1.4826 * float(np.median(r)))
        out = r > rep['threshold']
    inl = ~out
    rep['mean'] = _mean3(A, every)
    rep['mean_inliers'] = _mean3(A, inl)
    std = np.zeros(3)
    std[:2] = A[inl, :2].std(axis=0)
    if (inl & has_z).any():
        std[2] = float(A[inl & has_z, 2].std())
    rep['std'] = std
    rep['rms'] = float(np.sqrt(np.mean(_deviations(A, rep['mean'], every) ** 2)))
    dev = _deviations(A, rep['mean_inliers'], inl)
    rep['rms_inliers'] = float(np.sqrt(np.mean(dev ** 2)))
    rep['max_dev'] = float(dev.max())
    rep['residuals'] = r
    rep['outliers'] = [int(k) for k in np.flatnonzero(out)]
    w = rep['warnings']
    for k in rep['outliers']:
        w.append('%s is %.1f mm from the median of the measurements (limit %.1f mm): '
                 're-measure it, or check that the right object was measured'
                 % (names[k], r[k] * 1e3, rep['threshold'] * 1e3))
    gap = float(_deviations(A[:1], A[1], every[:1])[0]) if n == 2 else 0.0
    if gap > outlier_m:
        w.append('the two measurements differ by %.1f mm: measure a third object to tell '
                 'which one is off' % (gap * 1e3))
    if n < min_samples:
        w.append('only %d measurement%s: measure 3 or more objects spread over the pick area'
                 % (n, '' if n == 1 else 's'))
    if rep['rms_inliers'] > spread_warn_m:
        w.append('the corrections differ between objects by %.1f mm RMS (limit %.1f mm): one '
                 'translation cannot fix that. Re-measure, or the extrinsics rotation is off '
                 '(re-run the hand-eye calibration)'
                 % (rep['rms_inliers'] * 1e3, spread_warn_m * 1e3))
    return rep


def _fmt_d(d: Sequence[float]) -> str:
    """Format a correction in mm; a NaN z (not measured) prints as '-'."""
    v = np.asarray(d, dtype=float)
    z = '-' if not math.isfinite(v[2]) else '%.1f' % (v[2] * 1e3)
    return '[%.1f, %.1f, %s]' % (v[0] * 1e3, v[1] * 1e3, z)


def summary_lines(rep: Dict[str, Any], point_mode: bool = False) -> List[str]:
    """Return the printed result of summarize_corrections (mm)."""
    n = rep['n']
    if n == 0:
        return ['nothing measured']
    m = rep['mean_inliers']
    n_in = n - len(rep['outliers'])
    z_note = ('z from %d height measurement%s' % (rep['n_z'], '' if rep['n_z'] == 1 else 's')
              if rep['n_z'] else 'z not measured, left at 0')
    ref = 'commanded point' if point_mode else 'camera estimate'
    lines = ['mean correction d (real - %s) = %s mm over %d measurement%s (%s)'
             % (ref, fmt_mm(m), n_in, '' if n_in == 1 else 's', z_note)]
    if rep['outliers']:
        lines.append('  (outliers left out; the mean over all %d is %s mm)'
                     % (n, fmt_mm(rep['mean'])))
    lines.append('spread: std %s mm, RMS %.1f mm, max %.1f mm from the mean'
                 % (fmt_mm(rep['std']), rep['rms_inliers'] * 1e3, rep['max_dev'] * 1e3))
    if point_mode:
        lines.append('fingertip error (real - reported position) = %s mm: a constant error '
                     'means robot.tcp_offset or the arm is off (or the point is not at X Y Z)'
                     % fmt_mm(-m))
    lines.extend('WARNING: %s' % w for w in rep['warnings'])
    if float(np.linalg.norm(m)) < SMALL_CORRECTION_M:
        lines.append('the mean correction is below %.0f mm: %s' % (
            SMALL_CORRECTION_M * 1e3, 'the TCP is accurate' if point_mode
            else 'the camera and the arm agree, no correction needed'))
    return lines


@dataclass
class Sample:
    """One measurement: the offset the operator typed and the correction it gives."""

    index: int                       # detection number, -1 for --point
    label: str
    target: np.ndarray               # (3,) where the fingertips aimed (camera estimate)
    tcp: np.ndarray                  # (3,) reported fingertip position at the hover
    dx_mm: float
    dy_mm: float
    h_mm: Optional[float]
    d: np.ndarray                    # (3,) correction (m), z NaN when not measured


SAMPLE_HEADERS = ('#', 'label', 'dx, dy mm', 'h mm', 'correction d mm', 'flag')


def samples_table(samples: Sequence[Sample], outliers: Sequence[int] = ()) -> str:
    """Return the plain-text table of the measurements (outliers flagged)."""
    rows = []
    for k, sm in enumerate(samples):
        rows.append(['point' if sm.index < 0 else str(sm.index), sm.label,
                     '%.1f, %.1f' % (sm.dx_mm, sm.dy_mm),
                     '-' if sm.h_mm is None else '%.1f' % sm.h_mm, _fmt_d(sm.d),
                     'OUTLIER' if k in outliers else ''])
    return format_table(SAMPLE_HEADERS, rows)


# ------------------------------------------------------------------ targets
@dataclass
class HoverPlan:
    """Where the fingertips go for one check (base frame, m)."""

    index: int                       # detection number, -1 for --point
    label: str
    target: np.ndarray               # (3,) [x, y, top z] the fingertips aim at
    table_z: Optional[float]         # estimated table (or point) z under the target
    closing_angle: float
    T_approach: np.ndarray           # TCP above the target, clear of every object
    T_hover: np.ndarray              # TCP --hover above the target
    note: str = ''


def approach_z(top: float, hover: float, approach_height: float,
               highest_top: Optional[float] = None, clearance: float = SAFE_CLEARANCE) -> float:
    """Return the approach TCP z: approach_height above the target, clear of every top."""
    z = float(top) + max(float(approach_height), float(hover))
    if highest_top is not None and math.isfinite(float(highest_top)):
        z = max(z, float(highest_top) + float(clearance))
    return z


def hover_plan(det, index: int, planner, hover: float, approach_height: float,
               highest_top: Optional[float] = None) -> Tuple[Optional[HoverPlan], str]:
    """
    Return (HoverPlan above the grasp point of det, '') or (None, why).

    x, y and the closing angle come from planner.plan(det) (GraspPlanner),
    the height from det.top_z. An infeasible grasp (e.g. too wide) still
    hovers; its reason goes to note.
    """
    plan = planner.plan(det)
    if plan.reason.startswith(('missing_geometry', 'bad_geometry')):
        return None, plan.reason
    top = float(det.top_z if det.top_z is not None else det.center_base[2])
    x, y, c = float(plan.T_grasp[0, 3]), float(plan.T_grasp[1, 3]), float(plan.closing_angle)
    za = approach_z(top, hover, approach_height, highest_top)
    return HoverPlan(index=int(index), label=str(det.label), target=np.array([x, y, top]),
                     table_z=detection_table_z(det), closing_angle=c,
                     T_approach=top_down_T(x, y, za, c),
                     T_hover=top_down_T(x, y, top + float(hover), c),
                     note='' if plan.feasible else plan.reason), ''


def point_plan(point: Sequence[float], hover: float, approach_height: float) -> HoverPlan:
    """Return the HoverPlan above a base point [x, y, z] (m) for --point."""
    x, y, z = (float(v) for v in point)
    c = neutral_closing_angle(x, y)
    return HoverPlan(index=-1, label='point', target=np.array([x, y, z]), table_z=z,
                     closing_angle=c,
                     T_approach=top_down_T(x, y, approach_z(z, hover, approach_height), c),
                     T_hover=top_down_T(x, y, z + float(hover), c))


def select_targets(dets: Sequence[Any], indices: Optional[Sequence[int]] = None
                   ) -> Tuple[List[int], List[str]]:
    """
    Return (detection numbers to check, messages).

    indices None: every valid detection in table order. Otherwise the given
    numbers (duplicates dropped); missing and rejected ones are left out with
    a message.
    """
    if indices is None:
        return [i for i, d in enumerate(dets) if d.valid], []
    out: List[int] = []
    msgs: List[str] = []
    for i in indices:
        i = int(i)
        if i in out:
            continue
        if not 0 <= i < len(dets):
            msgs.append('--index %d: no such detection (%d listed)' % (i, len(dets)))
        elif not dets[i].valid:
            msgs.append('--index %d: rejected (%s), skipped'
                        % (i, dets[i].reject_reason or 'invalid'))
        else:
            out.append(i)
    return out, msgs


def table_z_estimate(result, polygon: Optional[Sequence[Sequence[float]]] = None
                     ) -> Optional[float]:
    """Return the table z for the arm's clearance checks like the task runner, or None."""
    t = getattr(result, 'table', None)
    if t is not None and getattr(t, 'frame', '') == 'base' and polygon and len(polygon) >= 3:
        c = np.asarray(polygon, dtype=float).mean(axis=0)
        try:
            z = float(t.z_at(float(c[0]), float(c[1])))
            if math.isfinite(z):
                return z
        except ValueError:
            pass
    zs = [detection_table_z(d) for d in result.valid_detections()]
    zs = [z for z in zs if z is not None and math.isfinite(z)]
    return float(np.median(zs)) if zs else None


def sim_measurement(world, tcp_xyz: Sequence[float], point: Optional[Sequence[float]] = None,
                    max_dist: float = 0.05) -> Optional[Tuple[float, float, float]]:
    """
    Return the true (dx, dy, h) in mm at the fingertips (simulated operator, sim only).

    Object checks measure to the footprint center of the nearest sim object
    and h above the world's table; --point checks measure to the point. None
    when no object is within max_dist m.
    """
    tcp = np.asarray(tcp_xyz, dtype=float).reshape(3)
    if point is not None:
        p = np.asarray(point, dtype=float).reshape(3)
        xy, ground = p[:2], float(p[2])
    else:
        obj = nearest_sim_object(world, tcp[:2], max_dist)
        if obj is None:
            return None
        xy, ground = np.asarray(obj.center_xy(), dtype=float), float(world.table_z)
    return (float((xy[0] - tcp[0]) * 1e3), float((xy[1] - tcp[1]) * 1e3),
            float((tcp[2] - ground) * 1e3))


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    """Return the calib_verify argument parser."""
    p = argparse.ArgumentParser(prog='calib_verify', description=__doc__.split('\n\n')[0])
    add_common_args(p)
    add_onsite_args(p, site=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument('--index', type=int, nargs='+', default=None, metavar='I',
                     help='check only these detection numbers (from the printed table)')
    sel.add_argument('--all', action='store_true',
                     help='check every valid detection without asking per object')
    sel.add_argument('--point', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                     help='hover above this base-frame point (m) instead of detected objects '
                          '(TCP check, nothing written)')
    p.add_argument('--hover', type=float, default=DEFAULT_HOVER,
                   help='fingertip height above the detected top / point (m, default %.2f)'
                        % DEFAULT_HOVER)
    p.add_argument('--speed', type=int, default=None,
                   help='speed percent (default %d; never above robot.speed_percent)'
                        % DEFAULT_SPEED)
    p.add_argument('--apply', action='store_true',
                   help='write the corrected extrinsics after confirmation (old file backed up)')
    return p


class _Verifier:
    """State of one calib_verify run (thin I/O around the pure helpers)."""

    def __init__(self, args, s, ui, speed: int):
        from ..perception.grasp import max_gripper_width
        self.args, self.s, self.ui, self.cfg = args, s, ui, s.cfg
        self.arm = s.arm
        self.speed = int(speed)
        self.guard = MotionGuard(ui, args.mode, self.speed)
        self.T_ft = flange_to_tcp(s.cfg)
        self.point = None if args.point is None else [float(v) for v in args.point]
        self.hover = float(args.hover)
        self.approach = float(s.cfg.get('grasp.approach_height', 0.08))
        ow = s.cfg.get('place.open_width', None)
        self.open_width = max_gripper_width(s.cfg) if ow is None else float(ow)
        self.close_width = float(s.cfg.get('robot.gripper.close_width', 0.0) or 0.0)
        f = s.cfg.get('robot.gripper.force', None)
        self.force = None if f is None else float(f)
        self.ext = None
        self.T_bf_obs: Optional[np.ndarray] = None
        self.samples: List[Sample] = []
        self.summary: Optional[Dict[str, Any]] = None
        self.visited = 0
        self.fault = ''
        self.explained = False

    def run(self) -> int:
        """Do the whole check; return the exit code."""
        ui, s = self.ui, self.s
        if not connect_arm_readonly(s, ui, required=True, why='arm motions'):
            return EXIT_SETUP
        if self.point is None:
            self.ext = load_extrinsics_or_none(self.cfg, ui)
            if self.ext is None:
                return EXIT_SETUP
            ui.say('extrinsics: %s' % self.ext.describe())
        ui.say(plan_text(self.point, self.hover, self.speed,
                         str(self.cfg.get('camera.mount', 'eye_to_hand'))))
        if not ensure_motion_ready(ui, self.arm, purpose='move above the objects'):
            ui.say('cancelled: nothing moved')
            return EXIT_CANCELLED
        self.arm.set_speed_percent(self.speed)
        ok, why = go_observe(self.arm, self.cfg, self.guard)
        if not ok:
            return self._observe_failed(why)
        self._gripper(self.open_width, 'open')
        plans = self._detect() if self.point is None else self._point_plans()
        if plans is None:                 # camera / configuration problem, at the observe pose
            return EXIT_SETUP
        if plans:
            self._gripper(self.close_width, 'close the fingers: their tips meet at the TCP, a '
                                            'sharp point to measure from')
            self._visit_all(plans)
            self._finish()
        return self._conclude()

    def _observe_failed(self, why: str) -> int:
        if why == 'cancelled':
            self.ui.say('cancelled: nothing moved')
            return EXIT_CANCELLED
        self.ui.say('cannot reach the observe pose: %s' % why)
        if why.startswith(('no observe pose', 'observe_tcp not reachable', 'rejected')):
            return EXIT_SETUP
        return EXIT_FAULT

    def _gripper(self, width: float, what: str) -> bool:
        self.ui.say('GRIPPER: %s, width %.0f mm' % (what, width * 1e3))
        g = self.arm.gripper(width, force=self.force, wait=True)
        if not g.ok:
            self.ui.say('gripper command failed: %s %s' % (g.error, g.detail))
        return bool(g.ok)

    def _point_plans(self) -> List[HoverPlan]:
        tz = self.cfg.get('perception.table.expected_z', None)
        if tz is not None and hasattr(self.arm, 'set_table_z'):
            self.arm.set_table_z(float(tz))
        return [point_plan(self.point, self.hover, self.approach)]

    def _detect(self) -> Optional[List[HoverPlan]]:
        """Capture at the observe pose, list the detections, plan the hovers (None = error)."""
        from ..core.extrinsics import camera_pose_in_base
        from ..perception.grasp import GraspPlanner
        ui, s, cfg = self.ui, self.s, self.cfg
        perception = build_perception(cfg, s.task, self.args, ui)
        s.clock.sleep(float(cfg.get('camera.settle_s', 0.4)))
        frame = capture_view(s.camera, cfg, newer_than=s.clock.now())
        if frame is None:
            ui.say('no camera frame: check the camera with the view tool')
            return None
        T_bf = None
        if self.ext.mount == 'eye_in_hand':
            T_bf = recorded_flange_pose(frame)
            if T_bf is None:
                st = self.arm.get_flange_pose()
                T_bf = None if st is None else np.asarray(st.value, dtype=float)
            if T_bf is None:
                ui.say('eye_in_hand: no flange pose for the capture')
                return None
        self.T_bf_obs = T_bf
        try:
            T_bc = camera_pose_in_base(self.ext, T_bf, self.T_ft)
            planner = GraspPlanner(cfg, s.task or 'L1')
        except ValueError as e:
            ui.say('cannot plan: %s' % e)
            return None
        res = perception.process(frame, T_bc)
        self._show(frame, res)
        ui.say(detection_table(res.detections))
        if res.warnings:
            ui.say('perception warnings: %s' % ', '.join(res.warnings))
        tz = table_z_estimate(res, cfg.get('perception.roi.pick_polygon', None))
        if tz is not None:
            # The taught table (perception.table.expected_z) bounds the safety floor from
            # below: an extrinsics z offset must not let the fingertips into the table.
            exp_z = expected_table_z(cfg)
            warning = table_z_warning(tz, exp_z)
            if warning:
                ui.say('WARNING: %s' % warning)
            if hasattr(self.arm, 'set_table_z'):
                floor = table_floor_z(tz, exp_z)
                self.arm.set_table_z(floor)
                ui.say('table z %.1f mm measured by the camera; %.1f mm used by the motion '
                       'safety checks' % (tz * 1e3, floor * 1e3))
        idx, msgs = select_targets(res.detections, self.args.index)
        for m in msgs:
            ui.say(m)
        tops = [float(d.top_z) for d in res.detections
                if d.top_z is not None and math.isfinite(float(d.top_z))]
        highest = max(tops) if tops else None
        plans = []
        for i in idx:
            hp, why = hover_plan(res.detections[i], i, planner, self.hover, self.approach,
                                 highest)
            if hp is None:
                ui.say('#%d: no hover pose (%s); skipped' % (i, why))
            else:
                plans.append(hp)
        if not plans:
            ui.say('no valid detection to check (see the table above)')
        return plans

    def _show(self, frame, res) -> None:
        base = res.debug_image if res.debug_image is not None else frame.color
        img = draw_numbers(np.array(base, copy=True), res.detections)
        path = self.ui.show('calib_verify', compose_view(frame, img))
        if not self.ui.headless:
            self.ui.key(1)
        elif path is not None:
            self.ui.say('preview image: %s' % path)

    @staticmethod
    def _name(hp: HoverPlan) -> str:
        return 'point' if hp.index < 0 else '#%d %s' % (hp.index, hp.label)

    def _visit_all(self, plans: Sequence[HoverPlan]) -> None:
        ask = not (self.args.all or self.args.index or self.point is not None)
        for hp in plans:
            if ask:
                ans = self.ui.prompt('next: %s at %s mm. Enter = hover above it, s = skip, '
                                     'q = finish' % (self._name(hp), fmt_mm(hp.target, 0)))
                ans = ans.strip().lower()
                if self.ui.eof or ans.startswith('q'):
                    break
                if ans.startswith('s'):
                    continue
            if self._visit(hp) in ('stop', 'fault'):
                break

    def _failed(self, name: str, stage: str, m) -> str:
        """Report a failed motion; return 'skip' (the arm did not move) or 'fault'."""
        if m.error in SKIP_ERRORS:
            self.ui.say('%s: %s refused (%s %s); skipped' % (name, stage, m.error, m.detail))
            return 'skip'
        self.fault = '%s %s: %s %s' % (name, stage, m.error, m.detail)
        self.ui.say('FAULT: %s. No further arm motion.' % self.fault)
        return 'fault'

    def _visit(self, hp: HoverPlan) -> str:
        """Approach, descend, measure, go back up; return 'ok', 'skip', 'stop' or 'fault'."""
        ui, arm, name = self.ui, self.arm, self._name(hp)
        T_app, tilt_a, why = reach_adjust(arm, hp.T_approach)
        if T_app is None:
            ui.say('%s: approach pose %s mm not reachable (%s); skipped'
                   % (name, fmt_mm(hp.T_approach[:3, 3], 0), why))
            return 'skip'
        T_hov, tilt_h, why = reach_adjust(arm, hp.T_hover)
        if T_hov is None:
            ui.say('%s: hover pose %s mm not reachable (%s); skipped'
                   % (name, fmt_mm(hp.T_hover[:3, 3], 0), why))
            return 'skip'
        if hp.note:
            ui.say('%s: note: a grasp would be refused (%s); hovering anyway' % (name, hp.note))
        if not self.guard.approve('move_tcp (joint path) to %s mm above %s, tilt %.0f deg'
                                  % (fmt_mm(T_app[:3, 3], 0), name, tilt_a)):
            return 'stop'
        m = arm.move_tcp(T_app, linear=False)
        if not m.ok:
            return self._failed(name, 'approach', m)
        if not self.guard.approve('move_l down to %s mm (%.0f mm above the detected top), '
                                  'tilt %.0f deg' % (fmt_mm(T_hov[:3, 3], 0),
                                                     self.hover * 1e3, tilt_h)):
            return 'stop'
        m = arm.move_tcp(T_hov, linear=True)
        if not m.ok:
            return self._failed(name, 'descent', m)
        status = self._measure(hp, T_hov, tilt_h)
        self.guard.approve('move_l back up to %s mm' % fmt_mm(T_app[:3, 3], 0))
        m = arm.move_tcp(T_app, linear=True)
        if not m.ok:
            self.fault = '%s: moving back up failed: %s %s' % (name, m.error, m.detail)
            ui.say('FAULT: %s. No further arm motion: raise the arm by hand (teach button) '
                   'before the next run.' % self.fault)
            return 'fault'
        return status

    def _sim_default(self, tcp: np.ndarray) -> str:
        if self.args.mode != 'sim' or self.s.world is None:
            return ''
        meas = sim_measurement(self.s.world, tcp, self.point)
        if meas is None:
            return ''
        txt = '%.1f %.1f %.1f' % meas
        self.ui.say('[sim operator] measured dx dy h = %s mm' % txt)
        return txt

    def _measure(self, hp: HoverPlan, T_hov: np.ndarray, tilt: float) -> str:
        """Wait for the operator at the hover pose; return 'ok' or 'stop' (q / end of input)."""
        ui, name = self.ui, self._name(hp)
        self.visited += 1
        st = self.arm.get_tcp_pose()
        tcp = T_hov[:3, 3].copy() if st is None else np.asarray(st.value, dtype=float)[:3, 3]
        if not self.explained:
            ui.say(measure_help(self.point is not None))
            self.explained = True
        exp_h = '' if hp.table_z is None else ', expected h %.0f mm' % (
            (tcp[2] - hp.table_z) * 1e3)
        ui.say('%s: fingertips at %s mm (reported, %.1f mm from the command), %.0f mm above '
               'the %s%s, tilt %.0f deg' % (
                   name, fmt_mm(tcp), float(np.linalg.norm(tcp - T_hov[:3, 3])) * 1e3,
                   (tcp[2] - hp.target[2]) * 1e3, 'point' if hp.index < 0 else 'detected top',
                   exp_h, tilt))
        default = self._sim_default(tcp)
        enter = 'the simulated measurement' if default else 'not measured'
        for _ in range(MAX_TRIES):
            ans = ui.prompt('%s offset "dx dy [h]" mm (Enter = %s, s = skip, q = stop)'
                            % (name, enter), default)
            a = ans.strip().lower()
            if ui.eof or a.startswith('q'):
                return 'stop'
            if a in ('', 's', 'skip'):
                ui.say('%s: not measured' % name)
                return 'ok'
            val, why = parse_measurement(ans)
            if val is None:
                ui.say(why)
                continue
            dx, dy, h = val
            d = sample_correction(hp.target[:2], hp.table_z, tcp, dx, dy, h)
            self.samples.append(Sample(hp.index, hp.label, hp.target.copy(), tcp.copy(),
                                       dx, dy, h, d))
            ui.say('%s: correction d = %s mm (real - %s%s)' % (
                name, _fmt_d(d), 'commanded point' if hp.index < 0 else 'camera estimate',
                '' if math.isfinite(d[2]) else '; z not measured'))
            return 'ok'
        ui.say('%s: no valid answer, not measured' % name)
        return 'ok'

    def _finish(self) -> None:
        """Return to the observe pose (unless a fault stopped the arm), then open the gripper."""
        if self.fault:
            self.ui.say('not returning to the observe pose after the fault')
        else:
            ok, why = go_observe(self.arm, self.cfg, self.guard)
            if not ok:
                self.fault = 'return to the observe pose failed: %s' % why
                self.ui.say('FAULT: %s' % self.fault)
        self._gripper(self.open_width, 'open')

    def _conclude(self) -> int:
        """Print the results, apply on request; return the exit code."""
        ui, point_mode = self.ui, self.point is not None
        if self.samples:
            names = ['point' if sm.index < 0 else '#%d %s' % (sm.index, sm.label)
                     for sm in self.samples]
            self.summary = summarize_corrections([sm.d for sm in self.samples], names,
                                                 min_samples=1 if point_mode else 3)
            ui.say('results (correction d = real - %s, base frame):'
                   % ('commanded point' if point_mode else 'camera estimate'))
            ui.say(samples_table(self.samples, self.summary['outliers']))
            for line in summary_lines(self.summary, point_mode):
                ui.say(line)
        elif self.visited:
            ui.say('checked %d target%s by eye, no offsets entered'
                   % (self.visited, '' if self.visited == 1 else 's'))
        written = self._apply() if self.args.apply else False
        if not self.args.apply and self.samples and not point_mode:
            ui.say('dry check only: add --apply to write the corrected extrinsics')
        if self.fault:
            return EXIT_FAULT
        if self.visited == 0 or (self.args.apply and not written):
            return EXIT_CANCELLED
        return EXIT_OK

    def _apply(self) -> bool:
        """Write the corrected extrinsics after confirmation; return True when written."""
        ui, rep = self.ui, self.summary
        if self.ext is None or rep is None or rep['n'] == 0:
            ui.say('--apply: no offsets measured, nothing written')
            return False
        d, n_used, rms = rep['mean_inliers'], rep['n'] - len(rep['outliers']), rep['rms_inliers']
        if rep['outliers'] and not ui.confirm('Leave the outliers out of the correction '
                                              '(n = use all measurements)?', default=True):
            d, n_used, rms = rep['mean'], rep['n'], rep['rms']
        if float(np.linalg.norm(d)) > MAX_CORRECTION_M:
            ui.say('REFUSED: the correction %s mm is longer than %.0f mm. Check the signs and '
                   'units of the offsets, or re-run the hand-eye calibration.'
                   % (fmt_mm(d), MAX_CORRECTION_M * 1e3))
            return False
        try:
            new = corrected_extrinsics(self.ext, d, self.T_bf_obs, self.T_ft,
                                       rms=rms if n_used >= 2 else None, n=n_used)
        except ValueError as e:
            ui.say('cannot correct the extrinsics: %s' % e)
            return False
        path = extrinsics_out_path(self.cfg, self.args.mode, self.args.site_dir)
        ui.say('correction d = %s mm; %s' % (fmt_mm(d), correction_formula(self.ext)))
        ui.say('  current:   %s' % self.ext.describe())
        ui.say('  corrected: %s' % new.describe())
        if not ui.confirm('Write the corrected extrinsics to %s (the old file is backed up)?'
                          % path, default=False):
            ui.say('nothing written')
            return False
        out, backup, keys = write_extrinsics_site(path, new)
        ui.say(write_report(out, keys, backup))
        ui.say(use_hint(self.cfg, self.args.mode, out, 'camera.extrinsics_file'))
        ui.say('run calib_verify again to check: the offsets should now be close to 0')
        return True

    def interrupted(self) -> int:
        """Ctrl-C: stop the arm where it is (never disables it); return the exit code."""
        self.ui.say('\nCtrl-C: stopping the arm')
        try:
            self.arm.stop()
        except Exception as e:  # noqa: B902 - report, the operator has the e-stop
            self.ui.say('arm.stop() failed: %s (use the e-stop)' % e)
        self.ui.say('the arm stays where it is, nothing written. Move it clear by hand (teach '
                    'button) or re-run the tool, which starts with the observe pose.')
        return EXIT_INTERRUPTED


def main(argv=None, ui=None) -> int:
    """Run the hover check; return the process exit code."""
    from ..core.interfaces import ArmError, CameraError
    parser = build_parser()
    args = parser.parse_args(argv)
    err = check_args(args)
    if err:
        parser.error(err)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    cfg = cfg_world_info[0]
    ui = make_ui(args, cfg, 'calib_verify', ui)
    try:
        speed, note = resolve_speed(cfg, DEFAULT_SPEED if args.speed is None else args.speed)
    except ValueError as e:
        ui.say('REFUSED: %s' % e)
        return EXIT_SETUP
    if note and args.speed is not None:
        ui.say(note)
    try:
        s = open_session(args, arm=True, camera=args.point is None, perception=False,
                         connect=args.mode != 'real', cfg_world_info=cfg_world_info)
    except (ArmError, CameraError, ValueError, OSError, ImportError) as e:
        ui.say('REFUSED: setup failed: %s' % e)
        return EXIT_SETUP
    try:
        with s:
            try:
                v = _Verifier(args, s, ui, speed)
            except ValueError as e:
                ui.say('REFUSED: configuration: %s' % e)
                return EXIT_SETUP
            try:
                return v.run()
            except KeyboardInterrupt:
                return v.interrupted()
    finally:
        ui.close()


if __name__ == '__main__':
    raise SystemExit(main())

"""
Make planned TCP poses reachable by leaning the tool outward when needed.

PiPER's joint5 limit (+-70 deg) with a ~0.14 m TCP means a vertical tool only
reaches TCP heights up to about 0.06-0.10 m. Pre-grasp, transit and place poses
higher than that must lean the approach axis away from the base. The grasp pose
itself stays vertical whenever it is reachable, because the lean is chosen as
the smallest that works (SafeArm.find_reachable_tilt).

Free-space poses (pre-grasp, lift, pre-place, retreat) may lean up to
robot.workspace.max_tilt_free_deg when the arm supports a per-call tilt limit
(free_tilt_limit); the grasp and release poses keep max_tilt_from_down_deg. A
free-space pose that stays unreachable is lowered in REACH_STEP steps down to
its floor (adjust_plan_poses).
"""
from __future__ import annotations

import inspect
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: a pose that is out of reach is lowered by this much per try (m).
REACH_STEP = 0.01
#: free-space poses of one pick, in planning order.
FREE_POSES = ('pre_grasp', 'lift', 'pre_place', 'retreat')
#: free-space poses that keep the grasp / release orientation at any height before
#: leaning, so the move from the grasp / release pose is straight up.
VERTICAL_FIRST = ('lift', 'retreat')
#: default of robot.workspace.max_tilt_free_deg (deg).
DEFAULT_MAX_TILT_FREE_DEG = 40.0


def accepts_kwarg(fn, name: str) -> bool:
    """Return True when callable fn takes keyword argument name (or **kwargs)."""
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == p.VAR_KEYWORD for p in params)


def free_tilt_limit(arm, cfg) -> Optional[float]:
    """
    Return the lean limit (deg) for free-space poses, or None for the default limit.

    robot.workspace.max_tilt_free_deg (default 40) when arm.move_tcp and its
    reachability check accept ``max_tilt_deg`` (SafeArm); None otherwise, so a
    plan never contains a pose the arm would refuse to move to.
    """
    check = getattr(arm, 'find_reachable_tilt', None) or getattr(arm, 'check_reachable', None)
    if not (accepts_kwarg(getattr(arm, 'move_tcp', None), 'max_tilt_deg')
            and accepts_kwarg(check, 'max_tilt_deg')):
        return None
    v = cfg.get('robot.workspace.max_tilt_free_deg', DEFAULT_MAX_TILT_FREE_DEG)
    try:
        f = float(DEFAULT_MAX_TILT_FREE_DEG if v is None else v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0.0 else None


def reach_adjust(arm, T_base_tcp: np.ndarray, q_seed: Optional[Sequence[float]] = None,
                 max_tilt_deg: Optional[float] = None
                 ) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Return (pose to use or None, tilt in deg, reason) for one TCP target.

    Uses arm.find_reachable_tilt when the arm offers it (SafeArm), else
    arm.check_reachable, else accepts the pose unchanged. max_tilt_deg: lean
    limit (deg, capped by SafeArm at max_tilt_free_deg); None = the arm's
    default (max_tilt_from_down_deg).
    """
    T = np.asarray(T_base_tcp, dtype=float)
    fn = getattr(arm, 'find_reachable_tilt', None)
    if fn is not None:
        if max_tilt_deg is not None and accepts_kwarg(fn, 'max_tilt_deg'):
            T2, tilt, why = fn(T, q_seed=q_seed, max_tilt_deg=max_tilt_deg)
        else:
            T2, tilt, why = fn(T, q_seed=q_seed)
        if T2 is None:
            return None, 0.0, why
        return np.asarray(T2, dtype=float), math.degrees(float(tilt)), ''
    if getattr(arm, 'check_reachable', None) is not None:
        ok, why = _check(arm, T, max_tilt_deg)
        return (T, 0.0, '') if ok else (None, 0.0, why)
    return T, 0.0, ''


def _check(arm, T: np.ndarray, max_tilt_deg: Optional[float]) -> Tuple[bool, str]:
    chk = getattr(arm, 'check_reachable', None)
    if chk is None:
        return True, ''
    if max_tilt_deg is not None and accepts_kwarg(chk, 'max_tilt_deg'):
        return chk(T, max_tilt_deg=max_tilt_deg)
    return chk(T)


def _tilt_deg(T: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(T[2, 2])))))


def _at_z(T: np.ndarray, z: float, R: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.array(T, dtype=float, copy=True)
    out[2, 3] = z
    if R is not None:
        out[:3, :3] = R
    return out


def _heights(z0: float, floor_z: Optional[float], step: float) -> list:
    zs = [z0]
    if floor_z is None or not math.isfinite(float(floor_z)) or float(floor_z) >= z0 - 1e-9:
        return zs
    k = 1
    while z0 - k * step > float(floor_z) + 1e-9:
        zs.append(z0 - k * step)
        k += 1
    zs.append(float(floor_z))
    return zs


def reach_lowered(arm, T_base_tcp: np.ndarray, floor_z: Optional[float] = None,
                  max_tilt_deg: Optional[float] = None,
                  keep_R: Optional[np.ndarray] = None, vertical_first: bool = False,
                  step: float = REACH_STEP) -> Tuple[Optional[np.ndarray], float, float, str]:
    """
    Return (pose or None, tilt deg, lowered m, reason) for a free-space pose.

    Tries the planned z, then REACH_STEP lower down to floor_z (None = never
    lowered). At each z first keep_R (the orientation of the pose this one is
    joined to by a straight vertical move, e.g. the grasp for the pre-grasp and
    the lift), then the smallest outward lean up to max_tilt_deg.
    vertical_first: try keep_R at every height before any lean.
    """
    T0 = np.asarray(T_base_tcp, dtype=float)
    z0 = float(T0[2, 3])
    zs = _heights(z0, floor_z, step)
    why = ''
    if keep_R is not None and vertical_first:
        for z in zs:
            Tk = _at_z(T0, z, keep_R)
            ok, why = _check(arm, Tk, max_tilt_deg)
            if ok:
                return Tk, _tilt_deg(Tk), z0 - z, ''
    for z in zs:
        if keep_R is not None and not vertical_first:
            Tk = _at_z(T0, z, keep_R)
            ok, why = _check(arm, Tk, max_tilt_deg)
            if ok:
                return Tk, _tilt_deg(Tk), z0 - z, ''
        T2, tilt, why = reach_adjust(arm, _at_z(T0, z), max_tilt_deg=max_tilt_deg)
        if T2 is not None:
            return T2, tilt, z0 - z, ''
    return None, 0.0, 0.0, why


def adjust_plan_poses(arm, grasp, place, floors: Optional[Dict[str, float]] = None,
                      free_tilt_deg: Optional[float] = None,
                      report: Optional[dict] = None) -> Tuple[bool, str, dict]:
    """
    Replace the poses of a GraspPlan and PlacePlan in place with reachable ones.

    Returns (ok, reason, tilts_deg). On failure the plans are left unchanged.
    The grasp and release poses only lean (max_tilt_from_down_deg). The
    free-space poses (FREE_POSES) lean up to free_tilt_deg (None = the same
    limit, see free_tilt_limit), first keeping the orientation of the grasp
    (pre-grasp, lift) or the release (pre-place, retreat) so the moves between
    them are straight; lift and retreat stay vertical down to their floor before
    leaning. A free-space pose that stays out of reach is lowered down to
    floors[name] (m, base z; missing = never lowered). report (dict) gets
    'lowered' {name: m} of the lowered poses.
    """
    floors = dict(floors or {})
    new = {}
    tilts = {}
    lowered = {}
    for name, obj, attr in (('grasp', grasp, 'T_grasp'), ('release', place, 'T_release')):
        T2, tilt, why = reach_adjust(arm, getattr(obj, attr))
        if T2 is None:
            return False, '%s: %s' % (name, why), tilts
        new[name] = (obj, attr, T2)
        tilts[name] = round(tilt, 1)
    joined = {'pre_grasp': ('grasp', grasp, 'T_pre'), 'lift': ('grasp', grasp, 'T_lift'),
              'pre_place': ('release', place, 'T_pre'),
              'retreat': ('release', place, 'T_retreat')}
    for name in FREE_POSES:
        ref, obj, attr = joined[name]
        T2, tilt, dz, why = reach_lowered(arm, getattr(obj, attr), floors.get(name),
                                          free_tilt_deg, keep_R=new[ref][2][:3, :3],
                                          vertical_first=name in VERTICAL_FIRST)
        if T2 is None:
            return False, '%s: %s' % (name, why), tilts
        new[name] = (obj, attr, T2)
        tilts[name] = round(tilt, 1)
        if dz > 1e-9:
            lowered[name] = round(dz, 4)
    order = ('pre_grasp', 'grasp', 'lift', 'pre_place', 'release', 'retreat')
    tilts = {k: tilts[k] for k in order}
    for obj, attr, T2 in new.values():
        setattr(obj, attr, T2)
    if report is not None:
        report['lowered'] = lowered
    return True, '', tilts

"""
Make planned TCP poses reachable by leaning the tool outward when needed.

PiPER's joint5 limit (+-70 deg) with a ~0.14 m TCP means a vertical tool only
reaches TCP heights up to about 0.06-0.10 m. Pre-grasp, transit and place poses
higher than that must lean the approach axis away from the base. The grasp pose
itself stays vertical whenever it is reachable, because the lean is chosen as
the smallest that works (SafeArm.find_reachable_tilt).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def reach_adjust(arm, T_base_tcp: np.ndarray, q_seed: Optional[Sequence[float]] = None
                 ) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Return (pose to use or None, tilt in deg, reason) for one TCP target.

    Uses arm.find_reachable_tilt when the arm offers it (SafeArm), else
    arm.check_reachable, else accepts the pose unchanged.
    """
    T = np.asarray(T_base_tcp, dtype=float)
    fn = getattr(arm, 'find_reachable_tilt', None)
    if fn is not None:
        T2, tilt, why = fn(T, q_seed=q_seed)
        if T2 is None:
            return None, 0.0, why
        return np.asarray(T2, dtype=float), math.degrees(float(tilt)), ''
    chk = getattr(arm, 'check_reachable', None)
    if chk is not None:
        ok, why = chk(T)
        return (T, 0.0, '') if ok else (None, 0.0, why)
    return T, 0.0, ''


def adjust_plan_poses(arm, grasp, place) -> Tuple[bool, str, dict]:
    """
    Replace the poses of a GraspPlan and PlacePlan in place with reachable ones.

    Returns (ok, reason, tilts_deg). On failure the plans are left unchanged.
    """
    names = [('pre_grasp', grasp, 'T_pre'), ('grasp', grasp, 'T_grasp'),
             ('lift', grasp, 'T_lift'), ('pre_place', place, 'T_pre'),
             ('release', place, 'T_release'), ('retreat', place, 'T_retreat')]
    new = {}
    tilts = {}
    for name, obj, attr in names:
        T2, tilt, why = reach_adjust(arm, getattr(obj, attr))
        if T2 is None:
            return False, '%s: %s' % (name, why), tilts
        new[(id(obj), attr)] = (obj, attr, T2)
        tilts[name] = round(tilt, 1)
    for obj, attr, T2 in new.values():
        setattr(obj, attr, T2)
    return True, '', tilts

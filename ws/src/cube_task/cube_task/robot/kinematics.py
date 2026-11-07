"""
PiPER forward / inverse kinematics (modified DH), independent of the SDK.

Frames: ``base`` is the PiPER base frame (z up, x forward); the FK result is
``T_base_flange`` (link6), the same frame that pyAgxArm ``fk()`` and
``get_flange_pose()`` report. Units: meters and radians.

The MDH tables and joint limits are copied from the pyAgxArm SDK,
``pyAgxArm/api/constants.py`` (ROBOT_MDH_PRESET and
ROBOT_JOINT_LIMIT_PRESET_RAD, SDK commit d90e234). The link convention is the
one of ``pyAgxArm/utiles/mdh_kinematics.py``::

    T_(i-1)_i = Rx(alpha) Tx(a) Rz(q_i + theta_offset) Tz(d)

test/test_robot_kinematics.py checks these tables and the FK against the
installed SDK, so an SDK update that changes them is detected.

Reachability note (standard PiPER, joint5 limited to +-70 deg): with a
gripper of about 0.14 m the tool can point straight down only for TCP heights
up to roughly 0.06-0.10 m over the central pick area. Higher top-down poses
need the tool leaned outward (``tilt_outward`` / ``solve_tcp_with_tilt``).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import inv_T, is_rotation, make_T, pose6_to_T

# (d, a, alpha, theta_offset) per link, m / rad.
# Source: pyAgxArm/api/constants.py ROBOT_MDH_PRESET.
MDH_PRESETS: Dict[str, Tuple[Tuple[float, float, float, float], ...]] = {
    'piper': (
        (0.123, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.5707963267948966, -3.0058060377846343),
        (0.0, 0.28502999999999995, 0.0, -1.793849405199772),
        (0.25075, -0.02198, 1.5707963267948966, 0.0),
        (0.0, 0.0, -1.5707963267948966, 0.0),
        (0.091, 0.0, 1.5707963267948966, 0.0),
    ),
    'piper_h': (
        (0.123, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.5707963267948966, -3.0058060377846343),
        (0.0, 0.28502999999999995, 0.0, -1.793849405199772),
        (0.25575, -0.02198, 1.5707963267948966, 0.0),
        (0.0, 0.0, -1.5707963267948966, 0.0),
        (0.0963, 0.0, 1.5707963267948966, 0.0),
    ),
    'piper_l': (
        (0.123, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.5707963267948966, -3.0058060377846343),
        (0.0, 0.33823000000000003, 0.0, -1.793849405199772),
        (0.32075, -0.023010000000000003, 1.5707963267948966, 0.0),
        (0.0, 0.0, -1.5707963267948966, 0.0),
        (0.091, 0.0, 1.5707963267948966, 0.0),
    ),
    'piper_x': (
        (0.123, 0.0, 0.0, 3.141592653589793),
        (0.0, 0.0, 1.5707963267948966, 0.13578661580515886),
        (0.0, 0.28502999999999995, 0.0, 2.8380798966679794),
        (0.0, 0.27364, 0.0, 0.08063421144213803),
        (0.0, 0.07465999999999999, -1.5707963267948966, 1.5707963267948966),
        (0.03526, 0.0, 1.5707963267948966, 0.0),
    ),
}

# [lower, upper] per joint, rad.
# Source: pyAgxArm/api/constants.py ROBOT_JOINT_LIMIT_PRESET_RAD.
JOINT_LIMITS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    'piper': ((-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
              (-1.745330, 1.745330), (-1.221730, 1.221730), (-2.967060, 2.967060)),
    'piper_h': ((-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
                (-2.356195, 2.356195), (-1.562070, 1.562070), (-2.967060, 2.967060)),
    'piper_l': ((-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
                (-2.216569, 2.216569), (-1.562070, 1.562070), (-2.967060, 2.967060)),
    'piper_x': ((-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
                (-1.553344, 1.553344), (-1.553344, 1.553344), (-2.967060, 2.967060)),
}

MODELS = tuple(sorted(MDH_PRESETS))

# Typical (j2, j3, j5) of top-down poses over the table in front of a PiPER;
# combined with j1 from the target azimuth and j6 from the target rotation.
_TOP_DOWN_SEEDS = (
    (1.6, -1.2, 1.1), (1.0, -0.4, 0.9), (2.1, -1.7, 1.1), (1.3, -1.0, 1.2),
    (1.8, -1.2, 0.5), (0.8, -0.7, 0.6), (2.4, -2.2, 1.2),
)
_ROT_WEIGHT = 0.1          # m per rad when mixing position and rotation errors
_MAX_STEP = 0.35           # rad, largest joint change per iteration


def flange_to_tcp_from_cfg(cfg) -> np.ndarray:
    """
    Return T_flange_tcp (4x4) from config robot.tcp_offset [x y z roll pitch yaw].

    Raises ValueError when the key is missing or malformed: the TCP offset
    decides where the fingers go, so it is never defaulted.
    """
    raw = cfg.get('robot.tcp_offset', None)
    if raw is None:
        raise ValueError('config robot.tcp_offset is missing (m / rad, flange frame)')
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ValueError('config robot.tcp_offset must be 6 numbers, got %r' % (raw,))
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        raise ValueError('config robot.tcp_offset must be 6 finite numbers, got %r' % (raw,))
    return pose6_to_T(vals)


def _as_joints(q: Sequence[float], n: int) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError('expected %d joint angles, got %d' % (n, arr.shape[0]))
    if not np.all(np.isfinite(arr)):
        raise ValueError('joint angles must be finite, got %s' % (arr.tolist(),))
    return arr


def _check_pose(T: np.ndarray, name: str = 'pose') -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError('%s must be a 4x4 matrix, got shape %s' % (name, T.shape))
    if not np.all(np.isfinite(T)):
        raise ValueError('%s contains NaN / inf' % name)
    if not is_rotation(T[:3, :3], tol=1e-3):
        raise ValueError('%s rotation part is not a rotation matrix' % name)
    return T


def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of a rotation matrix."""
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return 0.5 * w
    if th > math.pi - 1e-4:
        # Near pi the antisymmetric part vanishes; use the symmetric part.
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        axis = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        if np.dot(axis, w) < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * th
    return th / (2.0 * math.sin(th)) * w


def tilt_outward(T_base_tcp: np.ndarray, tilt: float) -> np.ndarray:
    """
    Lean the tool z axis away from the base z axis by ``tilt`` rad.

    Rotates the orientation about the horizontal axis perpendicular to the
    radial direction of the TCP (in base XY), keeping the TCP position. For a
    top-down pose with the closing axis tangential (``neutral_closing_angle``)
    the closing axis stays horizontal. Positive tilt points the approach (tool
    z) axis away from the base, so the wrist moves toward the base and up;
    that is the direction joint5's limit requires for high poses.
    """
    T = np.asarray(T_base_tcp, dtype=float)
    x, y = float(T[0, 3]), float(T[1, 3])
    r = math.hypot(x, y)
    if r < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = np.array([-y / r, x / r, 0.0])
    # Rotation about the tangential axis by -tilt moves a down vector outward.
    a = -float(tilt)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    Rt = np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)
    return make_T(Rt @ T[:3, :3], T[:3, 3])


class PiperKinematics:
    """
    FK, geometric Jacobian and numeric IK of the PiPER arm family.

    model: 'piper' | 'piper_h' | 'piper_l' | 'piper_x'. joint_limits
    optionally overrides the SDK preset with six (lower, upper) pairs (rad).
    """

    n_joints = 6

    def __init__(self, model: str = 'piper',
                 joint_limits: Optional[Sequence[Sequence[float]]] = None):
        if model not in MDH_PRESETS:
            raise ValueError('unknown robot model %r, expected one of %s' % (model, MODELS))
        self.model = model
        self.mdh = MDH_PRESETS[model]
        lim = JOINT_LIMITS[model] if joint_limits is None else joint_limits
        lim = np.array(lim, dtype=float)
        if lim.shape != (6, 2) or not np.all(np.isfinite(lim)) or np.any(lim[:, 0] > lim[:, 1]):
            raise ValueError('joint_limits must be six finite (lower, upper) pairs')
        self._lim = lim
        # Precomputed per-link constants: (d, a, cos alpha, sin alpha, offset).
        self._links = [(d, a, math.cos(al), math.sin(al), off) for d, a, al, off in self.mdh]
        self._d6 = float(self.mdh[-1][0])

    # ------------------------------------------------------------------ limits
    @property
    def limits(self) -> np.ndarray:
        """Joint limits as a (6, 2) array [lower, upper] in rad (copy)."""
        return self._lim.copy()

    def within_limits(self, q: Sequence[float], tol: float = 1e-9) -> bool:
        """Return True when q has 6 finite values inside the joint limits."""
        try:
            arr = _as_joints(q, self.n_joints)
        except ValueError:
            return False
        return bool(np.all(arr >= self._lim[:, 0] - tol) and np.all(arr <= self._lim[:, 1] + tol))

    def limit_violations(self, q: Sequence[float], tol: float = 1e-9) -> List[str]:
        """Describe each joint outside its limit, e.g. 'joint5=1.300 > 1.222'."""
        arr = _as_joints(q, self.n_joints)
        out = []
        for i, v in enumerate(arr):
            lo, hi = self._lim[i]
            if v < lo - tol:
                out.append('joint%d=%.3f < %.3f' % (i + 1, v, lo))
            elif v > hi + tol:
                out.append('joint%d=%.3f > %.3f' % (i + 1, v, hi))
        return out

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        """Clip q into the joint limits."""
        return np.clip(_as_joints(q, self.n_joints), self._lim[:, 0], self._lim[:, 1])

    # ---------------------------------------------------------------------- FK
    def _chain(self, q: np.ndarray) -> List[np.ndarray]:
        frames = []
        T = np.eye(4)
        for (d, a, ca, sa, off), qi in zip(self._links, q):
            th = qi + off
            ct, st = math.cos(th), math.sin(th)
            L = np.array([[ct, -st, 0.0, a],
                          [ca * st, ca * ct, -sa, -sa * d],
                          [sa * st, sa * ct, ca, ca * d],
                          [0.0, 0.0, 0.0, 1.0]])
            T = T @ L
            frames.append(T)
        return frames

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """T_base_flange (4x4) for joint angles q (rad)."""
        return self._chain(_as_joints(q, self.n_joints))[-1].copy()

    def fk_all(self, q: Sequence[float]) -> List[np.ndarray]:
        """T_base_link_i for i = 1..6 (the last one is the flange)."""
        return [T.copy() for T in self._chain(_as_joints(q, self.n_joints))]

    @staticmethod
    def _jacobian_from(frames: List[np.ndarray]) -> np.ndarray:
        pe = frames[-1][:3, 3]
        J = np.empty((6, len(frames)))
        for i, T in enumerate(frames):
            z = T[:3, 2]
            r = pe - T[:3, 3]
            J[0, i] = z[1] * r[2] - z[2] * r[1]
            J[1, i] = z[2] * r[0] - z[0] * r[2]
            J[2, i] = z[0] * r[1] - z[1] * r[0]
            J[3:, i] = z
        return J

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """
        Geometric Jacobian (6x6) at the flange origin, base frame.

        Rows 0-2: linear velocity (m/s per rad/s); rows 3-5: angular velocity.
        """
        return self._jacobian_from(self._chain(_as_joints(q, self.n_joints)))

    # ---------------------------------------------------------------------- IK
    @staticmethod
    def _error(T: np.ndarray, Tt: np.ndarray) -> Tuple[np.ndarray, float, float]:
        ep = Tt[:3, 3] - T[:3, 3]
        eo = rot_log(Tt[:3, :3] @ T[:3, :3].T)
        return np.concatenate([ep, eo]), float(np.linalg.norm(ep)), float(np.linalg.norm(eo))

    def standard_seeds(self, T_base_flange: np.ndarray) -> List[np.ndarray]:
        """Seeds for top-down-like targets: j1 from the azimuth, j6 from the twist."""
        Tt = np.asarray(T_base_flange, dtype=float)
        wc = Tt[:3, 3] - self._d6 * Tt[:3, 2]
        j1 = math.atan2(wc[1], wc[0]) if math.hypot(wc[0], wc[1]) > 1e-6 else 0.0
        if self.model == 'piper_x':
            j1 = j1 + math.pi if j1 < 0 else j1 - math.pi   # theta_offset of pi on link 1
        seeds = []
        for j2, j3, j5 in _TOP_DOWN_SEEDS:
            for s5 in (1.0, -1.0):
                q = self.clamp([j1, j2, j3, 0.0, s5 * j5, 0.0])
                R6 = self._chain(q)[-1][:3, :3]
                Rres = R6.T @ Tt[:3, :3]
                if Rres[2, 2] > 0.3:
                    q[5] = math.atan2(Rres[1, 0], Rres[0, 0])
                    seeds.append(self.clamp(q))
        return seeds

    def _solve(self, Tt: np.ndarray, q0: np.ndarray, max_iter: int, tol_pos: float,
               tol_rot: float) -> Tuple[np.ndarray, float, float, bool]:
        lo, hi = self._lim[:, 0], self._lim[:, 1]
        w = np.array([1.0, 1.0, 1.0, _ROT_WEIGHT, _ROT_WEIGHT, _ROT_WEIGHT])
        q = np.clip(q0, lo, hi)
        frames = self._chain(q)
        e, ep, eo = self._error(frames[-1], Tt)
        cost = float(np.sum((w * e) ** 2))
        lam = 1e-3
        stall = 0
        for _ in range(max_iter):
            if ep < tol_pos and eo < tol_rot:
                return q, ep, eo, True
            J = self._jacobian_from(frames) * w[:, None]
            ew = w * e
            # Joints pinned at a limit and pushed outward are dropped from the step.
            g = J.T @ ew
            pinned = ((q <= lo + 1e-9) & (g < 0)) | ((q >= hi - 1e-9) & (g > 0))
            Ja = J.copy()
            Ja[:, pinned] = 0.0
            H = Ja.T @ Ja
            dq = np.linalg.solve(H + lam * np.eye(6), Ja.T @ ew)
            n = float(np.max(np.abs(dq)))
            if n > _MAX_STEP:
                dq *= _MAX_STEP / n
            q_new = np.clip(q + dq, lo, hi)
            frames_new = self._chain(q_new)
            e_new, ep_new, eo_new = self._error(frames_new[-1], Tt)
            cost_new = float(np.sum((w * e_new) ** 2))
            if cost_new < cost:
                stall = stall + 1 if cost - cost_new < 1e-12 + 1e-6 * cost else 0
                q, frames, e, ep, eo, cost = q_new, frames_new, e_new, ep_new, eo_new, cost_new
                lam = max(lam / 3.0, 1e-7)
            else:
                stall += 1
                lam = min(lam * 5.0, 1e3)
            if stall >= 8 or lam >= 1e3:
                break
        return q, ep, eo, (ep < tol_pos and eo < tol_rot)

    def ik(self, T_base_flange: np.ndarray, q_seed: Optional[Sequence[float]] = None,
           max_iter: int = 60, tol_pos: float = 1e-4, tol_rot: float = 1e-3,
           n_random: int = 4) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Solve joint angles for a flange pose, respecting the joint limits.

        Damped least squares (Levenberg-Marquardt) with joint-limit clamping.
        Tries q_seed first, then standard top-down seeds, then ``n_random``
        deterministic random seeds. Returns (q, pos_err m, rot_err rad); q is
        None when no seed converged within tol_pos / tol_rot, and the errors are
        then those of the best attempt.

        Raises ValueError for a malformed target (shape, NaN, not a rotation).
        """
        Tt = _check_pose(T_base_flange, 'T_base_flange')
        seeds: List[np.ndarray] = []
        if q_seed is not None:
            seeds.append(self.clamp(q_seed))
        seeds.extend(self.standard_seeds(Tt))
        if n_random > 0:
            rng = np.random.RandomState(1234)
            lo, hi = self._lim[:, 0], self._lim[:, 1]
            for _ in range(n_random):
                seeds.append(rng.uniform(lo, hi))
        best = (math.inf, math.inf, math.inf)
        for s in seeds:
            q, ep, eo, ok = self._solve(Tt, s, max_iter, tol_pos, tol_rot)
            if ok:
                return q, ep, eo
            c = ep * ep + (_ROT_WEIGHT * eo) ** 2
            if c < best[0]:
                best = (c, ep, eo)
        return None, best[1], best[2]

    def refine(self, T_base_flange: np.ndarray, q: Sequence[float], tol_pos: float = 1e-7,
               tol_rot: float = 1e-6, max_iter: int = 30) -> Tuple[np.ndarray, float, float]:
        """
        Polish an IK solution from q only (no other seeds).

        Returns (q, pos_err m, rot_err rad); q is the input unchanged when the
        polish does not reach the tighter tolerances.
        """
        Tt = _check_pose(T_base_flange, 'T_base_flange')
        q0 = self.clamp(q)
        qr, ep, eo, ok = self._solve(Tt, q0, max_iter, tol_pos, tol_rot)
        if ok:
            return qr, ep, eo
        _e, ep0, eo0 = self._error(self._chain(q0)[-1], Tt)
        return q0, ep0, eo0

    def ik_tcp(self, T_base_tcp: np.ndarray, T_flange_tcp: np.ndarray,
               q_seed: Optional[Sequence[float]] = None,
               **kwargs) -> Tuple[Optional[np.ndarray], float, float]:
        """IK for a TCP pose: solves T_base_flange = T_base_tcp @ inv(T_flange_tcp)."""
        T = _check_pose(T_base_tcp, 'T_base_tcp')
        return self.ik(T @ inv_T(T_flange_tcp), q_seed=q_seed, **kwargs)

    def solve_tcp_with_tilt(self, T_base_tcp: np.ndarray, T_flange_tcp: np.ndarray,
                            max_tilt: float = math.radians(30.0),
                            step: float = math.radians(5.0),
                            q_seed: Optional[Sequence[float]] = None
                            ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Find the least outward tilt (see tilt_outward) that makes a TCP pose reachable.

        Tries tilt = 0, step, 2 * step, ... up to max_tilt (rad). Returns
        (T_base_tcp actually used, q, tilt) or None when nothing is reachable.
        """
        if not (step > 0 and math.isfinite(step)) or not math.isfinite(max_tilt):
            raise ValueError('step must be > 0 and max_tilt finite')
        n = int(math.floor(max(0.0, max_tilt) / step + 1e-9))
        for k in range(n + 1):
            tilt = k * step
            Tc = tilt_outward(T_base_tcp, tilt) if tilt > 0 else np.asarray(T_base_tcp, float)
            q, _ep, _eo = self.ik_tcp(Tc, T_flange_tcp, q_seed=q_seed)
            if q is not None:
                return Tc, q, tilt
        return None

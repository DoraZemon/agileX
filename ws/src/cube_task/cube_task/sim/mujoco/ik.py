"""
Damped-least-squares inverse kinematics on a MuJoCo model (PiPER arm).

Inside a MuJoCo simulation the MuJoCo model is the robot: its kinematics
define where the flange and the TCP are, the way a well-calibrated real
controller defines them. ``MujocoIK`` solves joint angles for a 4x4 tool pose
(position and orientation) of a frame rigidly attached to one body: a site
(for example the TCP site), or a body plus an offset ``T_body_tool`` (the
flange plus ``robot.tcp_offset``). It works on a private ``MjData`` (only
``mj_kinematics`` + ``mj_comPos`` + ``mj_jac``), so it never touches the
simulated state.

Poses are expressed in the frame of ``base_body`` (the PiPER base frame,
default body ``base_link``; the world frame when that body does not exist).
The base is assumed fixed in the world. Units: meters, radians.

All mujoco imports are lazy (engine.require_mujoco).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...core.transforms import inv_T, is_rotation, make_T
from ...robot.kinematics import rot_log
from .engine import require_mujoco

ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')

# Typical (j2, j3, j5) of top-down poses over the table in front of a PiPER; j1
# comes from the target azimuth and j6 from the target twist (same templates as
# robot.kinematics, the MuJoCo model uses the SDK joint conventions).
_TOP_DOWN_SEEDS = (
    (1.6, -1.2, 1.1), (1.0, -0.4, 0.9), (2.1, -1.7, 1.1), (1.3, -1.0, 1.2),
    (1.8, -1.2, 0.5), (0.8, -0.7, 0.6), (2.4, -2.2, 1.2),
)
_ROT_WEIGHT = 0.1          # m per rad when mixing position and rotation errors
_MAX_STEP = 0.35           # rad, largest joint change per iteration
_WRIST_OFFSET = 0.2        # m, rough TCP -> wrist distance along the tool z (seeds only)


def _check_pose(T, name: str = 'pose') -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError('%s must be a 4x4 matrix, got shape %s' % (name, T.shape))
    if not np.all(np.isfinite(T)):
        raise ValueError('%s contains NaN / inf' % name)
    if not is_rotation(T[:3, :3], tol=1e-3):
        raise ValueError('%s rotation part is not a rotation matrix' % name)
    return T


def _wrap_into(v: float, lo: float, hi: float) -> float:
    """Shift v by multiples of 2 pi into [lo, hi] when possible (else unchanged)."""
    for k in (0, -1, 1, -2, 2):
        w = v + 2.0 * math.pi * k
        if lo <= w <= hi:
            return w
    return v


class MujocoIK:
    """
    Joint-space IK for a tool frame on a MuJoCo model.

    model: mujoco.MjModel. joints: the 6 hinge joints solved for (other joints
    stay at qpos0 in the private data). Tool frame: ``site`` (name; the
    site's body and local pose), or ``body`` (name) with ``T_body_tool``
    (4x4, default identity). base_body: frame of all poses (None or missing
    = world). limits: (6, 2) [lower, upper] in rad, default the model's
    joint ranges.
    """

    def __init__(self, model, joints: Sequence[str] = ARM_JOINTS, body: Optional[str] = None,
                 T_body_tool: Optional[np.ndarray] = None, site: Optional[str] = None,
                 base_body: Optional[str] = 'base_link',
                 limits: Optional[Sequence[Sequence[float]]] = None):
        mj = require_mujoco()
        self.mujoco = mj
        self.m = model
        self.d = mj.MjData(model)
        self.joint_names = tuple(joints)
        if len(self.joint_names) != 6:
            raise ValueError('MujocoIK needs 6 joints, got %d' % len(self.joint_names))
        jids = []
        for name in self.joint_names:
            j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise KeyError('no joint named %r' % name)
            if int(model.jnt_type[j]) not in (int(mj.mjtJoint.mjJNT_HINGE),
                                              int(mj.mjtJoint.mjJNT_SLIDE)):
                raise ValueError('joint %r is not a hinge / slide joint' % name)
            jids.append(int(j))
        self.joint_ids = np.array(jids, dtype=int)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
        if site is not None:
            s = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site)
            if s < 0:
                raise KeyError('no site named %r' % site)
            self.body_id = int(model.site_bodyid[s])
            R = np.zeros(9)
            mj.mju_quat2Mat(R, np.asarray(model.site_quat[s], dtype=float))   # w x y z
            self.T_body_tool = make_T(R.reshape(3, 3), np.asarray(model.site_pos[s], dtype=float))
        else:
            if body is None:
                raise ValueError('give a site or a body for the tool frame')
            b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
            if b < 0:
                raise KeyError('no body named %r' % body)
            self.body_id = int(b)
            self.T_body_tool = (np.eye(4) if T_body_tool is None
                                else _check_pose(T_body_tool, 'T_body_tool').copy())
        if limits is None:
            lim = np.array([model.jnt_range[j] for j in jids], dtype=float)
            hinge = int(mj.mjtJoint.mjJNT_HINGE)
            for i, j in enumerate(jids):
                if not model.jnt_limited[j]:
                    lim[i] = (-math.pi, math.pi) if int(model.jnt_type[j]) == hinge else (-1., 1.)
        else:
            lim = np.array(limits, dtype=float)
        if lim.shape != (6, 2) or not np.all(np.isfinite(lim)) or np.any(lim[:, 0] > lim[:, 1]):
            raise ValueError('limits must be six finite (lower, upper) pairs')
        self._lim = lim
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        mj.mj_kinematics(model, self.d)
        self.T_world_base = np.eye(4)
        if base_body is not None:
            b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_body)
            if b >= 0:
                self.T_world_base = self._body_T(int(b))
        self.T_base_world = inv_T(self.T_world_base)
        self.n_evals = 0

    # ------------------------------------------------------------------ limits
    @property
    def limits(self) -> np.ndarray:
        """Joint limits as a (6, 2) array [lower, upper] in rad (copy)."""
        return self._lim.copy()

    def within_limits(self, q: Sequence[float], tol: float = 1e-9) -> bool:
        """Return True when q has 6 finite values inside the limits."""
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape[0] != 6 or not np.all(np.isfinite(arr)):
            return False
        return bool(np.all(arr >= self._lim[:, 0] - tol) and np.all(arr <= self._lim[:, 1] + tol))

    def limit_violations(self, q: Sequence[float], tol: float = 1e-9) -> List[str]:
        """Describe each joint outside its limit, e.g. 'joint5=1.300 > 1.220'."""
        arr = np.asarray(q, dtype=float).reshape(-1)
        out = []
        for i, v in enumerate(arr[:6]):
            lo, hi = self._lim[i]
            if v < lo - tol:
                out.append('%s=%.3f < %.3f' % (self.joint_names[i], v, lo))
            elif v > hi + tol:
                out.append('%s=%.3f > %.3f' % (self.joint_names[i], v, hi))
        return out

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        """Clip q into the limits."""
        return np.clip(np.asarray(q, dtype=float).reshape(-1)[:6], self._lim[:, 0],
                       self._lim[:, 1])

    # -------------------------------------------------------------- kinematics
    def _body_T(self, b: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.d.xmat[b], dtype=float).reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def _set_q(self, q: np.ndarray, jac: bool = False) -> np.ndarray:
        """Run kinematics at q; return the world tool pose (Jacobian data with jac)."""
        mj = self.mujoco
        self.d.qpos[self.qadr] = q
        mj.mj_kinematics(self.m, self.d)
        if jac:
            mj.mj_comPos(self.m, self.d)
        self.n_evals += 1
        return self._body_T(self.body_id) @ self.T_body_tool

    def _jac_world(self, p_tool: np.ndarray) -> np.ndarray:
        """6x6 world Jacobian of the tool point (call after _set_q(..., jac=True))."""
        self.mujoco.mj_jac(self.m, self.d, self._jacp, self._jacr, p_tool, self.body_id)
        return np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Return T_base_tool (4x4) for joint angles q."""
        return self.T_base_world @ self._set_q(np.asarray(q, dtype=float).reshape(-1)[:6])

    def fk_body(self, q: Sequence[float]) -> np.ndarray:
        """Return T_base_body (4x4) of the tool's body (the flange) for joint angles q."""
        self._set_q(np.asarray(q, dtype=float).reshape(-1)[:6])
        return self.T_base_world @ self._body_T(self.body_id)

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """
        Geometric Jacobian (6x6) of the tool point in the base frame.

        Rows 0-2: linear velocity (m/s per rad/s); rows 3-5: angular velocity.
        """
        Tw = self._set_q(np.asarray(q, dtype=float).reshape(-1)[:6], jac=True)
        J = self._jac_world(Tw[:3, 3].copy())
        R = self.T_base_world[:3, :3]
        return np.vstack([R @ J[:3], R @ J[3:]])

    # ---------------------------------------------------------------------- IK
    @staticmethod
    def _error(Tw: np.ndarray, Tt: np.ndarray) -> Tuple[np.ndarray, float, float]:
        ep = Tt[:3, 3] - Tw[:3, 3]
        eo = rot_log(Tt[:3, :3] @ Tw[:3, :3].T)
        return np.concatenate([ep, eo]), float(np.linalg.norm(ep)), float(np.linalg.norm(eo))

    def standard_seeds(self, T_base_tool: np.ndarray) -> List[np.ndarray]:
        """Seeds for top-down-like targets: j1 from the azimuth, j6 from the twist."""
        Tt = np.asarray(T_base_tool, dtype=float)
        wc = Tt[:3, 3] - _WRIST_OFFSET * Tt[:3, 2]
        j1 = math.atan2(wc[1], wc[0]) if math.hypot(wc[0], wc[1]) > 1e-6 else 0.0
        seeds = []
        for j2, j3, j5 in _TOP_DOWN_SEEDS:
            for s5 in (1.0, -1.0):
                q = self.clamp([j1, j2, j3, 0.0, s5 * j5, 0.0])
                Rres = self.fk(q)[:3, :3].T @ Tt[:3, :3]
                if Rres[2, 2] > 0.3:
                    q[5] = _wrap_into(math.atan2(Rres[1, 0], Rres[0, 0]), *self._lim[5])
                    seeds.append(self.clamp(q))
        return seeds

    def _solve(self, Tt_world: np.ndarray, q0: np.ndarray, max_iter: int, tol_pos: float,
               tol_rot: float) -> Tuple[np.ndarray, float, float, bool]:
        lo, hi = self._lim[:, 0], self._lim[:, 1]
        w = np.array([1.0, 1.0, 1.0, _ROT_WEIGHT, _ROT_WEIGHT, _ROT_WEIGHT])
        q = np.clip(q0, lo, hi)
        Tw = self._set_q(q, jac=True)
        e, ep, eo = self._error(Tw, Tt_world)
        cost = float(np.sum((w * e) ** 2))
        lam = 1e-3
        stall = 0
        J = None
        for _ in range(max_iter):
            if ep < tol_pos and eo < tol_rot:
                return q, ep, eo, True
            if J is None:
                J = self._jac_world(Tw[:3, 3].copy()) * w[:, None]
            ew = w * e
            # Joints pinned at a limit and pushed outward are dropped from the step.
            g = J.T @ ew
            pinned = ((q <= lo + 1e-9) & (g < 0)) | ((q >= hi - 1e-9) & (g > 0))
            Ja = J.copy()
            Ja[:, pinned] = 0.0
            dq = np.linalg.solve(Ja.T @ Ja + lam * np.eye(6), Ja.T @ ew)
            n = float(np.max(np.abs(dq)))
            if n > _MAX_STEP:
                dq *= _MAX_STEP / n
            q_new = np.clip(q + dq, lo, hi)
            Tw_new = self._set_q(q_new, jac=True)
            e_new, ep_new, eo_new = self._error(Tw_new, Tt_world)
            cost_new = float(np.sum((w * e_new) ** 2))
            if cost_new < cost:
                stall = stall + 1 if cost - cost_new < 1e-14 + 1e-6 * cost else 0
                q, Tw, e, ep, eo, cost = q_new, Tw_new, e_new, ep_new, eo_new, cost_new
                J = None
                lam = max(lam / 3.0, 1e-7)
            else:   # rejected: keep q and its cached Jacobian, damp more
                stall += 1
                lam = min(lam * 5.0, 1e3)
            if stall >= 8 or lam >= 1e3:
                break
        return q, ep, eo, (ep < tol_pos and eo < tol_rot)

    def solve(self, T_base_tool: np.ndarray, q_seed: Optional[Sequence[float]] = None,
              max_iter: int = 80, tol_pos: float = 1e-4, tol_rot: float = 1e-3,
              n_random: int = 4, seeds: Optional[Sequence[Sequence[float]]] = None,
              standard: bool = True, accept_pos: Optional[float] = None,
              accept_rot: Optional[float] = None
              ) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Solve joint angles for a tool pose in the base frame, inside the limits.

        Levenberg-Marquardt (damped least squares) with limit clamping. Seeds in
        order: q_seed, ``seeds``, the standard top-down seeds (standard=True),
        then ``n_random`` deterministic random seeds; the first converged
        solution wins. Returns (q, pos_err m, rot_err rad); q is None when no
        seed converged within tol_pos / tol_rot, and the errors are then those
        of the best attempt. With accept_pos / accept_rot (m / rad), the best
        attempt is returned instead of None when its residual is within them
        (the closest pose for a target just outside the reachable set, e.g.
        at a joint limit).

        Raises ValueError for a malformed target (shape, NaN, not a rotation).
        """
        Tt = self.T_world_base @ _check_pose(T_base_tool, 'T_base_tool')
        cand: List[np.ndarray] = []
        if q_seed is not None:
            cand.append(self.clamp(q_seed))
        for s in seeds or ():
            cand.append(self.clamp(s))
        if standard:
            cand.extend(self.standard_seeds(T_base_tool))
        if n_random > 0:
            rng = np.random.RandomState(1234)
            for _ in range(n_random):
                cand.append(rng.uniform(self._lim[:, 0], self._lim[:, 1]))
        best = (math.inf, None, math.inf, math.inf)
        for s in cand:
            q, ep, eo, ok = self._solve(Tt, s, max_iter, tol_pos, tol_rot)
            if ok:
                return q, ep, eo
            c = ep * ep + (_ROT_WEIGHT * eo) ** 2
            if c < best[0]:
                best = (c, q, ep, eo)
        _c, q, ep, eo = best
        if (q is not None and accept_pos is not None and accept_rot is not None
                and ep <= accept_pos and eo <= accept_rot):
            return q, ep, eo
        return None, ep, eo

    def refine(self, T_base_tool: np.ndarray, q: Sequence[float], tol_pos: float = 1e-7,
               tol_rot: float = 1e-6, max_iter: int = 30) -> Tuple[np.ndarray, float, float]:
        """
        Polish a solution from q only (no other seeds).

        Returns (q, pos_err m, rot_err rad); q is the input (clamped) when the
        polish does not reach the tighter tolerances.
        """
        Tt = self.T_world_base @ _check_pose(T_base_tool, 'T_base_tool')
        q0 = self.clamp(q)
        qr, ep, eo, ok = self._solve(Tt, q0, max_iter, tol_pos, tol_rot)
        if ok:
            return qr, ep, eo
        _e, ep0, eo0 = self._error(self._set_q(q0), Tt)
        return q0, ep0, eo0

    def solve_with_tilt(self, T_base_tool: np.ndarray, max_tilt: float = math.radians(30.0),
                        step: float = math.radians(5.0),
                        q_seed: Optional[Sequence[float]] = None, **kw
                        ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Find the least outward tilt (robot.kinematics.tilt_outward) that is solvable.

        Tries tilt = 0, step, 2 * step, ... up to max_tilt (rad); kw go to
        solve() (default n_random=0). Returns (T_base_tool used, q, tilt) or
        None when nothing converges.
        """
        from ...robot.kinematics import tilt_outward
        kw.setdefault('n_random', 0)
        if not (step > 0 and math.isfinite(step)) or not math.isfinite(max_tilt):
            raise ValueError('step must be > 0 and max_tilt finite')
        n = int(math.floor(max(0.0, max_tilt) / step + 1e-9))
        for k in range(n + 1):
            tilt = k * step
            Tc = tilt_outward(T_base_tool, tilt) if tilt > 0 else np.asarray(T_base_tool, float)
            q, _ep, _eo = self.solve(Tc, q_seed=q_seed, **kw)
            if q is not None:
                return Tc, q, tilt
        return None

    def solve_path(self, poses: Sequence[np.ndarray], q_start: Sequence[float],
                   max_joint_step: float = 0.2, tol_pos: float = 1e-4,
                   tol_rot: float = 1e-3, accept_pos: Optional[float] = None,
                   accept_rot: Optional[float] = None) -> Tuple[Optional[np.ndarray], str]:
        """
        Solve a dense sequence of tool poses by continuation from q_start.

        Each pose is seeded only with the previous solution (no other seeds, so
        the arm cannot switch to another IK branch mid-path; accept_pos /
        accept_rot as in solve()). Returns (q array (n, 6), '') or (None,
        reason) when a pose has no solution or a joint would jump more than
        max_joint_step rad between neighbors.
        """
        q_prev = self.clamp(q_start)
        out = np.empty((len(poses), 6))
        for k, T in enumerate(poses):
            q, ep, eo = self.solve(T, q_seed=q_prev, max_iter=40, tol_pos=tol_pos,
                                   tol_rot=tol_rot, n_random=0, standard=False,
                                   accept_pos=accept_pos, accept_rot=accept_rot)
            if q is None:
                return None, ('no IK at waypoint %d/%d (residual %.1f mm, %.2f deg)'
                              % (k, len(poses) - 1, ep * 1e3, math.degrees(eo)))
            jump = float(np.max(np.abs(q - q_prev)))
            if k > 0 and jump > max_joint_step:
                return None, ('joint jump %.3f rad at waypoint %d/%d (singularity or limit)'
                              % (jump, k, len(poses) - 1))
            out[k] = q
            q_prev = q
        return out, ''

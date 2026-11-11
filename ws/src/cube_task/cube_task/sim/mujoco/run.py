"""
Helpers for running the task stack on the MuJoCo physics scene.

- scene_config(info, base_cfg): the config a MuJoCo scenario run uses
  (scenario overlay + MujocoWorld.config_overlay), with the world.
- observe_frame / collect_samples: perception features of MuJoCo renders
  labeled with the scene's ground truth (as sim/teach.py does for the
  kinematic renderer). Used to check that the simulation KNN model, trained
  on the kinematic renderer, still classifies MuJoCo renders (measured
  2026-11-06: 95/95 eye_to_hand and 96/96 eye_in_hand samples of seeds
  200-215 correct), so --mode mujoco uses that model.
- PhysicsMonitor: a frame hook that records what the physics did to every
  object (grasped, released, landed, pushed, rolled, dropped) as run-log
  events and a summary; failure_causes() turns it plus the runner's pick
  results into one cause per failed object.
- final_truth(world, info): let the scene come to rest, then score it.
- Viewer: optional mujoco.viewer passive window synced at a frame rate,
  paced to wall time with an injected real clock.
- RunExtras: monitor + optional video + viewer of one run (run_task --mode
  mujoco, sim_bench --backend mujoco).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_LOG = logging.getLogger('cube_task.sim.mujoco.run')

#: seconds of physics after a round before the ground truth is read (objects come to rest).
FINAL_SETTLE_S = 1.0


# ------------------------------------------------------------------ config
def scene_config(info: Dict[str, Any], base_cfg: Any = None, sim_world: Any = None,
                 extrinsics_error: Sequence[float] = (0.0, 0.0)):
    """
    Return (cfg, MujocoWorld) for a scenario (no site overlays).

    base_cfg: core Config to start from (default: load_config(use_overlays=False)).
    The scenario overlay and the scene overlay are merged over it.
    """
    from ...core.config import Config, deep_merge, load_config
    from ..scenarios import scenario_config_overlay
    from .scene import build_scene
    base = base_cfg if base_cfg is not None else load_config(use_overlays=False)
    err = tuple(float(v) for v in extrinsics_error)
    cfg = Config(deep_merge(base.data, scenario_config_overlay(info, extrinsics_error=err)),
                 base.root, list(base.sources))
    mw = build_scene(info, cfg, world=sim_world)
    cfg = Config(deep_merge(cfg.data, mw.config_overlay(extrinsics_error=err)), cfg.root,
                 list(cfg.sources))
    return cfg, mw


# --------------------------------------------------------------------- KNN
def observe_frame(cfg: Any, mw: Any, clock: Any, settle_s: float = 0.3):
    """
    Put the arm at robot.poses.observe_joints, let the scene settle, render one frame.

    Returns (frame, T_base_camera from the configured extrinsics). Creates
    and frees a MujocoArm and a MujocoCamera on mw (attached to clock).
    """
    from ...core.extrinsics import camera_pose_in_base, extrinsics_from_cfg
    from .arm import MujocoArm
    from .camera import MujocoCamera
    sim = mw.attach(clock)
    arm = MujocoArm(cfg, clock, sim)
    q = cfg.get('robot.poses.observe_joints', None)
    if q is not None:
        arm.set_joints(q)
    mw.settle(settle_s)
    cam = MujocoCamera(mw, clock, cfg)
    cam.start()
    try:
        frame = cam.get_frame(timeout=1.0, newer_than=clock.now())
    finally:
        cam.stop()
    arm.connect()                       # feedback only; nothing moves
    T_bf = arm.get_flange_pose().value if mw.info['mount'] == 'eye_in_hand' else None
    return frame, camera_pose_in_base(extrinsics_from_cfg(cfg), T_bf)


def match_objects(mw: Any, detections: Sequence[Any], match_dist: float = 0.02
                  ) -> List[Tuple[Any, Optional[str]]]:
    """Return [(detection, object name or None)] by the nearest object frame origin in XY."""
    out = []
    poses = mw.object_poses()
    for d in detections:
        if d.center_base is None:
            out.append((d, None))
            continue
        c = np.asarray(d.center_base[:2], dtype=float)
        name, dist = None, float('inf')
        for n, T in poses.items():
            dd = float(np.hypot(*(T[:2, 3] - c)))
            if dd < dist:
                name, dist = n, dd
        out.append((d, name if dist <= match_dist else None))
    return out


def collect_samples(seeds: Iterable[int], mount: str = 'eye_to_hand', task: str = 'L2',
                    match_dist: float = 0.02, base_cfg: Any = None):
    """
    Return (X, y, feature_names) from perception on MuJoCo renders of scenarios.

    Like sim.teach.collect_sim_samples: odd seeds use the hard difficulty, the
    label of a detection is the label of the nearest scene object.
    """
    from ...core.clock import SimClock
    from ...perception.features import feature_vector
    from ...perception.pipeline import Perception
    from ..scenarios import make_scenario
    X, y, names = [], [], None
    for s in seeds:
        world, info = make_scenario(task, int(s), mount=mount,
                                    difficulty='hard' if int(s) % 2 else 'normal')
        cfg, mw = scene_config(info, base_cfg, sim_world=world)
        frame, T_bc = observe_frame(cfg, mw, SimClock())
        if frame is None:
            continue
        res = Perception(cfg).process(frame, T_bc)
        for d, name in match_objects(mw, res.valid_detections(), match_dist):
            if name is None:
                continue
            v, names = feature_vector(frame, d)
            if np.all(np.isfinite(v)):
                X.append(v)
                y.append(mw.object(name).label)
    return np.array(X), y, names


# ----------------------------------------------------------------- monitor
MONITOR_HZ = 25.0
MOVE_TOL = 0.01              # m, an object at rest that ends this far away was moved
HOLD_TICKS = 2               # consecutive ticks touching both fingers = grasped
RELEASE_TICKS = 3            # consecutive ticks not held = released


class _ObjState:
    def __init__(self, com: np.ndarray):
        self.state = 'rest'                 # rest | moving | held | falling
        self.rest = com.copy()
        self.held_ticks = 0
        self.free_ticks = 0
        self.release_com: Optional[np.ndarray] = None
        self.touch: set = set()
        self.grasps = 0
        self.off = False


class _EventTap:
    """Delegate to a logger (or nothing) and feed every event to a listener first."""

    def __init__(self, inner: Any, listener):
        from .video import _NullLogger
        self._inner = inner if inner is not None else _NullLogger()
        self._listener = listener

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def event(self, kind: str, **fields: Any) -> None:
        try:
            self._listener(kind, fields)
        except Exception:  # noqa: B902 - analysis must never break logging
            _LOG.exception('physics monitor event failed')
        self._inner.event(kind, **fields)


class PhysicsMonitor:
    """
    Record what the physics did to every object during a run.

    A frame hook at MONITOR_HZ follows each object through rest -> held ->
    falling -> rest (or rest -> moving -> rest when something pushed it) and
    records 'grasped', 'released', 'landed', 'disturbed' and 'off_table'
    events with the sim time, the contacts seen meanwhile ('arm' = robot
    links, 'finger', 'bin' = zone floor / walls, 'object', 'table', 'floor')
    and the zone it ended in. Wrap the run logger with wrap_logger() to also
    record the runner's pick decisions and results (picks), and to write the
    physics events into the run log (kind 'physics').
    """

    def __init__(self, world: Any, logger: Any = None, hz: float = MONITOR_HZ):
        self.world = world
        self.logger = logger
        self.hz = float(hz)
        self.events: List[Dict[str, Any]] = []
        self.picks: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []
        self._obj: Dict[str, _ObjState] = {}
        self._started = False
        from .engine import require_mujoco
        mj = require_mujoco()
        m = world.model
        base_name = str(world.meta.get('base_body', 'base_link'))
        base = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, base_name))
        fingers = set(int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b))
                      for b in world.meta['finger_bodies'])
        zone_bodies = set(int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b))
                          for b in world.meta.get('zone_bodies', {}).values())
        self._kind = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in fingers:
                self._kind[g] = 'finger'
            elif int(m.body_rootid[b]) == base:
                self._kind[g] = 'arm'
            elif b in zone_bodies:
                self._kind[g] = 'bin'
            elif world.geom_object(g) is not None:
                self._kind[g] = 'object'
        self._table = int(world.meta['table_geom'])
        self._floor = int(world.meta['floor_geom'])

    # ------------------------------------------------------------- wiring
    def wrap_logger(self, logger: Any) -> Any:
        """
        Return a logger proxy that records the runner events here (logger may be None).

        Physics events go to self.logger (the constructor argument; set it to
        the real run logger to have them in events.jsonl).
        """
        return _EventTap(logger, self.on_event)

    def start(self) -> 'PhysicsMonitor':
        """Begin monitoring at the current state (objects are taken as at rest)."""
        if self._started:
            return self
        sim = self.world.sim
        if sim is None:
            raise RuntimeError('PhysicsMonitor: attach the MujocoWorld to its clock first')
        for rec in self.world.objects:
            self._obj[rec.name] = _ObjState(self.world.object_com(rec.name))
        sim.add_frame_hook(self._tick, self.hz)
        self._started = True
        return self

    def stop(self) -> None:
        """Stop the frame hook."""
        if self._started and self.world.sim is not None:
            self.world.sim.remove_frame_hook(self._tick)
        self._started = False

    # ------------------------------------------------------- runner events
    def _nearest(self, xy: Sequence[float]) -> Tuple[Optional[str], float]:
        best, dist = None, float('inf')
        for rec in self.world.objects:
            c = self.world.object_com(rec.name)
            d = float(math.hypot(c[0] - float(xy[0]), c[1] - float(xy[1])))
            if d < dist:
                best, dist = rec.name, d
        return best, dist

    def on_event(self, kind: str, fields: Dict[str, Any]) -> None:
        """Record runner / skill events that the failure analysis needs."""
        t = round(float(self.world.sim.time), 3) if self.world.sim is not None else None
        if kind == 'decision' and fields.get('action') == 'pick':
            g = fields.get('grasp_xyz') or [float('nan')] * 3
            name, dist = self._nearest(g[:2])
            true = self.world.object(name).label if name is not None else None
            self.picks.append({'t': t, 'label': fields.get('label'), 'zone': fields.get('zone'),
                               'attempt': fields.get('attempt'), 'grasp_xyz': list(g),
                               'release_xyz': fields.get('release_xyz'),
                               'target': name, 'target_dist': round(dist, 4),
                               'true_label': true, 'steps': [], 'grasped': [],
                               'result': None})
        elif kind == 'skill_step' and self.picks and self.picks[-1]['result'] is None:
            self.picks[-1]['steps'].append({k: fields.get(k) for k in
                                            ('stage', 'ok', 'error', 'duration')})
        elif kind == 'pick_result' and self.picks and self.picks[-1]['result'] is None:
            self.picks[-1]['result'] = {k: fields.get(k) for k in
                                        ('ok', 'stage', 'error', 'grasp_width', 'fault')}
            self.picks[-1]['t_end'] = t
        elif kind == 'observation' and fields.get('ok'):
            self.observations.append({'t': t, 'detections': [
                {'label': d.get('label'), 'center_base': d.get('center_base'),
                 'valid': d.get('valid'), 'reject_reason': d.get('reject_reason')}
                for d in fields.get('detections') or []]})
            del self.observations[:-3]

    # ------------------------------------------------------------- physics
    def _contacts(self) -> Dict[str, set]:
        d = self.world.data
        out: Dict[str, set] = {}
        n = int(d.ncon)
        if n == 0:
            return out
        geoms = np.asarray(d.contact.geom[:n], dtype=int)
        for g1, g2 in geoms:
            for ga, gb in ((g1, g2), (g2, g1)):
                obj = self.world.geom_object(int(ga))
                if obj is None:
                    continue
                if gb == self._table:
                    k = 'table'
                elif gb == self._floor:
                    k = 'floor'
                else:
                    k = self._kind.get(int(gb), 'other')
                out.setdefault(obj, set()).add(k)
        return out

    def _emit(self, what: str, name: str, t: float, **fields: Any) -> None:
        ev = {'t': round(float(t), 3), 'what': what, 'object': name,
              'label': self.world.object(name).label}
        ev.update(fields)
        self.events.append(ev)
        if self.picks and self.picks[-1]['result'] is None and what == 'grasped':
            self.picks[-1]['grasped'].append(name)
        if self.logger is not None:
            fields = dict(ev)
            fields['sim_t'] = fields.pop('t')       # 't' is the logger's own time stamp
            try:
                self.logger.event('physics', **fields)
            except Exception:  # noqa: B902 - logging must never stop the physics
                _LOG.exception('logger.event failed')

    def _tick(self, sim: Any, t: float) -> None:
        w = self.world
        held = set(w.held_objects())
        touch = self._contacts()
        for rec in w.objects:
            st = self._obj[rec.name]
            com = w.object_com(rec.name)
            lin, ang = w.object_speed(rec.name)
            rest = lin < 0.02 and ang < 0.5
            st.touch |= touch.get(rec.name, set())
            is_held = rec.name in held
            st.held_ticks = st.held_ticks + 1 if is_held else 0
            st.free_ticks = 0 if is_held else st.free_ticks + 1
            if st.state != 'held' and st.held_ticks >= HOLD_TICKS:
                if st.state == 'moving' and float(np.linalg.norm(com - st.rest)) > MOVE_TOL:
                    self._emit('disturbed', rec.name, t, cause=self._push_cause(st),
                               dist=round(float(np.linalg.norm(com[:2] - st.rest[:2])), 4),
                               touch=sorted(st.touch), before_grasp=True)
                st.state = 'held'
                st.grasps += 1
                st.touch = set()
                self._emit('grasped', rec.name, t, com=_r(com), n=st.grasps)
                continue
            if st.state == 'held':
                if st.free_ticks == 1:
                    st.release_com = com.copy()
                if st.free_ticks >= RELEASE_TICKS:
                    st.state = 'falling'
                    p = st.release_com if st.release_com is not None else com
                    over = [z for z, zn in w.zones.items() if zn.contains_xy(p[0], p[1])]
                    tcp = self._tcp_z()
                    self._emit('released', rec.name, t, com=_r(p), over_zone=over[0]
                               if over else None, height=round(float(p[2]) - w.table_z, 4),
                               tcp_z=tcp, touch_while_held=sorted(st.touch))
                    st.touch = set()
                continue
            if st.state == 'rest':
                if (float(np.linalg.norm(com - st.rest)) > 0.003 or not rest):
                    st.state = 'moving'
                    st.touch = set(touch.get(rec.name, set()))
                continue
            # moving or falling: wait until it is at rest again
            if not w.on_table(com) and float(com[2]) < w.table_z - 0.05:
                if not st.off:
                    st.off = True
                    self._emit('off_table', rec.name, t, com=_r(com), was=st.state,
                               touch=sorted(st.touch))
            if rest:
                zone = w.zone_at(com)
                over = [zn for zn in w.zones.values() if zn.contains_xy(com[0], com[1])]
                if st.state == 'falling':
                    p = st.release_com if st.release_com is not None else com
                    self._emit('landed', rec.name, t, com=_r(com), zone=zone,
                               on_table=w.on_table(com),
                               dist=round(float(np.linalg.norm(com[:2] - p[:2])), 4),
                               touch=sorted(st.touch),
                               footprint=over[0].name if over else None,
                               above_floor=round(float(com[2] - over[0].center[2]), 4)
                               if over else None)
                elif float(np.linalg.norm(com - st.rest)) > MOVE_TOL:
                    self._emit('disturbed', rec.name, t, cause=self._push_cause(st),
                               dist=round(float(np.linalg.norm(com[:2] - st.rest[:2])), 4),
                               zone=zone, on_table=w.on_table(com), touch=sorted(st.touch),
                               footprint=over[0].name if over else None)
                st.state = 'rest'
                st.rest = com.copy()
                st.touch = set()

    @staticmethod
    def _push_cause(st: _ObjState) -> str:
        if 'finger' in st.touch:
            return 'pushed by the fingers'
        if 'arm' in st.touch:
            return 'pushed by the arm'
        if 'object' in st.touch:
            return 'knocked by another object'
        return 'rolled'

    def _tcp_z(self) -> Optional[float]:
        try:
            T = self.world.sim.site_pose(self.world.meta['tcp_site'])
        except (KeyError, AttributeError):
            return None
        return round(float(T[2, 3]), 4)

    def summary(self) -> Dict[str, Any]:
        """Return {events, picks, last_observation} (JSON-friendly)."""
        return {'events': list(self.events), 'picks': list(self.picks),
                'last_observation': self.observations[-1] if self.observations else None}


def _r(v: Any) -> List[float]:
    return [round(float(x), 4) for x in np.asarray(v).ravel()]


# --------------------------------------------------------------- analysis
def expected_zones(world: Any, info: Dict[str, Any]) -> Dict[str, str]:
    """Return object name -> the zone it must end in (targets only)."""
    c2z = dict(info.get('class_to_zone') or {})
    return {rec.name: c2z[rec.label] for rec in world.objects if rec.label in c2z}


def final_truth(world: Any, info: Dict[str, Any], settle_s: float = FINAL_SETTLE_S
                ) -> Dict[str, int]:
    """
    Let the scene come to rest for settle_s, then score it by physics.

    Returns {correct, wrong, remaining, dropped}: correct = target at rest in
    its zone; wrong = any object at rest in another zone (non-targets
    included); remaining = targets over the table or held; dropped = objects
    on the floor.
    """
    if settle_s > 0:
        world.settle(float(settle_s))
    return world.score(dict(info.get('class_to_zone') or {}))


_STEP_CAUSE = {'grasp_miss': 'grasp miss (closed on nothing)',
               'dropped': 'slipped out of the gripper during the lift',
               'dropped_in_transit': 'dropped in transit'}
_MOTION_CAUSE = {'unreachable': 'unreachable', 'rejected': 'unreachable (rejected by SafeArm)',
                 'stalled': 'arm stalled (collision)', 'timeout': 'motion timeout'}


def _pick_cause(p: Dict[str, Any], evs: List[Dict[str, Any]], name: str) -> Tuple[str, str]:
    """Return (cause, detail) of one failed or misplaced pick involving object name."""
    res = p.get('result') or {}
    t0, t1 = p['t'], p.get('t_end') or float('inf')
    during = [e for e in evs if t0 <= e['t'] <= t1 + 3.0]
    pushed = [e for e in during if e['what'] == 'disturbed']
    push_txt = '; %s %.0f mm' % (pushed[0]['cause'], 1e3 * pushed[0]['dist']) if pushed else ''
    if res.get('ok'):
        if p.get('label') != p.get('true_label') and p.get('target') == name:
            return ('perception: %s classified as %s' % (p.get('true_label'), p.get('label')),
                    'put into %s' % p.get('zone'))
        if p.get('grasped') and name not in p['grasped']:
            return 'grasped another object', 'grasped %s' % ', '.join(p['grasped'])
        rel = [e for e in during if e['what'] == 'released']
        land = [e for e in evs if e['what'] == 'landed' and e['t'] >= t0]
        if rel and rel[0]['over_zone'] != p.get('zone'):
            return 'released outside the zone', 'released at %s over %s' % (
                rel[0]['com'][:2], rel[0]['over_zone'])
        if land and land[0]['zone'] != p.get('zone'):
            lnd = land[0]
            if lnd.get('footprint') == p.get('zone'):
                # At rest over the zone but too high to count (ZONE_TOP): on top of the
                # earlier objects, or still hanging on a finger.
                after = [e for e in evs if e['what'] == 'disturbed' and e['t'] > lnd['t']
                         and e.get('footprint') != p.get('zone')]
                high = 'center of mass %.0f mm above the zone floor after the release' % (
                    1e3 * float(lnd['above_floor']))
                touch = lnd.get('touch') or []
                if 'finger' in touch and after:
                    return ('carried out of the zone by the fingers after the release (%s)'
                            % ('it lay on the objects piled in the zone' if 'object' in touch
                               else 'stuck to a finger'),
                            '%s; moved %.0f mm' % (high, 1e3 * after[0]['dist']))
                if after:
                    return ('piled up on the objects in the zone, then %s out'
                            % after[0]['cause'], '%s; moved %.0f mm'
                            % (high, 1e3 * after[0]['dist']))
                return 'piled up on the objects in the zone (too high to count)', high
            touch = lnd.get('touch') or []
            return ('bounced out of the zone%s' % (' (hit the bin wall)' if 'bin' in touch
                                                   else ''),
                    'landed %.0f mm from the release point, %s' % (
                        1e3 * land[0]['dist'], 'on the table' if land[0]['on_table']
                        else 'off the table'))
        later = [e for e in evs if e['what'] == 'disturbed' and e['t'] > t1]
        if later:
            return 'moved out of the zone later (%s)' % later[-1]['cause'], ''
        return 'placed but not counted (not at rest in the zone)', ''
    stage, err = str(res.get('stage') or ''), str(res.get('error') or '')
    if stage in _STEP_CAUSE:
        return _STEP_CAUSE[stage] + push_txt, 'grasp width %s' % res.get('grasp_width')
    if err in _MOTION_CAUSE:
        return '%s at %s' % (_MOTION_CAUSE[err], stage), push_txt.lstrip('; ')
    if stage == 'plan':
        return 'no feasible plan', err
    if res.get('fault'):
        land = [e for e in during if e['what'] == 'landed' and e.get('footprint')]
        if stage == 'open_release' and land:
            return ('fault at open_release: %s (fingers blocked by the objects piled in the '
                    'zone)' % err, 'center of mass %.0f mm above the zone floor'
                    % (1e3 * float(land[0]['above_floor'])))
        return 'fault at %s: %s' % (stage, err), ''
    if not res:
        return 'pick did not finish (round ended)', ''
    return '%s at %s' % (err or 'failed', stage), push_txt.lstrip('; ')


def failure_causes(world: Any, info: Dict[str, Any], monitor: Any,
                   stop_reason: str = '') -> List[Dict[str, Any]]:
    """
    Return one entry per object that ended wrong (a target outside its zone, any wrong zone).

    Entry: {object, label, final (status), zone, expected, kind (wrong |
    dropped | remaining | held), cause, detail, attempts}. The cause comes
    from the last pick that targeted or grasped the object (runner result +
    physics events), else from the physics events (pushed / rolled off) and
    the last observation (not detected / misclassified / not attempted).
    """
    exp = expected_zones(world, info)
    empty = {'events': [], 'picks': [], 'last_observation': None}
    summ = monitor.summary() if monitor is not None else empty
    out = []
    for name, st in world.object_status().items():
        want = exp.get(name)
        if st['status'] == 'zone' and st['zone'] == want:
            continue
        if want is None and st['status'] != 'zone':
            continue
        kind = ('wrong' if st['status'] == 'zone' else
                {'dropped': 'dropped', 'held': 'held'}.get(st['status'], 'remaining'))
        evs = [e for e in summ['events'] if e['object'] == name]
        picks = [p for p in summ['picks'] if p.get('target') == name or name in p['grasped']]
        cause, detail = _object_cause(name, st, kind, evs, picks, summ['last_observation'],
                                      stop_reason)
        out.append({'object': name, 'label': st['label'], 'final': st['status'],
                    'zone': st['zone'], 'expected': want, 'kind': kind, 'cause': cause,
                    'detail': detail, 'attempts': len(picks)})
    return out


def _object_cause(name: str, st: Dict[str, Any], kind: str, evs: List[Dict[str, Any]],
                  picks: List[Dict[str, Any]], obs: Optional[Dict[str, Any]],
                  stop_reason: str) -> Tuple[str, str]:
    if kind == 'held':
        why = _pick_cause(picks[-1], evs, name)[0] if picks else 'never picked'
        return 'still held at the end (%s)' % why, ''
    off = [e for e in evs if e['what'] == 'off_table']
    if picks:
        cause, detail = _pick_cause(picks[-1], evs, name)
        if len(picks) > 1:
            earlier = [_pick_cause(p, evs, name)[0] for p in picks[:-1]]
            detail = '; '.join([d for d in (detail, 'earlier: ' + ', '.join(earlier)) if d])
        return cause, detail
    if off:
        pushed = [e for e in evs if e['what'] == 'disturbed']
        return ('%s off the table' % (pushed[-1]['cause'] if pushed else off[-1]['was']),
                'from %s' % off[-1]['com'])
    pushed = [e for e in evs if e['what'] == 'disturbed']
    note = ('; %s %.0f mm earlier' % (pushed[-1]['cause'], 1e3 * pushed[-1]['dist'])
            if pushed else '')
    if kind == 'wrong':
        return ('%s into a zone' % pushed[-1]['cause']) if pushed else 'in a zone', ''
    if obs is not None:
        c = np.asarray(st['com'][:2], dtype=float)
        near = [d for d in obs['detections'] if d.get('center_base') is not None
                and float(np.hypot(*(np.asarray(d['center_base'][:2]) - c))) < 0.03]
        if not near:
            return 'perception: not detected' + note, 'last observation t %.1f' % obs['t']
        d = near[0]
        if not d.get('valid', True):
            return 'perception: rejected (%s)' % d.get('reject_reason') + note, ''
        if d.get('label') != st['label']:
            return 'perception: seen as %s' % d.get('label') + note, ''
    return 'not attempted (%s)' % (stop_reason or 'round ended') + note, ''


# ------------------------------------------------------------------ viewer
class Viewer:
    """
    Passive MuJoCo viewer window synced every 1/fps of virtual time.

    With wall_clock (core.clock.RealClock or similar), each sync waits until
    the wall time catches up with the virtual time x speed, so the window
    shows the run at real speed. open() returns False (and logs why) when
    no display or no GLFW is available; the run then continues without it.
    """

    def __init__(self, world: Any, fps: float = 30.0, wall_clock: Any = None,
                 speed: float = 1.0):
        self.world = world
        self.fps = float(fps)
        self.wall = wall_clock
        self.speed = float(speed)
        self._handle = None
        self._t0: Optional[Tuple[float, float]] = None

    def open(self) -> bool:
        """Open the window; return False when it is not possible here."""
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            _LOG.warning('--viewer: no DISPLAY; running without the viewer window')
            return False
        try:
            import mujoco.viewer
            self._handle = mujoco.viewer.launch_passive(self.world.model, self.world.data,
                                                        show_left_ui=False,
                                                        show_right_ui=False)
        except Exception as e:  # noqa: B902 - the viewer is optional
            _LOG.warning('--viewer: cannot open the MuJoCo viewer (%s); running without it', e)
            self._handle = None
            return False
        self.world.sim.add_frame_hook(self._sync, self.fps)
        return True

    def _sync(self, sim: Any, t: float) -> None:
        h = self._handle
        if h is None:
            return
        if not h.is_running():
            _LOG.info('viewer window closed; the run continues')
            self.close()
            return
        h.sync()
        if self.wall is not None:
            if self._t0 is None:
                self._t0 = (float(t), float(self.wall.now()))
            lag = (float(t) - self._t0[0]) / self.speed - (self.wall.now() - self._t0[1])
            if lag > 0:
                self.wall.sleep(min(lag, 1.0))

    def close(self) -> None:
        """Close the window and stop syncing."""
        h, self._handle = self._handle, None
        if self.world.sim is not None:
            self.world.sim.remove_frame_hook(self._sync)
        if h is not None:
            try:
                h.close()
            except Exception:  # noqa: B902 - teardown
                _LOG.exception('closing the viewer failed')


# ----------------------------------------------------------------- session
class RunExtras:
    """
    The MuJoCo-only parts of a run: physics monitor, optional video and viewer.

    attach(session) wraps session.logger (runner events -> monitor / HUD,
    physics events -> run log) and session.perception (debug image -> video
    inset); call it before building the TaskRunner. finish(...) settles and
    scores the scene, saves physics.json in the run log and returns the
    result; close() finalizes the video and closes the viewer.
    """

    def __init__(self, world: Any, video: Optional[str] = None, viewer: bool = False,
                 video_speed: float = 1.0, wall_clock: Any = None):
        self.world = world
        self.monitor = PhysicsMonitor(world)
        self.recorder = None
        self.viewer = None
        if video:
            from .video import VideoRecorder
            self.recorder = VideoRecorder(world, video, speed=video_speed)
        if viewer:
            v = Viewer(world, wall_clock=wall_clock)
            self.viewer = v if v.open() else None
        self.video_info: Optional[Dict[str, Any]] = None

    def attach(self, session: Any) -> None:
        """Wrap the session's logger and perception, start monitoring and recording."""
        logger = session.logger
        self.monitor.logger = logger
        tapped = logger
        if self.recorder is not None:
            tapped = self.recorder.wrap_logger(tapped)
            if session.perception is not None:
                session.perception = self.recorder.wrap_perception(session.perception)
        session.logger = self.monitor.wrap_logger(tapped)
        self.monitor.start()
        if self.recorder is not None:
            self.recorder.start()

    def finish(self, info: Dict[str, Any], stop_reason: str = '', logger: Any = None,
               targets: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Settle, score and analyse; return {truth, failures, objects}."""
        truth = final_truth(self.world, info)
        if targets:
            c2z = {k: v for k, v in (info.get('class_to_zone') or {}).items() if k in targets}
            truth = self.world.score(c2z)
        failures = failure_causes(self.world, info, self.monitor, stop_reason)
        out = {'truth': truth, 'failures': failures, 'objects': self.world.object_status()}
        if logger is not None:
            try:
                logger.save_json('physics.json', dict(out, **self.monitor.summary()))
            except Exception:  # noqa: B902 - logging must not stop the report
                _LOG.exception('saving physics.json failed')
        return out

    def close(self) -> Optional[Dict[str, Any]]:
        """Finalize the video, close the viewer, stop the monitor; return the video summary."""
        self.monitor.stop()
        if self.recorder is not None and self.video_info is None:
            self.video_info = self.recorder.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.video_info

r"""
Run many simulated rounds and report success, timing and wrong placements.

Each run builds a seeded scenario (sim/scenarios.py) with the simulated arm and
camera through tools.common (the same plumbing as ``run_task --mode sim``),
runs the TaskRunner and scores the result against the simulation ground
truth. A wrong placement (an object in the wrong zone, or a non-target put
anywhere) is the worst outcome and is listed separately.

Examples::

    python3 -m cube_task sim_bench --tasks L1,L2 --seeds 0-19 --jobs 8
    python3 -m cube_task sim_bench --tasks L1 --seeds 0-9 --mounts eye_in_hand \\
        --difficulty normal,hard --faults none,grasp_miss,drop --json /tmp/bench.json
    python3 -m cube_task sim_bench --tasks L2 --seeds 0-4 --extrinsics-error 0.003 0.5
    bash scripts/mujoco_run.sh sim_bench --backend mujoco --tasks L1,L2 --seeds 0-9 \
        --jobs 6 --save-logs

Faults are injected at a seeded random point of each run: drop, timeout and
comm_loss through SimArm.inject(at_call=N); grasp_miss right before the first
real gripper close from command N on (GraspMissArm); camera_none as 1-3
missing frames (SimCamera.inject('none_frames')) at the N-th frame request.

--backend mujoco runs the same rounds on the MuJoCo physics scene (tools.common
--mode mujoco: MuJoCo arm, rendered camera, contact physics). The ground truth
is then read after the scene came to rest (sim/mujoco/run.final_truth, which
also counts objects dropped on the floor), and every object that did not end
in its zone gets a cause from the physics monitor and the runner's pick
results (grasp miss, slip, drop in transit, bounced out, pushed, not
detected ...), listed with the run's log directory (use --save-logs).

Success = every required object in its correct zone, no wrong placement and
the round finished within the time limit (ground truth, not the runner's own
report). Exit code: 0 when no run placed an object wrongly or crashed, 1
otherwise, 2 for bad arguments, 130 on Ctrl-C.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import add_common_args, ensure_sim_knn, load_cfg_for, open_session, TASKS

_LOG = logging.getLogger('cube_task.tools.sim_bench')

MOUNTS = ('eye_to_hand', 'eye_in_hand')
DIFFICULTIES = ('normal', 'hard')
#: bench fault name -> SimArm.inject kind
ARM_FAULTS = {'grasp_miss': 'grasp_miss', 'drop': 'drop', 'timeout': 'timeout',
              'comm_loss': 'comm_loss'}
#: bench fault name -> SimCamera.inject kind
CAMERA_FAULTS = {'camera_none': 'none_frames'}
FAULTS = ('none',) + tuple(ARM_FAULTS) + tuple(CAMERA_FAULTS)
#: about this many arm commands per pick-and-place (open, 7 motions, close, observe).
CALLS_PER_PICK = 10


# ------------------------------------------------------------------ parsing
def parse_list(text: str, allowed: Sequence[str], what: str) -> List[str]:
    """Split a comma list and check every item is allowed (raises ValueError)."""
    items = [t.strip() for t in str(text).split(',') if t.strip()]
    if not items:
        raise ValueError('%s: empty list' % what)
    bad = [t for t in items if t not in allowed]
    if bad:
        raise ValueError('%s: unknown %s (use %s)' % (what, ', '.join(bad), ', '.join(allowed)))
    out: List[str] = []
    for t in items:
        if t not in out:
            out.append(t)
    return out


def parse_seeds(text: str) -> List[int]:
    """Parse '0-19', '3', '0,5,7-9' into a sorted list of unique seeds."""
    seeds = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                a, b = part.split('-', 1)
                lo, hi = int(a), int(b)
            else:
                lo = hi = int(part)
            if lo < 0 or hi < lo:
                raise ValueError(part)
            seeds.update(range(lo, hi + 1))
        except ValueError:
            raise ValueError('--seeds: cannot parse %r (examples: 0-19, 3, 0,5,7-9)' % part)
    if not seeds:
        raise ValueError('--seeds: no seeds given')
    return sorted(seeds)


# ------------------------------------------------------------- ground truth
def ground_truth(world, info: Dict[str, Any], task: str,
                 targets: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """
    Score the simulated world: dict(correct, wrong, remaining).

    world.score(class_to_zone), restricted to targets when given.
    """
    c2z = dict(info.get('class_to_zone') or {})
    if targets:
        c2z = {k: v for k, v in c2z.items() if k in set(targets)}
    return world.score(c2z)


def is_success(run: Dict[str, Any]) -> bool:
    """Return True when all required objects are correct, none wrong, within the limit."""
    if run.get('error'):
        return False
    return (int(run.get('correct', 0)) >= int(run.get('required', 0)) > 0
            and int(run.get('wrong', 0)) == 0
            and float(run.get('elapsed', 0.0)) <= float(run.get('time_limit', 0.0)))


# ------------------------------------------------------------- fault camera
class FaultCamera:
    """Delegate to a SimCamera; inject 'none_frames' at the at_call-th get_frame call."""

    def __init__(self, camera, at_call: int, count: int, kind: str = 'none_frames'):
        self._cam = camera
        self.at_call = int(at_call)
        self.count = int(count)
        self.kind = kind
        self.calls = 0
        self.fired = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cam, name)

    def start(self) -> None:
        self._cam.start()

    def stop(self) -> None:
        self._cam.stop()

    def intrinsics(self):
        return self._cam.intrinsics()

    def get_frame(self, timeout: float = 1.0, newer_than: Optional[float] = None):
        self.calls += 1
        if not self.fired and self.calls >= self.at_call:
            self._cam.inject(self.kind, count=self.count)
            self.fired = True
        return self._cam.get_frame(timeout=timeout, newer_than=newer_than)


class GraspMissArm:
    """
    Delegate to a SafeArm(SimArm); arm a 'grasp_miss' right before a real close.

    SimArm takes a grasp_miss fault on any narrowing gripper command, including
    the pre-open to object width + margin, where it has no effect. This proxy
    injects it just before the first gripper command at or after the
    at_call-th command whose width is at most robot.gripper.close_width + 1 mm.
    """

    def __init__(self, arm, at_call: int, close_width: float):
        self._arm = arm
        self.at_call = int(at_call)
        self.close_width = float(close_width)
        self.calls = 0
        self.fired = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)

    def _count(self) -> None:
        self.calls += 1

    def move_tcp(self, T_base_tcp, linear: bool = False, timeout: Optional[float] = None):
        self._count()
        return self._arm.move_tcp(T_base_tcp, linear=linear, timeout=timeout)

    def move_joints(self, q, timeout: Optional[float] = None):
        self._count()
        return self._arm.move_joints(q, timeout=timeout)

    def gripper(self, width: float, force: Optional[float] = None, wait: bool = True):
        self._count()
        if (not self.fired and self.calls >= self.at_call
                and float(width) <= self.close_width + 0.001):
            self._arm.inner.inject('grasp_miss')
            self.fired = True
        return self._arm.gripper(width, force=force, wait=wait)


def fault_point(spec: Dict[str, Any], required: int) -> Dict[str, int]:
    """Return the seeded injection point for spec['fault'] (empty dict for 'none')."""
    fault = spec['fault']
    if fault == 'none':
        return {}
    rng = np.random.default_rng([int(spec['seed']), FAULTS.index(fault),
                                 TASKS.index(spec['task']), MOUNTS.index(spec['mount']),
                                 DIFFICULTIES.index(spec['difficulty'])])
    n = max(1, int(required))
    if fault == 'grasp_miss':
        # Arm before a close even when fixed-camera picks omit the observe move.
        # A continuous pick has 6 motions and 3 gripper commands, after 2 prep calls.
        return {'at_call': 3 + 9 * int(rng.integers(n))}
    if fault in ARM_FAULTS:
        # 2 preparation commands, then about CALLS_PER_PICK per pick: hit the first 80 %.
        return {'at_call': int(rng.integers(3, 3 + max(1, int(0.8 * CALLS_PER_PICK * n))))}
    return {'at_call': int(rng.integers(1, 1 + 3 * n)), 'count': int(rng.integers(1, 4))}


# ------------------------------------------------------------------ one run
def _namespace(spec: Dict[str, Any], mode: Optional[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    add_common_args(p, mujoco=True)
    mode = mode or spec.get('backend', 'sim')
    argv = ['--task', spec['task'], '--mode', mode, '--seed', str(spec['seed']),
            '--mount', spec['mount'], '--difficulty', spec['difficulty'],
            '--extrinsics-error', repr(float(spec['extrinsics_error'][0])),
            repr(float(spec['extrinsics_error'][1]))]
    for c in spec.get('config') or []:
        argv += ['--config', c]
    for s in spec.get('set') or []:
        argv += ['--set', s]
    if spec.get('log_dir'):
        argv += ['--log-dir', spec['log_dir']]
    return p.parse_args(argv)


def reproduce_command(spec: Dict[str, Any]) -> str:
    """Return a shell command that repeats one run with its logs saved."""
    cmd = ['python3 -m cube_task sim_bench', '--tasks', spec['task'], '--seeds',
           str(spec['seed']), '--mounts', spec['mount'], '--difficulty', spec['difficulty'],
           '--faults', spec['fault']]
    if spec.get('backend', 'sim') == 'mujoco':
        cmd[:1] = ['bash scripts/mujoco_run.sh sim_bench', '--backend', 'mujoco']
    err = spec.get('extrinsics_error') or (0.0, 0.0)
    if float(err[0]) or float(err[1]):
        cmd += ['--extrinsics-error', '%g' % float(err[0]), '%g' % float(err[1])]
    for c in spec.get('config') or []:
        cmd += ['--config', c]
    for s in spec.get('set') or []:
        cmd += ['--set', "'%s'" % s]
    return ' '.join(cmd + ['--save-logs'])


def run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run one simulated round described by spec; never raises (errors go to 'error')."""
    from ..task.runner import TaskRunner
    wall0 = time.monotonic()          # wall time for the report only, not for any logic
    task = spec['task']
    out: Dict[str, Any] = {k: spec[k] for k in ('task', 'seed', 'mount', 'difficulty', 'fault')}
    out.update(extrinsics_error=[float(v) for v in spec['extrinsics_error']], error='',
               placed=0, required=0, correct=0, wrong=0, remaining=0, elapsed=0.0,
               time_limit=0.0, attempts=0, stop_reason='', report_ok=False,
               fault_at=None, fault_fired=False, log_dir='')
    mujoco = spec.get('backend', 'sim') == 'mujoco'
    if mujoco:
        out.update(backend='mujoco', dropped=0, failures=[])
    try:
        args = _namespace(spec)
        cfg, world, info = load_cfg_for(args, task)
        with open_session(args, task, logger=bool(spec.get('save_logs')),
                          cfg_world_info=(cfg, world, info)) as s:
            required = int(s.cfg.get('tasks.%s.required' % task, 0) or 0)
            point = fault_point(spec, required)
            camera, arm = s.camera, s.arm
            if spec['fault'] == 'grasp_miss':
                arm = GraspMissArm(s.arm, point['at_call'],
                                   float(s.cfg.get('robot.gripper.close_width', 0.0) or 0.0))
            elif spec['fault'] in ARM_FAULTS:
                s.arm.inner.inject(ARM_FAULTS[spec['fault']], at_call=point['at_call'])
            elif spec['fault'] in CAMERA_FAULTS:
                camera = FaultCamera(s.camera, point['at_call'], point['count'],
                                     CAMERA_FAULTS[spec['fault']])
            out['fault_at'] = point or None
            log = s.logger
            extras = None
            if mujoco:
                from ..sim.mujoco.run import RunExtras
                extras = RunExtras(world)
                extras.attach(s)            # wraps s.logger: runner events -> monitor
            runner = TaskRunner(s.cfg, task, arm, camera, s.perception, s.clock, s.logger,
                                mode='sim')
            rep = runner.run()
            if extras is not None:
                phys = extras.finish(info, rep.stop_reason, logger=log)
                extras.close()
                gt = phys['truth']
                out['failures'] = phys['failures']
            else:
                gt = ground_truth(world, info, task)
            out.update(placed=rep.placed, required=rep.required, elapsed=round(rep.elapsed, 2),
                       time_limit=rep.time_limit, attempts=rep.attempts,
                       stop_reason=rep.stop_reason, report_ok=bool(rep.ok),
                       fault_detail=runner.fault_detail, **gt)
            if spec['fault'] in ARM_FAULTS:
                out['fault_fired'] = bool(getattr(s.arm.inner, 'fault_log', [])) and (
                    spec['fault'] != 'grasp_miss' or arm.fired)
            elif spec['fault'] in CAMERA_FAULTS:
                out['fault_fired'] = bool(camera.fired)
            if log is not None:
                out['log_dir'] = str(log.dir)
                log.close({'bench': out, 'placements': world.placements()})
    except Exception as e:  # noqa: B902 - one broken run must not stop the bench
        _LOG.exception('run %s failed', reproduce_command(spec))
        out['error'] = '%s: %s' % (type(e).__name__, e)
    out['success'] = is_success(out)
    out['wall_s'] = round(time.monotonic() - wall0, 2)
    out['reproduce'] = reproduce_command(spec)
    return out


# --------------------------------------------------------------- aggregate
def _stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(runs)
    el = [float(r['elapsed']) for r in runs if not r.get('error')]
    ok = sum(1 for r in runs if r.get('success'))
    return {'runs': n, 'success': ok, 'success_rate': round(ok / float(n), 3) if n else 0.0,
            'mean_elapsed': round(float(np.mean(el)), 2) if el else None,
            'max_elapsed': round(float(np.max(el)), 2) if el else None,
            'wrong_placements': sum(int(r.get('wrong', 0)) for r in runs),
            'runs_with_wrong': sum(1 for r in runs if int(r.get('wrong', 0)) > 0),
            'errors': sum(1 for r in runs if r.get('error')),
            'dropped': sum(int(r.get('dropped', 0)) for r in runs),
            'mean_attempts': round(float(np.mean([r.get('attempts', 0) for r in runs])), 2)
            if runs else None}


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return overall and per (task, mount, difficulty, fault) statistics."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r['task'], r['mount'], r['difficulty'], r['fault']), []).append(r)
    rows = []
    for key in sorted(groups):
        st = _stats(groups[key])
        st.update(task=key[0], mount=key[1], difficulty=key[2], fault=key[3])
        rows.append(st)
    return {'overall': _stats(runs), 'groups': rows}


# ----------------------------------------------------------------- output
def _row(r: Dict[str, Any]) -> str:
    flag = 'OK' if r.get('success') else 'FAIL'
    if int(r.get('wrong', 0)) > 0:
        flag = '!! WRONG'
    if r.get('error'):
        flag = 'ERROR'
    if int(r.get('dropped', 0)) > 0:
        flag += ' (dropped %d)' % int(r['dropped'])
    return ('%-3s %4d %-11s %-6s %-11s %3d/%-3d %4d %5d %5d %7.1f %4d  %-10s %s' % (
        r['task'], r['seed'], r['mount'], r['difficulty'], r['fault'], r['placed'],
        r['required'], r['correct'], r['wrong'], r['remaining'], r['elapsed'], r['attempts'],
        r['stop_reason'] or '-', flag))


HEADER = ('%-3s %4s %-11s %-6s %-11s %7s %4s %5s %5s %7s %4s  %-10s %s' % (
    'tsk', 'seed', 'mount', 'diff', 'fault', 'placed', 'ok', 'wrong', 'left', 'time_s', 'att',
    'stop', 'result'))


def format_report(runs: List[Dict[str, Any]], agg: Dict[str, Any]) -> str:
    """Return the aggregate table, the wrong-placement list and the error list."""
    lines = ['', 'aggregate (success = all required correct, none wrong, within the limit):',
             '%-3s %-11s %-6s %-11s %5s %8s %8s %8s %6s %6s' % (
                 'tsk', 'mount', 'diff', 'fault', 'runs', 'success', 'mean_s', 'max_s',
                 'WRONG', 'errors')]

    def fmt(st: Dict[str, Any], label: str) -> str:
        return '%-34s %5d %7.0f%% %8s %8s %6d %6d%s' % (
            label, st['runs'], 100.0 * st['success_rate'],
            '-' if st['mean_elapsed'] is None else '%.1f' % st['mean_elapsed'],
            '-' if st['max_elapsed'] is None else '%.1f' % st['max_elapsed'],
            st['wrong_placements'], st['errors'], '  !!' if st['wrong_placements'] else '')

    for g in agg['groups']:
        key = '%-3s %-11s %-6s %-11s' % (g['task'], g['mount'], g['difficulty'], g['fault'])
        lines.append(fmt(g, key))
    lines.append(fmt(agg['overall'], 'ALL'))
    wrong = [r for r in runs if int(r.get('wrong', 0)) > 0]
    lines.append('')
    if wrong:
        lines.append('!! WRONG PLACEMENTS (worst outcome) in %d run(s), %d object(s):'
                     % (len(wrong), sum(int(r['wrong']) for r in wrong)))
        for r in wrong:
            lines.append('   %s seed %d %s %s fault=%s: wrong %d, stop %s' % (
                r['task'], r['seed'], r['mount'], r['difficulty'], r['fault'], r['wrong'],
                r['stop_reason']))
            lines.append('      reproduce: %s' % r['reproduce'])
    else:
        lines.append('wrong placements: none')
    lines.extend(failure_report(runs))
    errors = [r for r in runs if r.get('error')]
    for r in errors:
        lines.append('ERROR %s seed %d %s: %s' % (r['task'], r['seed'], r['mount'], r['error']))
        lines.append('      reproduce: %s' % r['reproduce'])
    return '\n'.join(lines)


def failure_report(runs: List[Dict[str, Any]]) -> List[str]:
    """Return the per-cause counts and the list of failed objects (mujoco runs only)."""
    fails = [(r, f) for r in runs for f in (r.get('failures') or [])]
    if not any('failures' in r for r in runs):
        return []
    if not fails:
        return ['physics failures: none']
    counts: Dict[str, int] = {}
    for _r, f in fails:
        key = _cause_key(f['cause'])
        counts[key] = counts.get(key, 0) + 1
    lines = ['physics failures by cause (%d object(s) not in their zone):' % len(fails)]
    for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append('  %3d  %s' % (n, key))
    lines.append('failed objects:')
    for r, f in fails:
        lines.append('  %s seed %d %s %s: %s (%s) %s: %s%s' % (
            r['task'], r['seed'], r['mount'], r['difficulty'], f['object'], f['label'],
            f['kind'], f['cause'], ' [%s]' % f['detail'] if f.get('detail') else ''))
        lines.append('      log: %s' % (r.get('log_dir') or '- (use --save-logs)'))
    return lines


def _cause_key(cause: str) -> str:
    """Group causes: drop the details after ';' and the class names of misclassifications."""
    key = cause.split(';')[0].strip()
    if key.startswith('perception:') and (' classified as ' in key or ' seen as ' in key):
        return 'perception: misclassified'
    return key


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='sim_bench', description='Run many simulated rounds and score them by ground '
        'truth (success rate, time, wrong placements).')
    p.add_argument('--tasks', default='L1,L2', help='comma list of L1, L2 (default L1,L2)')
    p.add_argument('--seeds', default='0-9', help="seeds, e.g. '0-19' or '0,3,5-7'")
    p.add_argument('--mounts', default='eye_in_hand',
                   help='comma list of eye_in_hand, eye_to_hand (default eye_in_hand, the '
                   'competition camera)')
    p.add_argument('--difficulty', default='normal', help='comma list of normal, hard')
    p.add_argument('--extrinsics-error', type=float, nargs=2, default=(0.0, 0.0),
                   metavar=('M', 'DEG'), help='error of the extrinsics estimate (m, deg)')
    p.add_argument('--faults', default='none', help='comma list of %s' % ', '.join(FAULTS))
    p.add_argument('--backend', choices=('sim', 'mujoco'), default='sim',
                   help='sim: kinematic simulator (default); mujoco: MuJoCo physics '
                   '(run with scripts/mujoco_run.sh)')
    p.add_argument('--jobs', type=int, default=1, help='parallel processes (default 1)')
    p.add_argument('--json', default=None, metavar='PATH',
                   help="write results here ('-' = stdout; default "
                   "<logging.dir>/sim_bench/<time>.json)")
    p.add_argument('--save-logs', action='store_true',
                   help='write a run log directory per run (slower)')
    p.add_argument('--log-dir', default=None, help='override logging.dir')
    p.add_argument('--config', action='append', default=[], metavar='YAML',
                   help='extra YAML merged into every run (repeatable)')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='config override for every run, e.g. robot.speed_percent=50')
    p.add_argument('-v', '--verbose', action='store_true', help='show runner logging')
    return p


def _worker_init(verbose: bool) -> None:
    _quiet_logging(verbose)


def _quiet_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('cube_task').setLevel(logging.INFO if verbose else logging.ERROR)


def make_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Expand the command line into one spec dict per run (raises ValueError)."""
    tasks = parse_list(args.tasks, TASKS, '--tasks')
    mounts = parse_list(args.mounts, MOUNTS, '--mounts')
    diffs = parse_list(args.difficulty, DIFFICULTIES, '--difficulty')
    faults = parse_list(args.faults, FAULTS, '--faults')
    seeds = parse_seeds(args.seeds)
    err = [float(v) for v in args.extrinsics_error]
    if any(not np.isfinite(v) or v < 0 for v in err):
        raise ValueError('--extrinsics-error needs two finite values >= 0')
    specs = []
    for task in tasks:
        for mount in mounts:
            for diff in diffs:
                for fault in faults:
                    for seed in seeds:
                        specs.append({'task': task, 'seed': seed, 'mount': mount,
                                      'difficulty': diff, 'fault': fault,
                                      'backend': args.backend,
                                      'extrinsics_error': err, 'config': list(args.config),
                                      'set': list(args.set), 'save_logs': bool(args.save_logs),
                                      'log_dir': args.log_dir})
    return specs


def prepare_knn(specs: List[Dict[str, Any]]) -> None:
    """Train (once, cached) the simulation KNN models the runs need, before forking."""
    done = set()
    for spec in specs:
        key = (spec['task'], spec['mount'])
        if spec['task'] == 'L1' or key in done:
            continue
        done.add(key)
        args = _namespace(spec, mode='sim')     # the same KNN serves --backend mujoco
        cfg, _world, _info = load_cfg_for(args, spec['task'])
        ensure_sim_knn(cfg, spec['task'], spec['mount'])


def _json_path(args: argparse.Namespace) -> Optional[Path]:
    if args.json == '-':
        return None
    if args.json:
        return Path(args.json).expanduser()
    from ..core.config import load_config
    base = args.log_dir or load_config(use_overlays=False).get('logging.dir', '~/vision_runs')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(base).expanduser() / 'sim_bench' / ('%s.json' % stamp)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the bench; see the module docstring for the exit codes."""
    args = build_parser().parse_args(argv)
    _quiet_logging(args.verbose)
    try:
        specs = make_specs(args)
    except ValueError as e:
        print('sim_bench: %s' % e, file=sys.stderr)
        return 2
    if args.jobs < 1:
        print('sim_bench: --jobs must be >= 1', file=sys.stderr)
        return 2
    print('sim_bench: %d run(s), %d job(s)' % (len(specs), args.jobs))
    runs: List[Dict[str, Any]] = []
    interrupted = False
    try:
        prepare_knn(specs)
        print(HEADER)
        if args.jobs == 1 or len(specs) == 1:
            for spec in specs:
                runs.append(run_one(spec))
                print(_row(runs[-1]), flush=True)
        else:
            methods = multiprocessing.get_all_start_methods()
            # MuJoCo renders through EGL / OpenGL, which must not be inherited by a fork.
            fork = 'fork' in methods and args.backend != 'mujoco'
            ctx = multiprocessing.get_context('fork' if fork else 'spawn')
            with ctx.Pool(min(args.jobs, len(specs)), initializer=_worker_init,
                          initargs=(args.verbose,)) as pool:
                for res in pool.imap(run_one, specs):
                    runs.append(res)
                    print(_row(res), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\ninterrupted: reporting the %d finished run(s)' % len(runs))
    agg = aggregate(runs)
    print(format_report(runs, agg))
    result = {'created': datetime.datetime.now().isoformat(timespec='seconds'),
              'argv': list(sys.argv[1:] if argv is None else argv),
              'pid': os.getpid(), 'interrupted': interrupted, 'runs': runs, 'aggregate': agg}
    path = _json_path(args)
    text = json.dumps(result, indent=1, ensure_ascii=False, default=str)
    if path is None:
        print(text)
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            print('results: %s' % path)
        except OSError as e:
            print('sim_bench: cannot write %s: %s' % (path, e), file=sys.stderr)
    if interrupted:
        return 130
    bad = agg['overall']['wrong_placements'] > 0 or agg['overall']['errors'] > 0
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

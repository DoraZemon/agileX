"""
Competition entry point: run one L1 / L2 round.

Examples::

    run_task --task L1                              # real robot, waits for Enter
    run_task --task L1 --targets red,blue           # referee-specified labels / order
    run_task --task L2 --dry-run --speed 20         # hover above grasp / release poses
    run_task --task L2 --mode sim --seed 3 --mount eye_in_hand
    run_task --task L1 --mode mujoco --seed 0 --video ~/vision_runs/mujoco/L1_s0.mp4

(``python3 -m cube_task run_task ...`` or ``ros2 run cube_task run_task ...``;
scripts/start.sh runs the pre-flight checks first.)

Flow: config -> arm, camera, perception, run log and operator keys
(tools.common.open_session) -> connect the arm (read-only) -> summary -> real
mode: full self check (a FAIL refuses, nothing enabled), then ask before the
first motion (--yes skips) -> enable -> TaskRunner.run() -> report table (sim:
also the ground truth). During the run: p / space =
abort and hold, e = electronic e-stop, Ctrl-C = stop and hold, save the log.

Exit codes: 0 round complete (sim: also no wrong placement by ground truth;
dry run: every required object visited); 1 finished incomplete (timeout, no
targets left, operator abort); 2 refused before any motion (self check,
config, arguments, operator did not confirm); 3 fault (arm, camera, e-stop,
crash); 130 Ctrl-C.

--mode mujoco runs the same stack on the MuJoCo physics scene of the sim
scenario (MuJoCo arm and rendered camera, sim/mujoco; needs the MuJoCo venv,
see scripts/mujoco_run.sh). The report then shows the physics ground truth
after the scene came to rest (correct / wrong / remaining / dropped) and one
cause per object that did not end in its zone; --video PATH records an MP4,
--viewer opens the MuJoCo viewer (paced to real time) when a display exists.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from .common import add_common_args, load_cfg_for, open_session, setup_logging

_LOG = logging.getLogger('cube_task.tools.run_task')

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_REFUSED = 2
EXIT_FAULT = 3
EXIT_INTERRUPTED = 130
EXIT_MEANING = {EXIT_OK: 'round complete', EXIT_INCOMPLETE: 'finished incomplete',
                EXIT_REFUSED: 'refused before any motion', EXIT_FAULT: 'fault',
                EXIT_INTERRUPTED: 'interrupted'}
_RULE = '=' * 78


class Refused(Exception):
    """The run is refused before anything moves (exit code 2)."""


def _say(text: str = '') -> None:
    try:
        print(text, flush=True)
    except (OSError, ValueError):  # broken pipe / closed stdout must not stop the cleanup
        pass


# ---------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='run_task', description='Run one competition round (L1 / L2).',
        epilog='Exit codes: 0 complete, 1 incomplete, 2 refused before motion, 3 fault, '
        '130 Ctrl-C.')
    add_common_args(p, task_required=True, mujoco=True)
    p.add_argument('--dry-run', action='store_true',
                   help='hover above every grasp / release pose; never close the gripper')
    p.add_argument('--targets', default='', metavar='LABELS',
                   help='L1 / L2: referee-specified labels in order, e.g. red,blue '
                   '(only these are picked)')
    p.add_argument('--no-wait', action='store_true',
                   help='start the timed run without waiting for Enter (sim never waits)')
    p.add_argument('--speed', type=int, default=None, metavar='PCT',
                   help='speed percent for this run; may only lower robot.speed_percent')
    p.add_argument('--allow-faster', action='store_true',
                   help='allow --speed above robot.speed_percent')
    p.add_argument('-y', '--yes', action='store_true',
                   help='real mode: do not ask before the first motion')
    p.add_argument('--video', default='', metavar='MP4',
                   help='mujoco mode: record an H.264 video (overview, perception inset, HUD)')
    p.add_argument('--video-speed', type=float, default=1.0, metavar='X',
                   help='mujoco mode: video playback speed-up (default 1 = real time)')
    p.add_argument('--viewer', action='store_true',
                   help='mujoco mode: show the MuJoCo viewer at real speed (needs a display)')
    return p


def parse_targets(text: str, cfg, task: str) -> List[str]:
    """Return the referee target labels (config spelling); raise Refused when invalid."""
    items = [t for t in re.split(r'[,，;；\s]+', str(text or '').strip()) if t]
    if not items:
        return []
    c2z = dict(cfg.get('tasks.%s.class_to_zone' % task, {}) or {})
    labels = [str(v) for v in (cfg.get('tasks.%s.labels' % task, []) or [])]
    known = {lab.lower(): lab for lab in labels + [str(k) for k in c2z]}
    out: List[str] = []
    for t in items:
        lab = known.get(t.lower())
        if lab is None:
            raise Refused('target %r is not a %s label (labels: %s)'
                          % (t, task, ', '.join(labels) or 'none configured'))
        if lab not in c2z:
            raise Refused('target %s has no zone in tasks.%s.class_to_zone' % (lab, task))
        if lab not in out:
            out.append(lab)
    return out


def resolve_speed(args: argparse.Namespace, cfg) -> SimpleNamespace:
    """Apply --speed to robot.speed_percent; raise Refused when it is not allowed."""
    conf = cfg.get('robot.speed_percent', None)
    try:
        conf_i = int(conf)
    except (TypeError, ValueError):
        raise Refused('robot.speed_percent must be an integer 1..100, got %r' % (conf,))
    if conf_i != conf or not 1 <= conf_i <= 100:
        raise Refused('robot.speed_percent must be an integer 1..100, got %r' % (conf,))
    if args.speed is None:
        return SimpleNamespace(speed=conf_i, configured=conf_i)
    if not 1 <= int(args.speed) <= 100:
        raise Refused('--speed must be in 1..100, got %d' % args.speed)
    if args.speed > conf_i and not args.allow_faster:
        raise Refused('--speed %d is above robot.speed_percent %d; add --allow-faster to '
                      'raise it' % (args.speed, conf_i))
    cfg.set('robot.speed_percent', int(args.speed))
    return SimpleNamespace(speed=int(args.speed), configured=conf_i)


def prepare_options(args: argparse.Namespace, cfg, info: Optional[dict]) -> SimpleNamespace:
    """Validate the run options and apply them to cfg (raises Refused)."""
    if args.mode != 'mujoco' and (args.video or args.viewer):
        raise Refused('--video / --viewer need --mode mujoco')
    if not (args.video_speed > 0 and math.isfinite(args.video_speed)):
        raise Refused('--video-speed must be > 0')
    sp = resolve_speed(args, cfg)
    if args.no_wait:
        cfg.set('runner.start_on_enter', False)
    targets = parse_targets(args.targets, cfg, args.task)
    required = int(cfg.get('tasks.%s.required' % args.task, 0) or 0)
    if targets and len(set(targets)) < required:
        n = len(set(targets))
        _say('NOTE: --targets lists %d label(s) but tasks.%s.required is %d. The run ends with '
             "'no_targets' once no listed label is left; set --set tasks.%s.required=N if the "
             'referee asked for fewer objects.' % (n, args.task, required, args.task))
    return SimpleNamespace(speed=sp.speed, configured_speed=sp.configured, targets=targets,
                           real=args.mode == 'real', enabled=False)


# ------------------------------------------------------------------ summary
def _deg(q: Sequence[float]) -> str:
    return '[%s] deg' % ', '.join('%.1f' % math.degrees(float(v)) for v in q)


def _sources(cfg) -> str:
    out = []
    for src in cfg.sources:
        p = str(src)
        if os.path.isabs(p):
            try:
                p = os.path.relpath(p, str(cfg.root))
            except ValueError:
                pass
        out.append(p)
    return ' + '.join(out)


def _zones(cfg) -> Dict[str, Any]:
    from ..core.zones import parse_zones
    try:
        return parse_zones(cfg)
    except (ValueError, TypeError) as e:
        _LOG.error('zones: %s', e)
        return {}


def _extrinsics_text(cfg) -> str:
    try:
        from ..core.extrinsics import extrinsics_from_cfg
        ext = extrinsics_from_cfg(cfg)
    except Exception as e:  # noqa: B902 - shown to the operator, the runner refuses
        return 'INVALID: %s' % e
    if ext is None:
        return 'MISSING (camera.extrinsics_file %s)' % cfg.get('camera.extrinsics_file', '')
    return ext.describe() + (', rms unknown' if ext.rms is None else '')


def _observe_text(cfg) -> str:
    qj = cfg.get('robot.poses.observe_joints', None)
    if qj is not None:
        try:
            return 'taught joints %s' % _deg(qj)
        except (TypeError, ValueError):
            return 'INVALID observe_joints %r' % (qj,)
    t = cfg.get('robot.poses.observe_tcp', None)
    return 'not taught; fallback observe_tcp %s' % (t,) if t else 'NONE'


def _classifier_text(cfg, task: str) -> str:
    from ..perception.factory import classifier_backend
    try:
        b = classifier_backend(cfg, task)
    except ValueError as e:
        return 'INVALID: %s' % e
    if 'knn' in b:
        return '%s (model %s)' % (b, cfg.path('perception.classifier.knn.model_file'))
    if 'onnx' in b:
        return '%s (model %s)' % (b, cfg.path('perception.classifier.onnx.model_file'))
    return '%s (HSV ranges: %s)' % (b, ', '.join(sorted(
        cfg.get('perception.classifier.color.classes', {}) or {})))


def _arm_text(arm) -> str:
    try:
        h = arm.health()
    except Exception as e:  # noqa: B902 - diagnostics only
        return 'health() raised: %s' % e
    d = h.detail or {}
    state = 'ready' if h.ok else 'NOT READY: %s' % h.reason
    extra = [str(d[k]) for k in ('ctrl_mode_name', 'arm_status_name') if d.get(k)]
    q = arm.get_joints()
    joints = '' if q is None else ', joints %s' % _deg(q.value)
    return '%s%s%s' % (state, ' (%s)' % ', '.join(extra) if extra else '', joints)


def summary_lines(args: argparse.Namespace, s, opts: SimpleNamespace) -> List[str]:
    """Return the pre-run summary shown to the operator."""
    cfg, task = s.cfg, args.task
    out = [_RULE, 'run_task %s | %s mode%s' % (
        task, args.mode.upper(),
        ' | DRY RUN: hover only, the gripper never closes' if args.dry_run else ''), _RULE]

    def row(key: str, val: str) -> None:
        out.append('%-13s %s' % (key + ':', val))

    zones = _zones(cfg)
    row('config', _sources(cfg))
    req = cfg.get('tasks.%s.required' % task, None)
    row('round', 'time limit %s s, required %s' % (
        cfg.get('tasks.%s.time_limit_s' % task, None), req))
    c2z = dict(cfg.get('tasks.%s.class_to_zone' % task, {}) or {})
    labels = [str(v) for v in (cfg.get('tasks.%s.labels' % task, []) or [])]
    pairs = []
    for lab in labels + [k for k in c2z if k not in labels]:
        z = c2z.get(lab)
        pairs.append('%s -> %s' % (lab, 'not picked (no zone)' if not z else
                                   z if z in zones else '%s (ZONE MISSING)' % z))
    row('labels', ', '.join(pairs) or 'NONE')
    if opts.targets:
        row('targets', '%s (only these, first available first)' % ', '.join(opts.targets))
    elif cfg.get('tasks.%s.order' % task, None):
        row('order', ', '.join(str(v) for v in cfg.get('tasks.%s.order' % task)))
    row('zones', ', '.join('%s (%.3f, %.3f)' % (n, float(z.center[0]), float(z.center[1]))
                           for n, z in sorted(zones.items())) or 'NONE')
    row('camera', '%s, backend %s' % (cfg.get('camera.mount', None),
                                      cfg.get('camera.backend', None)))
    row('extrinsics', _extrinsics_text(cfg))
    note = ''
    if opts.speed < opts.configured_speed:
        note = ' (lowered from %d %%)' % opts.configured_speed
    elif opts.speed > opts.configured_speed:
        note = ' (RAISED from %d %% with --allow-faster)' % opts.configured_speed
    row('speed', '%d %%%s' % (opts.speed, note))
    row('observe pose', _observe_text(cfg))
    row('classifier', _classifier_text(cfg, task))
    if not opts.real:
        row('start', 'at once (simulation)')
    elif bool(cfg.get('runner.start_on_enter', True)):
        row('start', 'waits for Enter after moving to the observe pose')
    else:
        row('start', 'at once after moving to the observe pose (--no-wait)')
    row('arm', _arm_text(s.arm))
    if args.mode in ('sim', 'mujoco'):
        row('sim', 'seed %d, difficulty %s, extrinsics error %g m / %g deg%s' % (
            args.seed, args.difficulty, args.extrinsics_error[0], args.extrinsics_error[1],
            ', MuJoCo physics' if args.mode == 'mujoco' else ''))
    if args.mode == 'mujoco':
        row('video', str(Path(args.video).expanduser()) if args.video else '-')
    row('log dir', str(s.logger.dir) if s.logger is not None else '-')
    return out


# ------------------------------------------------------------------- report
def _sim_truth(s, task: str, opts: SimpleNamespace, stop_reason: str = ''
               ) -> Optional[Dict[str, int]]:
    if s.world is None:
        return None
    from .sim_bench import ground_truth
    extras = getattr(opts, 'mujoco', None)
    if extras is not None:              # physics: settle, score, explain the failures
        res = extras.finish(s.info, stop_reason, logger=s.logger,
                            targets=opts.targets or None)
        opts.failures = res['failures']
        return res['truth']
    return ground_truth(s.world, s.info, task, targets=opts.targets or None)


def failure_lines(failures: Sequence[Dict[str, Any]]) -> List[str]:
    """Return the per-object failure causes of a MuJoCo run (empty when none)."""
    out = []
    for f in failures:
        out.append('  %-14s %-9s %-8s %s%s' % (
            f['object'][:14], f['kind'], f['expected'] or '-', f['cause'],
            ' (%s)' % f['detail'] if f.get('detail') else ''))
    if out:
        out.insert(0, 'physics: objects not in their zone (object, kind, zone, cause):')
    return out


def report_lines(rep, fault_detail: str, truth: Optional[Dict[str, int]], task: str,
                 dry_run: bool) -> List[str]:
    """Return the report table printed after the run."""
    status = 'COMPLETE' if rep.ok else ('DRY RUN' if dry_run else 'INCOMPLETE')
    out = ['', _RULE, '%s %s: placed %d/%d, %.1f s of %.0f s, attempts %d, stop reason: %s' % (
        task, status, rep.placed, rep.required, rep.elapsed, rep.time_limit, rep.attempts,
        rep.stop_reason or '-'), _RULE]
    out.append('%-3s %-14s %-12s %-4s %-14s %-22s %7s' % (
        '#', 'label', 'zone', 'ok', 'stage', 'error', 'time_s'))
    for i, p in enumerate(rep.picks, 1):
        out.append('%-3d %-14s %-12s %-4s %-14s %-22s %7.1f' % (
            i, (p.label or '-')[:14], (p.zone or '-')[:12], 'yes' if p.ok else 'no',
            (p.stage or '-')[:14], (p.error or '')[:22], float(p.duration)))
    if not rep.picks:
        out.append('    (no pick attempted)')
    if fault_detail:
        out.append('detail: %s' % fault_detail)
    if rep.stop_reason == 'self_check':
        out.append('the self check failed: see the [FAIL] lines above or run check_setup')
    if truth is not None:
        out.append('ground truth (%s): correct %d, wrong %d, remaining %d%s%s' % (
            'physics' if 'dropped' in truth else 'sim', truth['correct'], truth['wrong'],
            truth['remaining'],
            ', dropped %d' % truth['dropped'] if 'dropped' in truth else '',
            '   !! WRONG PLACEMENT' if truth['wrong'] else ''))
    out.append('log: %s' % (rep.log_dir or '-'))
    return out


def exit_code_for(rep, dry_run: bool, estopped: bool,
                  truth: Optional[Dict[str, int]] = None) -> int:
    """Map a TaskReport (+ sim ground truth) to the documented exit code."""
    if rep.stop_reason in ('self_check', 'config'):
        return EXIT_REFUSED
    if rep.stop_reason == 'fault' or estopped:
        return EXIT_FAULT
    if dry_run:
        visited = sum(1 for p in rep.picks if p.stage == 'dry_run_done')
        ok = visited >= max(1, int(rep.required)) and rep.stop_reason != 'aborted'
        return EXIT_OK if ok else EXIT_INCOMPLETE
    if not rep.ok:
        return EXIT_INCOMPLETE
    if truth is not None and (truth['wrong'] > 0 or truth['correct'] < rep.required):
        return EXIT_INCOMPLETE
    return EXIT_OK


# --------------------------------------------------------------------- flow
def _close_log(s, summary: Dict[str, Any]) -> None:
    if s.logger is not None:
        s.logger.close(summary)


def _stop_arm(s, opts: SimpleNamespace) -> None:
    """Stop issuing commands and hold position (never disable); only after enable()."""
    if s.arm is None or not opts.enabled:
        return
    try:
        s.arm.stop()
    except Exception:  # noqa: B902 - best effort while shutting down
        _LOG.exception('arm.stop() raised')


def _confirm(args: argparse.Namespace, cfg, opts: SimpleNamespace) -> bool:
    from .ui import Ui
    start = ('wait for Enter' if bool(cfg.get('runner.start_on_enter', True))
             else 'start the timed run at once')
    _say('About to: enable the motors, move to the observe pose, open the gripper, %s, '
         'then run %s for up to %s s at %d %% speed%s.' % (
             start, args.task, cfg.get('tasks.%s.time_limit_s' % args.task, '?'), opts.speed,
             ' as a DRY RUN (hover only, the gripper never closes)' if args.dry_run else ''))
    _say('Keep a hand on the e-stop. During the run: p / space = abort and hold, '
         'e = electronic e-stop, Ctrl-C = stop.')
    if bool(cfg.get('runner.start_on_enter', True)):
        # The Enter at the observe pose is the real start gate (nothing is timed or
        # picked before it): no extra y first. Ctrl-C / e still stop the arm.
        _say('Moving to the observe pose now; the run starts only when you press Enter there.')
        return True
    return Ui(headless=True).confirm('Start on the REAL robot?')


def _leave_teach_mode(s) -> bool:
    """
    Real mode: when the controller is in drag-teach mode, offer the way back to CAN control.

    The self check fails on a teaching controller and used to end the run there; this
    reuses the typed-confirmation helpers of the on-site tools (teach button,
    c = take CAN control, x = exit_teach_mode). Does nothing when the arm is not teaching.
    """
    from .onsite import ensure_motion_ready, is_teaching
    from .ui import Ui
    try:
        if not is_teaching(s.arm):
            return True
    except Exception:  # noqa: B902 - the self check reports an unreadable arm
        return True
    return ensure_motion_ready(Ui(headless=True), s.arm, 'run the task')


def _precheck(args: argparse.Namespace, s) -> bool:
    """Real mode: run the full self check before asking, so a refusal enables nothing."""
    from ..task.selfcheck import self_check
    if not _leave_teach_mode(s):
        _say('REFUSED: the arm is still in drag-teach mode; nothing was enabled or moved.')
        return False
    ok, msgs = self_check(s.cfg, args.task, s.arm, s.camera, 'real', clock=s.clock)
    for m in msgs:
        if not m.startswith('[ OK ]'):
            _say('self check %s' % m)
    if s.logger is not None:
        s.logger.event('precheck', ok=ok, messages=msgs)
    if not ok:
        _say('REFUSED: the self check failed; nothing was enabled or moved. Fix the [FAIL] '
             'lines (check_setup --task %s --live lists them with fixes).' % args.task)
    return ok


def _base_summary(args: argparse.Namespace, opts: SimpleNamespace) -> Dict[str, Any]:
    return {'task': args.task, 'mode': args.mode, 'dry_run': bool(args.dry_run),
            'targets': list(opts.targets),
            'speed_percent': opts.speed, 'argv': list(sys.argv[1:])}


def _run_session(args: argparse.Namespace, s, opts: SimpleNamespace) -> int:
    from ..core.interfaces import ArmError
    from ..task.runner import TaskRunner
    cfg = s.cfg
    summary = _base_summary(args, opts)
    try:
        s.arm.connect()                 # opens communication; nothing moves
    except ArmError as e:
        _say('FAULT: cannot connect to the arm: %s' % e)
        _close_log(s, dict(summary, stop_reason='arm_connect', exit_code=EXIT_FAULT))
        return EXIT_FAULT
    lines = summary_lines(args, s, opts)
    _say('\n'.join(lines))
    if s.logger is not None:
        s.logger.save_text('config.yaml', cfg.to_yaml())
        s.logger.event('run_task', summary=lines, **summary)
    try:
        arm = s.arm
        runner = TaskRunner(cfg, args.task, arm, s.camera, s.perception, s.clock, s.logger,
                            operator=s.operator, dry_run=args.dry_run,
                            targets=opts.targets or None,
                            mode='real' if opts.real else 'sim')
    except ValueError as e:
        _say('REFUSED: configuration: %s' % e)
        _close_log(s, dict(summary, stop_reason='config', exit_code=EXIT_REFUSED))
        return EXIT_REFUSED
    if opts.real and not _precheck(args, s):
        _close_log(s, dict(summary, stop_reason='self_check', exit_code=EXIT_REFUSED))
        return EXIT_REFUSED
    if opts.real and not args.yes and not _confirm(args, cfg, opts):
        _say('Not started (no confirmation). Nothing moved.')
        _close_log(s, dict(summary, stop_reason='not_confirmed', exit_code=EXIT_REFUSED))
        return EXIT_REFUSED
    if not s.arm.enable(timeout=float(cfg.get('robot.enable_timeout_s', 5.0))):
        _say('FAULT: the arm did not enable; check the e-stop button and the CAN link')
        _close_log(s, dict(summary, stop_reason='enable', exit_code=EXIT_FAULT))
        return EXIT_FAULT
    opts.enabled = True
    s.operator.set_estop_callback(s.arm.emergency_stop)
    s.arm.set_abort_event(s.operator.abort_event)
    rep = runner.run()
    truth = _sim_truth(s, args.task, opts, rep.stop_reason)
    code = exit_code_for(rep, args.dry_run, bool(s.operator.estopped), truth)
    _say('\n'.join(report_lines(rep, runner.fault_detail, truth, args.task, args.dry_run)
                   + failure_lines(getattr(opts, 'failures', None) or [])))
    if s.world is not None and truth is None:
        _say('sim placements: %s' % s.world.placements())
    _say('exit code %d (%s)' % (code, EXIT_MEANING[code]))
    _close_log(s, dict(summary, stop_reason=rep.stop_reason, ok=rep.ok, placed=rep.placed,
                       required=rep.required, elapsed=round(rep.elapsed, 3),
                       attempts=rep.attempts, fault_detail=runner.fault_detail,
                       ground_truth=truth, exit_code=code))
    return code


def _ignore_sigint():
    try:
        return True, signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):        # not the main thread
        return False, None


def _restore_sigint(saved) -> None:
    ok, prev = saved
    if ok:
        try:
            signal.signal(signal.SIGINT, prev if prev is not None else signal.SIG_DFL)
        except (ValueError, OSError, TypeError):
            pass


def _interrupted(args: argparse.Namespace, s, opts: SimpleNamespace) -> int:
    saved = _ignore_sigint()            # a second Ctrl-C must not skip the cleanup
    try:
        # Stop first: printing can fail (closed pipe) and must not delay the stop.
        if s.operator is not None:
            s.operator.abort('Ctrl-C')
        _stop_arm(s, opts)
        if opts.enabled:
            _say('\nCtrl-C: arm stopped (it holds its position; the motors stay enabled)')
        else:
            _say('\nCtrl-C before the motors were enabled: nothing moved')
        truth = None
        try:
            truth = _sim_truth(s, args.task, opts, 'interrupted')
        except Exception:  # noqa: B902 - diagnostics only
            _LOG.exception('ground truth failed')
        _close_log(s, dict(_base_summary(args, opts), stop_reason='interrupted',
                           ground_truth=truth, exit_code=EXIT_INTERRUPTED))
        if truth is not None:
            _say('ground truth (%s): correct %d, wrong %d, remaining %d%s'
                 % ('physics' if 'dropped' in truth else 'sim', truth['correct'],
                    truth['wrong'], truth['remaining'],
                    ', dropped %d' % truth['dropped'] if 'dropped' in truth else ''))
        _say('log saved: %s' % (s.logger.dir if s.logger is not None else '-'))
    finally:
        _restore_sigint(saved)
    return EXIT_INTERRUPTED


def _open(args: argparse.Namespace):
    """Return (session, opts) or an exit code when the run is refused / faulted."""
    from ..core.interfaces import ArmError, CameraError
    try:
        cfg, world, info = load_cfg_for(args)
        opts = prepare_options(args, cfg, info)
    except Refused as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    except SystemExit as e:             # load_cfg_for reports bad --set / --replay-dir
        _say('REFUSED: %s' % (e.code,))
        return EXIT_REFUSED
    except (ValueError, TypeError, KeyError, OSError, ImportError) as e:
        _say('REFUSED: configuration: %s' % e)
        return EXIT_REFUSED
    try:
        s = open_session(args, args.task, arm=True, camera=True, perception=True, logger=True,
                         operator=True, connect=False, cfg_world_info=(cfg, world, info))
    except (ArmError, CameraError) as e:
        _say('FAULT: %s' % e)
        return EXIT_FAULT
    except SystemExit as e:
        _say('REFUSED: %s' % (e.code,))
        return EXIT_REFUSED
    except (ValueError, TypeError, KeyError, OSError, ImportError) as e:
        _say('REFUSED: setup: %s' % e)
        return EXIT_REFUSED
    if args.mode == 'mujoco':
        try:
            _attach_mujoco(args, s, opts)
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            s.close()
            _say('REFUSED: setup: %s' % e)
            return EXIT_REFUSED
    return s, opts


def _attach_mujoco(args: argparse.Namespace, s, opts: SimpleNamespace) -> None:
    """Attach the physics monitor, the video recorder and the viewer of --mode mujoco."""
    from ..core.clock import RealClock
    from ..sim.mujoco.run import RunExtras
    video = str(Path(args.video).expanduser()) if args.video else None
    opts.mujoco = RunExtras(s.world, video=video, viewer=bool(args.viewer),
                            video_speed=float(args.video_speed), wall_clock=RealClock())
    opts.mujoco.attach(s)


def _close_mujoco(opts: SimpleNamespace) -> None:
    extras = getattr(opts, 'mujoco', None)
    if extras is None:
        return
    try:
        info = extras.close()
    except Exception:  # noqa: B902 - the video must not hide the run result
        _LOG.exception('closing the video / viewer failed')
        return
    if info:
        _say('video: %s (%.1f s, %d frames; poster %s)' % (
            info['path'], info['duration_s'], info['frames'], info['poster'] or '-'))


def main(argv: Optional[List[str]] = None) -> int:
    """Run one round; returns the exit code documented in the module docstring."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        opened = _open(args)
    except KeyboardInterrupt:
        _say('\nCtrl-C before the arm was connected: nothing moved')
        return EXIT_INTERRUPTED
    if isinstance(opened, int):
        return opened
    s, opts = opened
    try:
        code = _run_session(args, s, opts)
    except KeyboardInterrupt:
        code = _interrupted(args, s, opts)
    except Exception as e:  # noqa: B902 - a bug here must still stop the arm
        _LOG.exception('run_task crashed')
        _stop_arm(s, opts)
        _close_log(s, dict(_base_summary(args, opts), stop_reason='crash', error=str(e),
                           exit_code=EXIT_FAULT))
        _say('FAULT: run_task crashed: %s (arm told to stop)' % e)
        code = EXIT_FAULT
    finally:
        if s.operator is not None:
            s.operator.close()          # restore the terminal first
        _close_mujoco(opts)
        s.close()
    return code


if __name__ == '__main__':
    sys.exit(main())

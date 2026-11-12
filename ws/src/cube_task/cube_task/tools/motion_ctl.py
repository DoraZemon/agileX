"""
Motion-control entry point: drive the arm without the timed competition loop.

    python3 -m cube_task motion_ctl status                      # read-only arm state
    python3 -m cube_task motion_ctl pick --label red --xyz 0.30 0.05 0.05 ...
    python3 -m cube_task motion_ctl vision --label red          # camera -> detection -> pick
    python3 -m cube_task motion_ctl gripper --width 0.08
    python3 -m cube_task motion_ctl joints --joint 1 --delta 0.05   # small joint jog
    python3 -m cube_task motion_ctl home [--observe]            # back to the home pose
    python3 -m cube_task motion_ctl moveit [--execute]          # RViz / MoveIt bridge

Mode 1, vision target input: ``pick`` takes a base-frame target typed by hand, ``vision``
takes it from the camera + perception (same code path as the competition run). Both end in
TaskRunner.pick_detection(): grasp plan -> reach adjust -> PickPlaceSkill -> SafeArm.
Mode 2, MoveIt: ``moveit`` serves MoveIt's FollowJointTrajectory goals through SafeArm
(plan-only unless --execute). docs/运控使用说明.md has the full walkthrough.

Exit codes: 0 done, 1 the pick failed, 2 refused before any motion, 3 fault, 130 Ctrl-C.
The arm is stopped (held) on exit, never disabled or reset.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .run_task import (_arm_text, _attach_mujoco, _close_mujoco, _precheck, _say, _stop_arm,
                       EXIT_FAULT, EXIT_INCOMPLETE, EXIT_INTERRUPTED, EXIT_OK, EXIT_REFUSED,
                       Refused, resolve_speed)

_LOG = logging.getLogger('cube_task.tools.motion_ctl')

MAX_JOG_RAD = 0.10       # default cap per joint for one `joints` move (about 5.7 deg)
HARD_JOG_RAD = 3.0       # --max-step above this is refused


def _common_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    add_common_args(p, task_required=False, default_task='L1', mujoco=True)
    p.add_argument('--speed', type=int, default=None, metavar='PCT',
                   help='arm speed percent 1..100 (default robot.speed_percent; use 20 on a '
                        'first real run)')
    p.add_argument('--allow-faster', action='store_true',
                   help='allow --speed above robot.speed_percent')
    p.add_argument('-y', '--yes', action='store_true',
                   help='real mode: skip the confirmation question')
    p.add_argument('--video', default=None, help='mujoco: record a video here')
    p.add_argument('--viewer', action='store_true', help='mujoco: open the live viewer')
    p.add_argument('--video-speed', type=float, default=1.0, help='mujoco: video speed factor')
    return p


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with the sub-commands."""
    parent = _common_parent()
    p = argparse.ArgumentParser(prog='cube_task motion_ctl', description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status', parents=[parent], help='connect read-only and print the arm state')

    pk = sub.add_parser('pick', parents=[parent], help='pick one hand-typed base-frame target')
    pk.add_argument('--label', required=True, help='class label, e.g. red (selects the zone)')
    pk.add_argument('--xyz', type=float, nargs=3, required=True, metavar=('X', 'Y', 'TOP_Z'),
                    help='target top-face centre in the robot base frame (m)')
    pk.add_argument('--size', type=float, nargs=3, default=(0.05, 0.05, 0.05),
                    metavar=('LEN', 'WID', 'HEIGHT'), help='object size (m), default a 50 mm cube')
    pk.add_argument('--yaw', type=float, default=0.0, help='object yaw about base z (rad)')
    pk.add_argument('--table-z', type=float, default=None,
                    help='table height in the base frame (m); default perception.table.expected_z '
                         'or top_z - height')
    pk.add_argument('--zone', default=None, help='place zone name (default: from the label)')
    pk.add_argument('--dry-run', action='store_true',
                    help='plan and hover over the target, do not close the gripper or lift')

    vi = sub.add_parser('vision', parents=[parent],
                        help='observe with the camera, then pick the chosen detection')
    vi.add_argument('--label', default=None, help='only this class (default: any task class)')
    vi.add_argument('--count', type=int, default=1, help='observe + pick this many times')
    vi.add_argument('--list-only', action='store_true',
                    help='observe and print the detections, do not pick')
    vi.add_argument('--dry-run', action='store_true',
                    help='plan and hover over the target, do not close the gripper or lift')

    gr = sub.add_parser('gripper', parents=[parent], help='move the gripper only')
    gr.add_argument('--width', type=float, required=True, help='opening width in metres')
    gr.add_argument('--force', type=float, default=None, help='gripper force (N, SDK units)')

    mv = sub.add_parser('moveit', parents=[parent],
                        help='MoveIt bridge: serve FollowJointTrajectory + joint feedback')
    mv.add_argument('--execute', action='store_true',
                    help='really move the arm; without it trajectory goals are refused')
    mv.add_argument('--waypoint-step', type=float, default=0.0, metavar='RAD',
                    help='0 (default): send only the final point, one smooth move; > 0: '
                         'follow the planned path with blocking waypoints this far apart '
                         '(stop-and-go)')
    jt = sub.add_parser('joints', parents=[parent],
                        help='small joint move: --joint N --delta RAD, or --to q1..q6')
    jt.add_argument('--joint', type=int, choices=range(1, 7), metavar='N',
                    help='joint number 1..6 (with --delta)')
    jt.add_argument('--delta', type=float, default=None, metavar='RAD',
                    help='relative move of --joint, rad')
    jt.add_argument('--to', type=float, nargs=6, default=None, metavar='Q',
                    help='absolute joint target q1..q6, rad (every joint is capped too)')
    jt.add_argument('--max-step', type=float, default=MAX_JOG_RAD, metavar='RAD',
                    help='refuse any joint moving more than this (default %.2f)' % MAX_JOG_RAD)
    jt.add_argument('--dry-run', action='store_true',
                    help='read the arm, check the joint limits, never move')
    hm = sub.add_parser('home', parents=[parent],
                        help='reset: straight up when low, then one joint move to '
                             'robot.poses.home_joints (or the observe pose)')
    hm.add_argument('--observe', action='store_true',
                    help='go to the observe pose (first observe view) instead of home_joints')
    hm.add_argument('--gripper', choices=('keep', 'open', 'close'), default='keep',
                    help='gripper after arriving (default keep; open drops a held object)')
    hm.add_argument('--dry-run', action='store_true',
                    help='read the arm, print the target and check the limits, never move')
    return p


def _open(args):
    """Return (session, opts) or an exit code."""
    from ..core.interfaces import ArmError, CameraError
    need_cam = args.cmd == 'vision'
    try:
        cfg, world, info = load_cfg_for(args)
        sp = resolve_speed(args, cfg)
    except Refused as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    except SystemExit as e:
        _say('REFUSED: %s' % (e.code,))
        return EXIT_REFUSED
    except (ValueError, TypeError, KeyError, OSError, ImportError) as e:
        _say('REFUSED: configuration: %s' % e)
        return EXIT_REFUSED
    if args.cmd == 'moveit' and args.mode not in ('real', 'sim'):
        _say('REFUSED: moveit supports --mode real or sim (rclpy needs the system python)')
        return EXIT_REFUSED
    try:
        s = open_session(args, args.task, arm=True, camera=need_cam, perception=need_cam,
                         logger=True, operator=args.cmd in ('pick', 'vision', 'home'),
                         connect=False, cfg_world_info=(cfg, world, info))
    except (ArmError, CameraError) as e:
        _say('FAULT: %s' % e)
        return EXIT_FAULT
    except SystemExit as e:
        _say('REFUSED: %s' % (e.code,))
        return EXIT_REFUSED
    except (ValueError, TypeError, KeyError, OSError, ImportError) as e:
        _say('REFUSED: setup: %s' % e)
        return EXIT_REFUSED
    opts = SimpleNamespace(real=args.mode == 'real', enabled=False, speed=sp.speed)
    if args.mode == 'mujoco':
        try:
            _attach_mujoco(args, s, opts)
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            s.close()
            _say('REFUSED: setup: %s' % e)
            return EXIT_REFUSED
    return s, opts


def _confirm(args, s, what: str) -> bool:
    """Real mode: show what will happen and ask once. Sim and -y never ask."""
    if args.mode != 'real' or args.yes:
        return True
    _say('About to ENABLE the motors and %s at speed %d%%.'
         % (what, s.cfg.get('robot.speed_percent')))
    _say('Clear the table edge, keep a hand on the e-stop; keys: p/space abort, e e-stop.')
    try:
        ans = input('Type yes to continue: ').strip().lower()
    except EOFError:
        return False
    return ans in ('y', 'yes')


def _connect(s) -> Optional[int]:
    from ..core.interfaces import ArmError
    try:
        s.arm.connect()
    except ArmError as e:
        _say('FAULT: cannot connect to the arm: %s' % e)
        return EXIT_FAULT
    return None


def _enable(s, opts) -> Optional[int]:
    if not s.arm.enable(timeout=float(s.cfg.get('robot.enable_timeout_s', 5.0))):
        _say('FAULT: the arm did not enable; check the e-stop button and the CAN link')
        return EXIT_FAULT
    opts.enabled = True
    if s.operator is not None:
        s.operator.set_estop_callback(s.arm.emergency_stop)
        s.arm.set_abort_event(s.operator.abort_event)
    return None


def _make_runner(args, s, dry_run: bool = False):
    from ..task.runner import TaskRunner
    return TaskRunner(s.cfg, args.task, s.arm, s.camera, s.perception, s.clock, s.logger,
                      operator=s.operator, dry_run=dry_run,
                      mode='real' if args.mode == 'real' else 'sim')


def cmd_status(args, s, opts) -> int:
    """Connect read-only and print health, joints, TCP pose and gripper."""
    code = _connect(s)
    if code is not None:
        return code
    _say('mode %s, speed %s%%' % (args.mode, s.cfg.get('robot.speed_percent')))
    _say('arm: ' + _arm_text(s.arm))
    tcp = s.arm.get_tcp_pose()
    if tcp is not None and tcp.value is not None:
        T = np.asarray(tcp.value, dtype=float).reshape(4, 4)
        _say('tcp position in base (m): x %.4f y %.4f z %.4f; tool z axis %s'
             % (T[0, 3], T[1, 3], T[2, 3], np.round(T[:3, 2], 3).tolist()))
    w = s.arm.gripper_width()
    if w is not None and w.value is not None:
        _say('gripper width: %.1f mm' % (1000.0 * float(w.value)))
    return EXIT_OK


def _manual_detection(args, cfg):
    from ..core.types import Detection
    from ..perception.grasp import expected_table_z
    x, y, top = (float(v) for v in args.xyz)
    length, width, height = (float(v) for v in args.size)
    vals = [x, y, top, length, width, height, float(args.yaw)]
    if not all(math.isfinite(v) for v in vals) or min(length, width, height) <= 0.0:
        raise Refused('--xyz/--size/--yaw must be finite and the size positive')
    table_z = args.table_z
    if table_z is None:
        table_z = expected_table_z(cfg)
    if table_z is None:
        table_z = top - height
    return Detection(label=args.label, score=1.0, score_kind='manual', valid=True,
                     center_px=(float('nan'), float('nan')),
                     center_base=np.array([x, y, top]), top_z=top, height=height,
                     yaw=float(args.yaw), length=length, width=width,
                     extra={'table_z': float(table_z), 'source': 'manual'})


def _print_result(res, why: str) -> int:
    if res is None:
        _say('NOT PICKED: %s' % why)
        return EXIT_INCOMPLETE
    _say('pick %s: ok=%s stage=%s placed=%s%s' % (
        res.label, res.ok, res.stage, getattr(res, 'placed', None),
        '' if res.ok else ' error=%s' % res.error))
    return EXIT_OK if res.ok else EXIT_INCOMPLETE


def cmd_pick(args, s, opts) -> int:
    """Pick one hand-typed Detection through the same path the competition run uses."""
    try:
        det = _manual_detection(args, s.cfg)
        runner = _make_runner(args, s, args.dry_run)
    except (Refused, ValueError) as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    code = _connect(s)
    if code is not None:
        return code
    if args.mode == 'real' and not _precheck_arm_only(args, s):
        return EXIT_REFUSED
    _say('target %s at base xyz (%.3f, %.3f, %.3f) size %s yaw %.2f' % (
        det.label, *det.center_base, tuple(args.size), det.yaw))
    if not _confirm(args, s, 'pick it%s' % (' (dry run)' if args.dry_run else '')):
        _say('Not started (no confirmation). Nothing moved.')
        return EXIT_REFUSED
    code = _enable(s, opts)
    if code is not None:
        return code
    ok, why = runner.ready_for_pick(go_observe=False)
    if not ok:
        _say('FAULT: %s' % why)
        return EXIT_FAULT
    res, why = runner.pick_detection(det, zone_name=args.zone)
    return _print_result(res, why)


def _precheck_arm_only(args, s) -> bool:
    """Real mode self check; the camera check is skipped for arm-only commands."""
    return _precheck(args, s) if s.camera is not None else _arm_selfcheck(args, s)


def _arm_selfcheck(args, s) -> bool:
    from ..task.selfcheck import arm_self_check
    ok, msgs = arm_self_check(s.cfg, s.arm, 'real', clock=s.clock)
    for m in msgs:
        if not m.startswith('[ OK ]'):
            _say('self check %s' % m)
    if not ok:
        _say('REFUSED: the self check failed; nothing was enabled or moved.')
    return ok


def cmd_vision(args, s, opts) -> int:
    """Observe, show the detections, pick the one the task's selection policy chooses."""
    from .onsite import detection_table
    try:
        runner = _make_runner(args, s, args.dry_run)
    except ValueError as e:
        _say('REFUSED: configuration: %s' % e)
        return EXIT_REFUSED
    code = _connect(s)
    if code is not None:
        return code
    if args.mode == 'real' and not _precheck(args, s):
        return EXIT_REFUSED
    what = 'move to the observe pose and ' + ('list detections' if args.list_only else 'pick')
    if not _confirm(args, s, what):
        _say('Not started (no confirmation). Nothing moved.')
        return EXIT_REFUSED
    code = _enable(s, opts)
    if code is not None:
        return code
    ok, why = runner.ready_for_pick(go_observe=True)
    if not ok:
        _say('FAULT: %s' % why)
        return EXIT_FAULT
    last = EXIT_OK
    for i in range(max(1, int(args.count))):
        runner._at_observe = False          # re-observe from the observe pose each round
        result, why = runner.observe_once()
        if result is None:
            _say('NOT OBSERVED: %s' % why)
            return EXIT_INCOMPLETE
        valid = [d for d in result.detections if d.valid and d.center_base is not None]
        _say('detections (round %d):\n%s' % (i + 1, detection_table(result.detections)))
        cands = [d for d in valid if d.label in runner.class_to_zone
                 and (args.label is None or d.label == args.label)]
        if not cands:
            _say('no valid target%s among %d detections' % (
                '' if args.label is None else ' with label %s' % args.label, len(valid)))
            return EXIT_INCOMPLETE
        if args.list_only:
            return EXIT_OK
        target = runner._select(cands, valid) or cands[0]
        res, why = runner.pick_detection(target, neighbors=valid)
        last = _print_result(res, why)
        if last != EXIT_OK:
            return last
    return last


def cmd_gripper(args, s, opts) -> int:
    """Open / close the gripper to --width."""
    w = float(args.width)
    from ..perception.grasp import max_gripper_width
    top = max_gripper_width(s.cfg)
    if not math.isfinite(w) or not 0.0 <= w <= top:
        _say('REFUSED: --width must be within 0..%.3f m (gripper max width)' % top)
        return EXIT_REFUSED
    code = _connect(s)
    if code is not None:
        return code
    if not _confirm(args, s, 'move the gripper to %.0f mm' % (1000 * w)):
        return EXIT_REFUSED
    code = _enable(s, opts)
    if code is not None:
        return code
    g = s.arm.gripper(w, force=args.force, wait=True)
    _say('gripper: ok=%s error=%s %s' % (g.ok, g.error, g.detail))
    return EXIT_OK if (g.ok or g.error == 'timeout') else EXIT_FAULT


def jog_target(q_now, joint, delta, to, max_step):
    """Return the 6 joint target of ``joints``; raise Refused when it is unsafe or unclear."""
    q_now = np.asarray(q_now, dtype=float).reshape(-1)
    if q_now.shape[0] != 6 or not np.all(np.isfinite(q_now)):
        raise Refused('current joint angles are unavailable')
    if not math.isfinite(max_step) or not 0.0 < max_step <= HARD_JOG_RAD:
        raise Refused('--max-step must be in (0, %.2f] rad' % HARD_JOG_RAD)
    if (to is None) == (joint is None):
        raise Refused('give either --joint N --delta RAD or --to q1..q6')
    if to is not None:
        q = np.asarray(to, dtype=float)
    else:
        if delta is None:
            raise Refused('--joint needs --delta RAD')
        q = q_now.copy()
        q[int(joint) - 1] += float(delta)
    if q.shape[0] != 6 or not np.all(np.isfinite(q)):
        raise Refused('the joint target must be 6 finite values')
    big = np.abs(q - q_now)
    if float(big.max()) > max_step + 1e-9:
        raise Refused('joint %d would move %.3f rad, more than --max-step %.3f rad'
                      % (int(big.argmax()) + 1, float(big.max()), max_step))
    return q


def cmd_joints(args, s, opts) -> int:
    """Move the joints by a small step (SafeArm checks limits, table height and path)."""
    code = _connect(s)
    if code is not None:
        return code
    jq = s.arm.get_joints()
    if jq is None or jq.value is None:
        _say('FAULT: cannot read the joint angles')
        return EXIT_FAULT
    q_now = np.asarray(jq.value, dtype=float).reshape(-1)
    try:
        q = jog_target(q_now, args.joint, args.delta, args.to, args.max_step)
    except Refused as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    _say('joints now    (rad): %s' % np.round(q_now, 4).tolist())
    _say('joints target (rad): %s' % np.round(q, 4).tolist())
    if args.dry_run:
        kin = getattr(s.arm, '_kin', None)
        bad = kin.limit_violations(q) if kin is not None else []
        if bad:
            _say('REFUSED: joint limits: %s' % ', '.join(bad))
            return EXIT_REFUSED
        _say('dry run: within the joint limits; nothing moved.')
        return EXIT_OK
    step = float(np.abs(q - q_now).max())
    if not _confirm(args, s, 'jog the joints by up to %.3f rad' % step):
        _say('Not started (no confirmation). Nothing moved.')
        return EXIT_REFUSED
    code = _enable(s, opts)
    if code is not None:
        return code
    res = s.arm.move_joints(q)
    after = s.arm.get_joints()
    _say('move: ok=%s error=%s' % (res.ok, res.error))
    if after is not None and after.value is not None:
        _say('joints after (rad): %s' % np.round(np.asarray(after.value), 4).tolist())
    return EXIT_OK if res.ok else EXIT_FAULT


def home_target(cfg, observe: bool = False) -> Tuple[np.ndarray, str]:
    """Return (joint target, name) of the reset: robot.poses.home_joints or the observe pose."""
    if observe:
        from ..task.views import observe_views
        try:
            views = observe_views(cfg)
        except ValueError as e:
            raise Refused(str(e))
        if not views:
            raise Refused('no observe pose taught: teach it with teach_poses (o)')
        return np.asarray(views[0], dtype=float), 'observe pose'
    q = cfg.get('robot.poses.home_joints', None)
    try:
        vals = [float(v) for v in (q if q is not None else [0.0] * 6)]
    except (TypeError, ValueError):
        raise Refused('robot.poses.home_joints must be 6 joint angles (rad), got %r' % (q,))
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        raise Refused('robot.poses.home_joints must be 6 finite joint angles (rad), got %r'
                      % (q,))
    return np.asarray(vals, dtype=float), 'home pose'


def cmd_home(args, s, opts) -> int:
    """Reset: go straight up when the TCP is low, then one joint move to the home pose."""
    try:
        q_goal, what = home_target(s.cfg, args.observe)
    except Refused as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    code = _connect(s)
    if code is not None:
        return code
    jq = s.arm.get_joints()
    if jq is None or jq.value is None:
        _say('FAULT: cannot read the joint angles')
        return EXIT_FAULT
    q_now = np.asarray(jq.value, dtype=float).reshape(-1)
    kin = getattr(s.arm, '_kin', None)
    bad = kin.limit_violations(q_goal) if kin is not None else []
    if bad:
        _say('REFUSED: the %s is outside the joint limits: %s' % (what, ', '.join(bad)))
        return EXIT_REFUSED
    step = float(np.max(np.abs(q_goal - q_now)))
    _say('joints now    (rad): %s' % np.round(q_now, 4).tolist())
    _say('%-13s (rad): %s' % (what, np.round(q_goal, 4).tolist()))
    _say('largest joint change %.1f deg; low TCP goes straight up first' % math.degrees(step))
    if args.dry_run:
        _say('dry run: the %s is within the joint limits; nothing moved.' % what)
        return EXIT_OK
    if args.mode == 'real' and not _arm_selfcheck(args, s):
        return EXIT_REFUSED
    if not _confirm(args, s, 'move to the %s (largest joint change %.0f deg)'
                    % (what, math.degrees(step))):
        _say('Not started (no confirmation). Nothing moved.')
        return EXIT_REFUSED
    code = _enable(s, opts)
    if code is not None:
        return code
    if s.operator is not None and hasattr(s.operator, 'start'):
        s.operator.start()
    runner = _make_runner(args, s)
    ok_up, err = runner._lift_if_low()
    if not ok_up:
        _say('FAULT: cannot go straight up first: %s' % err)
        return EXIT_FAULT
    res = s.arm.move_joints(q_goal)
    after = s.arm.get_joints()
    _say('move to the %s: ok=%s error=%s %s' % (what, res.ok, res.error, res.detail or ''))
    if after is not None and after.value is not None:
        _say('joints after (rad): %s' % np.round(np.asarray(after.value), 4).tolist())
    if not res.ok:
        return EXIT_FAULT
    if args.gripper != 'keep':
        from ..perception.grasp import max_gripper_width
        width = max_gripper_width(s.cfg) if args.gripper == 'open' else 0.0
        g = s.arm.gripper(width, wait=True)
        _say('gripper %s: ok=%s error=%s' % (args.gripper, g.ok, g.error))
    return EXIT_OK


def cmd_moveit(args, s, opts) -> int:
    """Run the MoveIt bridge until Ctrl-C (see robot/moveit_bridge.py)."""
    try:
        from ..robot.moveit_bridge import MoveItBridge
    except ImportError as e:
        _say('REFUSED: %s (source /opt/ros/humble/setup.bash and use /usr/bin/python3)' % e)
        return EXIT_REFUSED
    from ..task.skills import gripper_force_for
    code = _connect(s)
    if code is not None:
        return code
    if args.execute:
        if args.mode == 'real' and not _arm_selfcheck(args, s):
            return EXIT_REFUSED
        if not _confirm(args, s, 'execute MoveIt trajectories on the arm'):
            _say('Not started (no confirmation). Nothing moved.')
            return EXIT_REFUSED
        code = _enable(s, opts)
        if code is not None:
            return code
    try:
        bridge = MoveItBridge(s.arm, force=gripper_force_for(s.cfg, args.task),
                              execute=args.execute, waypoint_step=args.waypoint_step)
    except (ImportError, ValueError) as e:
        _say('REFUSED: %s' % e)
        return EXIT_REFUSED
    _say('bridge running (%s). Start MoveIt with scripts/moveit_rviz.sh; Ctrl-C to quit.'
         % ('EXECUTE' if args.execute else 'plan-only'))
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
    return EXIT_OK


_CMDS = {'status': cmd_status, 'pick': cmd_pick, 'vision': cmd_vision, 'gripper': cmd_gripper,
         'joints': cmd_joints, 'home': cmd_home, 'moveit': cmd_moveit}


def main(argv: Optional[List[str]] = None) -> int:
    """Run one sub-command; returns the exit code documented in the module docstring."""
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
        code = _CMDS[args.cmd](args, s, opts)
    except KeyboardInterrupt:
        _stop_arm(s, opts)
        _say('interrupted: the arm was told to stop (held, not disabled)')
        code = EXIT_INTERRUPTED
    except Exception as e:  # noqa: B902 - a bug here must still stop the arm
        _LOG.exception('motion_ctl crashed')
        _stop_arm(s, opts)
        _say('FAULT: motion_ctl crashed: %s (arm told to stop)' % e)
        code = EXIT_FAULT
    finally:
        if s.operator is not None:
            s.operator.close()
        _close_mujoco(opts)
        s.close()
    if opts.enabled and args.cmd != 'moveit':
        _say('done. The arm is still enabled and holds its pose (never disabled by this tool).')
    return code


if __name__ == '__main__':
    sys.exit(main())

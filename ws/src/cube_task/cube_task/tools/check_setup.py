"""
Pre-flight checks before a round. Nothing moves and nothing is enabled.

Examples::

    check_setup                        # real mode: modules, CAN, ROS topics, L1 / L2 config
    check_setup --task L2 --live       # also read the arm (no enable) and one camera frame
    check_setup --mode sim --task L1   # config of a simulated scenario
    check_setup --skip can,ros         # only modules and config (laptop without hardware)

Groups:

- modules: numpy, cv2, yaml (required); pyAgxArm, python-can (required for
  robot.backend pyagxarm in real mode); rclpy (required for camera.backend ros
  in real mode).
- can: ``ip -details link show <robot.can_channel>`` must exist, be UP at
  1000000 bit/s and not bus-off.
- ros: the configured camera topics must appear in ``ros2 topic list`` (only
  when ROS is sourced and camera.backend is ros).
- config: task.selfcheck.self_check without hardware for every task (or
  --task), plus reachability of the observe and zone poses with a kinematic
  SafeArm built with robot.backend=sim (nothing connects).
- live (--live): arm health, joints, TCP pose and gripper width read after
  connect() only; one camera frame: resolution, encodings, valid depth (image
  and pick area), intrinsics, frame rate.

Every FAIL comes with a one-line fix. Exit code 0 when nothing FAILed, else 1.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import add_common_args, ensure_sim_knn, load_cfg_for, open_session, TASKS

_LOG = logging.getLogger('cube_task.tools.check_setup')

GROUPS = ('modules', 'can', 'ros', 'config', 'live')
CAN_BITRATE = 1000000
ROS_TIMEOUT_S = 10.0
FPS_WINDOW_S = 1.0
NUMPY_TESTED = '1.21.5'
CV2_TESTED = '4.5.'
_NO_HW = ('[WARN] no arm given: arm checks skipped',
          '[WARN] no camera given: camera checks skipped')
_LEVEL_OF = {'[FAIL]': 'FAIL', '[WARN]': 'WARN', '[ OK ]': 'PASS'}


# ------------------------------------------------------------------ report
class Item:
    """One check result: level PASS / WARN / FAIL, group, text and fix hint."""

    __slots__ = ('level', 'group', 'text', 'fix')

    def __init__(self, level: str, group: str, text: str, fix: str = ''):
        self.level = level
        self.group = group
        self.text = text
        self.fix = fix


class Report:
    """Collect check results; every FAIL gets a fix hint."""

    def __init__(self) -> None:
        self.items: List[Item] = []
        self.skipped: List[Tuple[str, str]] = []

    def add(self, level: str, group: str, text: str, fix: str = '') -> None:
        if level == 'FAIL' and not fix:
            fix = DEFAULT_FIX.get(group, 'see the message above')
        self.items.append(Item(level, group, text, fix))

    def ok(self, group: str, text: str) -> None:
        self.add('PASS', group, text)

    def warn(self, group: str, text: str, fix: str = '') -> None:
        self.add('WARN', group, text, fix)

    def fail(self, group: str, text: str, fix: str = '') -> None:
        self.add('FAIL', group, text, fix)

    def skip(self, group: str, why: str) -> None:
        self.skipped.append((group, why))

    def count(self, level: str) -> int:
        return sum(1 for i in self.items if i.level == level)

    @property
    def failed(self) -> bool:
        return self.count('FAIL') > 0

    def format(self) -> str:
        out: List[str] = []
        for g in GROUPS:
            items = [i for i in self.items if i.group == g]
            why = [w for grp, w in self.skipped if grp == g]
            if not items and not why:
                continue
            out.append('== %s ==' % g)
            for w in why:
                out.append('  -    skipped: %s' % w)
            for i in items:
                out.append('[%s] %s' % (i.level, i.text))
                if i.fix and i.level != 'PASS':
                    out.append('       fix: %s' % i.fix)
        verdict = 'NOT READY: fix the FAIL items above' if self.failed else 'ready'
        out.append('== summary: %d PASS, %d WARN, %d FAIL -> %s ==' % (
            self.count('PASS'), self.count('WARN'), self.count('FAIL'), verdict))
        return '\n'.join(out)


DEFAULT_FIX = {
    'modules': 'install it for this python3 (pyAgxArm README); keep numpy 1.21.5',
    'can': 'bash scripts/can_up.sh <robot.can_channel>',
    'ros': 'start the camera driver and source /opt/ros/humble/setup.bash',
    'config': 'edit config/site/*.yaml (see config/site/README.md)',
    'live': 'see the message; rerun check_setup --live after fixing it',
}


def run_cmd(cmd: Sequence[str], timeout: float) -> Tuple[Optional[int], str, str]:
    """Run a command; return (returncode or None when it could not run, stdout, stderr)."""
    try:
        p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, universal_newlines=True)
    except FileNotFoundError:
        return None, '', '%s not found' % cmd[0]
    except subprocess.TimeoutExpired:
        return None, '', '"%s" timed out after %.0f s' % (' '.join(cmd), timeout)
    except OSError as e:
        return None, '', str(e)
    return p.returncode, p.stdout or '', p.stderr or ''


def _pkg_root() -> Path:
    from ..core.config import package_root
    return package_root()


def can_up_command(channel: str, bitrate: Optional[int] = None) -> str:
    """Return the command that brings the CAN interface up (scripts/can_up.sh)."""
    cmd = 'bash %s %s' % (_pkg_root() / 'scripts' / 'can_up.sh', channel)
    return cmd + (' %d' % bitrate if bitrate else '')


def official_can_script() -> Optional[Path]:
    """Return the AgileX can_activate.sh (third_party checkout or installed pyAgxArm)."""
    cands = [_pkg_root().parents[2] / 'third_party' / 'pyAgxArm' / 'pyAgxArm' / 'scripts' /
             'ubuntu' / 'can_activate.sh']
    try:
        import pyAgxArm
        cands.append(Path(pyAgxArm.__file__).parent / 'scripts' / 'ubuntu' / 'can_activate.sh')
    except Exception:  # noqa: B902 - optional
        pass
    for c in cands:
        if c.is_file():
            return c
    return None


# ----------------------------------------------------------------- modules
def _version(mod: Any, dist: Optional[str] = None) -> str:
    v = getattr(mod, '__version__', None)
    if v:
        return str(v)
    try:
        from importlib import metadata
        return metadata.version(dist or mod.__name__)
    except Exception:  # noqa: B902 - version is informational
        return '?'


def check_modules(rep: Report, need_arm: bool, need_ros: bool) -> None:
    """Check the Python version and the runtime modules."""
    g = 'modules'
    if sys.version_info < (3, 8):
        rep.fail(g, 'python %s is too old (need 3.8+)' % platform.python_version(),
                 'use the Ubuntu 22.04 python3 (3.10)')
    else:
        rep.ok(g, 'python %s (%s)' % (platform.python_version(), sys.executable))
    specs = [
        ('numpy', 'numpy', None, True, 'sudo apt install python3-numpy (keep 1.21.5)'),
        ('cv2', 'OpenCV', None, True, 'sudo apt install python3-opencv'),
        ('yaml', 'PyYAML', None, True, 'sudo apt install python3-yaml'),
        ('pyAgxArm', 'pyAgxArm', 'pyAgxArm', need_arm,
         'install pyAgxArm for this python3: cd third_party/pyAgxArm && '
         'python3 -m pip install --user -c ~/constraints.txt .'),
        ('can', 'python-can', 'python-can', need_arm,
         'python3 -m pip install --user -c ~/constraints.txt python-can'),
        ('rclpy', 'rclpy', None, need_ros, 'source /opt/ros/humble/setup.bash'),
    ]
    for name, label, dist, required, fix in specs:
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # noqa: B902 - ImportError or a broken install
            why = '%s not importable: %s' % (label, e)
            if required:
                rep.fail(g, why, fix)
            else:
                rep.warn(g, why + ' (only needed on the real robot)')
            continue
        ver = _version(mod, dist)
        rep.ok(g, '%s %s' % (label, ver))
        if name == 'numpy' and ver != NUMPY_TESTED:
            rep.warn(g, 'numpy %s: the code is tested with %s (~/constraints.txt)'
                     % (ver, NUMPY_TESTED))
        if name == 'cv2' and not ver.startswith(CV2_TESTED):
            rep.warn(g, 'OpenCV %s: tested with 4.5.4 (ONNX export patches target it)' % ver)


# --------------------------------------------------------------------- CAN
def parse_ip_link(text: str) -> Dict[str, Any]:
    """
    Parse ``ip -details link show <if>`` output.

    Returns exists, name, flags, up (UP flag), operstate, can_state
    (ERROR-ACTIVE / ERROR-PASSIVE / BUS-OFF / STOPPED / None), bitrate (int
    or None), virtual (vcan), parentdev (USB address or None).
    """
    info: Dict[str, Any] = {'exists': False, 'name': None, 'flags': [], 'up': False,
                            'operstate': None, 'can_state': None, 'bitrate': None,
                            'virtual': False, 'parentdev': None}
    lines = [ln for ln in str(text or '').splitlines() if ln.strip()]
    if not lines:
        return info
    m = re.match(r'^\s*\d+:\s*([^:@\s]+)', lines[0])
    if not m:
        return info
    info['exists'] = True
    info['name'] = m.group(1)
    f = re.search(r'<([^>]*)>', lines[0])
    info['flags'] = [x for x in (f.group(1).split(',') if f else []) if x]
    info['up'] = 'UP' in info['flags']
    s = re.search(r'\bstate\s+(\S+)', lines[0])
    info['operstate'] = s.group(1) if s else None
    body = '\n'.join(lines)
    c = re.search(r'\bcan\s+(?:<[^>]*>\s+)?state\s+([A-Z-]+)', body)
    info['can_state'] = c.group(1) if c else None
    b = re.search(r'\bbitrate\s+(\d+)', body)
    info['bitrate'] = int(b.group(1)) if b else None
    info['virtual'] = bool(re.search(r'^\s*vcan\b', body, re.M))
    p = re.search(r'\bparentdev\s+(\S+)', body)
    info['parentdev'] = p.group(1) if p else None
    return info


def check_can(rep: Report, channel: str, bitrate: int = CAN_BITRATE, run=None) -> None:
    """Check that the SocketCAN interface exists, is UP at bitrate and not bus-off."""
    run = run or run_cmd
    g = 'can'
    fix = can_up_command(channel)
    rc, out, err = run(['ip', '-details', 'link', 'show', channel], 5.0)
    if rc is None:
        rep.warn(g, 'cannot run "ip -details link show %s": %s' % (channel, err),
                 'sudo apt install iproute2')
        return
    info = parse_ip_link(out)
    if not info['exists']:
        _rc, br, _err = run(['ip', '-br', 'link', 'show', 'type', 'can'], 5.0)
        others = [ln.split()[0] for ln in (br or '').splitlines() if ln.strip()]
        official = official_can_script()
        hint = 'plug in the USB-CAN adapter, then: %s' % fix
        if others:
            hint += ' (or set robot.can_channel: %s in config/site/local.yaml)' % others[0]
        if official is not None:
            hint += '; official AgileX script: bash %s %s %d' % (official, channel, bitrate)
        rep.fail(g, '%s does not exist; CAN interfaces present: %s'
                 % (channel, ', '.join(others) or 'none'), hint)
        return
    where = ' (USB %s)' % info['parentdev'] if info['parentdev'] else ''
    if info['virtual']:
        rep.warn(g, '%s is a virtual CAN interface: no real arm on it' % channel,
                 'set robot.can_channel to the USB-CAN interface')
        return
    if not info['up']:
        rep.fail(g, '%s exists%s but is DOWN' % (channel, where), fix)
    elif info['bitrate'] != bitrate:
        rep.fail(g, '%s is UP at bitrate %s, the PiPER needs %d'
                 % (channel, info['bitrate'], bitrate), can_up_command(channel, bitrate))
    elif info['can_state'] == 'BUS-OFF':
        rep.fail(g, '%s is BUS-OFF (too many errors)' % channel,
                 'check the CAN cable and the arm power, then: %s (restarts the link)' % fix)
    elif info['can_state'] in ('ERROR-PASSIVE', 'ERROR-WARNING'):
        rep.warn(g, '%s can state %s: frames are not acknowledged' % (channel, info['can_state']),
                 'power on the arm and check the CAN cable')
    else:
        rep.ok(g, '%s UP at %d bit/s, can state %s%s' % (channel, bitrate,
                                                         info['can_state'] or '?', where))


# --------------------------------------------------------------------- ROS
def ros_sourced() -> bool:
    """Return True when a ROS 2 environment is sourced (ROS_DISTRO set and ros2 on PATH)."""
    return bool(os.environ.get('ROS_DISTRO')) and shutil.which('ros2') is not None


def camera_topics(cfg) -> List[Tuple[str, str]]:
    """Return [(config key, topic)] the RosCamera subscribes to."""
    keys = ['color_topic', 'depth_topic', 'info_topic']
    if not bool(cfg.get('camera.ros.depth_registered', True)):
        keys.append('depth_info_topic')
    return [(k, str(cfg.get('camera.ros.' + k, '') or '')) for k in keys]


def check_ros(rep: Report, cfg, run=None) -> None:
    """Check that the configured camera topics are advertised."""
    run = run or run_cmd
    g = 'ros'
    rc, out, err = run(['ros2', 'topic', 'list'], ROS_TIMEOUT_S)
    if rc is None or rc != 0:
        rep.fail(g, '"ros2 topic list" failed: %s' % (err.strip() or 'exit code %s' % rc),
                 'restart the ROS daemon: ros2 daemon stop; ros2 daemon start')
        return
    topics = set(ln.strip() for ln in out.splitlines() if ln.strip().startswith('/'))
    seen = sorted(t for t in topics if 'image' in t or 'camera_info' in t or 'depth' in t)
    dom = os.environ.get('ROS_DOMAIN_ID', '0 (unset)')
    for key, topic in camera_topics(cfg):
        if topic in topics:
            rep.ok(g, 'camera.ros.%s %s is advertised' % (key, topic))
        else:
            rep.fail(g, 'camera.ros.%s %s not found; camera topics seen: %s'
                     % (key, topic or '(empty)', ', '.join(seen[:6]) or 'none'),
                     'start the camera driver (e.g. ros2 launch orbbec_camera dabai.launch.py) '
                     'with ROS_DOMAIN_ID %s, or set camera.ros.%s in config/site/local.yaml'
                     % (dom, key))


# ------------------------------------------------------------------ config
_CONFIG_FIXES = [
    ('no zones configured', 'teach the zones with teach_poses (writes config/site/poses.yaml)'),
    ('zones:', 'fix the zones (center [x, y, z], size [sx, sy]) in config/site/poses.yaml'),
    ('labels is empty', 'set tasks.{T}.labels in config/site/tasks.yaml '
     '(copy config/examples/tasks.example.yaml)'),
    ('class_to_zone is empty', 'set tasks.{T}.class_to_zone in config/site/tasks.yaml'),
    ('class_to_zone: label', 'zone names in config/site/tasks.yaml must match the taught zones'),
    ('no extrinsics', 'copy the official calibration to config/site/extrinsics.yaml '
     '(see config/examples/extrinsics.example.yaml) or run calib_points'),
    ('extrinsics mount', 'set camera.mount in config/site/local.yaml to match the calibration'),
    ('extrinsics', 'check the calibration file format (docs/CONTRACTS.md, core/extrinsics.py)'),
    ('knn model', 'record the classes with teach_classes --task {T}'),
    ('onnx asset', 'export the detector as described in docs/yolo.md'),
    ('HSV ranges', 'run color_calib for the missing colors'),
    ('unknown classifier', 'set tasks.{T}.classifier to color, knn, onnx or onnx+knn'),
    ('observe', 'teach robot.poses.observe_joints with teach_poses'),
    ('not reachable', 'move the zone closer to the robot (re-teach with teach_poses)'),
    ('outside the workspace', 're-teach the pose with teach_poses or check robot.workspace'),
    ('tcp_offset', 'measure flange -> fingertip center; set robot.tcp_offset in local.yaml'),
    ('gripper', 'set robot.gripper.* in config/site/local.yaml (max_width 0.070 or 0.100)'),
    ('reachability', 'fix the robot section of the config (robot.model, robot.tcp_offset)'),
]


def config_fix(text: str, task: str) -> str:
    """Return the one-line fix for a config FAIL message."""
    low = text.lower()
    for key, fix in _CONFIG_FIXES:
        if key.lower() in low:
            return fix.replace('{T}', task)
    return DEFAULT_FIX['config']


def kinematic_arm(cfg):
    """Return a SafeArm around a bare kinematic SimArm (robot.backend=sim, never connects)."""
    from ..core.clock import SimClock
    from ..robot.factory import make_arm
    c = cfg.copy()
    c.set('robot.backend', 'sim')
    return make_arm(c, SimClock(), sim_world=None)


def config_messages(cfg, task: str) -> List[str]:
    """Return self_check lines (config only) plus kinematic reachability lines."""
    from ..task.selfcheck import self_check
    _ok, msgs = self_check(cfg, task, arm=None, camera=None, mode='real')
    msgs = [m for m in msgs if m not in _NO_HW]
    try:
        arm = kinematic_arm(cfg)
    except (ValueError, TypeError, KeyError) as e:
        return msgs + ['[FAIL] reachability not checked: %s' % e]
    _ok, sim_msgs = self_check(cfg, task, arm=arm, camera=None, mode='sim')
    msgs += [m for m in sim_msgs if m not in msgs and m not in _NO_HW]
    qj = cfg.get('robot.poses.observe_joints', None)
    kin = getattr(arm, 'kin', None)
    if qj is not None and kin is not None:
        try:
            q = [float(v) for v in qj]
        except (TypeError, ValueError):
            q = []
        if len(q) == 6 and not kin.within_limits(q):
            msgs.append('[FAIL] robot.poses.observe_joints outside the joint limits: %s'
                        % ', '.join(kin.limit_violations(q)))
    return msgs


def check_config(rep: Report, args: argparse.Namespace, tasks: Sequence[str],
                 explicit: bool) -> None:
    """Run the config-only self check for each task; lines common to tasks print once."""
    seen: Dict[str, List[str]] = {}
    order: List[str] = []
    for task in tasks:
        try:
            cfg, _world, _info = load_cfg_for(args, task)
            if args.mode == 'sim':
                ensure_sim_knn(cfg, task, args.mount)
        except SystemExit as e:
            rep.fail('config', '%s: config could not be loaded: %s' % (task, e.code))
            continue
        except Exception as e:  # noqa: B902 - any load problem is a finding
            rep.fail('config', '%s: config could not be loaded: %s' % (task, e))
            continue
        try:
            msgs = config_messages(cfg, task)
        except Exception as e:  # noqa: B902 - a crash in a check is a finding, not a crash
            _LOG.exception('self check crashed')
            msgs = ['[FAIL] self check crashed: %s' % e]
        for m in msgs:
            if m not in seen:
                seen[m] = []
                order.append(m)
            seen[m].append(task)
    for m in order:
        level = _LEVEL_OF.get(m[:6], 'WARN')
        text = m[7:] if m[:6] in _LEVEL_OF else m
        ts = seen[m]
        label = '%s: ' % ' '.join(ts) if len(tasks) > 1 else ''
        fix = config_fix(text, ts[0]) if level != 'PASS' else ''
        rep.add(level, 'config', label + text, fix if level == 'FAIL' else '')


# -------------------------------------------------------------------- live
def _arm_fix(reason: str, firmware_version: Optional[str] = None) -> str:
    """Return the fix for an arm FAIL; firmware_version tailors the drag-teach advice."""
    low = reason.lower()
    if 'teach' in low:
        from ..robot.pyagxarm_arm import teach_exit_hint
        return ('press the teach button on the arm again and re-run check_setup --live; if '
                'TEACHING_MODE stays: %s' % teach_exit_hint(firmware_version))
    if 'emergency' in low:
        return 'release the e-stop; after an electronic e-stop the arm needs a reset ' \
               '(power-cycle it while it rests on its support)'
    if 'connect' in low or 'feedback' in low or 'comm' in low:
        return 'check the CAN link (bash scripts/can_up.sh) and the arm power'
    return 'check the arm status; power-cycle the arm while it rests on its support'


def _live_arm(rep: Report, s) -> bool:
    from ..core.transforms import T_to_pose6
    g = 'live'
    arm = s.arm
    try:
        arm.connect()                  # read-only: opens CAN, waits for feedback
    except Exception as e:  # noqa: B902 - ArmError or SDK trouble
        rep.fail(g, 'arm: cannot connect: %s' % e,
                 'bash scripts/can_up.sh %s; power on the arm; stop other programs that use '
                 'it (agx_arm_ctrl)' % s.cfg.get('robot.can_channel', 'can0'))
        return False
    try:
        h = arm.health()
    except Exception as e:  # noqa: B902
        rep.fail(g, 'arm: health() raised: %s' % e)
        return True
    from ..robot.pyagxarm_arm import seamless_teach_switch
    d = h.detail or {}
    state = ', '.join(str(d[k]) for k in ('ctrl_mode_name', 'arm_status_name') if d.get(k))
    fw = getattr(getattr(arm, 'inner', None), 'firmware_info', None)
    fw = fw if isinstance(fw, dict) else {}
    sw = fw.get('software_version')
    if h.ok:
        rep.ok(g, 'arm: ready%s' % (' (%s)' % state if state else ''))
    else:
        rep.fail(g, 'arm: not ready: %s' % h.reason, _arm_fix(h.reason, sw))
    if h.enabled is False:
        rep.ok(g, 'arm: motors not enabled (run_task enables them after confirmation)')
    if fw:
        seamless = seamless_teach_switch(sw)
        teach = {True: 'leaves drag-teach on a CAN frame (S-V1.8-5 or later)',
                 False: 'needs a reset to leave drag-teach (below S-V1.8-5)'}.get(
                     seamless, 'drag-teach exit behaviour unknown for this version')
        rep.ok(g, 'arm: firmware %s (hardware %s); %s' % (sw or '?',
                                                          fw.get('hardware_version', '?'), teach))
    q = arm.get_joints()
    if q is None:
        rep.fail(g, 'arm: no fresh joint feedback', _arm_fix('feedback'))
    else:
        deg = ', '.join('%.1f' % math.degrees(float(v)) for v in q.value)
        rep.ok(g, 'arm: joints [%s] deg' % deg)
        kin = getattr(arm, 'kin', None)
        if kin is not None and not kin.within_limits(q.value):
            rep.warn(g, 'arm: joints outside the configured limits: %s'
                     % ', '.join(kin.limit_violations(q.value)))
    t = arm.get_tcp_pose()
    if t is None:
        rep.fail(g, 'arm: no fresh TCP pose', _arm_fix('feedback'))
    else:
        p = T_to_pose6(t.value)
        rep.ok(g, 'arm: TCP [%.1f, %.1f, %.1f] mm, rpy [%.1f, %.1f, %.1f] deg' % (
            p[0] * 1000, p[1] * 1000, p[2] * 1000, math.degrees(p[3]), math.degrees(p[4]),
            math.degrees(p[5])))
    if str(s.cfg.get('robot.effector', 'agx_gripper')) in ('none', 'None', ''):
        return True
    w = arm.gripper_width()
    mw = float(s.cfg.get('robot.gripper.max_width', 0.07) or 0.07)
    if w is None or w.value is None:
        rep.fail(g, 'gripper: no width feedback',
                 'check the gripper cable and robot.effector: agx_gripper')
    else:
        rep.ok(g, 'gripper: width %.1f mm (robot.gripper.max_width %.0f mm)'
               % (float(w.value) * 1000, mw * 1000))
        if float(w.value) > mw + 0.005:
            rep.warn(g, 'gripper opens wider than robot.gripper.max_width: set 0.100 for the '
                     '100 mm gripper')
    return True


def _live_camera(rep: Report, s, arm_ok: bool) -> None:
    import numpy as np
    from ..camera.factory import make_camera
    from ..core.extrinsics import extrinsics_from_cfg
    from ..task.selfcheck import camera_pose_for_check, pick_area_mask
    g = 'live'
    cfg = s.cfg
    try:
        cam = make_camera(cfg, s.clock, sim_world=s.world, arm=s.arm)
        cam.start()
    except Exception as e:  # noqa: B902 - CameraError, ImportError, config errors
        rep.fail(g, 'camera: cannot start: %s' % e,
                 'start the camera driver, then check the topics with check_setup (ros group)')
        return
    try:
        timeout = max(2.0, float(cfg.get('camera.frame_timeout_s', 2.0)))
        frame = cam.get_frame(timeout=timeout)
        if frame is None:
            diag = cam.diagnose() if hasattr(cam, 'diagnose') else ''
            rep.fail(g, 'camera: no frame within %.1f s%s'
                     % (timeout, '; ' + diag if diag else ''))
            return
        h, w = frame.color.shape[:2]
        enc = '%s / %s' % (frame.meta.get('color_encoding', '-'),
                           frame.meta.get('depth_encoding', '-'))
        rep.ok(g, 'camera: %dx%d color from %s, encodings %s' % (w, h, frame.source, enc))
        intr = frame.intrinsics
        if intr is None or not (intr.fx > 0 and intr.fy > 0):
            rep.fail(g, 'camera: intrinsics missing or invalid',
                     'check camera.ros.info_topic (CameraInfo of the color image)')
        else:
            dist = [round(float(v), 3) for v in np.asarray(intr.dist).ravel()]
            rep.ok(g, 'camera: intrinsics fx %.1f fy %.1f cx %.1f cy %.1f (%dx%d), dist %s'
                   % (intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height, dist))
        if frame.depth is None:
            rep.fail(g, 'camera: frame has no depth', 'enable depth in the camera driver')
        else:
            depth = np.asarray(frame.depth, dtype=np.float32)
            if depth.shape[:2] != (h, w):
                rep.fail(g, 'camera: depth %s does not match color %dx%d' % (depth.shape, w, h),
                         'enable depth_registration in the driver (camera.ros.depth_registered)')
            valid = np.isfinite(depth) & (depth > 0)
            frac = float(np.count_nonzero(valid)) / max(1, valid.size)
            (rep.ok if frac >= 0.5 else rep.warn)(
                g, 'camera: %.0f %% valid depth in the image' % (100 * frac))
            _pick_area_depth(rep, cfg, frame, depth, extrinsics_from_cfg, camera_pose_for_check,
                             pick_area_mask, s.arm if arm_ok else None)
        age = s.clock.now() - float(frame.stamp)
        if age > float(cfg.get('camera.max_frame_age_s', 1.0)):
            rep.warn(g, 'camera: frame is %.1f s old' % age, 'check the driver frame rate')
        t0 = s.clock.now()
        n, last = 0, float(frame.stamp)
        while s.clock.now() - t0 < FPS_WINDOW_S:
            f = cam.get_frame(timeout=1.0, newer_than=last)
            if f is None:
                break
            n += 1
            last = float(f.stamp)
        dt = max(1e-6, s.clock.now() - t0)
        fps = n / dt
        (rep.ok if fps >= 5.0 else rep.warn)(g, 'camera: %.1f frames/s' % fps)
    finally:
        try:
            cam.stop()
        except Exception:  # noqa: B902 - best effort
            _LOG.exception('camera.stop() failed')


def _pick_area_depth(rep, cfg, frame, depth, extrinsics_from_cfg, camera_pose_for_check,
                     pick_area_mask, arm) -> None:
    import numpy as np
    g = 'live'
    poly = cfg.get('perception.roi.pick_polygon', None) or []
    try:
        ext = extrinsics_from_cfg(cfg)
    except Exception as e:  # noqa: B902 - reported by the config group
        rep.warn(g, 'camera: pick-area depth not checked (extrinsics: %s)' % e)
        return
    if ext is None or len(poly) < 3:
        rep.warn(g, 'camera: pick-area depth not checked (no extrinsics or pick polygon)')
        return
    T_bc, why = camera_pose_for_check(cfg, ext, arm)
    if T_bc is None:
        rep.warn(g, 'camera: pick-area depth not checked: %s' % why)
        return
    # A wrist camera sees the pick area only from the observe pose, and nothing moves here.
    bad = rep.warn
    note = ' (wrist camera: valid only at the observe pose)'
    tz = cfg.get('perception.table.expected_z', None)
    try:
        mask = pick_area_mask(poly, 0.0 if tz is None else float(tz), T_bc, frame.intrinsics)
    except ValueError as e:
        bad(g, 'camera: %s%s' % (e, note), 'check the extrinsics direction (T_flange_camera)')
        return
    n = int(mask.sum())
    if n == 0:
        bad(g, 'camera: the pick polygon projects outside the image%s' % note,
            'check the extrinsics and perception.roi.pick_polygon (calib_verify)')
        return
    d = depth[mask]
    frac = float(np.count_nonzero(np.isfinite(d) & (d > 0))) / n
    if frac < 0.5:
        bad(g, 'camera: only %.0f %% valid depth in the pick area%s' % (100 * frac, note),
            'check the camera view, lighting and the minimum depth range of the camera')
    else:
        rep.ok(g, 'camera: %.0f %% valid depth in the pick area' % (100 * frac))


def check_live(rep: Report, args: argparse.Namespace, task: str) -> None:
    """Connect to the arm read-only and grab camera frames."""
    try:
        cwi = load_cfg_for(args, task)
        s = open_session(args, task, arm=True, camera=False, perception=False, connect=False,
                         cfg_world_info=cwi)
    except SystemExit as e:
        rep.fail('live', 'cannot build the arm: %s' % (e.code,))
        return
    except Exception as e:  # noqa: B902 - config / SDK problems are findings
        rep.fail('live', 'cannot build the arm: %s' % e)
        return
    with s:
        arm_ok = _live_arm(rep, s)
        _live_camera(rep, s, arm_ok)


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='check_setup', description='Pre-flight checks (nothing moves). Exit code 0 when '
        'no check FAILs.')
    add_common_args(p, default_task=None)
    p.add_argument('--live', action='store_true',
                   help='also connect to the arm (read-only, never enabled) and the camera')
    p.add_argument('--skip', default='', metavar='GROUPS',
                   help='comma list of groups to skip: modules, can, ros, config')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Run the checks and print the PASS / WARN / FAIL list; see the module docstring."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    if not args.verbose:
        logging.getLogger('cube_task').setLevel(logging.ERROR)
    skip = set(t.strip() for t in args.skip.split(',') if t.strip())
    bad = sorted(skip - set(GROUPS))
    if bad:
        print('check_setup: unknown --skip group(s) %s (use %s)' % (', '.join(bad),
                                                                    ', '.join(GROUPS[:4])))
        return 2
    tasks = [args.task] if args.task else list(TASKS)
    rep = Report()
    try:
        base, _w, _i = load_cfg_for(args, tasks[0])
    except SystemExit as e:
        rep.fail('config', 'config could not be loaded: %s' % (e.code,))
        base = None
    except Exception as e:  # noqa: B902 - reported, not raised
        rep.fail('config', 'config could not be loaded: %s' % e)
        base = None
    real = args.mode == 'real'
    arm_backend = str(base.get('robot.backend', '')) if base is not None else ''
    cam_backend = str(base.get('camera.backend', '')) if base is not None else ''
    print('check_setup: mode %s, task(s) %s, config %s' % (
        args.mode, ' '.join(tasks), ' + '.join(base.sources) if base is not None else '?'))
    if 'modules' in skip:
        rep.skip('modules', '--skip')
    else:
        check_modules(rep, need_arm=real and arm_backend == 'pyagxarm',
                      need_ros=real and cam_backend == 'ros')
    if 'can' in skip:
        rep.skip('can', '--skip')
    elif arm_backend != 'pyagxarm' or base is None:
        rep.skip('can', 'robot.backend is %s' % (arm_backend or '?'))
    else:
        check_can(rep, str(base.get('robot.can_channel', 'can0')))
    if 'ros' in skip:
        rep.skip('ros', '--skip')
    elif cam_backend != 'ros' or base is None:
        rep.skip('ros', 'camera.backend is %s' % (cam_backend or '?'))
    elif not ros_sourced():
        rep.warn('ros', 'ROS 2 is not sourced: camera topics not checked',
                 'source /opt/ros/humble/setup.bash (scripts/start.sh does this)')
    else:
        check_ros(rep, base)
    if 'config' in skip:
        rep.skip('config', '--skip')
    elif base is not None:
        check_config(rep, args, tasks, explicit=bool(args.task))
    if args.live:
        check_live(rep, args, tasks[0])
    else:
        rep.skip('live', 'pass --live to read the arm (no enable) and the camera')
    print(rep.format())
    return 1 if rep.failed else 0


if __name__ == '__main__':
    sys.exit(main())

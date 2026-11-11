"""
Shared helpers for the on-site setup tools (capture, view, calib_*, teach_*, color_calib).

Most functions are pure (no I/O) and unit tested. The few that talk to the
arm, the camera or the operator are thin wrappers and say so in their
docstring. Units: meters, radians; printed values are millimeters / degrees.

Site files: every tool writes with write_site_yaml() (timestamped backup of
the previous file next to it, then an atomic core.config.write_yaml) and
prints the dotted keys it wrote. In sim mode the default output directory is
``<logging.dir>/sim_site`` so a rehearsal never overwrites config/site.
"""
from __future__ import annotations

import datetime
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import read_yaml, write_yaml

_LOG = logging.getLogger('cube_task.tools.onsite')

#: pyAgxArm ctrl_mode of drag-teach mode (robot/pyagxarm_arm.CTRL_TEACHING).
CTRL_TEACHING = 0x02
EXIT_TEACH_WORD = 'EXIT'
EXIT_TEACH_WARNING = (
    'WARNING: exit_teach_mode() runs the official agx_arm_ctrl sequence in this order: '
    '(1) move_js to joint zero in MIT mode: NO smoothing, NO trajectory planning, the arm '
    'JUMPS to zero as fast as the motors allow, then 2 s wait; (2) electronic e-stop; '
    '(3) reset(), which CUTS MOTOR POWER at once (a raised arm drops); (4) enable (the motors '
    'stiffen); (5) move_j to joint zero WITHOUT any safety check. It is only run when every '
    'joint is within 0.05 rad (2.9 deg) of zero, i.e. the arm rests folded on its support. '
    'Keep your hands off the arm until the tool prints "exit_teach_mode done". Try the teach '
    'button first; firmware S-V1.8-5 and later only need c = take CAN control.')
EXIT_TEACH_WARNING_ZH = (
    '警告：exit_teach_mode() 按官方 agx_arm_ctrl 的顺序执行：(1) move_js 到全零关节位，MIT 模式，'
    '没有平滑、没有轨迹规划，机械臂以电机能达到的最快速度直接跳到零位，然后等 2 秒；(2) 电子急停；'
    '(3) reset()，电机立即断电（抬起的臂会掉下来）；(4) 重新使能（电机变硬）；(5) move_j 到全零位，'
    '不经过任何安全检查。只有六个关节都在零位 0.05 rad（2.9°）以内（臂折叠放在支架上）时才会执行。'
    '工具打印 "exit_teach_mode done" 之前，手不要碰机械臂。先试示教按钮；S-V1.8-5 及以上固件只需要 '
    'c（接管 CAN 控制）。')
TAKE_CAN_WORD = 'CAN'
TAKE_CAN_WARNING = (
    'c = take CAN control: sends the enable frame and the speed / mode frame 0x151 with '
    'ctrl_mode CAN (what the official agx_arm_ctrl node sends at start-up), NO motion command. '
    'Firmware S-V1.8-5 and later switch from drag-teach to CAN control on such a frame '
    '(official README: seamless switching); below S-V1.8-5 the controller most likely stays '
    'in TEACHING_MODE until a reset. Not tested on this arm yet: the arm should stiffen and '
    'hold its pose. Hands off the arm, e-stop within reach.')


# ------------------------------------------------------------------ arguments / ui
def add_onsite_args(p, site: bool = True) -> None:
    """Add --headless, --preview-dir and (site=True) --site-dir to an argparse parser."""
    p.add_argument('--headless', action='store_true',
                   help='no windows: terminal prompts, preview images saved to --preview-dir')
    p.add_argument('--preview-dir', default=None,
                   help='headless preview images (default <logging.dir>/preview/<tool>)')
    if site:
        p.add_argument('--site-dir', default=None,
                       help='write site files here instead of the configured paths '
                            '(default in sim mode: <logging.dir>/sim_site)')


def logging_dir(cfg) -> Path:
    """Return the expanded logging.dir (default ~/vision_runs)."""
    return Path(str(cfg.get('logging.dir', '~/vision_runs') or '~/vision_runs')).expanduser()


def make_ui(args, cfg, tool: str, ui=None):
    """Return ui, or a new tools.ui.Ui honoring --headless / --preview-dir."""
    if ui is not None:
        return ui
    from .ui import Ui
    preview = getattr(args, 'preview_dir', None) or str(logging_dir(cfg) / 'preview' / tool)
    return Ui(headless=bool(getattr(args, 'headless', False)), preview_dir=preview)


# ------------------------------------------------------------------ site files
def wall_stamp() -> str:
    """Return the local wall time as YYYYmmdd_HHMMSS (file names only)."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def backup_path(path: Path, stamp: str) -> Path:
    """Return <dir>/<stem>.bak_<stamp><suffix> for path."""
    p = Path(path)
    return p.with_name('%s.bak_%s%s' % (p.stem, stamp, p.suffix))


def backup_file(path: Path, stamp: Optional[str] = None) -> Optional[Path]:
    """Copy an existing file to a timestamped backup next to it; return it (None if absent)."""
    p = Path(path)
    if not p.is_file():
        return None
    stamp = stamp or wall_stamp()
    dst = backup_path(p, stamp)
    k = 1
    while dst.exists():
        dst = backup_path(p, '%s_%d' % (stamp, k))
        k += 1
    shutil.copy2(str(p), str(dst))
    return dst


def write_site_yaml(path: Path, data: Dict[str, Any], header: str = '') -> Optional[Path]:
    """Back up the previous file, then write data atomically; return the backup path."""
    backup = backup_file(path)
    write_yaml(Path(path), data, header=header)
    return backup


def read_yaml_or_empty(path: Optional[Path]) -> Dict[str, Any]:
    """Return the mapping in path, or {} when the file does not exist."""
    if path is None or not Path(path).is_file():
        return {}
    return read_yaml(Path(path))


def dotted_keys(d: Dict[str, Any], prefix: str = '') -> List[str]:
    """Return the dotted leaf keys of a nested mapping (lists are leaves)."""
    out: List[str] = []
    for k, v in d.items():
        key = '%s.%s' % (prefix, k) if prefix else str(k)
        if isinstance(v, dict) and v:
            out.extend(dotted_keys(v, key))
        else:
            out.append(key)
    return out


def write_report(path: Path, keys: Sequence[str], backup: Optional[Path]) -> str:
    """Return the message printed after writing a site file."""
    lines = ['wrote %s' % path]
    if backup is not None:
        lines.append('  previous file kept as %s' % backup)
    lines.append('  keys written:')
    lines.extend('    %s' % k for k in keys)
    return '\n'.join(lines)


def overlay_file(cfg, basename: str) -> Path:
    """Return the site overlay path named basename from cfg site_overlays."""
    for rel in cfg.get('site_overlays', []) or []:
        if Path(str(rel)).name == basename:
            return cfg.resolve(str(rel))
    return cfg.resolve('config/site/%s' % basename)


def sim_site_dir(cfg) -> Path:
    """Return the directory sim-mode tools write site files to by default."""
    return logging_dir(cfg) / 'sim_site'


def output_path(cfg, mode: str, site_dir: Optional[str], default: Path) -> Path:
    """
    Return where a tool writes the file whose configured path is default.

    --site-dir wins (same file name inside it); sim mode uses sim_site_dir;
    otherwise the configured path itself.
    """
    if site_dir:
        return Path(site_dir).expanduser() / Path(default).name
    if mode == 'sim':
        return sim_site_dir(cfg) / Path(default).name
    return Path(default)


# ------------------------------------------------------------------ speed / motion
def resolve_speed(cfg, requested: Optional[int] = None) -> Tuple[int, str]:
    """
    Return (speed percent, note) for a tool that moves the arm.

    The default is robot.speed_percent. --speed can only lower it; a higher
    request is clamped to the config value (note says so).
    """
    base = cfg.get('robot.speed_percent', 30)
    try:
        base = int(base)
    except (TypeError, ValueError):
        raise ValueError('robot.speed_percent must be an integer, got %r' % (base,))
    if not 1 <= base <= 100:
        raise ValueError('robot.speed_percent must be in [1, 100], got %d' % base)
    if requested is None:
        return base, ''
    r = int(requested)
    if r < 1:
        raise ValueError('--speed must be >= 1, got %d' % r)
    if r > base:
        return base, ('--speed %d is above robot.speed_percent %d; using %d (raise the config '
                      'value to go faster)' % (r, base, base))
    return r, ''


class MotionGuard:
    """
    Print every motion and ask once before the first motion on the real arm.

    In sim / replay mode motions are only printed.
    """

    def __init__(self, ui, mode: str, speed: int):
        self.ui = ui
        self.mode = mode
        self.speed = int(speed)
        self.approved = mode != 'real'

    def approve(self, what: str) -> bool:
        """Print what is about to happen; return False when the operator declines."""
        self.ui.say('MOTION: %s' % what)
        if self.approved:
            return True
        self.ui.say('The arm will move at %d %% speed. Keep the e-stop within reach and '
                    'the workspace clear of hands.' % self.speed)
        self.approved = self.ui.confirm('Start moving?', default=False)
        if not self.approved:
            self.ui.say('motion cancelled')
        return self.approved


def health_of(arm):
    """Return arm.health(), or None when the call raises (thin wrapper)."""
    try:
        return arm.health()
    except Exception as e:  # noqa: B902 - a broken link must not crash the tool
        _LOG.warning('arm.health() failed: %s', e)
        return None


def is_teaching(arm) -> bool:
    """Return True when the controller reports drag-teach mode."""
    h = health_of(arm)
    return h is not None and h.ctrl_mode == CTRL_TEACHING


def _say_zh(ui, text: str) -> None:
    """Print a Chinese copy of a warning; a terminal that cannot encode it gets none."""
    try:
        ui.say(text)
    except UnicodeError:
        pass


def arm_firmware_version(arm) -> Optional[str]:
    """Return the controller software version read at connect (e.g. 'S-V1.8-5'), or None."""
    for a in (arm, getattr(arm, 'inner', None)):
        fw = getattr(a, 'firmware_info', None)
        if isinstance(fw, dict) and fw.get('software_version'):
            return str(fw['software_version']).strip()
    return None


def exit_teach_mode_confirmed(ui, arm) -> bool:
    """
    Offer PyAgxArm.exit_teach_mode() behind a typed confirmation (operator I/O).

    Refused without asking unless the arm reports every joint within
    EXIT_TEACH_MAX_JOINT_RAD of 0 (folded on its support): the sequence starts
    with an unsmoothed MIT jump to joint zero. Returns True when it ran.
    """
    from ..robot.pyagxarm_arm import zero_pose_problem
    p = read_pose(arm)
    problem = 'no fresh joint feedback' if p is None else zero_pose_problem(p.joints)
    if problem:
        ui.say('exit_teach_mode refused: %s. Its first command jumps to joint zero without '
               'smoothing. While drag mode is still on, drag the arm folded onto its support '
               '(every joint at 0), let go, then retry.' % problem)
        return False
    ui.say(EXIT_TEACH_WARNING)
    _say_zh(ui, EXIT_TEACH_WARNING_ZH)
    if not ui.confirm_word('Is the arm resting folded on its support and is nobody touching it?',
                           EXIT_TEACH_WORD):
        ui.say('exit_teach_mode not run')
        return False
    try:
        ran = bool(arm.exit_teach_mode())
    except Exception as e:  # noqa: B902 - report and let the operator decide
        ui.say('exit_teach_mode failed: %s' % e)
        return False
    ui.say('exit_teach_mode %s' % ('done' if ran else 'skipped: the arm was not in teach mode'))
    return ran


def take_can_control_confirmed(ui, arm) -> bool:
    """
    Behind a typed confirmation, call arm.enable() to leave drag-teach mode (operator I/O).

    enable() sends the enable frame, then set_speed_percent (mode frame 0x151
    with ctrl_mode CAN); no motion command. Firmware S-V1.8-5 and later switch
    to CAN control on it. Re-checks health(); returns True when the arm is
    ready afterwards.
    """
    ui.say(TAKE_CAN_WARNING)
    if not ui.confirm_word('Hands off the arm and the e-stop within reach?', TAKE_CAN_WORD):
        ui.say('CAN control not requested')
        return False
    try:
        enabled = bool(arm.enable())
    except Exception as e:  # noqa: B902 - report and let the operator decide
        ui.say('enable failed: %s' % e)
        return False
    h = health_of(arm)
    if h is not None and h.ok:
        ui.say('CAN control taken: the arm is ready (ctrl_mode %s)' % (h.ctrl_mode,))
        return True
    from ..robot.pyagxarm_arm import teach_exit_hint
    ui.say('still not ready after enable (%s): %s' % (
        'enable ok' if enabled else 'joints did not report enabled',
        'cannot read the arm health' if h is None else (h.reason or 'unknown problem')))
    if h is not None and h.ctrl_mode == CTRL_TEACHING:
        ui.say('The controller stays in TEACHING_MODE: %s' % teach_exit_hint(
            arm_firmware_version(arm)))
    return False


def ensure_motion_ready(ui, arm, purpose: str = 'move the arm', max_checks: int = 20) -> bool:
    """
    Make sure the arm accepts commands before a tool moves it (operator I/O).

    Checks health(). In drag-teach mode asks the operator to press the teach
    button, and offers (each behind a typed confirmation) c = take CAN
    control (enable only, firmware S-V1.8-5 and later) and x = exit_teach_mode
    (only with the arm folded at joint zero). Enables disabled motors after
    asking. Returns False when the operator cancels.
    """
    noted = False
    for _ in range(max_checks):
        h = health_of(arm)
        if h is None:
            ui.say('cannot read the arm health; not moving')
            return False
        if h.ok:
            if h.enabled is False:
                if not ui.confirm('The motors are disabled. Enable them now (the arm stiffens '
                                  'and holds its pose)?', default=True):
                    return False
                if not arm.enable():
                    ui.say('enable failed: check the e-stop and the CAN link')
                    return False
                continue
            return True
        teaching = h.ctrl_mode == CTRL_TEACHING
        if teaching:
            ui.say('The arm is in drag-teach mode, so the tool cannot %s.' % purpose)
            if not noted:
                from ..robot.pyagxarm_arm import teach_exit_hint
                ui.say('If it stays in drag-teach mode after the teach button: %s.'
                       % teach_exit_hint(arm_firmware_version(arm)))
                noted = True
            ans = ui.prompt('Press the teach button on the arm to leave drag-teach mode, then '
                            'Enter [Enter = check again, c = take CAN control, '
                            'x = exit_teach_mode (arm folded at zero only), q = cancel]')
        else:
            ui.say('The arm is not ready: %s' % (h.reason or 'unknown problem'))
            ans = ui.prompt('Fix it (e-stop released? CAN link up?), then Enter '
                            '[Enter = check again, q = cancel]')
        ans = ans.strip().lower()
        if ui.eof or ans.startswith('q'):
            return False
        if teaching and ans.startswith('c'):
            take_can_control_confirmed(ui, arm)
        elif teaching and ans.startswith('x'):
            exit_teach_mode_confirmed(ui, arm)
    return False


def teach_mode_reminder(ui, arm) -> None:
    """At the end of a teach tool: tell the operator if the arm is still in teach mode."""
    if not is_teaching(arm):
        return
    ui.say('The arm is still in drag-teach mode. Press the teach button on the arm to leave '
           'it (preferred), or let the next tool that moves the arm (calib_verify, color_calib '
           '--observe) handle it: it offers c = take CAN control.')
    ans = ui.prompt('Enter = leave it as is, x = exit_teach_mode now (arm folded at zero '
                    'only)').strip().lower()
    if ans.startswith('x'):
        exit_teach_mode_confirmed(ui, arm)


def observe_target(cfg) -> Tuple[str, Any]:
    """Return ('joints', q), ('tcp', T_base_tcp) or ('none', None) for the observe pose."""
    q = cfg.get('robot.poses.observe_joints', None)
    if q is not None:
        return 'joints', [float(v) for v in q]
    from ..task.selfcheck import observe_tcp_pose
    T = observe_tcp_pose(cfg)
    return ('none', None) if T is None else ('tcp', T)


def go_observe(arm, cfg, guard: MotionGuard) -> Tuple[bool, str]:
    """Move to the observe pose like the task runner (thin wrapper); return (ok, why)."""
    kind, target = observe_target(cfg)
    if kind == 'none':
        return False, 'no observe pose configured (robot.poses.observe_joints / observe_tcp)'
    if kind == 'joints':
        if not guard.approve('move_joints to the observe pose %s deg'
                             % fmt_list(np.degrees(target), 1)):
            return False, 'cancelled'
        m = arm.move_joints(target)
    else:
        from ..task.reach import reach_adjust
        T2, tilt, why = reach_adjust(arm, target)
        if T2 is None:
            return False, 'observe_tcp not reachable: %s' % why
        if not guard.approve('move_tcp to the observe pose %s mm (tilt %.0f deg)'
                             % (fmt_mm(T2[:3, 3]), tilt)):
            return False, 'cancelled'
        m = arm.move_tcp(T2, linear=False)
    if not m.ok:
        return False, '%s %s' % (m.error, m.detail)
    return True, ''


# ------------------------------------------------------------------ arm poses
@dataclass
class ArmPose:
    """Arm state read for teaching: joints (6,) rad, 4x4 base-frame poses."""

    joints: np.ndarray
    T_base_flange: np.ndarray
    T_base_tcp: np.ndarray
    ctrl_mode: Optional[int] = None
    teach_status: Optional[int] = None

    def to_extra(self) -> Dict[str, Any]:
        """Return JSON-friendly fields for frame metadata."""
        return {'joints': [float(v) for v in self.joints],
                'T_base_flange': np.asarray(self.T_base_flange, float).tolist(),
                'T_base_tcp': np.asarray(self.T_base_tcp, float).tolist()}


def read_pose(arm) -> Optional[ArmPose]:
    """
    Read the current arm pose (thin wrapper), also in drag-teach mode.

    Uses read_teach_pose() of the real arm, else get_joints / get_flange_pose /
    get_tcp_pose. Returns None without fresh feedback.
    """
    fn = getattr(arm, 'read_teach_pose', None)
    if fn is not None:
        try:
            d = fn()
        except NotImplementedError:
            d = False
        if d is None:
            return None
        if d is not False:
            return ArmPose(np.asarray(d['joints'], float), np.asarray(d['T_base_flange'], float),
                           np.asarray(d['T_base_tcp'], float), d.get('ctrl_mode'),
                           d.get('teach_status'))
    j, f, t = arm.get_joints(), arm.get_flange_pose(), arm.get_tcp_pose()
    if j is None or f is None or t is None:
        return None
    return ArmPose(np.asarray(j.value, float), np.asarray(f.value, float),
                   np.asarray(t.value, float))


def average_poses(poses: Sequence[ArmPose]) -> Tuple[ArmPose, float]:
    """
    Average several readings of a held-still arm.

    Returns (pose with mean joints / positions and the last orientation, max
    TCP deviation from the mean in m).
    """
    if not poses:
        raise ValueError('average_poses needs at least one pose')
    tcp = np.array([p.T_base_tcp[:3, 3] for p in poses])
    fl = np.array([p.T_base_flange[:3, 3] for p in poses])
    mean_t, mean_f = tcp.mean(axis=0), fl.mean(axis=0)
    last = poses[-1]
    T_t = np.array(last.T_base_tcp, float)
    T_f = np.array(last.T_base_flange, float)
    T_t[:3, 3] = mean_t
    T_f[:3, 3] = mean_f
    dev = float(np.max(np.linalg.norm(tcp - mean_t, axis=1)))
    q = np.mean([p.joints for p in poses], axis=0)
    return ArmPose(q, T_f, T_t, last.ctrl_mode, last.teach_status), dev


def read_pose_averaged(arm, clock, n: int = 8, dt: float = 0.02
                       ) -> Tuple[Optional[ArmPose], float]:
    """Read n poses dt apart (injected clock) and average them (thin wrapper)."""
    poses = []
    for k in range(max(1, int(n))):
        p = read_pose(arm)
        if p is not None:
            poses.append(p)
        if k + 1 < n:
            clock.sleep(dt)
    if not poses:
        return None, float('nan')
    return average_poses(poses)


def sim_arm(arm):
    """Return the simulated arm behind a SafeArm, or None when it is not a SimArm."""
    inner = getattr(arm, 'inner', arm)
    return inner if hasattr(inner, 'set_joints') else None


def sim_drag_joints(arm, q: Sequence[float]) -> None:
    """Simulate operator: drag the sim arm to joint angles q (teleport, no checks)."""
    inner = sim_arm(arm)
    if inner is None:
        raise RuntimeError('sim_drag_joints needs the simulated arm')
    inner.set_joints([float(v) for v in q])


def sim_drag_tcp(arm, cfg, T_base_tcp: np.ndarray, max_tilt_deg: float = 45.0) -> bool:
    """
    Simulate operator: drag the sim arm so its TCP reaches T_base_tcp.

    The position is kept exactly; the tool is leaned outward when the
    vertical pose is out of reach (a person dragging the arm does the same).
    Returns False when no pose up to max_tilt_deg is reachable.
    """
    from ..robot.kinematics import flange_to_tcp_from_cfg
    kin = getattr(arm, 'kin', None)
    if kin is None:
        raise RuntimeError('sim_drag_tcp needs an arm with a kinematic model (SafeArm)')
    st = arm.get_joints()
    seed = None if st is None else np.asarray(st.value, float)
    T_ft = flange_to_tcp_from_cfg(cfg)
    sol = kin.solve_tcp_with_tilt(np.asarray(T_base_tcp, float), T_ft,
                                  max_tilt=math.radians(max_tilt_deg), q_seed=seed)
    if sol is None:
        return False
    from ..core.transforms import inv_T
    q, _ep, _eo = kin.refine(sol[0] @ inv_T(T_ft), sol[1])   # sub-micrometer touch
    sim_drag_joints(arm, q)
    return True


def nearest_sim_object(world, xy: Sequence[float], max_dist: float = 0.05):
    """Return the sim object whose footprint center is nearest to xy (m), or None (sim only)."""
    if world is None or not world.objects:
        return None
    c = np.asarray(xy, dtype=float)[:2]
    best = min(world.objects, key=lambda o: float(np.hypot(*(o.center_xy() - c))))
    return best if float(np.hypot(*(best.center_xy() - c))) <= max_dist else None


def sim_relayout(world, task: str, seed: int, mount: str = 'eye_to_hand') -> int:
    """
    Simulate operator: replace the objects with the layout of scenario seed.

    Keeps the zones. Returns the number of objects now on the table.
    """
    from ..sim.scenarios import make_scenario
    w2, _info = make_scenario(task if task in ('L1', 'L2') else 'L1', int(seed),
                              mount=mount)
    world.objects = list(w2.objects)
    world.held = None
    return len(world.objects)


def operator_action(ui, mode: str, text: str, sim_action=None) -> bool:
    """
    Ask the operator to do something by hand, or do it in the simulation (operator I/O).

    Real / replay: print text and wait for Enter; False on 'q' or end of input.
    Sim with sim_action: run it (it returns True on success) and report.
    """
    if mode == 'sim' and sim_action is not None:
        ok = bool(sim_action())
        ui.say('[sim operator] %s%s' % (text, '' if ok else '  -> FAILED (out of reach)'))
        return ok
    ans = ui.prompt('%s, then Enter (q = cancel)' % text).strip().lower()
    return not (ui.eof or ans.startswith('q'))


def connect_arm_readonly(s, ui, required: bool = False, why: str = 'arm poses') -> bool:
    """
    Real mode: connect the arm without enabling it (reading poses only; thin wrapper).

    Drag-teach mode is fine. When the connection fails, returns False if
    required, else drops the arm (s.arm = None) and returns True.
    """
    if s.arm is None or s.mode != 'real':
        return True
    from ..core.interfaces import ArmError
    try:
        s.arm.connect()
        return True
    except ArmError as e:
        if required:
            ui.say('the arm did not connect (%s), but %s are needed. Check the CAN link '
                   '(scripts/can_up.sh) and the arm power.' % (e, why))
            return False
        ui.say('arm not connected (%s); continuing without %s' % (e, why))
        s.arm = None
        return True


def teach_hint(ui, arm) -> None:
    """Real arm: explain drag-teach mode when the arm is not in it yet (operator I/O)."""
    if arm is None or sim_arm(arm) is not None or is_teaching(arm):
        return
    ui.say('To move the arm by hand: hold it firmly, then press the teach button on the arm '
           '(drag-teach mode). Press the button again to leave drag-teach mode.')


def kinematic_arm(cfg):
    """Return a SafeArm around a bare kinematic SimArm (robot.backend=sim, never connects)."""
    from ..core.clock import SimClock
    from ..robot.factory import make_arm
    c = cfg.copy()
    c.set('robot.backend', 'sim')
    arm = make_arm(c, SimClock(), sim_world=None)
    tz = cfg.get('perception.table.expected_z', None)
    if tz is not None:
        arm.set_table_z(float(tz))
    return arm


# ------------------------------------------------------------------ commands
def next_command(ui, help_text: str, wait_ms: int = 30) -> Optional[str]:
    """
    Return the next operator command line, or None when no key was pressed (operator I/O).

    Headless: a terminal line. Window: the pressed key; Enter asks for a whole
    command line in the terminal (for commands with arguments such as '2 apple').
    At the end of the input stream returns 'q'.
    """
    if ui.headless:
        line = ui.prompt(help_text)
        return 'q' if ui.eof else line.strip()
    k = ui.key(wait_ms)
    if k is None:
        return None
    if k == '\n':
        line = ui.prompt('command (%s)' % help_text)
        return 'q' if ui.eof else line.strip()
    return 'q' if k == 'esc' else k


def parse_floats(text: str, n: int, default: Optional[Sequence[float]] = None
                 ) -> Optional[List[float]]:
    """
    Parse exactly n numbers separated by spaces or commas; '' gives default.

    Returns None when the text does not hold n finite numbers.
    """
    s = str(text or '').replace(',', ' ').split()
    if not s:
        return None if default is None else [float(v) for v in default]
    if len(s) != n:
        return None
    try:
        vals = [float(v) for v in s]
    except ValueError:
        return None
    return vals if all(math.isfinite(v) for v in vals) else None


def assign_command(text: str, n_instances: int) -> Tuple[Optional[List[int]], str, str]:
    """
    Parse 'all <label>' or '<n>[,<m>...] <label>' (teach_classes / color_calib).

    Returns (instance indices, label, '') or (None, '', error message).
    """
    parts = str(text or '').split()
    if len(parts) != 2:
        return None, '', 'expected "<n> <label>" or "all <label>"'
    which, label = parts[0].lower(), parts[1].strip()
    if not label or label == 'unknown' or not all(c.isalnum() or c in '_-' for c in label):
        return None, '', 'label %r: use letters, digits, _ or -' % label
    if which == 'all':
        if n_instances == 0:
            return None, '', 'no instances in this view'
        return list(range(n_instances)), label, ''
    try:
        idx = sorted({int(v) for v in which.split(',') if v != ''})
    except ValueError:
        return None, '', 'expected an instance number, got %r' % parts[0]
    bad = [i for i in idx if not 0 <= i < n_instances]
    if not idx or bad:
        return None, '', 'instance numbers must be in 0..%d' % (n_instances - 1)
    return idx, label, ''


# ------------------------------------------------------------------ previews
def add_depth_range_arg(p) -> None:
    """Add --depth-range ZMIN ZMAX (m) to an argparse parser (default: per-frame auto)."""
    p.add_argument('--depth-range', type=float, nargs=2, default=None, metavar=('ZMIN', 'ZMAX'),
                   help='depth preview color range in m (default: 2-98 %% percentiles of each '
                        'frame)')


def depth_range_auto(depth: Optional[np.ndarray], lo_pct: float = 2.0, hi_pct: float = 98.0,
                     min_span: float = 0.05) -> Tuple[float, float]:
    """
    Return (zmin, zmax) m for a depth colormap from percentiles of the valid depth.

    The span is at least min_span (centered). No valid depth -> (0.2, 1.5).
    """
    if depth is None:
        return 0.2, 1.5
    d = np.asarray(depth, dtype=np.float32)
    with np.errstate(invalid='ignore'):
        v = d[np.isfinite(d) & (d > 0)]
    if v.size == 0:
        return 0.2, 1.5
    lo, hi = (float(x) for x in np.percentile(v, [lo_pct, hi_pct]))
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    return max(0.0, lo), hi


def compose_view(frame, overlay: Optional[np.ndarray] = None,
                 zrange: Optional[Sequence[float]] = None) -> np.ndarray:
    """Return color (or overlay) | depth colormap side by side (zrange None = auto)."""
    from .ui import depth_colormap, side_by_side
    left = overlay if overlay is not None else frame.color
    lo, hi = depth_range_auto(frame.depth) if zrange is None else (float(zrange[0]),
                                                                   float(zrange[1]))
    return side_by_side(left, depth_colormap(frame.depth, lo, hi))


def capture_view(camera, cfg, n: Optional[int] = None, newer_than: Optional[float] = None):
    """Grab n frames (default camera.frames_to_average) with median depth (thin wrapper)."""
    from ..camera.utils import capture_averaged
    n = int(cfg.get('camera.frames_to_average', 3)) if n is None else int(n)
    return capture_averaged(camera, max(1, n), newer_than=newer_than,
                            timeout=float(cfg.get('camera.frame_timeout_s', 2.0)))


def write_extrinsics_site(path: Path, ext) -> Tuple[Path, Optional[Path], List[str]]:
    """Back up the previous file, save ext atomically; return (path, backup, keys written)."""
    from ..core.extrinsics import save_extrinsics
    backup = backup_file(path)
    save_extrinsics(path, ext)
    return Path(path), backup, dotted_keys(ext.to_dict())


def extrinsics_out_path(cfg, mode: str, site_dir: Optional[str]) -> Path:
    """Return where a calibration tool writes the extrinsics file."""
    default = cfg.path('camera.extrinsics_file') or cfg.resolve('config/site/extrinsics.yaml')
    if Path(default).suffix.lower() not in ('.yaml', '.yml', '.json'):
        # save_extrinsics writes YAML or JSON only (e.g. not .npy): write a .yaml sibling.
        default = Path(default).with_suffix('.yaml')
    return output_path(cfg, mode, site_dir, default)


def use_hint(cfg, mode: str, path: Path, key: str) -> str:
    """Return how to make the tools use a file written outside its configured place."""
    configured = cfg.path(key)
    if configured is not None and Path(configured).resolve() == Path(path).resolve():
        if key == 'camera.extrinsics_file' and cfg.get('camera.extrinsics_inline', None):
            return ('note: camera.extrinsics_inline is set and wins over the file; '
                    'add --set camera.extrinsics_inline=null')
        return 'the tools read it from %s (config key %s)' % (path, key)
    extra = ' --set camera.extrinsics_inline=null' if key == 'camera.extrinsics_file' else ''
    return 'to use it: --set %s=%s%s' % (key, path, extra)


# ------------------------------------------------------------------ frames
def grab_frames(camera, n: int, timeout: float = 2.0, newer_than: Optional[float] = None):
    """Return up to n consecutive new frames (list, may be shorter on timeout)."""
    frames = []
    last = newer_than
    for _ in range(max(1, int(n))):
        f = camera.get_frame(timeout=timeout, newer_than=last)
        if f is None:
            break
        frames.append(f)
        last = float(f.stamp)
    return frames


def recorded_flange_pose(frame) -> Optional[np.ndarray]:
    """Return T_base_flange stored with a recorded frame (replay), else None."""
    extra = (getattr(frame, 'meta', None) or {}).get('extra') or {}
    T = extra.get('T_base_flange')
    if T is None:
        return None
    a = np.asarray(T, dtype=float)
    return a if a.shape == (4, 4) and np.all(np.isfinite(a)) else None


def camera_pose_for_frame(ext, frame, arm, T_flange_tcp=None) -> Tuple[Optional[np.ndarray], str]:
    """
    Return (T_base_camera for this frame, '') or (None, why).

    Eye-in-hand uses the flange pose recorded with the frame (replay) or the
    arm's current flange pose (the arm must not have moved since capture).
    """
    if ext is None:
        return None, 'no extrinsics configured'
    from ..core.extrinsics import camera_pose_in_base
    T_bf = None
    if ext.mount == 'eye_in_hand':
        T_bf = recorded_flange_pose(frame)
        if T_bf is None and arm is not None:
            st = arm.get_flange_pose()
            T_bf = None if st is None else np.asarray(st.value, float)
        if T_bf is None:
            return None, 'eye_in_hand needs the flange pose and none is available'
    try:
        return camera_pose_in_base(ext, T_bf, T_flange_tcp), ''
    except ValueError as e:
        return None, str(e)


def load_extrinsics_or_none(cfg, ui=None):
    """Return extrinsics_from_cfg(cfg), or None (message to ui) when invalid / missing."""
    from ..core.extrinsics import extrinsics_from_cfg
    try:
        ext = extrinsics_from_cfg(cfg)
    except (ValueError, OSError) as e:
        if ui is not None:
            ui.say('extrinsics unusable: %s' % e)
        return None
    if ext is None and ui is not None:
        ui.say('no extrinsics configured (camera.extrinsics_file)')
    return ext


def build_perception(cfg, task: Optional[str], args, ui, classify: bool = True):
    """
    Return a Perception for a viewer tool; fall back to unlabeled instances.

    classify=False gives class-agnostic segmentation (teach tools). In sim
    mode a KNN backend uses the cached simulation model (ensure_sim_knn).
    """
    from ..perception.pipeline import Perception
    if not classify:
        return Perception(cfg)
    from ..perception.factory import make_perception
    from .common import ensure_sim_knn
    try:
        if args.mode == 'sim':
            ensure_sim_knn(cfg, task, args.mount)
        return make_perception(cfg, task)
    except (ValueError, OSError, ImportError, RuntimeError) as e:
        ui.say('classifier unavailable (%s); showing unlabeled instances' % e)
        return Perception(cfg)


def flange_to_tcp(cfg) -> np.ndarray:
    """Return T_flange_tcp from robot.tcp_offset."""
    from ..robot.kinematics import flange_to_tcp_from_cfg
    return flange_to_tcp_from_cfg(cfg)


# ------------------------------------------------------------------ formatting
def fmt_mm(v: Any, prec: int = 1) -> str:
    """Format a vector in meters as '[x, y, z]' millimeters."""
    return fmt_list(np.asarray(v, dtype=float) * 1000.0, prec)


def fmt_list(v: Any, prec: int = 3) -> str:
    """Format numbers as '[a, b, c]' with prec decimals."""
    return '[%s]' % ', '.join('%.*f' % (prec, float(x)) for x in np.asarray(v).ravel())


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return a plain-text table with left-aligned columns."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(len(headers))]
    lines = []
    for k, r in enumerate(cells):
        lines.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())
        if k == 0:
            lines.append('  '.join('-' * w for w in widths))
    return '\n'.join(lines)


DETECTION_HEADERS = ('#', 'label', 'score', 'center_base mm', 'top_z mm', 'LxWxH mm',
                     'yaw deg', 'status')


def detection_rows(dets) -> List[List[str]]:
    """Return table rows (see DETECTION_HEADERS) for a list of Detections."""
    rows = []
    for i, d in enumerate(dets):
        c = '-' if d.center_base is None else fmt_mm(d.center_base, 1)
        top = '-' if d.top_z is None else '%.1f' % (d.top_z * 1000.0)
        if d.length is not None and d.width is not None and d.height is not None:
            size = '%.0fx%.0fx%.0f' % (d.length * 1e3, d.width * 1e3, d.height * 1e3)
        else:
            size = '-'
        yaw = '-' if d.yaw is None else '%.0f' % math.degrees(d.yaw)
        status = 'ok' if d.valid else 'rejected: %s' % (d.reject_reason or 'invalid')
        rows.append([str(i), d.label, '%.2f' % float(d.score), c, top, size, yaw, status])
    return rows


def detection_table(dets) -> str:
    """Return the plain-text detection table (DETECTION_HEADERS) for dets."""
    if not dets:
        return '(no detections)'
    return format_table(DETECTION_HEADERS, detection_rows(dets))


def draw_numbers(img: np.ndarray, dets, color=(0, 255, 255)) -> np.ndarray:
    """Write '#i' next to each detection's center (in place); return img."""
    import cv2
    for i, d in enumerate(dets):
        u, v = d.center_px
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        org = (int(u) + 8, int(v) - 8)
        cv2.putText(img, '#%d' % i, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(img, '#%d' % i, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def text_panel(img: np.ndarray, lines: Sequence[str], line_h: int = 16) -> np.ndarray:
    """Return img with a black text panel appended below it."""
    import cv2
    w = img.shape[1]
    panel = np.zeros((line_h * max(1, len(lines)) + 6, w, 3), dtype=np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(panel, s, (4, line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


# ------------------------------------------------------------------ simulation helpers
def ground_truth_label(world, det, max_dist: float = 0.03) -> Optional[str]:
    """Return the label of the sim object nearest to det in base XY, or None (sim only)."""
    if world is None or det.center_base is None or not world.objects:
        return None
    c = np.asarray(det.center_base[:2], dtype=float)
    best = min(world.objects, key=lambda o: float(np.hypot(*(o.center_xy() - c))))
    if float(np.hypot(*(best.center_xy() - c))) > max_dist:
        return None
    return best.label


# ------------------------------------------------------------------ point coverage
def coverage_check(points: Any, polygon: Optional[Sequence[Sequence[float]]] = None,
                   min_points: int = 4, min_minor_rms: float = 0.015,
                   grid: int = 3) -> Dict[str, Any]:
    """
    Check that touched points are numerous and spread out (not on one line).

    points: Nx2 or Nx3 base-frame positions (m); only x, y are used.
    min_minor_rms: required RMS distance (m) of the points from their best-fit
    line. Returns {'n', 'ok', 'reason', 'major_rms', 'minor_rms', 'cells'
    (grid x grid counts over the polygon / point bounding box, row 0 = +y),
    'text'}.
    """
    P = np.asarray(points, dtype=float)
    xy = np.zeros((0, 2)) if P.size == 0 else P.reshape(-1, P.shape[-1])[:, :2]
    n = int(xy.shape[0])
    rep: Dict[str, Any] = {'n': n, 'ok': False, 'reason': '', 'major_rms': 0.0,
                           'minor_rms': 0.0, 'cells': [], 'text': ''}
    if n >= 2:
        q = xy - xy.mean(axis=0)
        s = np.linalg.svd(q, compute_uv=False)
        rep['major_rms'] = float(s[0] / math.sqrt(n))
        rep['minor_rms'] = float((s[1] if s.size > 1 else 0.0) / math.sqrt(n))
    if n < min_points:
        rep['reason'] = 'need at least %d points, have %d' % (min_points, n)
    elif rep['minor_rms'] < min_minor_rms:
        rep['reason'] = ('points lie almost on one line (%.1f mm off the line, need %.0f mm): '
                         'add points to the side' % (rep['minor_rms'] * 1e3, min_minor_rms * 1e3))
    else:
        rep['ok'] = True
    if polygon is not None and len(polygon) >= 3:
        box = np.asarray(polygon, dtype=float)
    elif n:
        box = xy
    else:
        box = np.zeros((1, 2))
    lo, hi = box.min(axis=0), box.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    cells = np.zeros((grid, grid), dtype=int)
    for x, y in xy:
        i = int(np.clip((x - lo[0]) / span[0] * grid, 0, grid - 1))
        j = int(np.clip((hi[1] - y) / span[1] * grid, 0, grid - 1))
        cells[j, i] += 1
    rep['cells'] = cells.tolist()
    rows = ['  ' + ''.join('[%s]' % (str(c) if c else ' ') for c in row) for row in cells]
    rep['text'] = ('coverage of the pick area (top row = +y side, columns = increasing x):\n'
                   + '\n'.join(rows))
    return rep

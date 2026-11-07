"""
Live RGB-D view and frame capture (datasets, KNN teaching, offline replay).

Shows color | depth colormap side by side (--overlay: the perception debug
image instead of the plain color image) and saves frames with
camera/recorder.save_frame. Each frame's meta 'extra' holds T_base_flange,
T_base_tcp and joints (when the arm is connected) and 'session' (the label
used by tools/yolo/make_dataset.py and teach_classes --from-dir).

Window keys: s / space save, a toggle auto-save every --interval s, q quit.
Headless: Enter refreshes the preview, s saves, 'a N' auto-saves N frames,
q quits. --count N: unattended capture of N frames every --interval s.
The arm is never moved in real mode, only read (drag-teach mode is fine).

Example commands
    capture --session apple_1
    capture --session apple_1 --count 20 --interval 1 --headless
    capture --mode sim --task L2 --count 5 --out /tmp/sim_capture
"""
from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_depth_range_arg, add_onsite_args, build_perception,
                     camera_pose_for_frame, compose_view, flange_to_tcp, go_observe,
                     load_extrinsics_or_none, make_ui, MotionGuard, read_pose)

__all__ = ['auto_due', 'build_extra', 'build_parser', 'compose_view', 'default_out_dir', 'main']

HEADLESS_HELP = '[Enter] refresh preview, s save, a N auto-save N frames, q quit'
WINDOW_HELP = 'keys in the window: s / space save, a toggle auto-save, q quit'


def default_out_dir(session: str, date: Optional[str] = None,
                    root: str = '~/vision_data') -> Path:
    """Return <root>/<YYYYmmdd>_<session> (session made file-name safe)."""
    from ..core.logger import safe_filename
    date = date or datetime.date.today().strftime('%Y%m%d')
    return Path(root).expanduser() / ('%s_%s' % (date, safe_filename(session, 'capture')))


def build_extra(pose, session: str, mount: str, task: Optional[str]) -> Dict[str, Any]:
    """Return the 'extra' metadata saved with a frame (pose may be None)."""
    extra: Dict[str, Any] = {'session': str(session), 'mount': str(mount), 'task': task}
    if pose is not None:
        extra.update(pose.to_extra())
    return extra


def auto_due(now: float, last_save: float, interval: float) -> bool:
    """Return True when an auto-save is due (interval seconds of clock time passed)."""
    return now - last_save >= max(0.0, float(interval)) - 1e-9


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='capture', description=__doc__.split('\n\n')[0])
    add_common_args(p)
    add_onsite_args(p, site=False)
    add_depth_range_arg(p)
    p.add_argument('--out', default=None, help='output directory (default '
                   '~/vision_data/<date>_<session>; frames are appended)')
    p.add_argument('--session', default='capture',
                   help='session label stored with every frame (class name for one-class '
                        'sessions)')
    p.add_argument('--count', type=int, default=0, help='unattended: save N frames and exit')
    p.add_argument('--interval', type=float, default=1.0, help='auto-save period (s)')
    p.add_argument('--overlay', action='store_true', help='show the perception overlay')
    p.add_argument('--average', type=int, default=1,
                   help='save the median depth of N frames (static scenes only)')
    p.add_argument('--npy', action='store_true', help='also save float32 depth as .npy')
    p.add_argument('--no-arm', action='store_true', help='do not connect the arm (no poses)')
    return p


class _Capture:
    """State of one capture run (thin I/O around the pure helpers above)."""

    def __init__(self, args, s, ui, out: Path):
        from ..camera.recorder import next_index
        self.args, self.s, self.ui, self.out = args, s, ui, out
        self.idx = next_index(out)
        self.saved = 0
        self.last_save = -math.inf
        self.last_stamp: Optional[float] = None
        self.mount = str(s.cfg.get('camera.mount', 'eye_to_hand'))
        self.ext = None
        self.perception = None
        self.T_ft = flange_to_tcp(s.cfg)
        if args.overlay:
            self.ext = load_extrinsics_or_none(s.cfg, ui)
            self.perception = build_perception(s.cfg, s.task, args, ui)

    def grab(self):
        from ..camera.utils import capture_averaged
        timeout = float(self.s.cfg.get('camera.frame_timeout_s', 2.0))
        f = capture_averaged(self.s.camera, max(1, int(self.args.average)),
                             newer_than=self.last_stamp, timeout=timeout)
        if f is not None:
            self.last_stamp = float(f.stamp)
        return f

    def view(self, frame) -> np.ndarray:
        overlay = None
        if self.perception is not None:
            T_bc, why = camera_pose_for_frame(self.ext, frame, self.s.arm, self.T_ft)
            res = self.perception.process(frame, T_bc)
            overlay = res.debug_image
        return compose_view(frame, overlay, self.args.depth_range)

    def save(self, frame) -> None:
        from ..camera.recorder import save_frame
        pose = read_pose(self.s.arm) if self.s.arm is not None else None
        extra = build_extra(pose, self.args.session, self.mount, self.s.task)
        path = save_frame(self.out, self.idx, frame, extra=extra, save_depth_npy=self.args.npy)
        self.ui.say('saved frame %d -> %s%s' % (self.idx, path.parent,
                                                '' if pose else ' (no arm pose)'))
        self.idx += 1
        self.saved += 1
        self.last_save = self.s.clock.now()


def _connect_arm_readonly(args, s, ui) -> bool:
    """Real mode: connect without enabling (poses only). Returns False to abort."""
    if s.arm is None or args.mode != 'real':
        return True
    from ..core.interfaces import ArmError
    try:
        s.arm.connect()
        return True
    except ArmError as e:
        if str(s.cfg.get('camera.mount', '')) == 'eye_in_hand':
            ui.say('eye_in_hand frames need the arm pose, but the arm did not connect: %s\n'
                   'Fix the CAN link or pass --no-arm to capture without poses.' % e)
            return False
        ui.say('arm not connected (%s); frames are saved without arm poses' % e)
        s.arm = None
        return True


def main(argv=None, ui=None) -> int:
    """Run the capture tool; return the process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    cfg = cfg_world_info[0]
    ui = make_ui(args, cfg, 'capture', ui)
    out = Path(args.out).expanduser() if args.out else default_out_dir(args.session)
    use_arm = args.mode == 'sim' or (args.mode == 'real' and not args.no_arm)
    with open_session(args, arm=use_arm, camera=True, perception=False,
                      connect=args.mode != 'real', cfg_world_info=cfg_world_info) as s:
        if not _connect_arm_readonly(args, s, ui):
            return 2
        if args.mode == 'sim':
            ok, why = go_observe(s.arm, cfg, MotionGuard(ui, 'sim', 100))
            if not ok:
                ui.say('sim: could not reach the observe pose: %s' % why)
        cap = _Capture(args, s, ui, out)
        auto = args.count > 0
        target = args.count if auto else None
        ui.say('capture: session %r -> %s (next index %d)' % (args.session, out, cap.idx))
        ui.say(HEADLESS_HELP if ui.headless and not auto else
               ('unattended: %d frames every %.1f s' % (args.count, args.interval) if auto
                else WINDOW_HELP))
        if ui.headless:
            ui.say('preview image: %s' % ui.preview_path('capture'))
        misses = 0
        while True:
            if auto and ui.headless:
                wait = cap.last_save + args.interval - s.clock.now()
                if math.isfinite(wait) and wait > 0:
                    s.clock.sleep(wait)
            frame = cap.grab()
            if frame is None:
                misses += 1
                ui.say('no camera frame (%d)' % misses)
                if misses >= 5 or ui.eof:
                    break
                continue
            misses = 0
            ui.show('capture', cap.view(frame))
            save_now = auto and auto_due(s.clock.now(), cap.last_save, args.interval)
            if ui.headless:
                if not auto:
                    cmd = ui.prompt(HEADLESS_HELP).strip().lower()
                    if cmd.startswith('q') or ui.eof:
                        break
                    if cmd.startswith('a'):
                        parts = cmd.split()
                        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                        auto, target = True, cap.saved + n
                    save_now = save_now or cmd.startswith('s')
            else:
                k = ui.key(30)
                if k in ('q', 'esc'):
                    break
                if k == 'a':
                    auto = not auto
                    ui.say('auto-save %s' % ('on' if auto else 'off'))
                save_now = save_now or k in ('s', ' ')
            if save_now:
                cap.save(frame)
            if target is not None and cap.saved >= target:
                if args.count > 0:
                    break
                auto, target = False, None
        ui.say('captured %d frames in %s' % (cap.saved, out))
        if cap.saved:
            ui.say('replay: view --mode replay --replay-dir %s' % out)
    ui.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""
Live perception debug view: detections, table plane, timings, FPS.

The left image is the perception overlay (instance masks, labels and scores,
reject reasons in gray, grasp axis, pick polygon and zones projected from the
base frame), the right image the depth colormap. A text panel below shows the
table plane (z at the pick-area center, tilt, rms, inlier ratio), per-stage
timings and FPS. --task selects the classifier (tasks.<T>.classifier).

Window keys: d detection table, s save frame + overlay, q quit.
Headless: Enter next frame, d table, s save, q quit; the preview image is
written to --preview-dir. --count N: process N frames, print a line per frame
and exit (useful with --mode replay on captured data).

Example commands
    view --task L1
    view --mode replay --replay-dir ~/vision_data/20261112_blocks --task L1 --headless
    view --mode sim --task L2 --count 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_depth_range_arg, add_onsite_args, build_perception,
                     camera_pose_for_frame, capture_view, compose_view, connect_arm_readonly,
                     detection_table, flange_to_tcp, go_observe, load_extrinsics_or_none,
                     logging_dir, make_ui, MotionGuard, next_command, read_pose, text_panel,
                     wall_stamp)

HEADLESS_HELP = '[Enter] next frame, d detection table, s save, q quit'
WINDOW_HELP = 'keys in the window: d detection table, s save, q quit'


# ------------------------------------------------------------------ pure helpers
def polygon_center(polygon: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """Return the vertex mean of a base-XY polygon, or None."""
    if polygon is None or len(polygon) < 3:
        return None
    return np.asarray(polygon, dtype=float).mean(axis=0)


def table_status(res, polygon: Optional[Sequence[Sequence[float]]] = None) -> str:
    """Return a one-line table plane summary (z mm, tilt deg, rms mm, inlier %)."""
    t = getattr(res, 'table', None)
    if t is None:
        why = [w for w in getattr(res, 'warnings', []) if w.startswith('table:')]
        return 'table: NOT FOUND%s' % (' (%s)' % why[0][6:] if why else '')
    n = np.asarray(t.normal, dtype=float)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    c = polygon_center(polygon)
    try:
        z = t.z_at(float(c[0]), float(c[1])) if c is not None else t.z_at(0.3, 0.0)
        ztxt = '%.1f mm' % (z * 1000.0)
    except ValueError:
        ztxt = 'n/a'
    return ('table: z %s at the pick-area center, tilt %.2f deg, rms %.1f mm, inliers %.0f %%'
            % (ztxt, tilt, t.rms * 1000.0, t.inlier_ratio * 100.0))


TIMING_ORDER = ('backproject', 'table', 'segment', 'color_split', 'detector', 'classify',
                'merged_blocks', 'draw', 'total')


def timing_line(timings: Dict[str, float]) -> str:
    """Return 'stage ms | ...' in pipeline order (unknown stages last)."""
    keys = [k for k in TIMING_ORDER if k in timings]
    keys += sorted(k for k in timings if k not in TIMING_ORDER)
    return 'ms: ' + ' | '.join('%s %.0f' % (k, float(timings[k])) for k in keys)


def counts_line(res) -> str:
    """Return 'N valid / M instances' plus reject reasons and labels."""
    dets = list(getattr(res, 'detections', []))
    valid = [d for d in dets if d.valid]
    reasons: Dict[str, int] = {}
    for d in dets:
        if not d.valid:
            reasons[d.reject_reason or 'invalid'] = reasons.get(d.reject_reason or 'invalid',
                                                                0) + 1
    labels: Dict[str, int] = {}
    for d in valid:
        labels[d.label] = labels.get(d.label, 0) + 1
    s = '%d valid / %d instances' % (len(valid), len(dets))
    if labels:
        s += ' | ' + ', '.join('%s x%d' % (k, v) for k, v in sorted(labels.items()))
    if reasons:
        s += ' | rejected: ' + ', '.join('%s x%d' % (k, v) for k, v in sorted(reasons.items()))
    return s


def status_lines(res, polygon=None, fps: Optional[float] = None,
                 note: str = '') -> List[str]:
    """Return the text panel lines shown under the view."""
    lines = [counts_line(res), table_status(res, polygon), timing_line(res.timings)
             + ('' if fps is None else ' | %.1f FPS' % fps)]
    if res.warnings:
        lines.append('warnings: ' + ', '.join(res.warnings))
    if note:
        lines.append(note)
    return lines


class FpsMeter:
    """Exponential moving average of the frame rate (wall time, display only)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.last: Optional[float] = None
        self.fps: Optional[float] = None

    def tick(self, t: float) -> Optional[float]:
        """Register a frame at wall time t (s); return the smoothed FPS (None at first)."""
        if self.last is not None and t > self.last:
            inst = 1.0 / (t - self.last)
            self.fps = inst if self.fps is None else (
                self.alpha * inst + (1.0 - self.alpha) * self.fps)
        self.last = t
        return self.fps


def frame_report(res) -> Dict[str, Any]:
    """Return the JSON saved next to a saved frame (detections, table, warnings, timings)."""
    t = res.table
    table = None if t is None else {'normal': [float(v) for v in t.normal], 'd': float(t.d),
                                    'rms': float(t.rms), 'inlier_ratio': float(t.inlier_ratio)}
    return {'detections': [d.summary() for d in res.detections], 'table': table,
            'warnings': list(res.warnings), 'timings': dict(res.timings)}


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='view', description=__doc__.split('\n\n')[0])
    add_common_args(p)
    add_onsite_args(p, site=False)
    add_depth_range_arg(p)
    p.add_argument('--out', default=None,
                   help='where s saves frames (default <logging.dir>/view/<time>)')
    p.add_argument('--count', type=int, default=0, help='process N frames, then exit')
    p.add_argument('--average', type=int, default=1,
                   help='median depth of N frames per view (static scenes)')
    p.add_argument('--no-arm', action='store_true',
                   help='real mode: do not connect the arm (eye_to_hand only)')
    return p


class _Viewer:
    """State of one view run (thin I/O around the pure helpers)."""

    def __init__(self, args, s, ui):
        self.args, self.s, self.ui = args, s, ui
        self.cfg = s.cfg
        self.ext = load_extrinsics_or_none(s.cfg, ui)
        self.perception = build_perception(s.cfg, s.task, args, ui)
        self.T_ft = flange_to_tcp(s.cfg)
        self.polygon = s.cfg.get('perception.roi.pick_polygon', None)
        self.fps = FpsMeter()
        self.last_stamp: Optional[float] = None
        self.out: Optional[Path] = Path(args.out).expanduser() if args.out else None
        self.saved = 0

    def step(self):
        """Grab, process and show one frame; return (frame, result) or None."""
        frame = capture_view(self.s.camera, self.cfg, self.args.average, self.last_stamp)
        if frame is None:
            return None
        self.last_stamp = float(frame.stamp)
        T_bc, why = camera_pose_for_frame(self.ext, frame, self.s.arm, self.T_ft)
        res = self.perception.process(frame, T_bc)
        fps = self.fps.tick(time.perf_counter())
        note = '' if T_bc is not None else 'no camera pose: %s' % why
        img = compose_view(frame, res.debug_image, self.args.depth_range)
        self.ui.show('view', text_panel(img, status_lines(res, self.polygon, fps, note)))
        return frame, res

    def save(self, frame, res) -> Path:
        import cv2
        from ..camera.recorder import next_index, save_frame
        if self.out is None:
            self.out = logging_dir(self.cfg) / 'view' / wall_stamp()
        idx = next_index(self.out)
        pose = read_pose(self.s.arm) if self.s.arm is not None else None
        extra: Dict[str, Any] = {'session': 'view', 'task': self.s.task,
                                 'mount': str(self.cfg.get('camera.mount', 'eye_to_hand'))}
        if pose is not None:
            extra.update(pose.to_extra())
        save_frame(self.out, idx, frame, extra=extra)
        if res.debug_image is not None:
            cv2.imwrite(str(self.out / ('%06d_overlay.jpg' % idx)), res.debug_image)
        (self.out / ('%06d_detections.json' % idx)).write_text(
            json.dumps(frame_report(res), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        self.saved += 1
        return self.out / ('%06d_meta.json' % idx)


def main(argv=None, ui=None) -> int:
    """Run the live view; return the process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    cfg = cfg_world_info[0]
    ui = make_ui(args, cfg, 'view', ui)
    mount = str(cfg.get('camera.mount', 'eye_to_hand'))
    use_arm = args.mode != 'real' or not args.no_arm
    with open_session(args, arm=use_arm, camera=True, perception=False,
                      connect=args.mode != 'real', cfg_world_info=cfg_world_info) as s:
        if not connect_arm_readonly(s, ui, required=mount == 'eye_in_hand',
                                    why='eye_in_hand camera poses'):
            return 2
        if args.mode == 'sim':
            ok, why = go_observe(s.arm, cfg, MotionGuard(ui, 'sim', 100))
            if not ok:
                ui.say('sim: could not reach the observe pose: %s' % why)
        v = _Viewer(args, s, ui)
        ui.say('view: task %s, mode %s, classifier %s' % (
            s.task, args.mode, type(v.perception.classifier).__name__
            if v.perception.classifier is not None else 'none (unlabeled instances)'))
        if ui.headless:
            ui.say('preview image: %s' % ui.preview_path('view'))
        if args.count <= 0:
            ui.say(HEADLESS_HELP if ui.headless else WINDOW_HELP)
        n, misses = 0, 0
        while True:
            got = v.step()
            if got is None:
                misses += 1
                ui.say('no camera frame (%d)' % misses)
                if misses >= 5 or ui.eof:
                    break
                continue
            misses = 0
            frame, res = got
            n += 1
            if args.count > 0:
                ui.say('frame %d: %s' % (n, counts_line(res)))
                if n >= args.count:
                    ui.say(detection_table(res.detections))
                    break
                continue
            cmd = next_command(ui, HEADLESS_HELP)
            if cmd is None:
                continue
            c = cmd.lower()
            if c.startswith('q'):
                break
            if c.startswith('d'):
                ui.say(detection_table(res.detections))
                ui.say(table_status(res, v.polygon))
                if res.warnings:
                    ui.say('warnings: %s' % ', '.join(res.warnings))
            elif c.startswith('s'):
                ui.say('saved -> %s' % v.save(frame, res))
        if v.saved:
            ui.say('saved %d frames in %s (replay: view --mode replay --replay-dir %s)'
                   % (v.saved, v.out, v.out))
    ui.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

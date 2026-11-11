"""
Command line tools: ``python3 -m piper_vision <command> ...``.

Commands:
    board      write a printable ChArUco board image
    handeye    hand-eye calibration from capture frames or live (camera + arm)
    l1         run L1 detection on a capture directory, an image pair or live
    record     record the ROS camera as training frames or a rosbag2 bag
    transform  convert a pixel (+ depth) or a camera-frame point to the base frame
    show       print the camera config and the resulting T_base_camera
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraConfig, load_camera_config, points_camera_to_base, write_extrinsics
from .frames import list_frames, load_frame, load_image_pair
from .handeye import (BoardSpec, HandEyeCalibrator, METHODS, coverage_warning, detect_board,
                      draw_detection, render_board)
from .l1 import L1Detector, L1Params, draw_result
from .transforms import T_to_pose6

LOG = logging.getLogger('piper_vision.cli')
PKG_ROOT = Path(__file__).resolve().parent.parent


def _default(name: str) -> str:
    return str(PKG_ROOT / 'config' / name)


def _load_cam(path: str, need_ext: bool = True) -> CameraConfig:
    if not os.path.exists(path):
        raise SystemExit('%s not found; copy config/camera.example.yaml to it and fill in '
                         'the site values (or pass --camera)' % path)
    cfg = load_camera_config(path)
    if need_ext and cfg.extrinsics is None:
        raise SystemExit('%s has no extrinsics section' % path)
    return cfg


def _load_l1(path: str) -> L1Params:
    import yaml
    with open(path, encoding='utf-8') as f:
        return L1Params.from_dict(yaml.safe_load(f) or {})


# ---------------------------------------------------------------------------

def cmd_board(args) -> int:
    """Write the board image."""
    spec = BoardSpec.from_dict(load_camera_config(args.camera).board) if args.camera \
        else BoardSpec()
    img = render_board(spec, dpi=args.dpi)
    cv2.imwrite(args.out, img)
    print('wrote %s (%dx%d px). Print at %d dpi, 100%% scale; one square should measure '
          '%.1f mm.' % (args.out, img.shape[1], img.shape[0], args.dpi, spec.square_m * 1000))
    return 0


def cmd_handeye(args) -> int:
    """Run hand-eye calibration and optionally write the result."""
    cam = _load_cam(args.camera, need_ext=False)
    mount = args.mount or cam.mount
    spec = BoardSpec.from_dict(cam.board)
    cal = HandEyeCalibrator(mount, method=args.method)
    if args.frames:
        _collect_offline(args, cam, spec, cal)
    else:
        _collect_live(args, cam, spec, cal)
    warn = coverage_warning(cal.samples)
    if warn:
        print('WARNING: ' + warn)
    try:
        res = cal.solve()
    except ValueError as exc:
        print('cannot solve: %s' % exc)
        return 1
    print('\n%d samples, method %s' % (res.n_samples, res.method))
    print(res.extrinsics.describe())
    print('board position spread: rms %.1f mm, max %.1f mm'
          % (res.rms_m * 1000, res.max_m * 1000))
    for i, e in enumerate(res.per_sample_m):
        print('  sample %2d: %.1f mm' % (i, e * 1000))
    others = {}
    for m in sorted(METHODS):
        c = HandEyeCalibrator(mount, m)
        c.samples = cal.samples
        try:
            r = c.solve()
        except Exception as exc:  # cv2 errors on degenerate data
            LOG.debug('method %s failed: %s', m, exc)
            continue
        others[m] = r.rms_m
    print('rms by method: ' + ', '.join('%s %.1f mm' % (k, v * 1000) for k, v in others.items()))
    if res.rms_m > args.max_rms:
        print('rms above --max-rms %.1f mm; not writing' % (args.max_rms * 1000))
        return 1
    if args.write:
        path = write_extrinsics(args.camera, res.extrinsics, mount)
        print('wrote extrinsics to %s' % path)
    else:
        print('dry run; add --write to save into %s' % args.camera)
    return 0


def _collect_offline(args, cam, spec, cal) -> None:
    for idx in list_frames(args.frames):
        fr = load_frame(args.frames, idx)
        intr = cam.intrinsics or fr.intrinsics
        if intr is None or fr.T_base_flange is None:
            print('%s: skipped (no intrinsics or no T_base_flange in meta)' % fr.name)
            continue
        det = detect_board(fr.color, intr, spec)
        if det is None:
            print('%s: board not found' % fr.name)
            continue
        T_cb, n, err = det
        if err > args.max_reproj:
            print('%s: skipped, reprojection %.2f px' % (fr.name, err))
            continue
        cal.add(fr.T_base_flange, T_cb, n, err)
        print('%s: %d corners, reprojection %.2f px, board at %.3f m'
              % (fr.name, n, err, np.linalg.norm(T_cb[:3, 3])))


def _collect_live(args, cam, spec, cal) -> None:
    from .live import ArmPoseReader, RosRgbd
    cap = RosRgbd(cam)
    arm = ArmPoseReader(channel=args.can)
    print('Move the arm by hand (drag teach) or with your own tool; this tool only reads '
          'the pose.\nEnter = capture, u = undo last, q = solve.')
    try:
        while True:
            key = input('[%d samples] > ' % len(cal.samples)).strip().lower()
            if key == 'q':
                break
            if key == 'u':
                if cal.samples:
                    cal.samples.pop()
                continue
            color, _, intr = cap.get()
            det = detect_board(color, intr, spec)
            if det is None:
                print('board not found')
                continue
            T_cb, n, err = det
            T_bf = arm.flange_averaged()
            cal.add(T_bf, T_cb, n, err)
            print('%d corners, reprojection %.2f px' % (n, err))
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                stem = Path(args.save_dir) / ('%03d' % len(cal.samples))
                cv2.imwrite(str(stem) + '.png', draw_detection(color, T_cb, intr))
                Path(str(stem) + '.json').write_text(json.dumps(
                    {'T_base_flange': T_bf.tolist(), 'T_camera_board': T_cb.tolist()}))
    finally:
        arm.close()
        cap.close()


# ---------------------------------------------------------------------------

def cmd_l1(args) -> int:
    """Run L1 detection and print the cubes and cards."""
    cam = _load_cam(args.camera)
    det = L1Detector(_load_l1(args.params))
    frames = []
    if args.frames:
        frames = [load_frame(args.frames, i) for i in list_frames(args.frames)]
    elif args.color:
        frames = [load_image_pair(args.color, args.depth, cam.depth_scale)]
    else:
        from .live import RosRgbd
        cap = RosRgbd(cam)
        try:
            color, depth, intr = cap.get()
        finally:
            cap.close()
        from .frames import RgbdFrame
        frames = [RgbdFrame(color, depth, intr, name='live')]
    out_all = []
    for fr in frames:
        intr = cam.intrinsics or fr.intrinsics
        if intr is None:
            print('%s: no intrinsics (put them in %s)' % (fr.name, args.camera))
            continue
        if cam.mount == 'eye_in_hand' and fr.T_base_flange is None and args.flange is None:
            print('%s: wrist camera needs T_base_flange (frame meta or --flange pose6)'
                  % fr.name)
            continue
        T_bf = fr.T_base_flange
        if T_bf is None and args.flange is not None:
            from .transforms import pose6_to_T
            T_bf = pose6_to_T(args.flange)
        res = det.detect(fr.color, fr.depth, intr, cam.T_base_camera(T_bf))
        print('%s: table z %s, %d cube(s), %d card(s)'
              % (fr.name, 'none' if res.table_z is None else '%.4f' % res.table_z,
                 len(res.cubes), len(res.zones)))
        for w in res.warnings:
            print('  warning: ' + w)
        for c in res.cubes:
            print('  cube %-6s center [%.4f %.4f %.4f] yaw %5.1f deg edge %.1f mm '
                  'height %.1f mm score %.2f %s'
                  % (c.label, c.center_base[0], c.center_base[1], c.center_base[2],
                     math.degrees(c.yaw), c.edge * 1000, c.height * 1000, c.score,
                     '; '.join(c.warnings)))
        for z in res.zones:
            print('  card %-11s center [%.4f %.4f] yaw %5.1f deg size %.0fx%.0f mm'
                  % (z.name, z.center_base[0], z.center_base[1], math.degrees(z.yaw),
                     z.size[0] * 1000, z.size[1] * 1000))
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.out) / ('%s_l1.png' % fr.name)), draw_result(fr.color, res))
        out_all.append({'frame': fr.name, 'table_z': res.table_z,
                        'cubes': [c.summary() for c in res.cubes],
                        'zones': [z.summary() for z in res.zones], 'warnings': res.warnings})
    if args.json:
        Path(args.json).write_text(json.dumps(out_all, indent=2, ensure_ascii=False))
    return 0 if out_all else 1


def cmd_transform(args) -> int:
    """Convert a pixel + depth or a camera point into the base frame."""
    cam = _load_cam(args.camera)
    T_bf = None
    if args.flange is not None:
        from .transforms import pose6_to_T
        T_bf = pose6_to_T(args.flange)
    T_bc = cam.T_base_camera(T_bf)
    if args.pixel is not None:
        if cam.intrinsics is None:
            raise SystemExit('no intrinsics in %s' % args.camera)
        u, v, d = args.pixel
        p_cam = cam.intrinsics.backproject(np.array([[u, v]]), np.array([d]))[0]
    else:
        p_cam = np.asarray(args.point, dtype=float)
    p_base = points_camera_to_base(T_bc, p_cam)
    print('camera [%.4f %.4f %.4f] -> base [%.4f %.4f %.4f]' % (tuple(p_cam) + tuple(p_base)))
    return 0


def cmd_record(args) -> int:
    """Record the ROS camera as frames (training set) or a rosbag2 bag."""
    from .record import GENERIC_SESSIONS, record_bag, record_frames, safe_name, session_dir
    cam = _load_cam(args.camera, need_ext=False)
    if args.bag:
        out = Path(args.out).expanduser() if args.out else session_dir(args.session + '_bag')
        if out.exists():
            print('%s already exists; pick another --out' % out)
            return 1
        n = record_bag(cam, out, duration=args.duration)
        return 0 if n else 1
    session = safe_name(args.session, '')
    if session.lower() in GENERIC_SESSIONS:
        print('session name %r cannot be used as a class label; pass e.g. --session banana'
              % args.session)
        return 1
    out = Path(args.out).expanduser() if args.out else session_dir(session)
    n = record_frames(cam, out, session, count=args.count, interval=args.interval,
                      headless=args.headless or args.count > 0, with_arm=args.arm, can=args.can)
    print('saved %d frame(s) in %s' % (n, out))
    return 0


def cmd_show(args) -> int:
    """Print the camera config."""
    cam = _load_cam(args.camera, need_ext=False)
    print('mount: %s' % cam.mount)
    print('intrinsics: %s' % (None if cam.intrinsics is None else cam.intrinsics.to_dict()))
    if cam.extrinsics is not None:
        print(cam.extrinsics.describe())
        if cam.mount == 'eye_to_hand':
            p = T_to_pose6(cam.T_base_camera())
            print('T_base_camera pose6: [%s]' % ', '.join('%.4f' % v for v in p))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    ap = argparse.ArgumentParser(prog='piper_vision', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--verbose', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)
    cam_help = 'camera config YAML (default config/camera.yaml)'

    p = sub.add_parser('board', help='write a printable ChArUco board')
    p.add_argument('--camera', default=None, help='take the board spec from this config')
    p.add_argument('--out', default='charuco_board.png')
    p.add_argument('--dpi', type=int, default=200)
    p.set_defaults(fn=cmd_board)

    p = sub.add_parser('handeye', help='hand-eye calibration')
    p.add_argument('--camera', default=_default('camera.yaml'), help=cam_help)
    p.add_argument('--mount', choices=['eye_to_hand', 'eye_in_hand'], default=None)
    p.add_argument('--frames', default=None,
                   help='cube_task capture directory (meta extra must hold T_base_flange); '
                        'omit for live ROS camera + pyAgxArm')
    p.add_argument('--method', choices=sorted(METHODS), default='park')
    p.add_argument('--max-reproj', type=float, default=1.5, help='px, drop worse views')
    p.add_argument('--max-rms', type=float, default=0.005, help='m, refuse to write above')
    p.add_argument('--can', default='can0')
    p.add_argument('--save-dir', default=None, help='live: save each view here')
    p.add_argument('--write', action='store_true', help='write the result into --camera')
    p.set_defaults(fn=cmd_handeye)

    p = sub.add_parser('l1', help='L1 cube / card detection')
    p.add_argument('--camera', default=_default('camera.yaml'), help=cam_help)
    p.add_argument('--params', default=_default('l1.yaml'))
    p.add_argument('--frames', default=None, help='cube_task capture directory')
    p.add_argument('--color', default=None, help='color image')
    p.add_argument('--depth', default=None, help='depth image (.png in depth units / .npy m)')
    p.add_argument('--flange', type=float, nargs=6, default=None,
                   help='wrist camera: flange pose6 at capture (x y z roll pitch yaw)')
    p.add_argument('--out', default=None, help='write overlay images here')
    p.add_argument('--json', default=None, help='write results as JSON')
    p.set_defaults(fn=cmd_l1)

    p = sub.add_parser('record', help='record the ROS camera as frames or a bag')
    p.add_argument('--camera', default=_default('camera.yaml'), help=cam_help)
    p.add_argument('--session', default='capture',
                   help='class label of this session, one class per directory (e.g. banana)')
    p.add_argument('--out', default=None,
                   help='output directory (default ~/vision_data/<date>_<session>)')
    p.add_argument('--count', type=int, default=0, help='save this many frames, then exit')
    p.add_argument('--interval', type=float, default=1.0, help='seconds between auto-saves')
    p.add_argument('--headless', action='store_true', help='no preview window')
    p.add_argument('--arm', action='store_true',
                   help='also read the flange pose over CAN (read only, no motion)')
    p.add_argument('--can', default='can0')
    p.add_argument('--bag', action='store_true',
                   help='record a rosbag2 of the camera topics instead of frames')
    p.add_argument('--duration', type=float, default=0.0,
                   help='bag: seconds to record (0 = until Ctrl-C)')
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser('transform', help='camera point / pixel -> base')
    p.add_argument('--camera', default=_default('camera.yaml'), help=cam_help)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pixel', type=float, nargs=3, metavar=('U', 'V', 'DEPTH_M'))
    g.add_argument('--point', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--flange', type=float, nargs=6, default=None,
                   help='wrist camera: flange pose6 at capture')
    p.set_defaults(fn=cmd_transform)

    p = sub.add_parser('show', help='print the camera config')
    p.add_argument('--camera', default=_default('camera.yaml'), help=cam_help)
    p.set_defaults(fn=cmd_show)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    try:
        return int(args.fn(args) or 0)
    except ValueError as exc:
        print('error: %s' % exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

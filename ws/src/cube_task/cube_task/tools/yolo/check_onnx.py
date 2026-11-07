"""
Check an ONNX detector with the runtime stack (system python3, OpenCV 4.5.4).

Loads the model exactly like the robot does (perception/detector_onnx.py),
prints the environment, model facts, detections and latency. Run it with the
system Python, not the training venv:

    python3 -m cube_task.tools.yolo.check_onnx --model models/detector.onnx \
        --labels models/detector_labels.txt --image some.jpg --repeat 20
    python3 -m cube_task.tools.yolo.check_onnx --model m.onnx --write-labels m_labels.txt

--json prints one JSON object (used by export_onnx.py --verify). Exit code 0
when the model loads and runs, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _env() -> Dict[str, Any]:
    import cv2
    import numpy as np
    env: Dict[str, Any] = {'python': sys.version.split()[0], 'executable': sys.executable,
                           'cv2': cv2.__version__, 'numpy': np.__version__}
    try:
        import onnxruntime as ort
        env['onnxruntime'] = ort.__version__
    except ImportError:
        env['onnxruntime'] = None
    info = cv2.getBuildInformation()
    env['cv2_cuda'] = any('NVIDIA CUDA' in ln and 'YES' in ln for ln in info.splitlines())
    try:
        env['cuda_devices'] = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # cv2.cuda missing or OpenCV built without CUDA
        env['cuda_devices'] = 0
    return env


def _synthetic_image(seed: int = 0):
    """640x480 BGR test image: table color with a few colored blobs."""
    import cv2
    import numpy as np
    rng = np.random.RandomState(seed)
    img = np.full((480, 640, 3), (170, 180, 190), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.randint(0, 255, 3))
        center = (int(rng.randint(60, 580)), int(rng.randint(60, 420)))
        cv2.circle(img, center, int(rng.randint(20, 50)), c, -1)
    return img


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the checks; return the JSON-friendly report."""
    import cv2
    import numpy as np
    from cube_task.perception.detector_onnx import OnnxDetector, read_onnx_info

    report: Dict[str, Any] = {'ok': False, 'env': _env(), 'model': str(args.model)}
    try:
        info = read_onnx_info(Path(args.model))
        report['onnx'] = {'ir_version': info.ir_version, 'opsets': info.opsets,
                          'input': [info.input_name, info.input_shape],
                          'outputs': info.output_names,
                          'task': info.metadata.get('task'),
                          'imgsz': info.metadata.get('imgsz'),
                          'ultralytics': info.metadata.get('version'),
                          'names': info.names(),
                          'const_first_ops': info.const_first_ops}
    except (OSError, ValueError) as e:
        report['error'] = 'cannot read %s: %s' % (args.model, e)
        return report
    if args.write_labels:
        names = info.names()
        if not names:
            report['error'] = 'the model has no embedded class names; write the labels by hand'
            return report
        out = Path(args.write_labels)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(names) + '\n', encoding='utf-8')
        report['labels_written'] = str(out)
        if not args.labels:
            args.labels = str(out)
    if not args.labels:
        report['error'] = '--labels is required (or --write-labels)'
        return report
    opts: Dict[str, Any] = {'backend': args.backend, 'conf_threshold': args.conf,
                            'nms_threshold': args.nms, 'input_size': args.input_size}
    try:
        det = OnnxDetector.from_paths(args.model, args.labels, **opts)
    except Exception as e:  # report every failure as text, this is a checker
        report['error'] = '%s: %s' % (type(e).__name__, e)
        return report
    report.update({'backend': det.backend, 'input_size': det.input_size,
                   'labels': det.labels, 'warmup_ms': round(det.warmup_ms, 1)})
    images: List[Any] = []
    if args.image:
        for p in args.image:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                report['error'] = 'cannot read image %s' % p
                return report
            images.append((str(p), img))
    else:
        images.append(('synthetic', _synthetic_image()))
    results = []
    lat: List[float] = []
    for name, img in images:
        try:
            dets = det.detect(img)
            lat.append(det.last_latency_ms)
            for _ in range(max(0, args.repeat - 1)):
                det.detect(img)
                lat.append(det.last_latency_ms)
        except Exception as e:  # report every failure as text, this is a checker
            report['error'] = 'detect(%s) failed: %s: %s' % (name, type(e).__name__, e)
            return report
        results.append({'image': name, 'shape': list(img.shape),
                        'detections': [{'box': list(b), 'label': lab, 'prob': round(p, 4)}
                                       for b, lab, p in dets],
                        'timings_ms': {k: round(v, 2) for k, v in det.last_timings_ms.items()}})
        if args.save_vis:
            out_dir = Path(args.save_vis)
            out_dir.mkdir(parents=True, exist_ok=True)
            vis = img.copy()
            for (x0, y0, x1, y1), lab, p in dets:
                cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
                cv2.putText(vis, '%s %.2f' % (lab, p), (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / (Path(name).stem + '_det.jpg')), vis)
    a = np.asarray(lat, dtype=float)
    report['latency_ms'] = {'n': int(a.size), 'median': round(float(np.median(a)), 2),
                            'p90': round(float(np.percentile(a, 90)), 2),
                            'min': round(float(a.min()), 2)}
    report['results'] = results
    report['ok'] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='ONNX file')
    p.add_argument('--labels', default=None, help='labels file (one class name per line)')
    p.add_argument('--write-labels', default=None, metavar='PATH',
                   help='write the class names embedded in the model to PATH first')
    p.add_argument('--image', nargs='*', default=None, help='images to run (default synthetic)')
    p.add_argument('--backend', default='opencv', choices=['auto', 'opencv', 'onnxruntime'])
    p.add_argument('--conf', type=float, default=0.35, help='conf_threshold')
    p.add_argument('--nms', type=float, default=0.45, help='nms_threshold')
    p.add_argument('--input-size', type=int, default=None,
                   help='input_size (default: from the model)')
    p.add_argument('--repeat', type=int, default=1, help='runs per image for latency')
    p.add_argument('--save-vis', default=None, metavar='DIR', help='write *_det.jpg overlays')
    p.add_argument('--json', action='store_true', help='print one JSON object')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    report = run(args)
    report['total_s'] = round(time.perf_counter() - t0, 2)
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report['ok'] else 1
    env = report['env']
    print('python %s (%s), cv2 %s (CUDA DNN: %s), numpy %s, onnxruntime %s'
          % (env['python'], env['executable'], env['cv2'], env['cv2_cuda'], env['numpy'],
             env['onnxruntime']))
    onnx = report.get('onnx')
    if onnx:
        print('model %s: ir %s opsets %s input %s outputs %s task %s, %d names, '
              'const-first ops %s' % (report['model'], onnx['ir_version'], onnx['opsets'],
                                      onnx['input'], onnx['outputs'], onnx['task'],
                                      len(onnx['names'] or []), onnx['const_first_ops']))
    if report.get('labels_written'):
        print('labels written to %s' % report['labels_written'])
    if not report['ok']:
        print('FAILED: %s' % report.get('error'))
        return 1
    print('backend %s, input %d, warm-up %.0f ms, output labels %s'
          % (report['backend'], report['input_size'], report['warmup_ms'], report['labels']))
    for r in report['results']:
        print('%s %s: %d detections, timings %s'
              % (r['image'], r['shape'], len(r['detections']), r['timings_ms']))
        for d in r['detections']:
            print('   %-14s %.3f  %s' % (d['label'], d['prob'], d['box']))
    lat = report['latency_ms']
    print('latency over %d runs: median %.1f ms, p90 %.1f ms, min %.1f ms'
          % (lat['n'], lat['median'], lat['p90'], lat['min']))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
Export an ultralytics .pt detector to an ONNX file the robot runtime can run.

Runs INSIDE the training venv (~/.venvs/piper-yolo: torch, ultralytics, onnx,
onnxruntime); never imported by the robot runtime. Steps:

1. ultralytics export with the verified settings (EXPORT_ARGS below).
2. patch_for_opencv45(): rewrite the two head operations OpenCV 4.5.4 gets
   wrong (const - x is computed as x - const; a rank-2 constant multiplied
   with a rank-3 tensor fails), then check with onnxruntime that the patched
   graph gives the same output as the original.
3. Write <out stem>_labels.txt from the model's class names.
4. --verify: load the result with the SYSTEM python3 (OpenCV 4.5.4, numpy
   1.21.5, the same code as the robot) and compare its boxes with
   ultralytics' own predictions (same class, IoU > 0.9).

Usage (from the package root, see docs/yolo.md):
    ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.export_onnx best.pt \
        --out models/detector.onnx --verify
    ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.export_onnx yolo11n.pt --sweep
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Verified with ultralytics 8.4.202, torch 2.6.0, onnx 1.23.2, onnxslim 0.1.98 and
# the system OpenCV 4.5.4 (docs/yolo.md). simplify=False does not load in
# OpenCV 4.5.4 (Shape / Gather anchor subgraph); opset 11 works as well.
EXPORT_ARGS: Dict[str, Any] = {
    'format': 'onnx', 'opset': 12, 'simplify': True, 'dynamic': False, 'batch': 1,
    'device': 'cpu',
}
DEFAULT_IMGSZ = 640
SYSTEM_PYTHON = '/usr/bin/python3'
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
VERIFY_IOU = 0.9
VERIFY_MARGIN = 0.1          # boxes within this of conf may legitimately flip
VERIFY_PX_TOL = 1.5          # runtime boxes are rounded to integer pixels
VERIFY_LOW_CONF = 0.05       # retry threshold when nothing is detected


# --------------------------------------------------------------------------
# Graph patch for OpenCV 4.5.4
# --------------------------------------------------------------------------

def patch_for_opencv45(src: Path, dst: Path) -> List[str]:
    """
    Rewrite src into dst so OpenCV 4.5.4 runs it correctly; return the changes.

    - Sub(c, x) with a constant first operand becomes Add(Mul(x, -1), c).
    - A non-scalar constant operand of Add / Sub / Mul / Div whose rank
      differs from the other operand is expanded to the full broadcast shape.
    Raise ValueError for constant-first Div (not used by YOLOv8 / YOLO11).
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    model = onnx.load(str(src))
    # Constant nodes -> initializers, so both kinds of constants are handled below.
    kept_nodes = []
    for n in model.graph.node:
        value = [a for a in n.attribute if a.name == 'value']
        if n.op_type == 'Constant' and len(value) == 1 and value[0].HasField('t'):
            t = onnx.TensorProto()
            t.CopyFrom(value[0].t)
            t.name = n.output[0]
            model.graph.initializer.append(t)
        else:
            kept_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    inferred = shape_inference.infer_shapes(model)
    shapes: Dict[str, Tuple[int, ...]] = {}
    for v in (list(inferred.graph.value_info) + list(inferred.graph.input)
              + list(inferred.graph.output)):
        dims = v.type.tensor_type.shape.dim
        if dims and all(d.HasField('dim_value') for d in dims):
            shapes[v.name] = tuple(int(d.dim_value) for d in dims)
    inits = {i.name: i for i in model.graph.initializer}
    changes: List[str] = []
    nodes = []
    for n in model.graph.node:
        if n.op_type not in ('Add', 'Sub', 'Mul', 'Div') or len(n.input) != 2:
            nodes.append(n)
            continue
        const_idx = [k for k, name in enumerate(n.input) if name in inits]
        if len(const_idx) != 1:
            nodes.append(n)
            continue
        k = const_idx[0]
        other = n.input[1 - k]
        c = numpy_helper.to_array(inits[n.input[k]])
        oshape = shapes.get(other)
        if c.size > 1 and oshape is not None and c.ndim != len(oshape):
            full = tuple(np.broadcast_shapes(c.shape, oshape))
            lead = (1,) * (len(full) - c.ndim)
            big = np.ascontiguousarray(np.broadcast_to(c.reshape(lead + c.shape), full))
            name = '%s_full_%s' % (n.input[k], (n.name or n.output[0]).replace('/', '_'))
            model.graph.initializer.append(numpy_helper.from_array(big.astype(c.dtype), name))
            inits[name] = model.graph.initializer[-1]
            n.input[k] = name
            changes.append('expand constant of %s %s -> %s' % (n.name, c.shape, full))
        if k == 0 and n.op_type == 'Div':
            raise ValueError('%s divides a constant by a tensor; not supported by the patch'
                             % n.name)
        if k == 0 and n.op_type == 'Sub':
            neg = (n.name or n.output[0]) + '_minus_one'
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(-1.0, dtype=c.dtype), neg))
            tmp = n.output[0] + '_negated'
            nodes.append(helper.make_node('Mul', [other, neg], [tmp], name=n.name + '_neg'))
            nodes.append(helper.make_node('Add', [tmp, n.input[0]], list(n.output),
                                          name=n.name + '_as_add'))
            changes.append('rewrite %s: c - x -> (-1 * x) + c' % n.name)
            continue
        nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for n in model.graph.node for name in n.input}
    used.update(v.name for v in list(model.graph.input) + list(model.graph.output))
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return changes


def compare_onnx_outputs(a: Path, b: Path, n: int = 2, seed: int = 0) -> float:
    """Return max |out_a - out_b| over n random inputs (onnxruntime, CPU)."""
    import numpy as np
    import onnxruntime as ort
    sa = ort.InferenceSession(str(a), providers=['CPUExecutionProvider'])
    sb = ort.InferenceSession(str(b), providers=['CPUExecutionProvider'])
    inp = sa.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    rng = np.random.RandomState(seed)
    worst = 0.0
    for _ in range(n):
        x = rng.rand(*shape).astype(np.float32)
        ya = sa.run(None, {inp.name: x})[0]
        yb = sb.run(None, {sb.get_inputs()[0].name: x})[0]
        worst = max(worst, float(np.abs(ya - yb).max()))
    return worst


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def write_labels(names: Any, path: Path) -> List[str]:
    """Write ultralytics model.names (dict or list) one per line; return the list."""
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names)
        if keys != list(range(len(keys))):
            raise ValueError('class indices are not 0..n-1: %s' % keys)
        labels = [str(names[k]).strip() for k in keys]
    else:
        labels = [str(v).strip() for v in names]
    if any(not lab or '\n' in lab for lab in labels):
        raise ValueError('empty or multi-line class name in %r' % (labels,))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(labels) + '\n', encoding='utf-8')
    return labels


def export(weights: Path, out: Path, imgsz: int = DEFAULT_IMGSZ, opset: Optional[int] = None,
           simplify: Optional[bool] = None, patch: bool = True,
           labels_out: Optional[Path] = None) -> Dict[str, Any]:
    """
    Export weights (.pt) to out (.onnx) plus a labels file; return a summary dict.

    The raw ultralytics export is kept as <out stem>.raw.onnx for reference.
    """
    from ultralytics import YOLO

    weights = Path(weights).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    labels_out = (Path(labels_out).expanduser().resolve() if labels_out
                  else out.with_name(out.stem + '_labels.txt'))
    args = dict(EXPORT_ARGS, imgsz=int(imgsz))
    if opset is not None:
        args['opset'] = int(opset)
    if simplify is not None:
        args['simplify'] = bool(simplify)
    with tempfile.TemporaryDirectory(prefix='yolo_export_') as tmp:
        # Export from a copy: ultralytics writes <stem>.onnx next to the weights.
        src = Path(tmp) / weights.name
        shutil.copy2(str(weights), str(src))
        model = YOLO(str(src))
        if getattr(model, 'task', 'detect') != 'detect':
            raise ValueError('%s is a %s model; need a detection model' % (weights, model.task))
        raw = Path(model.export(**args))
        out.parent.mkdir(parents=True, exist_ok=True)
        raw_keep = out.with_name(out.stem + '.raw.onnx')
        shutil.copy2(str(raw), str(raw_keep))
        changes: List[str] = []
        diff = 0.0
        if patch:
            changes = patch_for_opencv45(raw, out)
            diff = compare_onnx_outputs(raw, out)
            if diff > 1e-3:
                raise RuntimeError('patched graph differs from the export by %.3g' % diff)
        else:
            shutil.copy2(str(raw), str(out))
        labels = write_labels(model.names, labels_out)
    return {'weights': str(weights), 'onnx': str(out), 'raw_onnx': str(raw_keep),
            'labels_file': str(labels_out), 'labels': labels, 'export_args': args,
            'patch': changes, 'patch_max_abs_diff': diff}


# --------------------------------------------------------------------------
# Verification against the runtime stack
# --------------------------------------------------------------------------

def run_system_check(onnx_path: Path, labels_path: Path, images: Sequence[Path],
                     conf: float, iou: float, backend: str = 'opencv',
                     python: str = SYSTEM_PYTHON, repeat: int = 1) -> Dict[str, Any]:
    """
    Run check_onnx in the robot's Python (isolated mode, cwd outside the repo).

    Returns its JSON report (see check_onnx.py). The child gets only the package
    root on sys.path, so it uses the system cv2 / numpy, never the venv ones.
    """
    argv = ['--model', str(onnx_path), '--labels', str(labels_path), '--backend', backend,
            '--conf', str(conf), '--nms', str(iou), '--repeat', str(repeat), '--json']
    if images:
        argv += ['--image'] + [str(p) for p in images]
    code = ('import sys; sys.path.insert(0, %r); '
            'from cube_task.tools.yolo.check_onnx import main; '
            'sys.exit(main(%r))' % (str(PACKAGE_ROOT), argv))
    with tempfile.TemporaryDirectory(prefix='yolo_check_') as cwd:
        proc = subprocess.run([python, '-I', '-c', code], cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('{')]
    if not lines:
        return {'ok': False, 'error': 'no JSON from %s (exit %d): %s'
                % (python, proc.returncode, proc.stderr.strip()[-800:])}
    return json.loads(lines[-1])


def ultralytics_predictions(model_path: Path, images: Sequence[Path], imgsz: int,
                            conf: float, iou: float) -> List[List[Dict[str, Any]]]:
    """Return ultralytics' own predictions [[{box, label, prob}], ...] per image."""
    from ultralytics import YOLO
    model = YOLO(str(model_path), task='detect')
    out = []
    for img in images:
        r = model.predict(str(img), imgsz=imgsz, conf=conf, iou=iou, device='cpu',
                          verbose=False)[0]
        names = r.names
        dets = []
        for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            dets.append({'box': [float(v) for v in b], 'label': str(names[int(c)]),
                         'prob': float(p)})
        out.append(dets)
    return out


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the intersection over union of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(ours: List[Dict[str, Any]], ref: List[Dict[str, Any]], conf: float,
                     min_iou: float = VERIFY_IOU, margin: float = VERIFY_MARGIN,
                     px_tol: float = VERIFY_PX_TOL) -> Dict[str, Any]:
    """
    Greedily match same-label boxes by IoU.

    A pair agrees when IoU > min_iou or every corner differs by <= px_tol
    pixels (the runtime rounds boxes to integer pixels, which matters for
    tiny boxes). ok is True when every box with prob >= conf + margin and
    both sides >= 2 px, in either list, has an agreeing same-label partner.
    """
    def big(d: Dict[str, Any]) -> bool:
        b = d['box']
        return d['prob'] >= conf + margin and b[2] - b[0] >= 2 and b[3] - b[1] >= 2

    def agree(a: Dict[str, Any], b: Dict[str, Any], v: float) -> bool:
        return v > min_iou or max(abs(x - y) for x, y in zip(a['box'], b['box'])) <= px_tol

    pairs = []
    for i, o in enumerate(ours):
        for j, r in enumerate(ref):
            if o['label'] == r['label']:
                pairs.append((box_iou(o['box'], r['box']), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_o, used_r, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_o or j in used_r or v <= 0.0:
            continue
        used_o.add(i)
        used_r.add(j)
        matched.append((i, j, v))
    good = [(i, j, v) for i, j, v in matched if agree(ours[i], ref[j], v)]
    good_o = {i for i, _j, _v in good}
    good_r = {j for _i, j, _v in good}
    miss_o = [o for i, o in enumerate(ours) if big(o) and i not in good_o]
    miss_r = [r for j, r in enumerate(ref) if big(r) and j not in good_r]
    ious = [v for i, j, v in matched if big(ours[i]) or big(ref[j])]
    return {'ok': not miss_o and not miss_r, 'n_ours': len(ours), 'n_ref': len(ref),
            'n_matched': len(matched), 'min_iou': min(ious) if ious else None,
            'mean_iou': sum(ious) / len(ious) if ious else None,
            'unmatched_ours': miss_o, 'unmatched_ref': miss_r}


def default_images() -> List[Path]:
    """Ultralytics' bundled sample images (bus.jpg, zidane.jpg)."""
    from ultralytics.utils import ASSETS
    return [Path(ASSETS) / 'bus.jpg', Path(ASSETS) / 'zidane.jpg']


def verify(summary: Dict[str, Any], images: Sequence[Path], imgsz: int, conf: float,
           iou: float, backend: str = 'opencv', repeat: int = 5) -> Dict[str, Any]:
    """
    Compare the system-python detections with ultralytics (.pt and raw .onnx).

    A comparison needs boxes: when ultralytics finds no box with prob >=
    conf + VERIFY_MARGIN on any image, it is repeated once at conf
    VERIFY_LOW_CONF; if there is still nothing to compare the result is
    inconclusive (ok False).
    """
    check = run_system_check(Path(summary['onnx']), Path(summary['labels_file']), images,
                             conf, iou, backend=backend, repeat=repeat)
    result: Dict[str, Any] = {'system_check': check, 'ok': bool(check.get('ok')),
                              'conf': conf, 'inconclusive': False}
    if not check.get('ok'):
        return result
    refs = {'pt': ultralytics_predictions(Path(summary['weights']), images, imgsz, conf, iou),
            'onnx': ultralytics_predictions(Path(summary['raw_onnx']), images, imgsz, conf, iou)}
    strong = sum(1 for dets in refs['pt'] for d in dets if d['prob'] >= conf + VERIFY_MARGIN)
    if strong == 0:
        if conf > VERIFY_LOW_CONF:
            return verify(summary, images, imgsz, VERIFY_LOW_CONF, iou, backend, repeat)
        result.update(ok=False, inconclusive=True)
        return result
    result['compare'] = {}
    for name, ref in refs.items():
        per_image = []
        for k, img in enumerate(images):
            ours = check['results'][k]['detections']
            m = match_detections(ours, ref[k], conf)
            m['image'] = Path(img).name
            per_image.append(m)
            result['ok'] = result['ok'] and m['ok']
        result['compare'][name] = per_image
    return result


def print_verify(res: Dict[str, Any]) -> None:
    """Print a verification result in a few readable lines."""
    chk = res['system_check']
    if not chk.get('ok'):
        print('  system python check FAILED: %s' % chk.get('error'))
        return
    env = chk['env']
    print('  runtime: python %s, cv2 %s (CUDA DNN %s), numpy %s, backend %s, latency median '
          '%.1f ms (p90 %.1f, n=%d)' % (env['python'], env['cv2'], env['cv2_cuda'],
                                        env['numpy'], chk['backend'],
                                        chk['latency_ms']['median'], chk['latency_ms']['p90'],
                                        chk['latency_ms']['n']))
    if res.get('inconclusive'):
        print('  INCONCLUSIVE: ultralytics finds no box with prob >= %.2f on these images, so '
              'there is nothing to compare; use images with objects (--images) or a better '
              'trained model' % (res['conf'] + VERIFY_MARGIN))
    elif res.get('conf') is not None:
        print('  compared at conf %.2f (boxes with prob >= %.2f must match)'
              % (res['conf'], res['conf'] + VERIFY_MARGIN))
    for ref_name, items in res.get('compare', {}).items():
        for m in items:
            print('  vs ultralytics %-4s %-10s ours %2d ref %2d matched %2d min IoU %s -> %s'
                  % (ref_name, m['image'], m['n_ours'], m['n_ref'], m['n_matched'],
                     'n/a' if m['min_iou'] is None else '%.3f' % m['min_iou'],
                     'OK' if m['ok'] else
                     'MISMATCH %s %s' % (m['unmatched_ours'], m['unmatched_ref'])))
    print('  verification %s' % ('PASSED' if res['ok'] else 'FAILED'))


def sweep(weights: Path, work: Path, imgsz: int, conf: float, iou: float,
          images: Sequence[Path]) -> List[Dict[str, Any]]:
    """Try opset 12 / 11 x simplify on / off x patch on / off; verify each one."""
    rows = []
    for opset in (12, 11):
        for simplify in (True, False):
            for patch in (False, True):
                tag = 'op%d_simplify%d_patch%d' % (opset, int(simplify), int(patch))
                out = work / tag / (Path(weights).stem + '.onnx')
                row: Dict[str, Any] = {'tag': tag}
                try:
                    s = export(weights, out, imgsz=imgsz, opset=opset, simplify=simplify,
                               patch=patch)
                    v = verify(s, images, imgsz, conf, iou)
                    row.update(ok=v['ok'], verify=v)
                except Exception as e:  # report and continue with the next setting
                    row.update(ok=False, error='%s: %s' % (type(e).__name__, e))
                rows.append(row)
                print('%-28s %s' % (tag, 'PASS' if row['ok'] else 'FAIL'), flush=True)
                if 'verify' in row:
                    print_verify(row['verify'])
                else:
                    print('  %s' % row['error'][:300])
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('weights', help='ultralytics .pt detection weights')
    p.add_argument('--out', default=None,
                   help='output .onnx (default: next to the weights, <stem>.onnx)')
    p.add_argument('--labels-out', default=None,
                   help='labels file (default: <out stem>_labels.txt)')
    p.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    p.add_argument('--no-patch', action='store_true',
                   help='skip the OpenCV 4.5.4 graph patch (onnxruntime-only models)')
    p.add_argument('--verify', action='store_true',
                   help='run the result with the system python3 and compare with ultralytics')
    p.add_argument('--images', nargs='*', default=None,
                   help='images for --verify / --sweep (default: bus.jpg, zidane.jpg)')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--sweep', action='store_true',
                   help='try several export settings and verify each (writes under --work)')
    p.add_argument('--work', default=str(Path.home() / 'vision_runs' / 'yolo' / 'export_sweep'))
    p.add_argument('--report', default=None, help='write the JSON result here')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        print('weights not found: %s' % weights)
        return 1
    images = [Path(p) for p in args.images] if args.images else default_images()
    if args.sweep:
        rows = sweep(weights, Path(args.work).expanduser(), args.imgsz, args.conf, args.iou,
                     images)
        if args.report:
            Path(args.report).write_text(json.dumps(rows, indent=1, default=str))
        return 0 if any(r['ok'] for r in rows) else 1
    out = Path(args.out) if args.out else weights.with_suffix('.onnx')
    summary = export(weights, out, imgsz=args.imgsz, patch=not args.no_patch,
                     labels_out=Path(args.labels_out) if args.labels_out else None)
    print('exported %s (%d classes) -> %s, labels %s'
          % (weights.name, len(summary['labels']), summary['onnx'], summary['labels_file']))
    for c in summary['patch']:
        print('  patch: %s' % c)
    if summary['patch']:
        print('  patched vs raw max |diff| (onnxruntime): %.2g' % summary['patch_max_abs_diff'])
    ok = True
    if args.verify:
        res = verify(summary, images, args.imgsz, args.conf, args.iou)
        summary['verify'] = res
        print_verify(res)
        ok = res['ok']
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=1, default=str))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

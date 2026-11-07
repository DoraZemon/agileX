"""
Train a YOLO11 / YOLOv8 detector and deploy it as models/detector.onnx.

Runs INSIDE the training venv (~/.venvs/piper-yolo), never imported by the
robot runtime. After training it exports best.pt with the verified ONNX
settings (export_onnx.py), checks the ONNX file with the system python3 /
OpenCV 4.5.4 against ultralytics on a few val images, copies it to
<models>/detector.onnx + detector_labels.txt (older files are kept with a
time-stamp suffix) and prints the config lines for config/site/local.yaml.

Example (package root, ROS not sourced or PYTHONPATH unset):
    ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.train_yolo \
        --data ~/vision_runs/yolo/datasets/own/data.yaml --model yolo11n.pt \
        --epochs 100 --batch 16 --device 0
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PROJECT = Path.home() / 'vision_runs' / 'yolo'


def _export_module():
    try:
        from cube_task.tools.yolo import export_onnx
    except ImportError:  # run as a script path: put the package root on sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from cube_task.tools.yolo import export_onnx
    return export_onnx


def parse_extra(items: List[str]) -> Dict[str, Any]:
    """Parse ['key=value', ...] (YAML values) into train() keyword arguments."""
    import yaml
    out: Dict[str, Any] = {}
    for it in items:
        if '=' not in it:
            raise ValueError('--arg needs key=value, got %r' % it)
        k, v = it.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def dataset_images(data_yaml: Path, split: str = 'val', limit: int = 4) -> List[Path]:
    """Return up to limit images of the dataset split (sorted, for verification)."""
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    root = Path(str(data.get('path') or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entries = data.get(split) or data.get('train') or []
    if isinstance(entries, str):
        entries = [entries]
    images: List[Path] = []
    for e in entries:
        p = Path(str(e)).expanduser()
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            images.extend(sorted(q for q in p.rglob('*')
                                 if q.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')))
        elif p.suffix == '.txt' and p.is_file():
            for ln in p.read_text().splitlines():
                if ln.strip():
                    q = Path(ln.strip())
                    images.append(q if q.is_absolute() else root / q)
    return images[:limit]


def _backup(path: Path, stamp: str) -> Optional[Path]:
    if not path.exists():
        return None
    dst = path.with_name('%s.%s%s' % (path.stem, stamp, path.suffix))
    shutil.move(str(path), str(dst))
    return dst


def deploy(onnx_path: Path, labels_path: Path, models_dir: Path, name: str) -> Dict[str, Any]:
    """Copy the model and labels into models_dir as <name>.onnx / <name>_labels.txt."""
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst_model = models_dir / (name + '.onnx')
    dst_labels = models_dir / (name + '_labels.txt')
    backups = [b for b in (_backup(dst_model, stamp), _backup(dst_labels, stamp)) if b]
    shutil.copy2(str(onnx_path), str(dst_model))
    shutil.copy2(str(labels_path), str(dst_labels))
    return {'model': dst_model, 'labels': dst_labels, 'backups': backups}


def config_snippet(models_dir: Path, name: str, imgsz: int, labels: List[str],
                   package_root: Path) -> str:
    """Return the YAML lines for config/site/local.yaml (paths relative when possible)."""
    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(package_root.resolve()))
        except ValueError:
            return str(p.resolve())
    model = rel(models_dir / (name + '.onnx'))
    label_file = rel(models_dir / (name + '_labels.txt'))
    return '\n'.join([
        '# config/site/local.yaml  (model classes: %s)' % ', '.join(labels),
        'tasks:',
        '  L2:',
        '    classifier: onnx            # or onnx+knn: KNN labels what the detector misses',
        '    labels: [%s]' % ', '.join(labels),
        'perception:',
        '  classifier:',
        '    onnx:',
        '      model_file: %s' % model,
        '      labels_file: %s' % label_file,
        '      input_size: %d' % imgsz,
        '      backend: auto             # onnxruntime if importable, else opencv',
        '      label_map: {}             # model label -> task label, \'\' drops it',
    ])


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='dataset yaml (YOLO format)')
    p.add_argument('--model', default='yolo11n.pt',
                   help='base weights: yolo11n.pt (fast) or yolo11s.pt (more accurate)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16, help='16 fits yolo11n/s on a 6 GB GPU')
    p.add_argument('--device', default=None, help="0 (GPU) or cpu; default: GPU if available")
    p.add_argument('--project', default=str(DEFAULT_PROJECT), help='runs go to project/name')
    p.add_argument('--name', default=None, help='run name (default l2_<date>_<time>)')
    p.add_argument('--patience', type=int, default=30, help='early stop after N flat epochs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arg', action='append', default=[], metavar='KEY=VALUE',
                   help='extra ultralytics train argument, e.g. --arg fliplr=0.5')
    p.add_argument('--models-dir', default=None, help='default: <package>/models')
    p.add_argument('--deploy-name', default='detector',
                   help='deployed as <models>/<name>.onnx and <name>_labels.txt')
    p.add_argument('--no-deploy', action='store_true', help='train and export only')
    p.add_argument('--verify-images', type=int, default=4,
                   help='val images used for the system-python check')
    p.add_argument('--skip-verify', action='store_true',
                   help='deploy without comparing against ultralytics (not recommended)')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    exp = _export_module()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        print('dataset yaml not found: %s' % data)
        return 1
    import torch
    from ultralytics import YOLO
    device = args.device
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not torch.cuda.is_available():
        print('device %s requested but torch %s has no CUDA; install the CUDA build '
              '(docs/yolo.md) or use --device cpu' % (device, torch.__version__))
        return 1
    name = args.name or datetime.datetime.now().strftime('l2_%Y%m%d_%H%M%S')
    project = Path(args.project).expanduser().resolve()
    train_args = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      device=device, project=str(project), name=name, patience=args.patience,
                      workers=args.workers, seed=args.seed, exist_ok=False)
    train_args.update(parse_extra(args.arg))
    print('training %s on %s (device %s, torch %s), run dir %s/%s'
          % (args.model, data, device, torch.__version__, project, name))
    model = YOLO(args.model)
    model.train(**train_args)
    best = Path(str(getattr(model.trainer, 'best', '')))
    if not best.is_file():
        print('training finished without best.pt (looked for %s)' % best)
        return 1
    run_dir = best.parent.parent
    summary = exp.export(best, best.with_suffix('.onnx'), imgsz=args.imgsz)
    print('exported %s -> %s' % (best, summary['onnx']))
    images = dataset_images(data, 'val', args.verify_images)
    if not images:
        print('no val images found for verification; using the ultralytics samples')
        images = exp.default_images()
    if args.skip_verify:
        print('WARNING: --skip-verify: the ONNX file was not checked in the robot runtime')
        res = {'ok': True}
    else:
        res = exp.verify(summary, images, args.imgsz, 0.35, 0.45)
        exp.print_verify(res)
    if not res['ok']:
        print('NOT deployed: the ONNX file does not reproduce ultralytics in the robot runtime. '
              'Run it on its own: python3 -m cube_task.tools.yolo.check_onnx --model %s '
              '--labels %s' % (summary['onnx'], summary['labels_file']))
        return 1
    if args.no_deploy:
        print('run dir %s (not deployed: --no-deploy)' % run_dir)
        return 0
    models_dir = (Path(args.models_dir).expanduser().resolve() if args.models_dir
                  else exp.PACKAGE_ROOT / 'models')
    dep = deploy(Path(summary['onnx']), Path(summary['labels_file']), models_dir,
                 args.deploy_name)
    for b in dep['backups']:
        print('kept previous file as %s' % b)
    print('deployed %s and %s (run dir %s)' % (dep['model'], dep['labels'], run_dir))
    print()
    print(config_snippet(models_dir, args.deploy_name, args.imgsz, summary['labels'],
                         exp.PACKAGE_ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())

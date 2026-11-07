"""Tests for cube_task/tools/yolo (dataset builder, export helpers, training helpers)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cube_task.tools.yolo import check_onnx, export_onnx, make_dataset as md, train_yolo

ROOT = Path(__file__).resolve().parents[1]
COCO_MODEL = ROOT / 'models' / 'yolo11n_coco.onnx'
COCO_LABELS = ROOT / 'models' / 'yolo11n_coco_labels.txt'
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def synthetic_scene(rng, n_obj=2, h=480, w=640, tilt=0.0003, dropout=0.01):
    """Table plane at ~0.6 m with n_obj 4 cm high disks; return color, depth, true boxes."""
    color = np.full((h, w, 3), (150, 160, 170), np.uint8)
    vv, uu = np.mgrid[0:h, 0:w]
    depth = (0.6 + tilt * (vv - h / 2)).astype(np.float32)
    boxes, circles = [], []
    while len(circles) < n_obj:
        cx, cy, r = rng.randint(80, w - 80), rng.randint(80, h - 80), rng.randint(22, 40)
        if all((cx - a) ** 2 + (cy - b) ** 2 > (r + c + 10) ** 2 for a, b, c in circles):
            circles.append((cx, cy, r))
    for cx, cy, r in circles:
        m = (uu - cx) ** 2 + (vv - cy) ** 2 < r * r
        color[m] = (40, 40, 200)
        depth[m] -= 0.04
        ys, xs = np.nonzero(m)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    depth[rng.rand(h, w) < dropout] = 0.0
    depth[rng.rand(h, w) < dropout] = np.nan
    return color, depth, boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u


# --------------------------------------------------------------------------
# make_dataset: plane segmentation
# --------------------------------------------------------------------------

def test_fit_plane_orientation_and_ratio():
    rng = np.random.RandomState(0)
    xy = rng.uniform(-0.3, 0.3, (3000, 2))
    pts = np.column_stack([xy, 0.6 + 0.1 * xy[:, 1]])          # plane in front of the camera
    pts[:300, 2] -= 0.05                                        # 10 % objects
    n, d, ratio = md.fit_plane_ransac(pts, 0.004, 100, rng)
    assert d > 0                                                # normal faces the camera
    assert ratio == pytest.approx(0.9, abs=0.01)
    assert np.all(pts[:300] @ n + d > 0.04)                    # objects are above
    assert md.fit_plane_ransac(pts[:2], 0.004, 10, rng) is None


def test_plane_boxes_match_objects():
    rng = np.random.RandomState(1)
    color, depth, truth = synthetic_scene(rng, n_obj=3)
    boxes, diag = md.plane_boxes(depth, K, md.PlaneParams(), np.random.RandomState(0))
    assert diag['reason'] == '' and diag['inlier_ratio'] > 0.8
    assert len(boxes) == 3
    for t in truth:
        assert max(iou(b, t) for b in boxes) > 0.85


def test_plane_boxes_filters():
    rng = np.random.RandomState(2)
    _c, depth, truth = synthetic_scene(rng, n_obj=2)
    big = md.PlaneParams(min_area_m2=0.01)                      # both disks are smaller
    boxes, diag = md.plane_boxes(depth, K, big, np.random.RandomState(0))
    assert boxes == [] and diag['rejected'].get('area') == 2
    x0, y0, x1, y1 = truth[0]
    roi = md.PlaneParams(roi=(x0 - 5, y0 - 5, x1 + 5, y1 + 5))
    boxes, _d = md.plane_boxes(depth, K, roi, np.random.RandomState(0))
    assert len(boxes) == 1 and iou(boxes[0], truth[0]) > 0.85
    d2 = depth.copy()
    d2[:, :40] = 0.5                                            # tall thing at the image border
    boxes, diag = md.plane_boxes(d2, K, md.PlaneParams(max_area_m2=1.0),
                                 np.random.RandomState(0))
    assert diag['rejected'].get('border') == 1 and len(boxes) == 2


def test_plane_boxes_invalid_depth():
    nan = np.full((480, 640), np.nan, np.float32)
    boxes, diag = md.plane_boxes(nan, K, md.PlaneParams(), np.random.RandomState(0))
    assert boxes == [] and 'too few valid depth' in diag['reason']
    rng = np.random.RandomState(3)
    noise = rng.uniform(0.3, 2.0, (480, 640)).astype(np.float32)
    boxes, diag = md.plane_boxes(noise, K, md.PlaneParams(), np.random.RandomState(0))
    assert boxes == [] and 'dominant plane' in diag['reason']
    with pytest.raises(ValueError):
        md.plane_boxes(np.zeros((4, 4, 1), np.float32), K, md.PlaneParams(),
                       np.random.RandomState(0))
    with pytest.raises(ValueError):
        md.PlaneParams(min_height=0.3, max_height=0.2)


# --------------------------------------------------------------------------
# make_dataset: split, labels, dataset
# --------------------------------------------------------------------------

def test_parse_session_and_split():
    assert md.parse_session('sports ball=~/x') == ('sports ball', Path('~/x').expanduser())
    for bad in ('nolabel', '=dir'):
        with pytest.raises(ValueError):
            md.parse_session(bad)
    sessions = [('a', Path('a%d' % i)) for i in range(5)] + [('b', Path('b0'))] + \
        [('c', Path('c0')), ('c', Path('c1'))]
    plan = md.split_sessions(sessions, [('b', Path('bv'))], 0.2, seed=0)
    split = {str(p): s for _l, p, s in plan}
    assert split['bv'] == 'val' and split['b0'] == 'train'
    assert sum(split['a%d' % i] == 'val' for i in range(5)) == 1
    assert sorted([split['c0'], split['c1']]) == ['train', 'val']
    assert plan == md.split_sessions(sessions, [('b', Path('bv'))], 0.2, seed=0)
    with pytest.raises(ValueError):
        md.split_sessions(sessions, [], 1.0, 0)


def test_yolo_line():
    assert md.yolo_line(2, (0, 0, 64, 48), 640, 480) == '2 0.050000 0.050000 0.100000 0.100000'


def write_capture(d, idx, color, depth):
    """Write one frame in the camera/recorder.py file format (local fake writer)."""
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / ('%06d_color.png' % idx)), color)
    mm = np.where(np.isfinite(depth) & (depth > 0), depth * 1000.0, 0).astype(np.uint16)
    cv2.imwrite(str(d / ('%06d_depth.png' % idx)), mm)
    meta = {'intrinsics': {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0,
                           'width': 640, 'height': 480}, 'depth_scale': 0.001}
    (d / ('%06d_meta.json' % idx)).write_text(json.dumps(meta))


def fake_loader(d, idx):
    """Local stand-in for camera/recorder.load_frame (format of write_capture)."""
    color_path = d / ('%06d_color.png' % idx)
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(d / ('%06d_depth.png' % idx)), cv2.IMREAD_UNCHANGED)
    return md.SessionFrame(idx, color, raw.astype(np.float32) * 0.001, K, color_path)


def make_sessions(tmp_path, n_frames=3, seed=0):
    rng = np.random.RandomState(seed)
    truth = {}
    for name in ('apple_1', 'apple_2', 'banana_1', 'banana_2'):
        for i in range(n_frames):
            color, depth, boxes = synthetic_scene(rng, n_obj=2)
            write_capture(tmp_path / 'cap' / name, i, color, depth)
            truth[(name, i)] = boxes
    (tmp_path / 'cap' / 'apple_1' / 'notes.txt').write_text('ignored')
    return truth


def test_build_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(md, '_load_capture_frame', fake_loader)
    truth = make_sessions(tmp_path)
    out = tmp_path / 'ds'
    rc = md.main(['--out', str(out), '--classes', 'banana,apple',
                  '--session', 'apple=%s' % (tmp_path / 'cap' / 'apple_1'),
                  '--session', 'apple=%s' % (tmp_path / 'cap' / 'apple_2'),
                  '--session', 'banana=%s' % (tmp_path / 'cap' / 'banana_1'),
                  '--session', 'banana=%s' % (tmp_path / 'cap' / 'banana_2'),
                  '--preview'])
    assert rc == 0
    import yaml
    data = yaml.safe_load((out / 'data.yaml').read_text())
    assert data['names'] == {0: 'banana', 1: 'apple'} and data['nc'] == 2
    assert data['val'] == 'images/val' and Path(data['path']) == out.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    splits = {s['name']: s['split'] for s in manifest['sessions']}
    assert sorted(splits.values()) == ['train', 'train', 'val', 'val']
    assert len({splits['apple_apple_1'], splits['apple_apple_2']}) == 2   # one of each
    for s in manifest['sessions']:
        assert s['frames'] == 3 and s['boxes'] == 6 and s['skipped'] == {}
    lab = out / 'labels' / splits['banana_banana_1'] / 'banana_banana_1_000001.txt'
    rows = [ln.split() for ln in lab.read_text().splitlines()]
    assert len(rows) == 2 and all(r[0] == '0' for r in rows)
    for r in rows:
        cx, cy, bw, bh = (float(v) for v in r[1:])
        box = ((cx - bw / 2) * 640, (cy - bh / 2) * 480, (cx + bw / 2) * 640, (cy + bh / 2) * 480)
        assert max(iou(box, t) for t in truth[('banana_1', 1)]) > 0.85
    img = out / 'images' / splits['banana_banana_1'] / 'banana_banana_1_000001.png'
    assert img.read_bytes() == (tmp_path / 'cap' / 'banana_1' / '000001_color.png').read_bytes()
    assert len(list((out / 'preview').glob('*.jpg'))) == 12
    # Refuses to write into a non-empty directory.
    again = ['--out', str(out), '--session', 'apple=%s' % (tmp_path / 'cap' / 'apple_1')]
    assert md.main(again) == 1


def test_build_dataset_skips_and_single_session(tmp_path, monkeypatch):
    monkeypatch.setattr(md, '_load_capture_frame', fake_loader)
    d = tmp_path / 'cap' / 'empty_table'
    write_capture(d, 0, np.zeros((480, 640, 3), np.uint8),
                  np.full((480, 640), 0.6, np.float32))
    color, depth, _b = synthetic_scene(np.random.RandomState(0), n_obj=3)
    write_capture(d, 1, color, depth)
    out = tmp_path / 'ds'
    rc = md.main(['--out', str(out), '--session', 'apple=%s' % d, '--max-boxes', '2'])
    assert rc == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    s = manifest['sessions'][0]
    assert s['frames'] == 0 and s['skipped'] == {'no boxes': 1, 'more than 2 boxes': 1}
    assert manifest['val_is_train'] is True
    import yaml
    assert yaml.safe_load((out / 'data.yaml').read_text())['val'] == 'images/train'


def test_image_folder_needs_model_boxes(tmp_path):
    d = tmp_path / 'imgs'
    d.mkdir()
    cv2.imwrite(str(d / 'a.jpg'), np.zeros((48, 64, 3), np.uint8))
    out = tmp_path / 'ds'
    assert md.main(['--out', str(out), '--session', 'apple=%s' % d]) == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert list(manifest['sessions'][0]['skipped']) == [
        'box error: no depth / intrinsics (use --boxes model for image folders)']
    assert md.main(['--out', str(tmp_path / 'ds2'), '--session', 'apple=%s' % d,
                    '--boxes', 'model']) == 1                   # --model / --labels missing


def test_build_dataset_with_recorder_files(tmp_path):
    """Integration with camera/recorder.py (save_frame / load_frame) when it exists."""
    recorder = pytest.importorskip('cube_task.camera.recorder')
    from cube_task.core.types import CameraIntrinsics, Frame
    intr = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
    rng = np.random.RandomState(5)
    for name in ('apple_1', 'apple_2'):
        for i in range(2):
            color, depth, _b = synthetic_scene(rng, n_obj=2)
            recorder.save_frame(tmp_path / 'cap' / name, i,
                                Frame(color=color, depth=depth, intrinsics=intr,
                                      stamp=float(i), source='sim'))
    out = tmp_path / 'ds'
    rc = md.main(['--out', str(out), '--session', 'apple=%s' % (tmp_path / 'cap' / 'apple_1'),
                  '--session', 'apple=%s' % (tmp_path / 'cap' / 'apple_2')])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert rc == 0, manifest
    assert manifest['counts'] == {'train': {'images': 2, 'boxes': 4},
                                  'val': {'images': 2, 'boxes': 4}}, manifest


# --------------------------------------------------------------------------
# export_onnx / check_onnx / train_yolo helpers (no torch needed)
# --------------------------------------------------------------------------

def test_match_detections():
    ref = [{'box': [10, 10, 110, 110], 'label': 'apple', 'prob': 0.9},
           {'box': [200, 200, 202, 201], 'label': 'apple', 'prob': 0.9},   # tiny
           {'box': [300, 300, 340, 340], 'label': 'banana', 'prob': 0.40}]  # near threshold
    ours = [{'box': [11, 10, 110, 111], 'label': 'apple', 'prob': 0.89}]
    m = export_onnx.match_detections(ours, ref, 0.35)
    assert m['ok'] and m['n_matched'] == 1 and m['min_iou'] > 0.97
    bad = [{'box': [11, 10, 110, 111], 'label': 'banana', 'prob': 0.89}]
    m = export_onnx.match_detections(bad, ref, 0.35)
    assert not m['ok'] and m['unmatched_ref'][0]['label'] == 'apple'
    small = [{'box': [0, 0, 10, 3], 'label': 'a', 'prob': 0.9}]
    near = [{'box': [1, 0, 11, 4], 'label': 'a', 'prob': 0.9}]
    assert export_onnx.match_detections(small, near, 0.35)['ok']        # within 1.5 px
    assert export_onnx.box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_write_labels(tmp_path):
    p = tmp_path / 'l.txt'
    assert export_onnx.write_labels({1: 'b', 0: 'a'}, p) == ['a', 'b']
    assert p.read_text() == 'a\nb\n'
    with pytest.raises(ValueError):
        export_onnx.write_labels({0: 'a', 2: 'c'}, p)
    with pytest.raises(ValueError):
        export_onnx.write_labels(['a', ''], p)


def test_train_helpers(tmp_path):
    assert train_yolo.parse_extra(['mosaic=0.5', 'fliplr=0', 'cache=true']) == \
        {'mosaic': 0.5, 'fliplr': 0, 'cache': True}
    with pytest.raises(ValueError):
        train_yolo.parse_extra(['mosaic'])
    (tmp_path / 'images' / 'val').mkdir(parents=True)
    for i in range(3):
        (tmp_path / 'images' / 'val' / ('%d.png' % i)).write_bytes(b'x')
    data = tmp_path / 'data.yaml'
    data.write_text('path: .\ntrain: images/train\nval: images/val\nnames: {0: a}\n')
    imgs = train_yolo.dataset_images(data, 'val', 2)
    assert [p.name for p in imgs] == ['0.png', '1.png']
    m = tmp_path / 'm.onnx'
    lab = tmp_path / 'm_labels.txt'
    m.write_bytes(b'new')
    lab.write_text('a\n')
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'detector.onnx').write_bytes(b'old')
    dep = train_yolo.deploy(m, lab, models, 'detector')
    assert (models / 'detector.onnx').read_bytes() == b'new'
    assert len(dep['backups']) == 1 and dep['backups'][0].read_bytes() == b'old'
    snippet = train_yolo.config_snippet(models, 'detector', 640, ['a', 'b'], tmp_path)
    assert 'model_file: models/detector.onnx' in snippet
    assert 'labels_file: models/detector_labels.txt' in snippet
    assert 'classifier: onnx' in snippet and 'labels: [a, b]' in snippet


@pytest.mark.skipif(not (COCO_MODEL.is_file() and COCO_LABELS.is_file()),
                    reason='models/yolo11n_coco.onnx not exported (docs/yolo.md)')
def test_check_onnx_json(capsys, tmp_path):
    rc = check_onnx.main(['--model', str(COCO_MODEL), '--labels', str(COCO_LABELS),
                          '--repeat', '2', '--json', '--save-vis', str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0 and report['ok'] and report['backend'] == 'opencv'
    assert report['onnx']['const_first_ops'] == [] and len(report['onnx']['names']) == 80
    assert report['latency_ms']['n'] == 2 and report['env']['cv2'].startswith('4.')
    assert (tmp_path / 'synthetic_det.jpg').is_file()
    labels = tmp_path / 'labels.txt'
    rc = check_onnx.main(['--model', str(COCO_MODEL), '--write-labels', str(labels), '--json'])
    capsys.readouterr()
    assert rc == 0 and labels.read_text() == COCO_LABELS.read_text()

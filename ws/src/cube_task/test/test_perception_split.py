"""Touching L1 blocks (color split, same-color cuts) and completed footprints."""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.interfaces import InstanceClassifier
from cube_task.core.types import Detection
from cube_task.perception.classify_color import ColorClassifier
from cube_task.perception.pipeline import annotate_neighbors, Perception
from cube_task.perception.segment import (_snap_to_step, completed_extents, find_block_cuts,
                                          mark_invalid, SegmentParams, split_by_classes)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_perception_synth as S  # noqa: E402
from test_perception_synth import Box, Sphere  # noqa: E402

CAMS = {'overhead': S.overhead_camera(), 'oblique': S.oblique_camera()}
LABELS = ['red', 'green', 'blue', 'yellow']


def _cfg(**over):
    cfg = load_config(use_overlays=False)
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def _clf():
    return ColorClassifier(_cfg().get('perception.classifier.color.classes'), 0.35, LABELS)


def _run(objs, cam, classifier='color', **over):
    clf = _clf() if classifier == 'color' else classifier
    p = Perception(_cfg(**over), classifier=clf)
    fr = S.make_frame(objs, CAMS[cam], noise_std=0.001, seed=1)
    return p.process(fr, CAMS[cam])


def _err_mm(d, objs):
    b = min(objs, key=lambda o: math.hypot(o.x - d.center_base[0], o.y - d.center_base[1]))
    return 1000.0 * math.hypot(b.x - d.center_base[0], b.y - d.center_base[1]), b


def _flush(b: Box, size: float, color, lateral: float = 0.0) -> Box:
    """Return a cube of edge size touching b on its +x face (b's frame), shifted by lateral."""
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    off = (b.sx + size) / 2.0
    return Box(b.x + c * off - s * lateral, b.y + s * off + c * lateral, b.yaw, size, size, size,
               color)


# --------------------------------------------------------------------------- color split


@pytest.mark.parametrize('cam', sorted(CAMS))
def test_touching_blocks_of_two_colors_are_split(cam):
    a = Box(0.30, 0.0, 0.0, 0.035, 0.035, 0.035, S.RED)
    objs = [a, _flush(a, 0.035, S.GREEN), Box(0.25, -0.12, 0.4, 0.035, 0.035, 0.035, S.BLUE)]
    # Without color the flush pair is one instance centered between the blocks.
    plain = _run(objs, cam, classifier=None)
    assert len(plain.detections) == 2
    assert max(_err_mm(d, objs)[0] for d in plain.detections) > 12.0
    res = _run(objs, cam)
    assert not res.warnings
    assert sorted(d.label for d in res.valid_detections()) == ['blue', 'green', 'red']
    for d in res.detections:
        err, b = _err_mm(d, objs)
        assert err < 3.0, (d.label, err)
        assert d.label == {S.RED: 'red', S.GREEN: 'green', S.BLUE: 'blue'}[b.color]
        assert abs(d.width - 0.035) < 0.005 and abs(d.length - 0.035) < 0.005
        assert d.extra.get('split') == ('color' if b is not objs[2] else None)
        assert abs(d.extra['table_z']) < 0.002 and abs(d.height - 0.035) < 0.003
        assert 'cap_center_xy' in d.extra
    assert 'color_split' in res.timings


def test_color_split_can_be_disabled_and_needs_a_color_classifier():
    a = Box(0.30, 0.0, 0.0, 0.035, 0.035, 0.035, S.RED)
    objs = [a, _flush(a, 0.035, S.GREEN)]
    off = _run(objs, 'overhead', **{'perception.segmentation.color_split': False,
                                    'perception.segmentation.merged_blocks': 'off'})
    assert len(off.detections) == 1

    class Fixed(InstanceClassifier):
        labels = ['red']

        def classify(self, frame, detections):
            for d in detections:
                d.label, d.score, d.score_kind = 'red', 0.9, 'knn'

    # Other classifiers (KNN, L2) never trigger block splitting.
    knn = _run(objs, 'overhead', classifier=Fixed())
    assert len(knn.detections) == 1 and 'split' not in knn.detections[0].extra


def test_two_tone_single_block_is_not_split():
    # Half the top in each color but only one block long: not split.
    a = Box(0.30, 0.0, 0.0, 0.035, 0.0175, 0.035, S.RED)
    b = Box(0.30, 0.0175, 0.0, 0.035, 0.0175, 0.035, S.GREEN)
    res = _run([a, b], 'overhead', **{'perception.segmentation.merged_blocks': 'off'})
    assert len(res.detections) == 1 and 'split' not in res.detections[0].extra


# --------------------------------------------------------------------------- same color


@pytest.mark.parametrize('cam', sorted(CAMS))
@pytest.mark.parametrize('case', ['equal', 'offset', 'row3'])
def test_touching_blocks_of_one_color_are_cut(cam, case):
    a = Box(0.30, -0.02, 0.3, 0.035, 0.035, 0.035, S.RED)
    if case == 'equal':
        objs = [a, _flush(a, 0.035, S.RED)]
    elif case == 'offset':
        objs = [a, _flush(a, 0.035, S.RED, lateral=0.008)]
    else:
        b = _flush(a, 0.035, S.RED)
        objs = [a, b, _flush(b, 0.035, S.RED)]
    res = _run(objs, cam)
    assert not res.warnings
    assert len(res.detections) == len(objs)
    for d in res.detections:
        err, _ = _err_mm(d, objs)
        assert d.valid and d.label == 'red'
        assert err < 3.5, (case, err)
        if case != 'offset':            # an offset pair may already split on depth
            assert d.extra['split'] == 'blocks' and d.extra['split_parts'] == len(objs)
            assert d.extra['block_cut_error'] <= 0.2
        assert abs(d.length - 0.035) < 0.005 and abs(d.width - 0.035) < 0.005


def test_merged_blocks_modes():
    a = Box(0.30, 0.0, 0.3, 0.035, 0.035, 0.035, S.RED)
    objs = [a, _flush(a, 0.035, S.RED)]
    rej = _run(objs, 'oblique', **{'perception.segmentation.merged_blocks': 'reject'})
    assert len(rej.detections) == 1
    d = rej.detections[0]
    assert not d.valid and d.reject_reason == 'merged' and 'merged' in d.extra['reject_reasons']
    assert rej.warnings == ['merged_blocks']
    off = _run(objs, 'oblique', **{'perception.segmentation.merged_blocks': 'off'})
    assert len(off.detections) == 1 and off.detections[0].valid
    assert 'merged_blocks' not in off.timings


def test_objects_that_are_not_cubes_stay_whole():
    objs = [Box(0.30, -0.10, 0.2, 0.060, 0.025, 0.025, S.YELLOW),   # 2.4 : 1 bar
            Box(0.30, 0.10, -0.4, 0.050, 0.035, 0.045, S.GREEN),    # 1.4 : 1 brick
            Box(0.38, 0.00, 0.0, 0.060, 0.060, 0.050, S.RED)]       # big cube
    for cam in CAMS:
        res = _run(objs, cam)
        assert len(res.valid_detections()) == 3
        assert all('split' not in d.extra for d in res.detections)
        big = max(res.detections, key=lambda q: q.width)
        assert big.extra.get('oversize') is True and res.warnings == ['oversize_block']


def test_block_size_config_is_validated():
    for key, bad in (('perception.l1_block_size', 0.0),
                     ('perception.segmentation.color_split_min_fraction', 0.9),
                     ('perception.segmentation.merged_blocks', 'maybe')):
        with pytest.raises(ValueError):
            SegmentParams.from_cfg(_cfg(**{key: bad}))
    p = SegmentParams.from_cfg(_cfg(**{'perception.segmentation.merged_blocks': ' Reject'}))
    assert p.merged_blocks == 'reject' and p.block_size == pytest.approx(0.035)


def test_split_scene_time_bound():
    a = Box(0.26, -0.08, 0.2, 0.035, 0.035, 0.035, S.RED)
    b = Box(0.34, 0.06, -0.5, 0.035, 0.035, 0.035, S.BLUE)
    objs = [a, _flush(a, 0.035, S.GREEN), b, _flush(b, 0.035, S.BLUE),
            Sphere(0.42, -0.15, 0.03), Box(0.22, 0.15, 0.0, 0.03, 0.03, 0.03, S.YELLOW)]
    p = Perception(_cfg(), classifier=_clf())
    times = []
    for k, cam in enumerate(sorted(CAMS) * 2):
        fr = S.make_frame(objs, CAMS[cam], noise_std=0.0015, seed=k)
        t0 = time.perf_counter()
        r = p.process(fr, CAMS[cam])
        times.append(time.perf_counter() - t0)
        assert len(r.valid_detections()) == 6
    assert sorted(times)[len(times) // 2] < 0.150, times


# --------------------------------------------------------------------------- helpers


def test_pixel_classes_marks_ambiguous_and_unknown_pixels():
    clf = ColorClassifier({'a': [[0, 50, 50], [20, 255, 255]],
                           'b': [[15, 50, 50], [40, 255, 255]]}, 0.35)
    hsv = np.array([[5, 200, 200], [17, 200, 200], [30, 200, 200], [100, 200, 200],
                    [5, 10, 200]], dtype=np.uint8)
    assert clf.pixel_classes(hsv).tolist() == [0, -1, 1, -1, -1]
    assert clf.pixel_classes(np.zeros((0, 3), np.uint8)).shape == (0,)


def test_split_by_classes_seeds_and_fill():
    mask = np.zeros((30, 60), bool)
    mask[5:25, 5:55] = True
    cls = np.full(mask.shape, -1)
    cls[8:22, 8:28] = 0
    cls[8:22, 32:52] = 1
    cls[10, 40] = 0                     # a speckle of class 0 inside class 1
    up = mask.copy()
    labels, part_class = split_by_classes(mask, cls, up, 0.25, 15)
    assert sorted(part_class) == [0, 1]
    assert set(np.unique(labels[mask])) == {1, 2}          # every mask pixel assigned
    assert np.all(labels[~mask] == 0)
    left = labels[15, 10]
    assert np.all(labels[5:25, 5:28] == left) and np.all(labels[5:25, 33:55] != left)
    # One dominant class, or too small a second one: no split.
    cls2 = np.where(mask, 0, -1)
    assert split_by_classes(mask, cls2, up, 0.25, 15) is None
    cls3 = cls2.copy()
    cls3[8:12, 30:34] = 1
    assert split_by_classes(mask, cls3, up, 0.25, 15) is None
    assert split_by_classes(mask, cls, np.zeros_like(mask), 0.25, 15) is None


def _square_points(cx, cy, side, step=0.0015, yaw=0.0):
    g = np.arange(-side / 2.0, side / 2.0 + 1e-9, step)
    gx, gy = np.meshgrid(g, g)
    loc = np.column_stack([gx.ravel(), gy.ravel()])
    c, s = math.cos(yaw), math.sin(yaw)
    return loc @ np.array([[c, s], [-s, c]]) + [cx, cy]


@pytest.mark.parametrize('yaw', [0.0, 0.5, -1.2])
def test_find_block_cuts_equal_unequal_offset(yaw):
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    o = np.array([0.3, 0.05])

    def cube(t, lat, side):
        return _square_points(*(o + t * u + lat * v), side, yaw=yaw)

    for sides, lat in (((0.035, 0.035), 0.0), ((0.030, 0.040), 0.0), ((0.035, 0.035), 0.009)):
        a, b = sides
        pts = np.vstack([cube(0.0, 0.0, a), cube((a + b) / 2.0, lat, b)])
        found = find_block_cuts(pts, yaw, 0.035)
        assert found is not None
        cuts, err = found
        touch = float(o @ u) + a / 2.0
        # Cuts lie on the BLOCK_BIN_M grid; a bin across the boundary of blocks of
        # different widths belongs to neither, so allow 1.5 bins.
        assert len(cuts) == 1 and abs(cuts[0] - touch) < 0.0031 and err < 0.15
    row = np.vstack([cube(0.035 * k, 0.0, 0.035) for k in range(3)])
    cuts, _ = find_block_cuts(row, yaw, 0.035)
    assert len(cuts) == 2
    # A single cube, a 60 x 25 bar and too few points have no square cut.
    assert find_block_cuts(cube(0.0, 0.0, 0.035), yaw, 0.035) is None
    bar = np.vstack([cube(0.0, 0.0, 0.025), cube(0.0125, 0.0, 0.025), cube(0.035, 0.0, 0.025)])
    assert find_block_cuts(bar, yaw, 0.035) is None
    assert find_block_cuts(row[:10], yaw, 0.035) is None


def test_snap_to_height_step():
    rng = np.random.default_rng(0)
    top = np.r_[np.full(20, 0.032), np.full(20, 0.038)] + rng.normal(0.0, 0.0004, 40)
    assert _snap_to_step(top, 24) == 20 and _snap_to_step(top, 17) == 20
    assert _snap_to_step(top, 31) == 31                     # step beyond BLOCK_SNAP_M
    hidden = top.copy()
    hidden[17:20] = -np.inf                                 # strip hidden behind the tall one
    assert _snap_to_step(hidden, 23) == 20
    low_right = top[::-1].copy()
    low_right[20:23] = -np.inf
    assert _snap_to_step(low_right, 25) == 20
    flat = 0.035 + rng.normal(0.0, 0.0008, 40)
    assert _snap_to_step(flat, 20) == 20
    assert _snap_to_step(np.full(4, 0.03), 2) == 2


def test_find_block_cuts_uses_heights_only_at_steps():
    a = _square_points(0.0, 0.0, 0.035)
    b = _square_points(0.035, 0.0, 0.035)
    pts = np.vstack([a, b])
    plain = find_block_cuts(pts, 0.0, 0.035)
    flat = find_block_cuts(pts, 0.0, 0.035, z=np.full(len(pts), 0.035))
    assert flat[0] == plain[0]
    step = find_block_cuts(pts, 0.0, 0.035, z=np.r_[np.full(len(a), 0.035),
                                                    np.full(len(b), 0.040)])
    assert abs(step[0][0] - 0.0175) < 0.0021
    with pytest.raises(ValueError):
        find_block_cuts(pts, 0.0, 0.035, z=np.zeros(5))


def test_completed_extents_mirror_the_hidden_half():
    g = np.arange(-0.03, 0.0301, 0.0005)
    gx, gy = np.meshgrid(g, g)
    disk = np.column_stack([gx.ravel(), gy.ravel()])
    half = disk[(np.hypot(disk[:, 0], disk[:, 1]) <= 0.03) & (disk[:, 0] <= 0.0)]
    c = np.array([0.3, 0.1])
    length, width = completed_extents(half + c, c, 0.0, trim_pct=0.0)
    assert length == pytest.approx(0.06, abs=0.001) and width == pytest.approx(0.06, abs=0.001)
    # The default trim (0.5 % per side) cuts the thin tips of the half disk.
    length, width = completed_extents(half + c, c, 0.0)
    assert 0.055 < width <= length < 0.0605
    sq = _square_points(0.3, 0.1, 0.04)
    length, width = completed_extents(sq, np.array([0.3, 0.1]), 0.0)
    assert length == pytest.approx(0.04, abs=0.001) and width == pytest.approx(0.04, abs=0.001)


def test_width_estimate_for_round_and_box_objects():
    objs = [Sphere(0.32, 0.0, 0.031, S.RED), Box(0.25, -0.12, 0.4, 0.035, 0.035, 0.035, S.BLUE)]
    res = _run(objs, 'oblique', classifier=None)
    assert len(res.detections) == 2
    for d in res.detections:
        _, b = _err_mm(d, objs)
        assert d.extra['width_est'] >= d.width and d.extra['length_est'] >= d.length
        if isinstance(b, Sphere):
            assert d.width < 0.050                                  # far half hidden
            assert abs(d.extra['width_est'] - 0.062) < 0.005
            assert abs(d.extra['length_est'] - 0.062) < 0.005
        else:
            assert abs(d.extra['width_est'] - d.width) < 0.003
            assert abs(d.extra['width_est'] - 0.035) < 0.004


def test_annotate_neighbors_and_mark_invalid():
    def det(x, y, side=0.03):
        pts = _square_points(x, y, side, step=0.002).astype(np.float32)
        return Detection('red', 1.0, center_base=np.array([x, y, 0.03]), length=side,
                         width=side, extra={'_points_xy': pts})

    a, b, far = det(0.30, 0.0), det(0.33, 0.0), det(0.30, 0.20)
    nopts = Detection('red', 1.0, center_base=np.array([0.31, 0.0, 0.03]))
    annotate_neighbors([a, b, far, nopts], max_points=50)
    assert '_neighbor_xy' in a.extra and '_neighbor_xy' in b.extra
    assert '_neighbor_xy' not in far.extra and '_neighbor_xy' not in nopts.extra
    nb = a.extra['_neighbor_xy']
    assert nb.shape[1] == 2 and 0 < nb.shape[0] <= 50 and float(nb[:, 0].min()) > 0.31
    mark_invalid(a, 'merged')
    mark_invalid(a, 'occluded')
    assert not a.valid and a.reject_reason == 'occluded'
    assert a.extra['reject_reasons'] == ['occluded', 'merged']

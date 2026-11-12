"""Split barrier step scaled with the measured depth noise (far / noisy camera, L2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.camera.utils import capture_averaged
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.types import CameraIntrinsics
from cube_task.perception.pipeline import Perception
from cube_task.perception.segment import SegmentParams, split_jump, SPLIT_JUMP_MAX_M
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from test_task_fakes import still_wrist_camera

ELONGATED = ('carrot', 'cucumber', 'banana', 'corn')


def test_split_jump_follows_the_table_rms():
    p = SegmentParams()
    assert p.split_jump_noise_k == 6.0 and p.split_height_jump_m == 0.008
    assert split_jump(p, None) == 0.008 and split_jump(p, 0.0) == 0.008
    assert split_jump(p, float('nan')) == 0.008 and split_jump(p, 'x') == 0.008
    assert split_jump(p, 0.0009) == 0.008                       # sim rig: 7.6 mm < 8 mm
    assert split_jump(p, 0.002) == pytest.approx(6.0 * math.sqrt(2.0) * 0.002)
    assert split_jump(p, 0.01) == SPLIT_JUMP_MAX_M
    p.split_jump_noise_k = 0.0
    assert split_jump(p, 0.002) == 0.008
    cfg = load_config(use_overlays=False)
    cfg.set('perception.segmentation.split_jump_noise_k', 4.0)
    assert SegmentParams.from_cfg(cfg).split_jump_noise_k == 4.0
    for bad in (-1.0, 25.0, float('nan')):
        cfg.set('perception.segmentation.split_jump_noise_k', bad)
        with pytest.raises(ValueError, match='split_jump_noise_k'):
            SegmentParams.from_cfg(cfg)


def _fragments(seeds, back=0.0, fx=380.0, noise_mult=1.0, k=None):
    """Return (extra valid fragments of elongated objects, objects found, objects, jumps)."""
    base = load_config(use_overlays=False)
    extra, found, total, jumps = 0, 0, 0, []
    for s in seeds:
        world, info = make_scenario('L2', s)
        cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root,
                     base.sources)
        if k is not None:
            cfg.set('perception.segmentation.split_jump_noise_k', k)
        T = np.array(info['T_base_camera_observe'], dtype=float)
        T[:3, 3] -= back * T[:3, 2]                  # move the camera back along its axis
        intr = CameraIntrinsics(fx, fx, 319.5, 239.5, 640, 480)
        cam = still_wrist_camera(world, SimClock(), intr, T,
                                 noise_std=0.0008 * noise_mult, dropout=0.002, seed=s)
        cam.start()
        res = Perception(cfg).process(capture_averaged(cam, 3), T)
        assert res.table is not None, res.warnings
        jumps.append(split_jump(SegmentParams.from_cfg(cfg), res.table.rms))
        objs = world.remaining(info['pick_polygon'])
        per = {}
        for d in res.valid_detections():
            c = np.asarray(d.center_base[:2])
            o = min(objs, key=lambda q: float(np.hypot(*(q.T_base_obj[:2, 3] - c))))
            per.setdefault(o.name, []).append(d)
        extra += sum(len(v) - 1 for o, v in per.items() if o.split('_')[0] in ELONGATED)
        found += len(per)
        total += len(objs)
    return extra, found, total, jumps


@pytest.mark.parametrize('case', ['far', 'noisy'])
def test_elongated_l2_objects_do_not_fragment_on_a_far_or_noisy_camera(case):
    # far: the camera 0.35 m farther back (objects at 1.0-1.1 m, fx 560 keeps the view);
    # noisy: twice the sim depth noise at the default distance.
    kw = dict(back=0.35, fx=560.0) if case == 'far' else dict(noise_mult=2.0)
    seeds = range(8)
    extra, found, total, jumps = _fragments(seeds, **kw)
    assert extra == 0 and found == total
    assert min(jumps) > 0.012                    # table rms 1.8-2 mm -> step 15-17 mm
    old_extra, old_found, _, _ = _fragments(seeds, k=0.0, **kw)    # fixed 8 mm step
    assert old_extra >= 5 and old_found == total

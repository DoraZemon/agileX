"""
Train a KNN classifier from simulated scenes with ground-truth labels.

The on-site tool (tools/teach_classes) does the same with the real camera, where
the operator supplies the label. In simulation the label of each detection is
the label of the nearest ground-truth object.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

_LOG = logging.getLogger('cube_task.sim.teach')


def collect_sim_samples(seeds: Iterable[int], mount: str = 'eye_to_hand', task: str = 'L2',
                        match_dist: float = 0.02, layout: str = 'A'):
    """Return (X, y, feature_names) from perception on simulated scenes."""
    from ..camera.factory import make_camera
    from ..core.clock import SimClock
    from ..core.config import Config, deep_merge, load_config
    from ..core.extrinsics import camera_pose_in_base, extrinsics_from_cfg
    from ..perception.features import feature_vector
    from ..perception.pipeline import Perception
    from ..robot.factory import make_arm
    from .scenarios import make_scenario, scenario_config_overlay

    base = load_config(use_overlays=False)
    X, y, names = [], [], None
    for s in seeds:
        world, info = make_scenario(task, int(s), mount=mount,
                                    difficulty='hard' if int(s) % 2 else 'normal',
                                    layout=layout)
        cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root,
                     base.sources)
        clock = SimClock()
        arm = make_arm(cfg, clock, sim_world=world)
        arm.connect()
        arm.enable()
        arm.move_joints(cfg.get('robot.poses.observe_joints'))
        cam = make_camera(cfg, clock, sim_world=world, arm=arm)
        cam.start()
        frame = cam.get_frame()
        cam.stop()
        if frame is None:
            continue
        T_bf = arm.get_flange_pose().value if info['mount'] == 'eye_in_hand' else None
        res = Perception(cfg).process(frame, camera_pose_in_base(extrinsics_from_cfg(cfg), T_bf))
        for d in res.valid_detections():
            c = np.asarray(d.center_base[:2], dtype=float)
            best = min(world.objects, key=lambda o: float(np.hypot(*(o.T_base_obj[:2, 3] - c))))
            if float(np.hypot(*(best.T_base_obj[:2, 3] - c))) > match_dist:
                continue
            v, names = feature_vector(frame, d)
            if np.all(np.isfinite(v)):
                X.append(v)
                y.append(best.label)
    return np.array(X), y, names


def train_knn_from_sim(out: Path, mount: str = 'eye_to_hand', n_scenes: int = 40,
                       first_seed: int = 100, k: int = 3,
                       max_distance: float = 3.0,
                       layout: str = 'C') -> Tuple[Path, Optional[float]]:
    """
    Train and save a KNN model; return (path, leave-one-out accuracy).

    The features do not depend on the table layout; 'C' lays out all L2 models.
    """
    from ..perception.classify_knn import KnnClassifier
    X, y, names = collect_sim_samples(range(first_seed, first_seed + n_scenes), mount=mount,
                                      layout=layout)
    if len(y) == 0:
        raise RuntimeError('no simulated samples collected')
    knn = KnnClassifier(k=k, max_distance=max_distance)
    knn.fit(X, y, names)
    acc = knn.leave_one_out().get('accuracy')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    knn.save(out)
    _LOG.info('sim KNN: %d samples, %d classes, leave-one-out accuracy %s -> %s',
              len(y), len(set(y)), acc, out)
    return out, acc

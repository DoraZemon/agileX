"""
L1 perception backed by piper_vision.L1Detector, as a drop-in for Perception.

Selected by perception.backend: l1_new (see perception.factory.make_perception).
Implements the same process(frame, T_base_camera, zones) -> PerceptionResult
contract the task runner uses, and fills the Detection fields the grasp planner
reads (table_z, _points_base, _points_xy, cap_center_xy, _neighbor_xy).

Only the red / green / yellow cubes are returned as detections; the target
cards are reported in result.warnings and result.table is the fitted plane.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..core.types import Detection, PerceptionResult, TablePlane
from ..core.zones import parse_zones
from .geometry import points_in_zones
from .pipeline import annotate_neighbors

LOG = logging.getLogger('cube_task.perception.l1_new')

# ws/src/piper_vision, for runs from the source tree without a colcon install.
_PV_SRC = Path(__file__).resolve().parents[3] / 'piper_vision'


def _import_piper_vision():
    try:
        import piper_vision  # noqa: F401
    except ImportError:
        if not _PV_SRC.is_dir():
            raise ImportError('piper_vision not installed and %s not found' % _PV_SRC)
        sys.path.insert(0, str(_PV_SRC))
    from piper_vision.camera import Intrinsics
    from piper_vision.l1 import L1Detector, L1Params
    return Intrinsics, L1Detector, L1Params


class L1NewPerception:
    """L1 detection through piper_vision, shaped like cube_task.perception.Perception."""

    def __init__(self, cfg: Any):
        Intrinsics, L1Detector, L1Params = _import_piper_vision()
        self.cfg = cfg
        d = dict(cfg.get('perception.l1_new', {}) or {})
        if 'cube_size' not in d:
            d['cube_size'] = float(cfg.get('perception.l1_block_size', 0.05))
        if 'roi_polygon' not in d:
            d['roi_polygon'] = cfg.get('perception.roi.pick_polygon', None)
        if 'colors' not in d:
            classes = cfg.get('perception.classifier.color.classes', None) or {}
            d['colors'] = {k: v for k, v in classes.items() if k in ('red', 'green', 'yellow')}
        self.params = L1Params.from_dict(d)
        self.detector = L1Detector(self.params)
        self._Intrinsics = Intrinsics
        self.min_score = float(cfg.get('perception.classifier.min_score', 0.55))
        self.default_zones = parse_zones(cfg)

    def process(self, frame: Any, T_base_camera: Optional[np.ndarray],
                zones: Any = None) -> PerceptionResult:
        """Run L1 detection. Never raises; problems go into result.warnings."""
        res = PerceptionResult(detections=[], frame_stamp=float(getattr(frame, 'stamp', 0.0)))
        try:
            zones_used = self.default_zones if zones is None else zones
            return self._process(frame, T_base_camera, zones_used, res)
        except Exception as exc:  # noqa: B902 - a detection bug must not stop the round
            LOG.exception('l1_new failed')
            res.warnings.append('error:%s' % type(exc).__name__)
            return res

    def _process(self, frame, T_base_camera, zones, res) -> PerceptionResult:
        color = getattr(frame, 'color', None)
        depth = getattr(frame, 'depth', None)
        intr = getattr(frame, 'intrinsics', None)
        if color is None or intr is None:
            res.warnings.append('no_color')
            return res
        if T_base_camera is None:
            res.warnings.append('no_extrinsics')
            return res
        T = np.asarray(T_base_camera, dtype=float)
        if T.shape != (4, 4):
            res.warnings.append('bad_extrinsics')
            return res
        pv_intr = self._Intrinsics(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height,
                                   np.asarray(getattr(intr, 'dist', []), dtype=float))
        out = self.detector.detect(color, depth, pv_intr, T)
        res.warnings.extend(out.warnings)
        if out.table_normal is not None and out.table_z is not None:
            n = np.asarray(out.table_normal, dtype=float)
            res.table = TablePlane(normal=n, d=float(-out.table_z * n[2]), frame='base')
        for c in out.cubes:
            det = self._to_detection(c, out.table_z)
            if zones and points_in_zones(det.center_base[None, :2], zones)[0]:
                det.valid = False                       # already placed
                det.reject_reason = 'touches_zone'
            elif det.score < self.min_score:
                det.valid = False
                det.reject_reason = 'low_score'
            res.detections.append(det)
        annotate_neighbors(res.detections)
        for z in out.zones:
            res.warnings.append('zone_seen:%s:[%.3f,%.3f]' % (z.name, z.center_base[0],
                                                              z.center_base[1]))
        return res

    def _to_detection(self, c, table_z) -> Detection:
        pts = self._footprint(c, table_z)
        extra: Dict[str, Any] = {
            'table_z': None if table_z is None else float(table_z),
            'source': c.source,
            'edge_m': float(c.edge),
        }
        if pts is not None:
            extra['_points_base'] = pts
            extra['_points_xy'] = pts[:, :2].astype(np.float32)
            extra['cap_center_xy'] = [float(c.center_base[0]), float(c.center_base[1])]
        det = Detection(
            label=c.label, score=float(c.score), score_kind='color',
            bbox=tuple(int(v) for v in c.bbox),
            center_px=(float(c.center_px[0]), float(c.center_px[1])),
            mask=c.mask, center_base=np.asarray(c.center_base, dtype=float),
            top_z=float(c.center_base[2]), height=float(c.height),
            yaw=float(c.yaw), length=float(c.edge), width=float(c.edge),
            n_points=int(c.n_points), valid=True, extra=extra)
        if c.warnings:
            extra['warnings'] = list(c.warnings)       # already reflected in the score
        return det

    @staticmethod
    def _footprint(c, table_z, step: float = 0.005) -> Optional[np.ndarray]:
        """
        Return a grid of base-frame points covering the cube's footprint (Nx3).

        The grasp planner checks neighbor points inside the finger band, so
        the whole face must be covered; corners alone fall outside the band.
        """
        if c.center_base is None:
            return None
        half = float(c.edge) / 2.0
        n = max(2, int(round(2 * half / step)) + 1)
        g = np.linspace(-half, half, n)
        gx, gy = np.meshgrid(g, g)
        local = np.column_stack((gx.ravel(), gy.ravel()))
        ca, sa = math.cos(float(c.yaw)), math.sin(float(c.yaw))
        xy = local @ np.array([[ca, -sa], [sa, ca]]).T + np.asarray(c.center_base[:2], float)
        return np.column_stack((xy, np.full(len(xy), float(c.center_base[2])))).astype(
            np.float32)

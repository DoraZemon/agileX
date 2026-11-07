"""
Debug overlay of a PerceptionResult on the color image.

Draws instance masks (alpha-blended), bounding boxes, label and score, the
grasp center and yaw axis, invalid detections in gray with their reject
reason, the pick polygon and zone outlines projected from the base frame, and
a status line. Never raises for missing data; returns a BGR uint8 image.
"""
from __future__ import annotations

import logging
import math
import zlib
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .geometry import as_zone_list, densify_polyline, project_points
from ..core.types import Detection, Frame, PerceptionResult

LOG = logging.getLogger('cube_task.perception.visualize')

NAMED_COLORS = {
    'red': (40, 40, 230), 'green': (60, 200, 60), 'blue': (230, 120, 30),
    'yellow': (30, 220, 230), 'orange': (20, 140, 255), 'purple': (200, 60, 160),
    'white': (235, 235, 235), 'black': (60, 60, 60),
}
INVALID_COLOR = (150, 150, 150)
UNKNOWN_COLOR = (255, 255, 255)
POLY_COLOR = (0, 255, 255)
ZONE_COLOR = (255, 0, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_color(label: str) -> Tuple[int, int, int]:
    """Return a stable BGR color for a label (named colors for color labels)."""
    if label in NAMED_COLORS:
        return NAMED_COLORS[label]
    if label == 'unknown':
        return UNKNOWN_COLOR
    hue = zlib.crc32(label.encode('utf-8')) % 180
    hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _text(img: np.ndarray, s: str, org: Tuple[int, int], color: Tuple[int, int, int],
          scale: float = 0.45) -> None:
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x, y), _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)


def _draw_base_polygon(img: np.ndarray, xy: np.ndarray, z: float, T_base_camera: np.ndarray,
                       intrinsics: Any, color: Tuple[int, int, int]) -> None:
    pts = densify_polyline(np.asarray(xy, dtype=float), step=0.01, closed=True)
    p3 = np.column_stack([pts, np.full(pts.shape[0], float(z))])
    uv = project_points(p3, T_base_camera, intrinsics)
    ok = np.all(np.isfinite(uv), axis=1) & (np.abs(uv) < 1e5).all(axis=1)
    # Draw only consecutive visible pieces.
    run = []
    for i in range(uv.shape[0]):
        if ok[i]:
            run.append(uv[i])
        elif len(run) > 1:
            cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, color, 1,
                          cv2.LINE_AA)
            run = []
        else:
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, color, 1,
                      cv2.LINE_AA)


def _draw_detection(img: np.ndarray, det: Detection, T_base_camera: Optional[np.ndarray],
                    intrinsics: Any) -> None:
    color = label_color(det.label) if det.valid else INVALID_COLOR
    x0, y0, x1, y1 = [int(v) for v in det.bbox]
    if det.mask is not None and det.mask.shape == img.shape[:2] and x1 > x0 and y1 > y0:
        m = det.mask[y0:y1, x0:x1]
        roi = img[y0:y1, x0:x1]
        blend = (0.55 * roi[m] + 0.45 * np.array(color, dtype=np.float64))
        roi[m] = blend.astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 1)
    if det.valid:
        text = '%s %.2f' % (det.label, det.score)
    else:
        text = '%s [%s]' % (det.label, det.reject_reason or 'invalid')
    _text(img, text, (x0, max(12, y0 - 4)), color)
    cu, cv_ = det.center_px
    if math.isfinite(cu) and math.isfinite(cv_):
        cv2.drawMarker(img, (int(round(cu)), int(round(cv_))), color, cv2.MARKER_CROSS, 12, 2)
    if (T_base_camera is not None and intrinsics is not None and det.center_base is not None
            and det.yaw is not None and det.length is not None):
        c = np.asarray(det.center_base, dtype=float)
        half = 0.5 * float(det.length)
        a = np.array([math.cos(det.yaw), math.sin(det.yaw), 0.0]) * half
        uv = project_points(np.vstack([c - a, c + a]), T_base_camera, intrinsics)
        if np.all(np.isfinite(uv)) and np.all(np.abs(uv) < 1e5):
            p, q = np.round(uv).astype(int)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2, cv2.LINE_AA)
        if det.valid and det.height is not None:
            _text(img, 'h%.0f %.0fx%.0fmm' % (det.height * 1e3, det.length * 1e3,
                                              (det.width or 0.0) * 1e3),
                  (x0, min(img.shape[0] - 4, y1 + 12)), color, 0.38)


def draw_result(frame: Frame, result: PerceptionResult,
                T_base_camera: Optional[np.ndarray] = None, zones: Any = None,
                pick_polygon: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Render the debug overlay and return a new BGR uint8 image.

    frame: the processed frame (its color image is copied, not modified).
    T_base_camera: 4x4 camera pose in the base frame, needed for the yaw axis
    and the projected pick polygon (base XY, m) / zone outlines (dict or
    iterable of core Zone). Without a color image a black canvas of the
    intrinsics size is used.
    """
    intr = frame.intrinsics
    color = frame.color
    if color is not None and color.ndim == 3 and color.shape[2] == 3:
        img = np.ascontiguousarray(color, dtype=np.uint8).copy()
    else:
        img = np.zeros((int(intr.height), int(intr.width), 3), dtype=np.uint8)
    try:
        table_z = 0.0
        if T_base_camera is not None:
            if pick_polygon is not None:
                pz = table_z
                if result.table is not None:
                    c = np.asarray(pick_polygon, dtype=float).mean(axis=0)
                    pz = result.table.z_at(float(c[0]), float(c[1]))
                _draw_base_polygon(img, pick_polygon, pz, T_base_camera, intr, POLY_COLOR)
            for zone in as_zone_list(zones):
                _draw_base_polygon(img, zone.corners_xy(), float(zone.center[2]),
                                   T_base_camera, intr, ZONE_COLOR)
                uv = project_points(np.asarray(zone.center, dtype=float), T_base_camera, intr)
                if np.all(np.isfinite(uv)) and np.all(np.abs(uv) < 1e5):
                    _text(img, zone.name, (int(uv[0, 0]), int(uv[0, 1])), ZONE_COLOR, 0.4)
        dets = sorted(result.detections, key=lambda d: d.valid)
        for det in dets:
            _draw_detection(img, det, T_base_camera, intr)
        n_valid = len(result.valid_detections())
        status = '%d valid / %d' % (n_valid, len(result.detections))
        if result.table is not None:
            status += ' | table rms %.1fmm in %.0f%%' % (result.table.rms * 1e3,
                                                         result.table.inlier_ratio * 100.0)
        if 'total' in result.timings:
            status += ' | %.0f ms' % result.timings['total']
        if result.warnings:
            status += ' | ' + ','.join(result.warnings[:4])
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        _text(img, status, (4, 13), (0, 255, 0) if not result.warnings else (0, 165, 255))
    except Exception:  # drawing must never break perception
        LOG.exception('debug drawing failed')
    return img

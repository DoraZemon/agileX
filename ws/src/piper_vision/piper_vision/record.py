"""
Record the ROS 2 RGB-D camera to disk.

Two formats, both written without importing cube_task:

- frames (default): the cube_task capture format (NNNNNN_color.png /
  _depth.png in mm / _depth.npy in m / _meta.json), so the directories can be
  fed straight to ``python3 -m cube_task.tools.yolo.make_dataset`` and
  ``teach_classes --from-dir``. The session name is stored in meta extra and
  is also the directory name, which is how those tools read the class.
- bag: a rosbag2 (sqlite3) of the color, depth and camera_info topics.

The arm is only read, never moved, so recording during drag-teach is safe.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .camera import CameraConfig, Intrinsics
from .frames import list_frames
from .live import ArmPoseReader, RosRgbd

LOG = logging.getLogger('piper_vision.record')

FORMAT = 'cube_task_frame_v1'
DEFAULT_ROOT = '~/vision_data'
#: session names that the training tools treat as "no class".
GENERIC_SESSIONS = ('', 'capture', 'view', 'teach_classes', 'session', 'unknown', 'bag')
_SAFE = re.compile(r'[^\w\-]+')


def safe_name(text: str, fallback: str) -> str:
    """Return text usable as a file or directory name."""
    name = _SAFE.sub('_', str(text).strip()).strip('_')
    return name or fallback


def session_dir(session: str, root: str = DEFAULT_ROOT,
                date: Optional[str] = None) -> Path:
    """Return <root>/<YYYYmmdd>_<session>, matching the cube_task capture layout."""
    date = date or datetime.date.today().strftime('%Y%m%d')
    return Path(root).expanduser() / ('%s_%s' % (date, safe_name(session, 'capture')))


def next_index(directory: str) -> int:
    """Return the first unused frame index in a capture directory."""
    try:
        idx = list_frames(directory)
    except ValueError:
        return 0
    return idx[-1] + 1 if idx else 0


def _atomic_imwrite(path: Path, img: np.ndarray) -> None:
    tmp = path.with_name('.' + path.stem + '.tmp' + path.suffix)
    if not cv2.imwrite(str(tmp), np.ascontiguousarray(img)):
        raise IOError('cv2.imwrite failed for %s' % tmp)
    os.replace(str(tmp), str(path))


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(str(tmp), str(path))


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    """Convert depth in meters to uint16 millimeters (invalid or non-positive -> 0)."""
    d = np.asarray(depth, dtype=np.float64)
    mm = np.zeros(d.shape, dtype=np.uint16)
    valid = np.isfinite(d) & (d > 0.0)
    mm[valid] = np.clip(np.rint(d[valid] * 1000.0), 1, 65535).astype(np.uint16)
    return mm


def save_frame(directory: str, idx: int, color: np.ndarray, depth: Optional[np.ndarray],
               intrinsics: Intrinsics, session: str, mount: str,
               T_base_flange: Optional[np.ndarray] = None,
               stamp: Optional[float] = None) -> Path:
    """
    Write one frame in the cube_task capture format and return the meta path.

    The meta file is written last, so a frame only shows up in list_frames once
    every image file is complete. depth is float32 meters aligned to color.
    """
    if isinstance(idx, bool) or int(idx) != idx or int(idx) < 0:
        raise ValueError('frame index must be an integer >= 0, got %r' % (idx,))
    color = np.asarray(color)
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError('color must be HxWx3 uint8, got %s %s' % (color.shape, color.dtype))
    h, w = color.shape[:2]
    if depth is not None and np.asarray(depth).shape != (h, w):
        raise ValueError('depth shape %s does not match color %dx%d'
                         % (np.asarray(depth).shape, w, h))
    if (intrinsics.width, intrinsics.height) != (w, h):
        # Same camera at another resolution (config intrinsics vs. the published image).
        intrinsics = intrinsics.scaled(w, h)
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    stem = '%06d' % int(idx)
    _atomic_imwrite(d / ('%s_color.png' % stem), color)
    depth_name = None
    npy_name = None
    if depth is not None:
        depth_name = '%s_depth.png' % stem
        _atomic_imwrite(d / depth_name, depth_to_mm(depth))
        npy_name = '%s_depth.npy' % stem
        clean = np.where(np.isfinite(depth) & (np.asarray(depth) > 0.0), depth, 0.0)
        tmp = d / ('.' + npy_name + '.tmp')
        with open(tmp, 'wb') as fh:
            np.save(fh, clean.astype(np.float32))
        os.replace(str(tmp), str(d / npy_name))
    extra: Dict[str, Any] = {'session': str(session), 'mount': str(mount), 'task': None}
    if T_base_flange is not None:
        extra['T_base_flange'] = np.asarray(T_base_flange, dtype=float).tolist()
    meta = {
        'format': FORMAT,
        'index': int(idx),
        'color_file': '%s_color.png' % stem,
        'depth_file': depth_name,
        'depth_npy_file': npy_name,
        'depth_unit': 'mm',
        'depth_scale': 0.001,
        'width': int(w),
        'height': int(h),
        'intrinsics': intrinsics.to_dict(),
        'stamp': float(time.monotonic() if stamp is None else stamp),
        'frame_id': 'camera_color_optical_frame',
        'source': 'ros',
        'wall': datetime.datetime.now().isoformat(timespec='milliseconds'),
        'meta': {},
        'extra': extra,
    }
    meta_path = d / ('%s_meta.json' % stem)
    _atomic_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return meta_path


def depth_preview(depth: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Return a BGR view of a depth image (m), or a black image when there is none."""
    h, w = shape
    if depth is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    view = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], (2, 98))
        span = max(float(hi - lo), 1e-3)
        view[valid] = np.clip((depth[valid] - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(view, cv2.COLORMAP_TURBO)


def preview(color: np.ndarray, depth: Optional[np.ndarray], saved: int,
            session: str) -> np.ndarray:
    """Return color | depth side by side with a status line."""
    view = np.hstack((color, depth_preview(depth, color.shape[:2])))
    text = '%s  saved %d   s/space save, a auto, q quit' % (session, saved)
    cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return view


class _Pose:
    """Flange pose read in a background thread; None when no arm is connected."""

    def __init__(self, reader: Optional[ArmPoseReader]):
        self.reader = reader
        self._lock = threading.Lock()
        self._T: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._thread = None
        if reader is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                T = self.reader.flange(timeout=1.0)
            except Exception as exc:  # noqa: BLE001 - keep the last good pose
                LOG.debug('flange read failed: %s', exc)
                continue
            with self._lock:
                self._T = T

    def latest(self) -> Optional[np.ndarray]:
        """Return the most recent T_base_flange, or None."""
        with self._lock:
            return None if self._T is None else np.array(self._T)

    def close(self) -> None:
        """Stop the reader thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def record_frames(cam: CameraConfig, out: Path, session: str, count: int = 0,
                  interval: float = 1.0, headless: bool = False, timeout: float = 5.0,
                  with_arm: bool = False, can: str = 'can0') -> int:
    """
    Save frames from the ROS camera until count frames (0 = until quit).

    Returns the number of frames saved. Raises when no frame arrives.
    """
    rgbd = RosRgbd(cam)
    arm = None
    pose = None
    try:
        color, depth, intr = rgbd.get(timeout=timeout)
        if with_arm:
            arm = ArmPoseReader(channel=can)
        pose = _Pose(arm)
        idx = next_index(str(out))
        saved = 0
        auto = count > 0
        last = 0.0
        print('recording %r to %s (mount %s, %s)'
              % (session, out, cam.mount, 'headless' if headless else 'window'))
        while count <= 0 or saved < count:
            color, depth, intr = rgbd.get(timeout=timeout)
            now = time.monotonic()
            key = _wait_key(preview(color, depth, saved, session), headless,
                            0.05 if auto else 0.2)
            if key in ('q', 'esc'):
                break
            if key == 'a' and count <= 0:
                auto = not auto
                print('auto-save %s (every %.2f s)' % ('on' if auto else 'off', interval))
            due = auto and now - last >= interval
            if key in ('s', ' ') or due:
                save_frame(str(out), idx, color, depth, intr, session, cam.mount,
                           pose.latest())
                print('saved %06d (%s)'
                      % (idx, 'with depth' if depth is not None else 'color only'))
                idx += 1
                saved += 1
                last = now
        return saved
    finally:
        if pose is not None:
            pose.close()
        if arm is not None:
            arm.close()
        rgbd.close()
        if not headless:
            cv2.destroyAllWindows()


def _wait_key(view: np.ndarray, headless: bool, seconds: float) -> str:
    if headless:
        time.sleep(seconds)
        return ''
    cv2.imshow('piper_vision record', view)
    k = cv2.waitKey(max(1, int(seconds * 1000))) & 0xFF
    return {ord('q'): 'q', 27: 'esc', ord('s'): 's', ord(' '): ' ', ord('a'): 'a'}.get(k, '')


def record_bag(cam: CameraConfig, out: Path, duration: float = 0.0,
               storage: str = 'sqlite3') -> int:
    """
    Record the camera topics to a rosbag2 bag; return the message count.

    duration 0 records until Ctrl-C. Writes through rosbag2_py, so no shelling
    out and no dependency on the ros2 CLI. The bag can be replayed with
    ``ros2 bag play`` and then recorded as frames.
    """
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import CameraInfo, Image

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(out), storage_id=storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {cam.topics['color']: (Image, 'sensor_msgs/msg/Image'),
             cam.topics['depth']: (Image, 'sensor_msgs/msg/Image'),
             cam.topics['info']: (CameraInfo, 'sensor_msgs/msg/CameraInfo')}
    for topic, (_, type_name) in types.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=type_name, serialization_format='cdr'))
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node('piper_vision_record')
    count = {'n': 0}

    def on_msg(msg, topic) -> None:
        writer.write(topic, serialize_message(msg), node.get_clock().now().nanoseconds)
        count['n'] += 1

    for topic, (cls, _) in types.items():
        node.create_subscription(cls, topic, lambda m, t=topic: on_msg(m, t),
                                 qos_profile_sensor_data)
    print('recording %s -> %s (Ctrl-C to stop)' % (', '.join(types), out))
    t_end = time.monotonic() + duration if duration > 0 else None
    try:
        while t_end is None or time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    print('bag: %d messages in %s' % (count['n'], out))
    return count['n']

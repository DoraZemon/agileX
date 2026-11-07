"""
ROS 2 RGB-D camera source (Orbbec 大白 / OrbbecSDK_ROS2, RealSense) and message helpers.

rclpy, message_filters and sensor_msgs are imported lazily in start(), so this
module imports (and its pure helpers work on any object with the message
fields) without ROS.

Driver notes:
- Orbbec: ``ros2 launch orbbec_camera dabai.launch.py depth_registration:=true``
  publishes /camera/color/image_raw (rgb8), /camera/depth/image_raw (16UC1, mm,
  aligned to color) and /camera/color/camera_info.
- RealSense: ``ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true``;
  use /camera/camera/aligned_depth_to_color/image_raw.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.extrinsics import validate_transform
from ..core.interfaces import CameraError, CameraSource, Clock
from ..core.types import CameraIntrinsics, Frame

_LOG = logging.getLogger('cube_task.camera.ros')

_BYTES_PER_PIXEL = {'rgb8': 3, 'bgr8': 3, '8uc3': 3, 'rgba8': 4, 'bgra8': 4,
                    'mono8': 1, '8uc1': 1, 'yuv422': 2, 'uyvy': 2,
                    'yuv422_yuy2': 2, 'yuyv': 2, 'yuy2': 2}
_YUYV = ('yuv422_yuy2', 'yuyv', 'yuy2')
SUPPORTED_COLOR_ENCODINGS = tuple(sorted(_BYTES_PER_PIXEL))
SUPPORTED_DEPTH_ENCODINGS = ('16uc1', 'mono16', '32fc1')
SYNC_MODES = ('header', 'receipt')
CLOCK_OFFSET_S = 1.0            # header stamp offset above which the clocks differ
_WARNED: set = set()            # camera_info_to_intrinsics: warnings logged once


def _stamp_to_sec(stamp: Any) -> float:
    return float(getattr(stamp, 'sec', 0)) + float(getattr(stamp, 'nanosec', 0)) * 1e-9


def _image_view(msg: Any, channels: int, dtype: str) -> np.ndarray:
    """Return an HxWxC (or HxW for C=1) view of msg.data honoring step and endianness."""
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        raise ValueError('image has no size (%dx%d)' % (w, h))
    dt = np.dtype(dtype)
    if dt.itemsize > 1:
        dt = dt.newbyteorder('>' if bool(getattr(msg, 'is_bigendian', False)) else '<')
    row_bytes = w * channels * dt.itemsize
    step = int(getattr(msg, 'step', 0) or 0)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == 0 and buf.size == h * row_bytes:
        step = row_bytes
    if step < row_bytes:
        raise ValueError('image step %d < width * bytes per pixel = %d' % (step, row_bytes))
    need = step * (h - 1) + row_bytes
    if buf.size < need:
        raise ValueError('image data truncated: %d bytes, need %d (%dx%d, step %d)'
                         % (buf.size, need, w, h, step))
    shape: Tuple[int, ...] = (h, w) if channels == 1 else (h, w, channels)
    strides: Tuple[int, ...] = ((step, dt.itemsize) if channels == 1
                                else (step, channels * dt.itemsize, dt.itemsize))
    return np.ndarray(shape=shape, dtype=dt, buffer=buf, offset=0, strides=strides)


def image_msg_to_bgr(msg: Any, yuv422_order: str = 'uyvy') -> np.ndarray:
    """
    Convert a sensor_msgs/Image color message into an HxWx3 uint8 BGR array (a copy).

    Encodings: rgb8, bgr8 (8UC3), rgba8, bgra8, mono8 (8UC1), yuv422_yuy2 /
    yuyv / yuy2 (YUYV byte order) and yuv422 / uyvy. The ROS standard
    (sensor_msgs image_encodings.hpp, cv_bridge) defines "yuv422" as UYVY;
    pass yuv422_order='yuyv' for a driver that mislabels YUYV data as
    "yuv422". Row padding (step) is respected. Raises ValueError for other
    encodings or malformed messages.
    """
    import cv2
    enc = str(msg.encoding).strip().lower()
    if enc not in _BYTES_PER_PIXEL:
        supported = ', '.join(SUPPORTED_COLOR_ENCODINGS)
        raise ValueError('unsupported color encoding %r (supported: %s). Configure the '
                         'driver for rgb8 / bgr8.' % (msg.encoding, supported))
    ch = _BYTES_PER_PIXEL[enc]
    img = _image_view(msg, ch, 'u1')
    if enc in ('bgr8', '8uc3'):
        return np.ascontiguousarray(img).copy()
    if enc == 'rgb8':
        return np.ascontiguousarray(img[:, :, ::-1])
    if enc == 'rgba8':
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGBA2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGRA2BGR)
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
    if int(msg.width) % 2:
        raise ValueError('YUV 4:2:2 image width must be even, got %d' % int(msg.width))
    if enc in _YUYV:
        code = cv2.COLOR_YUV2BGR_YUYV
    elif enc == 'uyvy':
        code = cv2.COLOR_YUV2BGR_UYVY
    else:  # 'yuv422'
        order = str(yuv422_order).strip().lower()
        if order not in ('uyvy', 'yuyv'):
            raise ValueError('yuv422_order must be uyvy or yuyv, got %r' % (yuv422_order,))
        code = cv2.COLOR_YUV2BGR_UYVY if order == 'uyvy' else cv2.COLOR_YUV2BGR_YUYV
    return cv2.cvtColor(np.ascontiguousarray(img), code)


#: Name used in docs/CONTRACTS.md.
image_msg_to_numpy = image_msg_to_bgr


def check_depth_scale(depth_scale: Any) -> float:
    """Return depth_scale (m per 16-bit unit) as float, or raise ValueError if implausible."""
    try:
        s = float(depth_scale)
    except (TypeError, ValueError):
        raise ValueError('depth_scale must be a number (m per unit), got %r' % (depth_scale,))
    if not math.isfinite(s) or not (1e-5 <= s <= 0.1):
        raise ValueError('depth_scale %r m per unit is implausible (expected 0.001 for '
                         'millimeters)' % (depth_scale,))
    return s


def depth_msg_to_meters(msg: Any, depth_scale: float) -> np.ndarray:
    """
    Convert a sensor_msgs/Image depth message into HxW float32 meters (a copy).

    16UC1 / mono16: value * depth_scale (0 = invalid). 32FC1: already meters.
    Invalid pixels (0, NaN, inf, negative) become 0. Raises ValueError for
    other encodings, malformed messages or an implausible depth_scale.
    """
    enc = str(msg.encoding).strip().lower()
    if enc in ('16uc1', 'mono16'):
        scale = check_depth_scale(depth_scale)
        raw = _image_view(msg, 1, 'u2')
        return (raw.astype(np.float32) * np.float32(scale)).astype(np.float32)
    if enc == '32fc1':
        d = np.array(_image_view(msg, 1, 'f4'), dtype=np.float32)
        with np.errstate(invalid='ignore'):
            d[~(np.isfinite(d) & (d > 0))] = 0.0
        return d
    raise ValueError('unsupported depth encoding %r (supported: 16UC1, mono16, 32FC1)'
                     % (msg.encoding,))


def camera_info_to_intrinsics(msg: Any, ignore_distortion: bool = False) -> CameraIntrinsics:
    """
    Convert sensor_msgs/CameraInfo into CameraIntrinsics (pixels, OpenCV distortion).

    Uses K (fx, fy, cx, cy), width / height and D: plumb_bob -> (k1, k2, p1,
    p2, k3); rational_polynomial -> 8 coefficients (k1, k2, p1, p2, k3, k4,
    k5, k6), shortened to 5 when k4..k6 are zero. 8 (or 12 / 14 with a zero
    tail) coefficients labeled plumb_bob are read as the rational model too,
    as OpenCV reads any 8 coefficients (some OrbbecSDK_ROS2 versions publish
    that; logged once). ignore_distortion: D is not read and dist is all
    zeros. Raises ValueError for an uncalibrated K (zeros), a missing image
    size or an unsupported non-zero distortion model (e.g. equidistant).
    """
    k = getattr(msg, 'k', None)
    if k is None:
        k = getattr(msg, 'K', None)
    kk = np.array(list(k if k is not None else []), dtype=float).ravel()
    if kk.size != 9:
        raise ValueError('camera_info K must have 9 values, got %d' % kk.size)
    fx, fy, cx, cy = kk[0], kk[4], kk[2], kk[5]
    if not np.all(np.isfinite(kk)) or fx <= 0 or fy <= 0:
        raise ValueError('camera_info K is invalid (fx=%g, fy=%g): the driver publishes no '
                         'calibration' % (fx, fy))
    w, h = int(msg.width), int(msg.height)
    if w <= 0 or h <= 0:
        raise ValueError('camera_info has no image size (%dx%d)' % (w, h))
    if ignore_distortion:
        return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                                width=w, height=h, dist=np.zeros(5))
    d = getattr(msg, 'd', None)
    if d is None:
        d = getattr(msg, 'D', None)
    dd = np.array(list(d if d is not None else []), dtype=float).ravel()
    if not np.all(np.isfinite(dd)):
        raise ValueError('camera_info D contains NaN or inf')
    model = str(getattr(msg, 'distortion_model', '') or '').strip().lower()
    if dd.size == 0 or not np.any(dd != 0.0):
        dist = np.zeros(5)
    elif model in ('plumb_bob', '') and dd.size in (4, 5):
        dist = np.zeros(5)
        dist[:dd.size] = dd
    elif model in ('rational_polynomial', 'plumb_bob', '') and dd.size in (8, 12, 14):
        dist = dd[:8].copy() if np.any(dd[5:8] != 0.0) else dd[:5].copy()
        if dd.size > 8 and np.any(dd[8:] != 0.0):
            raise ValueError('camera_info D has thin-prism / tilt terms, not supported')
        if model == 'plumb_bob' and dist.size == 8 and 'plumb_bob8' not in _WARNED:
            _WARNED.add('plumb_bob8')
            _LOG.warning('camera_info: distortion_model plumb_bob with %d coefficients and '
                         'non-zero k4..k6; using them as the OpenCV rational model (k1, k2, '
                         'p1, p2, k3, k4, k5, k6)', dd.size)
    elif model == 'plumb_bob' and dd.size > 5 and not np.any(dd[5:] != 0.0):
        dist = dd[:5].copy()
    else:
        raise ValueError('unsupported distortion model %r with %d non-zero coefficients'
                         % (model, int(np.count_nonzero(dd))))
    return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                            width=w, height=h, dist=dist)


def fit_intrinsics(intr: CameraIntrinsics, width: int, height: int,
                   what: str = 'camera_info') -> CameraIntrinsics:
    """
    Return intr for an image of (width, height).

    Same size: intr. Same aspect ratio (1 %): intr.scaled(). Otherwise
    ValueError, because the calibration does not describe that image.
    """
    if (intr.width, intr.height) == (width, height):
        return intr
    ra = intr.width / float(intr.height)
    rb = width / float(height)
    if abs(ra - rb) / rb > 0.01:
        raise ValueError('%s is for %dx%d but the image is %dx%d (different aspect ratio)'
                         % (what, intr.width, intr.height, width, height))
    return intr.scaled(width, height)


class RosCamera(CameraSource):
    """
    CameraSource on ROS 2 topics: synchronized color + depth + camera_info.

    Config (camera.ros.*): color_topic, depth_topic, info_topic,
    depth_info_topic, depth_registered (true: depth already aligned to color;
    false: registered here with depth_to_color = 4x4 T_color_depth),
    depth_scale (m per 16UC1 unit), sync_slop_s, qos_reliable (false =
    sensor-data best-effort QoS), start_timeout_s (start() waits this long for
    a first frame and raises CameraError with a diagnosis; 0 = no wait,
    default 5), yuv422_order (uyvy | yuyv, default uyvy).

    sync_mode: 'header' (default) pairs color and depth whose header stamps
    are within sync_slop_s (message_filters ApproximateTimeSynchronizer).
    'receipt' pairs the latest color and depth messages received within
    sync_slop_s of each other (clock seconds), for drivers that stamp the
    two streams from different clocks; the scene is static while a frame
    is captured, so the receipt time is close enough (the older message of
    a pair arrived up to sync_slop_s before Frame.stamp: keep
    camera.settle_s above that for eye-in-hand captures).

    On-site escape hatches for a driver with an unusable CameraInfo:
    ignore_distortion (true: dist all zeros, D is not read) and
    intrinsics_override ({fx, fy, cx, cy, width, height, dist (optional)} of
    the color image; CameraInfo is then not needed and ignored).

    A daemon thread spins a SingleThreadedExecutor on a private rclpy Context
    (no signal handlers installed). Each synchronized pair is converted in
    that thread; the latest Frame is kept. Frame.stamp = clock.now() at
    receipt (Clock seconds); meta holds the header stamps (ROS seconds),
    the ROS time at receipt, encodings and the frame sequence number.
    get_frame() blocks on a condition variable for up to ``timeout`` seconds
    of real time.

    The image was exposed some tens of milliseconds before its receipt stamp.
    For eye-in-hand captures, ask for frames newer than (arm stopped +
    camera.settle_s) so no frame exposed during motion is used.
    """

    name = 'ros'

    def __init__(self, cfg: Any, clock: Clock):
        self._clock = clock

        def g(key: str, default: Any = None) -> Any:
            return cfg.get('camera.ros.' + key, default)

        self.color_topic = str(g('color_topic', '') or '')
        self.depth_topic = str(g('depth_topic', '') or '')
        self.info_topic = str(g('info_topic', '') or '')
        self.depth_info_topic = str(g('depth_info_topic', '') or '')
        reg = g('depth_registered', None)
        if reg is None:
            raise CameraError('camera.ros.depth_registered is not set (true when the driver '
                              'aligns depth to color)')
        self.depth_registered = bool(reg)
        try:
            self.depth_scale = check_depth_scale(g('depth_scale', None))
        except ValueError as e:
            raise CameraError('camera.ros.depth_scale: %s' % e)
        try:
            self.sync_slop = float(g('sync_slop_s', 0.05))
            self.start_timeout = float(g('start_timeout_s', 5.0))
        except (TypeError, ValueError) as e:
            raise CameraError('camera.ros: bad number: %s' % e)
        if not (0.0 < self.sync_slop < 1.0):
            raise CameraError('camera.ros.sync_slop_s must be in (0, 1) s, got %r'
                              % self.sync_slop)
        self.qos_reliable = bool(g('qos_reliable', False))
        self.yuv422_order = str(g('yuv422_order', 'uyvy') or 'uyvy')
        self.sync_mode = str(g('sync_mode', 'header') or 'header').strip().lower()
        if self.sync_mode not in SYNC_MODES:
            raise CameraError('camera.ros.sync_mode must be one of %s, got %r'
                              % (', '.join(SYNC_MODES), self.sync_mode))
        self.ignore_distortion = bool(g('ignore_distortion', False))
        self.intrinsics_override = self._parse_override(g('intrinsics_override', None))
        topics = [('color_topic', self.color_topic), ('depth_topic', self.depth_topic),
                  ('info_topic', self.info_topic)]
        self.T_color_depth: Optional[np.ndarray] = None
        if not self.depth_registered:
            topics.append(('depth_info_topic', self.depth_info_topic))
            m = g('depth_to_color', None)
            if m is None:
                raise CameraError(
                    'camera.ros.depth_registered is false but camera.ros.depth_to_color '
                    '(4x4 T_color_depth) is not set. Let the driver align depth instead: '
                    'Orbbec: launch orbbec_camera with depth_registration:=true; RealSense: '
                    'align_depth.enable:=true and depth_topic '
                    '/camera/camera/aligned_depth_to_color/image_raw; then set '
                    'camera.ros.depth_registered: true.')
            try:
                self.T_color_depth = validate_transform(m, 'camera.ros.depth_to_color',
                                                        max_translation=0.3)
            except ValueError as e:
                raise CameraError(str(e))
        for key, topic in topics:
            if not topic.startswith('/'):
                raise CameraError('camera.ros.%s must be an absolute topic name, got %r'
                                  % (key, topic))

        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._color_intr: Optional[CameraIntrinsics] = self.intrinsics_override
        self._depth_intr: Optional[CameraIntrinsics] = None
        # receipt sync: latest unpaired (message, clock time) per stream
        self._pending: Dict[str, Optional[Tuple[Any, float]]] = {'color': None, 'depth': None}
        self._last_stamps: Dict[str, Optional[float]] = {'color': None, 'depth': None}
        if self.intrinsics_override is not None:
            _LOG.warning('RosCamera: camera.ros.intrinsics_override is set; CameraInfo on %s '
                         'is ignored (fx %.1f, %dx%d)', self.info_topic,
                         self.intrinsics_override.fx, self.intrinsics_override.width,
                         self.intrinsics_override.height)
        elif self.ignore_distortion:
            _LOG.warning('RosCamera: camera.ros.ignore_distortion is set; the CameraInfo '
                         'distortion coefficients are not used')
        self._counts: Dict[str, int] = {'color': 0, 'depth': 0, 'info': 0, 'depth_info': 0,
                                        'pairs': 0, 'frames': 0, 'dropped': 0}
        self._last_error = ''
        self._n_errors = 0
        self._n_timeouts = 0
        self._once: set = set()
        self._started = False
        self._t_start = 0.0
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ctx = None
        self._node = None
        self._executor = None
        self._keep: List[Any] = []

    def _parse_override(self, d: Any) -> Optional[CameraIntrinsics]:
        """Return camera.ros.intrinsics_override as CameraIntrinsics (None when unset)."""
        if d is None or d == {}:
            return None
        key = 'camera.ros.intrinsics_override'
        if not isinstance(d, dict):
            raise CameraError('%s must be a mapping {fx, fy, cx, cy, width, height, dist}, '
                              'got %r' % (key, d))
        missing = [k for k in ('fx', 'fy', 'cx', 'cy', 'width', 'height') if d.get(k) is None]
        if missing:
            raise CameraError('%s is missing %s' % (key, ', '.join(missing)))
        try:
            intr = CameraIntrinsics.from_dict(dict(d, dist=d.get('dist') or [0.0] * 5))
        except (TypeError, ValueError) as e:
            raise CameraError('%s: bad value: %s' % (key, e))
        vals = [intr.fx, intr.fy, intr.cx, intr.cy]
        dist = np.asarray(intr.dist, dtype=float).ravel()
        if (not all(math.isfinite(v) for v in vals) or intr.fx <= 0 or intr.fy <= 0
                or intr.width <= 0 or intr.height <= 0):
            raise CameraError('%s needs fx, fy > 0 and width, height > 0, got %r' % (key, d))
        if dist.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(dist)):
            raise CameraError('%s.dist needs 4, 5, 8, 12 or 14 finite values, got %r'
                              % (key, d.get('dist')))
        if self.ignore_distortion:
            intr.dist = np.zeros(5)
        return intr

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """
        Create the node and subscriptions and start spinning.

        Raises CameraError when ROS 2 is not importable or, with
        start_timeout_s > 0, when no frame arrives in time.
        """
        if self._started:
            return
        try:
            import message_filters
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy,
                                   qos_profile_sensor_data)
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as e:
            raise CameraError('ROS 2 Python packages are not importable (%s). Run "source '
                              '/opt/ros/humble/setup.bash" first, or use camera.backend '
                              'file / sim.' % e)
        try:
            ctx = rclpy.Context()
            try:
                from rclpy.signals import SignalHandlerOptions
                rclpy.init(context=ctx, signal_handler_options=SignalHandlerOptions.NO)
            except (ImportError, TypeError):
                rclpy.init(context=ctx)
            self._ctx = ctx
            node = rclpy.create_node('cube_task_camera', context=ctx)
            self._node = node
            if self.qos_reliable:
                qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST)
            else:
                qos = qos_profile_sensor_data
            if self.sync_mode == 'receipt':
                color_sub = node.create_subscription(
                    Image, self.color_topic, lambda m: self._on_stream(m, 'color'), qos)
                depth_sub = node.create_subscription(
                    Image, self.depth_topic, lambda m: self._on_stream(m, 'depth'), qos)
                self._keep = [color_sub, depth_sub]
            else:
                color_sub = message_filters.Subscriber(node, Image, self.color_topic,
                                                       qos_profile=qos)
                depth_sub = message_filters.Subscriber(node, Image, self.depth_topic,
                                                       qos_profile=qos)
                color_sub.registerCallback(self._count, 'color')
                depth_sub.registerCallback(self._count, 'depth')
                sync = message_filters.ApproximateTimeSynchronizer(
                    [color_sub, depth_sub], 10, self.sync_slop)
                sync.registerCallback(self._on_pair)
                self._keep = [color_sub, depth_sub, sync]
            self._keep.append(node.create_subscription(CameraInfo, self.info_topic,
                                                       self._on_info, qos))
            if not self.depth_registered:
                self._keep.append(node.create_subscription(
                    CameraInfo, self.depth_info_topic, self._on_depth_info, qos))
            executor = SingleThreadedExecutor(context=ctx)
            executor.add_node(node)
            self._executor = executor
        except Exception as e:
            self.stop()
            raise CameraError('cannot create the ROS camera node: %s' % e)
        self._stop_evt.clear()
        self._t_start = time.monotonic()
        self._started = True
        self._thread = threading.Thread(target=self._spin, name='cube_task_camera_spin',
                                        daemon=True)
        self._thread.start()
        _LOG.info('RosCamera: color %s, depth %s (%s), info %s, %s sync slop %.3f s, QoS %s',
                  self.color_topic, self.depth_topic,
                  'registered' if self.depth_registered else 'registered in software',
                  self.info_topic, self.sync_mode, self.sync_slop,
                  'reliable' if self.qos_reliable else 'sensor data')
        if self.start_timeout > 0:
            if self.get_frame(timeout=self.start_timeout) is None:
                msg = self.diagnose()
                self.stop()
                raise CameraError('no RGB-D frame within %.1f s. %s'
                                  % (self.start_timeout, msg))

    def _spin(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self._stop_evt.is_set():
                    break
                self._note_error('ROS spin failed: %s: %s' % (type(e).__name__, e))
                ctx = self._ctx
                if ctx is not None and not ctx.ok():
                    break
                self._stop_evt.wait(0.1)

    def stop(self) -> None:
        """Stop spinning, drop the last frame, destroy node and context (safe twice)."""
        self._stop_evt.set()
        th, self._thread = self._thread, None
        if th is not None and th is not threading.current_thread():
            th.join(timeout=2.0)
        ex, self._executor = self._executor, None
        node, self._node = self._node, None
        ctx, self._ctx = self._ctx, None
        self._keep = []
        with self._cond:
            self._latest = None
        for fn in ((lambda: ex.shutdown(timeout_sec=1.0)) if ex is not None else None,
                   (lambda: node.destroy_node()) if node is not None else None,
                   (lambda: ctx.try_shutdown()) if ctx is not None else None):
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:
                _LOG.debug('RosCamera.stop: %s', e)
        self._started = False

    # ------------------------------------------------------------ callbacks
    def _count(self, msg: Any, which: str) -> None:
        stamp = self._header_stamp(msg)
        with self._cond:
            self._counts[which] += 1
            if stamp is not None:
                self._last_stamps[which] = stamp

    @staticmethod
    def _header_stamp(msg: Any) -> Optional[float]:
        try:
            return _stamp_to_sec(msg.header.stamp)
        except (AttributeError, TypeError, ValueError):
            return None

    def _on_stream(self, msg: Any, which: str) -> None:
        """Receipt sync: pair msg with the other stream's latest message within slop."""
        t = float(self._clock.now())
        other = 'depth' if which == 'color' else 'color'
        self._count(msg, which)
        with self._cond:
            self._pending[which] = (msg, t)
            mate = self._pending[other]
            if mate is None or abs(t - mate[1]) > self.sync_slop:
                return
            self._pending = {'color': None, 'depth': None}
        if which == 'color':
            self._on_pair(msg, mate[0])
        else:
            self._on_pair(mate[0], msg)

    def _log_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._once:
            self._once.add(key)
            _LOG.warning(msg, *args)

    def _note_error(self, text: str) -> None:
        with self._cond:
            self._last_error = text
            self._n_errors += 1
            n = self._n_errors
        if n <= 3 or n % 100 == 0:
            _LOG.warning('RosCamera: %s [%d errors]', text, n)

    def _on_info(self, msg: Any) -> None:
        with self._cond:
            self._counts['info'] += 1
        if self.intrinsics_override is not None:
            return
        try:
            intr = camera_info_to_intrinsics(msg, self.ignore_distortion)
        except ValueError as e:
            self._note_error('%s: %s (on-site escape hatches: camera.ros.ignore_distortion, '
                             'camera.ros.intrinsics_override)' % (self.info_topic, e))
            return
        with self._cond:
            self._color_intr = intr

    def _on_depth_info(self, msg: Any) -> None:
        with self._cond:
            self._counts['depth_info'] += 1
        try:
            intr = camera_info_to_intrinsics(msg, self.ignore_distortion)
        except ValueError as e:
            self._note_error('%s: %s' % (self.depth_info_topic, e))
            return
        with self._cond:
            self._depth_intr = intr

    def _on_pair(self, color_msg: Any, depth_msg: Any) -> None:
        t = float(self._clock.now())
        node = self._node
        ros_now = None
        if node is not None:
            try:
                ros_now = node.get_clock().now().nanoseconds * 1e-9
            except Exception:
                ros_now = None
        with self._cond:
            self._counts['pairs'] += 1
            seq = self._counts['pairs']
        try:
            frame = self._make_frame(color_msg, depth_msg, t, seq)
            if ros_now is not None:
                # ROS time at receipt; ros_receipt_stamp - color_stamp = driver latency
                # (meaningful when the driver stamps with the host clock).
                frame.meta['ros_receipt_stamp'] = ros_now
        except Exception as e:
            with self._cond:
                self._counts['dropped'] += 1
            self._note_error('frame dropped: %s' % e)
            return
        with self._cond:
            self._latest = frame
            self._counts['frames'] += 1
            self._cond.notify_all()

    def _make_frame(self, color_msg: Any, depth_msg: Any, t: float, seq: int) -> Frame:
        import cv2
        with self._cond:
            intr = self._color_intr
            intr_d = self._depth_intr
        if intr is None:
            raise ValueError('no valid CameraInfo on %s yet' % self.info_topic)
        color = image_msg_to_bgr(color_msg, self.yuv422_order)
        depth = depth_msg_to_meters(depth_msg, self.depth_scale)
        h, w = color.shape[:2]
        intr_c = fit_intrinsics(intr, w, h, self.info_topic)
        if intr_c is not intr:
            self._log_once('scale_c', 'RosCamera: %s is %dx%d, color image %dx%d; scaling '
                           'intrinsics', self.info_topic, intr.width, intr.height, w, h)
        if self.depth_registered:
            dh, dw = depth.shape
            if (dh, dw) != (h, w):
                if abs(dw / float(dh) - w / float(h)) / (w / float(h)) > 0.01:
                    raise ValueError(
                        'depth %dx%d and color %dx%d have different aspect ratios, so depth '
                        'is not registered to color. Enable depth_registration in the driver '
                        'or set camera.ros.depth_registered: false with depth_to_color.'
                        % (dw, dh, w, h))
                self._log_once('resize', 'RosCamera: depth %dx%d resized to color %dx%d '
                               '(nearest neighbor)', dw, dh, w, h)
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            from .registration import register_depth_to_color
            if intr_d is None:
                raise ValueError('no valid CameraInfo on %s yet' % self.depth_info_topic)
            dh, dw = depth.shape
            intr_d = fit_intrinsics(intr_d, dw, dh, self.depth_info_topic)
            depth = register_depth_to_color(depth, intr_d, intr_c, self.T_color_depth)
        meta = {'seq': seq,
                'color_stamp': _stamp_to_sec(color_msg.header.stamp),
                'depth_stamp': _stamp_to_sec(depth_msg.header.stamp),
                'color_encoding': str(color_msg.encoding),
                'depth_encoding': str(depth_msg.encoding),
                'depth_frame_id': str(depth_msg.header.frame_id),
                'depth_registered': self.depth_registered}
        return Frame(color=color, depth=depth, intrinsics=intr_c, stamp=t,
                     frame_id=str(color_msg.header.frame_id or 'camera_color_optical_frame'),
                     source='ros', meta=meta)

    # ---------------------------------------------------------------- access
    def get_frame(self, timeout: float = 1.0,
                  newer_than: Optional[float] = None) -> Optional[Frame]:
        """Return the latest frame with stamp > newer_than, waiting up to timeout s."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while True:
                f = self._latest
                if f is not None and (newer_than is None or f.stamp > newer_than):
                    return f
                if not self._started:
                    raise CameraError('RosCamera.get_frame() called before start()')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
        self._log_timeout(timeout)
        return None

    def _log_timeout(self, timeout: float) -> None:
        with self._cond:
            self._n_timeouts += 1
            n = self._n_timeouts
        if n <= 3 or n % 20 == 0:
            _LOG.warning('RosCamera: no new frame within %.2f s. %s', timeout, self.diagnose())

    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Return the color intrinsics for the delivered image size, None until known."""
        with self._cond:
            if self._latest is not None:
                return self._latest.intrinsics
            return self._color_intr

    def status(self) -> Dict[str, Any]:
        """Return message counts, last error and frame age (s) for self checks."""
        with self._cond:
            st: Dict[str, Any] = dict(self._counts)
            st['last_error'] = self._last_error
            st['have_intrinsics'] = self._color_intr is not None
            sc, sd = self._last_stamps['color'], self._last_stamps['depth']
            # color - depth header stamp of the last message of each stream (s)
            st['header_offset_s'] = None if sc is None or sd is None else sc - sd
            st['sync_mode'] = self.sync_mode
            st['last_frame_age_s'] = (None if self._latest is None
                                      else float(self._clock.now()) - self._latest.stamp)
        st['running_s'] = time.monotonic() - self._t_start if self._started else 0.0
        return st

    def diagnose(self) -> str:
        """Explain why no frames arrive, with the commands to check it."""
        st = self.status()
        since = st['running_s']
        problems = []
        checks = [('color', self.color_topic), ('depth', self.depth_topic)]
        if self.intrinsics_override is None:
            checks.append(('info', self.info_topic))
        if not self.depth_registered:
            checks.append(('depth_info', self.depth_info_topic))
        silent = [topic for key, topic in checks if st[key] == 0]
        for topic in silent:
            problems.append('no messages received on %s within %.1f s' % (topic, since))
        if st['color'] and st['depth'] and not st['pairs']:
            problems.append(self._sync_advice(st['color'], st['depth'], st['header_offset_s']))
        if st['info'] and not st['have_intrinsics']:
            problems.append('CameraInfo on %s is invalid' % self.info_topic)
        if st['last_error']:
            problems.append('last error: %s' % st['last_error'])
        if silent:
            seen = self._visible_topics()
            if seen is not None:
                problems.append('image topics visible: %s' % (', '.join(seen) or 'none'))
            problems.append('check "ros2 topic list" and "ros2 topic hz <topic>", that the '
                            'driver is running with the same ROS_DOMAIN_ID (here %s), and '
                            'camera.ros.qos_reliable'
                            % os.environ.get('ROS_DOMAIN_ID', '0 (unset)'))
        if not problems:
            if not self._started:
                return 'camera not started'
            return 'frames are arriving (%d so far)' % st['frames']
        return '; '.join(problems)

    def _sync_advice(self, n_color: int, n_depth: int, offset: Optional[float]) -> str:
        """Explain why color and depth arrive but never pair (offset: color - depth, s)."""
        head = ('color (%d msgs) and depth (%d msgs) arrive but never within sync slop %.3f s'
                % (n_color, n_depth, self.sync_slop))
        if self.sync_mode == 'receipt':
            return ('%s of receipt time; check "ros2 topic hz" of both topics or raise '
                    'camera.ros.sync_slop_s (< 1 s)' % head)
        if offset is None:
            return ('%s; compare header stamps (ros2 topic echo --field header.stamp <topic>) '
                    'or raise camera.ros.sync_slop_s' % head)
        if abs(offset) > CLOCK_OFFSET_S:
            return ('%s: their header stamps differ by %.3f s (color - depth), so the driver '
                    'stamps the two streams from different clocks and no sync slop can pair '
                    'them. Make the driver use one time source for both (OrbbecSDK_ROS2: a '
                    'parameter such as time_domain:=system or use_hardware_time:=false, '
                    'depending on the release; list them with "ros2 param list | grep -i '
                    'time"), or set camera.ros.sync_mode: receipt' % (head, offset))
        return ('%s (header stamps differ by %.3f s); raise camera.ros.sync_slop_s (< 1 s) or '
                'set camera.ros.sync_mode: receipt' % (head, offset))

    def _visible_topics(self) -> Optional[List[str]]:
        node = self._node
        if node is None:
            return None
        try:
            names = [n for n, _types in node.get_topic_names_and_types()]
        except Exception:
            return None
        keep = [n for n in names if 'image' in n or 'camera_info' in n or 'depth' in n]
        return sorted(keep)[:20]

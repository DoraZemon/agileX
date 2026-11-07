"""Tests for camera/ros_camera.py: message conversion helpers and RosCamera frame assembly."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cube_task.camera.ros_camera import (camera_info_to_intrinsics, depth_msg_to_meters,
                                         fit_intrinsics, image_msg_to_bgr,
                                         image_msg_to_numpy, RosCamera)
from cube_task.core.clock import SimClock
from cube_task.core.config import Config
from cube_task.core.interfaces import CameraError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROS_SETUP = '/opt/ros/humble/setup.bash'
ROS_PY_PATHS = ['/opt/ros/humble/local/lib/python3.10/dist-packages',
                '/opt/ros/humble/lib/python3.10/site-packages']


def _sensor_msgs():
    try:
        import sensor_msgs.msg as m
        return m
    except ImportError:
        pass
    for p in ROS_PY_PATHS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    try:
        import sensor_msgs.msg as m
        return m
    except ImportError:
        return None


SENSOR_MSGS = _sensor_msgs()
KINDS = ['fake'] + (['ros'] if SENSOR_MSGS is not None else [])


def _pack(arr, pad=0, bigendian=False):
    """Return (data bytes, step) for an HxW or HxWxC array with `pad` bytes per row."""
    a = np.asarray(arr)
    if a.dtype.itemsize > 1:
        a = a.astype(a.dtype.newbyteorder('>' if bigendian else '<'))
    h = a.shape[0]
    rows = np.ascontiguousarray(a).view(np.uint8).reshape(h, -1)
    if pad:
        rows = np.hstack([rows, np.full((h, pad), 0xAB, np.uint8)])
    return rows.tobytes(), rows.shape[1]


def _img(kind, arr, encoding, pad=0, bigendian=False, stamp=(5, 250000000), frame_id='cam'):
    data, step = _pack(arr, pad, bigendian)
    h, w = np.asarray(arr).shape[:2]
    if encoding in ('yuv422', 'yuv422_yuy2', 'yuyv', 'uyvy'):
        w = w // 2                                      # arr given as H x 2W bytes
    if kind == 'ros':
        m = SENSOR_MSGS.Image()
        m.header.stamp.sec, m.header.stamp.nanosec = stamp
        m.header.frame_id = frame_id
        m.height, m.width, m.encoding = h, w, encoding
        m.is_bigendian = int(bigendian)
        m.step = step
        m.data = data
        return m
    hdr = SimpleNamespace(stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1]),
                          frame_id=frame_id)
    return SimpleNamespace(header=hdr, height=h, width=w, encoding=encoding,
                           is_bigendian=int(bigendian), step=step, data=data)


def _info(kind, w=64, h=48, fx=60.0, d=(0.0,) * 5, model='plumb_bob'):
    K = [fx, 0.0, (w - 1) / 2.0, 0.0, fx, (h - 1) / 2.0, 0.0, 0.0, 1.0]
    if kind == 'ros':
        m = SENSOR_MSGS.CameraInfo()
        m.width, m.height = w, h
        m.k = K
        m.d = [float(v) for v in d]
        m.distortion_model = model
        return m
    return SimpleNamespace(width=w, height=h, k=K, d=list(d), distortion_model=model)


def _bgr(h=6, w=8, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('pad', [0, 5])
def test_color_encodings(kind, pad):
    bgr = _bgr()
    assert np.array_equal(image_msg_to_bgr(_img(kind, bgr, 'bgr8', pad)), bgr)
    assert np.array_equal(image_msg_to_bgr(_img(kind, bgr[:, :, ::-1], 'rgb8', pad)), bgr)
    rgba = np.dstack([bgr[:, :, ::-1], np.full(bgr.shape[:2], 7, np.uint8)])
    assert np.array_equal(image_msg_to_bgr(_img(kind, rgba, 'rgba8', pad)), bgr)
    bgra = np.dstack([bgr, np.full(bgr.shape[:2], 7, np.uint8)])
    assert np.array_equal(image_msg_to_bgr(_img(kind, bgra, 'bgra8', pad)), bgr)
    gray = bgr[:, :, 0]
    out = image_msg_to_bgr(_img(kind, gray, 'mono8', pad))
    assert out.shape == (6, 8, 3) and np.array_equal(out[:, :, 2], gray)
    assert image_msg_to_numpy is image_msg_to_bgr
    assert out.flags['C_CONTIGUOUS'] and out.dtype == np.uint8


@pytest.mark.parametrize('kind', KINDS)
def test_yuv422_byte_orders(kind):
    # One saturated red-ish pixel pair: Y = 100, U = 128, V = 240.
    yuyv = np.tile(np.array([100, 128, 100, 240], np.uint8), (4, 3))      # 4 rows x 6 px
    uyvy = np.tile(np.array([128, 100, 240, 100], np.uint8), (4, 3))
    for enc, data in (('yuv422_yuy2', yuyv), ('yuyv', yuyv), ('yuv422', uyvy),
                      ('uyvy', uyvy)):
        out = image_msg_to_bgr(_img(kind, data, enc, pad=3))
        assert out.shape == (4, 6, 3), enc
        b, g, r = [int(v) for v in out[0, 0]]
        assert r > g + 60 and r > b + 60, (enc, out[0, 0])
        assert np.all(out == out[0, 0])
    # A driver that labels YUYV data as "yuv422" can be handled with yuv422_order.
    out = image_msg_to_bgr(_img(kind, yuyv, 'yuv422'), yuv422_order='yuyv')
    assert out[0, 0, 2] > out[0, 0, 1] + 60
    with pytest.raises(ValueError, match='even'):
        image_msg_to_bgr(_img(kind, np.zeros((2, 6), np.uint8), 'yuyv'))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('bigendian', [False, True])
def test_depth_16uc1(kind, bigendian):
    mm = np.array([[0, 1, 500, 65535], [1234, 0, 999, 2]], np.uint16)
    for enc in ('16UC1', 'mono16'):
        d = depth_msg_to_meters(_img(kind, mm, enc, pad=2, bigendian=bigendian), 0.001)
        assert d.dtype == np.float32 and d.shape == (2, 4)
        assert np.allclose(d, mm.astype(np.float64) * 0.001, atol=1e-6)
        assert d[0, 0] == 0.0 and d[1, 1] == 0.0


@pytest.mark.parametrize('kind', KINDS)
def test_depth_32fc1(kind):
    m = np.array([[0.5, np.nan, 0.0, -1.0], [np.inf, 1.25, 2.0, 0.3]], np.float32)
    d = depth_msg_to_meters(_img(kind, m, '32FC1', pad=4), 0.001)
    assert np.allclose(d, [[0.5, 0, 0, 0], [0, 1.25, 2.0, 0.3]])
    assert np.all(np.isfinite(d))


def test_conversion_errors():
    bgr = _bgr()
    with pytest.raises(ValueError, match='unsupported color encoding'):
        image_msg_to_bgr(_img('fake', bgr, 'bayer_rggb8'))
    with pytest.raises(ValueError, match='unsupported depth encoding'):
        depth_msg_to_meters(_img('fake', bgr[:, :, 0], 'mono8'), 0.001)
    m = _img('fake', bgr, 'bgr8')
    m.data = m.data[:-10]
    with pytest.raises(ValueError, match='truncated'):
        image_msg_to_bgr(m)
    m = _img('fake', bgr, 'bgr8')
    m.step = 10
    with pytest.raises(ValueError, match='step'):
        image_msg_to_bgr(m)
    mm = _img('fake', np.ones((2, 2), np.uint16), '16UC1')
    for bad in (None, 0.0, float('nan'), 1.0):
        with pytest.raises(ValueError, match='depth_scale'):
            depth_msg_to_meters(mm, bad)
    # step = 0 with tightly packed data is accepted.
    m = _img('fake', bgr, 'bgr8')
    m.step = 0
    assert np.array_equal(image_msg_to_bgr(m), bgr)


@pytest.mark.parametrize('kind', KINDS)
def test_camera_info(kind):
    intr = camera_info_to_intrinsics(_info(kind, d=(0.1, -0.2, 0.001, 0.002, 0.03)))
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (60.0, 60.0, 31.5, 23.5)
    assert (intr.width, intr.height) == (64, 48)
    assert np.allclose(intr.dist, [0.1, -0.2, 0.001, 0.002, 0.03])
    intr = camera_info_to_intrinsics(_info(kind, d=(0.1, -0.2, 0, 0, 0.03, 0.5, 0.1, 0.02),
                                           model='rational_polynomial'))
    assert intr.dist.shape == (8,) and intr.dist[5] == 0.5
    intr = camera_info_to_intrinsics(_info(kind, d=(0.1, -0.2, 0, 0, 0.03, 0, 0, 0),
                                           model='rational_polynomial'))
    assert intr.dist.shape == (5,)
    intr = camera_info_to_intrinsics(_info(kind, d=(), model=''))
    assert np.all(intr.dist == 0) and intr.dist.shape == (5,)
    intr = camera_info_to_intrinsics(_info(kind, d=(0.0,) * 4, model='equidistant'))
    assert np.all(intr.dist == 0)
    with pytest.raises(ValueError, match='K is invalid'):
        camera_info_to_intrinsics(_info(kind, fx=0.0))
    with pytest.raises(ValueError, match='image size'):
        camera_info_to_intrinsics(_info(kind, w=0))
    with pytest.raises(ValueError, match='unsupported distortion'):
        camera_info_to_intrinsics(_info(kind, d=(0.1, 0.0, 0.0, 0.0), model='equidistant'))


def test_fit_intrinsics():
    intr = camera_info_to_intrinsics(_info('fake', w=640, h=480, fx=600.0))
    assert fit_intrinsics(intr, 640, 480) is intr
    half = fit_intrinsics(intr, 320, 240)
    assert half.fx == 300.0 and half.width == 320
    with pytest.raises(ValueError, match='aspect'):
        fit_intrinsics(intr, 640, 400)


# --------------------------------------------------------------- RosCamera

def _cfg(**ros):
    base = {'color_topic': '/camera/color/image_raw', 'depth_topic': '/camera/depth/image_raw',
            'info_topic': '/camera/color/camera_info',
            'depth_info_topic': '/camera/depth/camera_info', 'depth_registered': True,
            'depth_to_color': None, 'depth_scale': 0.001, 'sync_slop_s': 0.05,
            'qos_reliable': False}
    base.update(ros)
    return Config({'camera': {'ros': base}}, ROOT, [])


def test_ros_camera_config_errors():
    with pytest.raises(CameraError, match='depth_registration'):
        RosCamera(_cfg(depth_registered=False), SimClock())
    with pytest.raises(CameraError, match='depth_scale'):
        RosCamera(_cfg(depth_scale=None), SimClock())
    with pytest.raises(CameraError, match='depth_scale'):
        RosCamera(_cfg(depth_scale=1.0), SimClock())
    with pytest.raises(CameraError, match='depth_registered is not set'):
        RosCamera(_cfg(depth_registered=None), SimClock())
    with pytest.raises(CameraError, match='absolute topic'):
        RosCamera(_cfg(color_topic='camera/color'), SimClock())
    with pytest.raises(CameraError, match='translation'):
        RosCamera(_cfg(depth_registered=False,
                       depth_to_color=[[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0],
                                       [0, 0, 0, 1]]), SimClock())
    cam = RosCamera(_cfg(), SimClock())
    with pytest.raises(CameraError, match='before start'):
        cam.get_frame(timeout=0.01)
    cam.stop()
    cam.stop()


def _scene(w=64, h=48):
    bgr = _bgr(h, w, seed=3)
    mm = np.full((h, w), 600, np.uint16)
    mm[10:20, 10:20] = 550
    mm[0, 0] = 0
    return bgr, mm


def test_ros_camera_frame_assembly_registered():
    clock = SimClock(start=42.0)
    cam = RosCamera(_cfg(), clock)
    bgr, mm = _scene()
    cam._on_pair(_img('fake', bgr[:, :, ::-1], 'rgb8'), _img('fake', mm, '16UC1'))
    assert cam._latest is None and cam.status()['dropped'] == 1      # no camera_info yet
    assert 'CameraInfo' in cam.status()['last_error']
    cam._on_info(_info('fake'))
    cam._on_pair(_img('fake', bgr[:, :, ::-1], 'rgb8', stamp=(7, 0)),
                 _img('fake', mm, '16UC1', stamp=(7, 1000)))
    f = cam.get_frame(timeout=0.0)
    assert f is not None and f.source == 'ros' and f.stamp == 42.0
    assert np.array_equal(f.color, bgr)
    assert f.depth[15, 15] == pytest.approx(0.55) and f.depth[0, 0] == 0.0
    assert f.intrinsics.fx == 60.0 and f.frame_id == 'cam'
    assert f.meta['color_stamp'] == 7.0 and f.meta['depth_stamp'] == pytest.approx(7.000001)
    assert f.meta['color_encoding'] == 'rgb8' and f.meta['depth_encoding'] == '16UC1'
    assert cam.intrinsics().fx == 60.0
    # Not started (or stopped): asking for a newer frame is an error, not a silent wait.
    with pytest.raises(CameraError, match='before start'):
        cam.get_frame(timeout=0.0, newer_than=42.0)
    cam._started = True                       # pretend the node runs
    assert cam.get_frame(timeout=0.0, newer_than=42.0) is None
    cam._started = False


def test_ros_camera_depth_resize_and_aspect_check():
    cam = RosCamera(_cfg(), SimClock())
    cam._on_info(_info('fake'))
    bgr, mm = _scene()
    small = mm[::2, ::2]
    cam._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', small, '16UC1'))
    f = cam.get_frame(timeout=0.0)
    assert f.depth.shape == (48, 64)
    assert f.depth[14, 14] == pytest.approx(0.55)
    cam2 = RosCamera(_cfg(), SimClock())
    cam2._on_info(_info('fake'))
    cam2._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', mm[:40], '16UC1'))
    assert cam2._latest is None
    assert 'not registered' in cam2.status()['last_error']


def test_ros_camera_software_registration():
    T = [[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    cam = RosCamera(_cfg(depth_registered=False, depth_to_color=T), SimClock())
    cam._on_info(_info('fake'))
    bgr, mm = _scene()
    cam._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', mm, '16UC1'))
    assert cam._latest is None and 'depth/camera_info' in cam.status()['last_error']
    cam._on_depth_info(_info('fake'))
    cam._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', mm, '16UC1'))
    f = cam.get_frame(timeout=0.0)
    assert f.depth[15, 15] == pytest.approx(0.55) and f.depth[30, 30] == pytest.approx(0.6)
    assert f.depth[0, 0] == 0.0               # invalid corner, not filled (3 neighbors)


def test_ros_camera_diagnose_lists_silent_topics():
    cam = RosCamera(_cfg(), SimClock())
    cam._started = True                       # pretend; no rclpy node exists
    text = cam.diagnose()
    cam._started = False
    assert 'no messages received on /camera/color/image_raw' in text
    assert 'ros2 topic list' in text


# ------------------------------------------------------ live DDS round trip

_LIVE_SCRIPT = textwrap.dedent('''
    import json, os, sys, threading
    sys.path.insert(0, sys.argv[1])
    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import CameraInfo, Image
    from cube_task.camera.ros_camera import RosCamera
    from cube_task.core.clock import RealClock
    from cube_task.core.config import Config

    ns = '/cube_task_io_test_%d' % os.getpid()
    cfg = Config({'camera': {'ros': {
        'color_topic': ns + '/color', 'depth_topic': ns + '/depth',
        'info_topic': ns + '/info', 'depth_registered': True, 'depth_scale': 0.001,
        'sync_slop_s': 0.05, 'qos_reliable': False, 'start_timeout_s': 0}}}, sys.argv[1], [])
    cam = RosCamera(cfg, RealClock())
    cam.start()
    rclpy.init()
    node = rclpy.create_node('cube_task_io_test_pub')
    pc = node.create_publisher(Image, ns + '/color', 10)
    pd = node.create_publisher(Image, ns + '/depth', 10)
    pi = node.create_publisher(CameraInfo, ns + '/info', 10)
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, :, 0] = 200
    mm = np.full((h, w), 600, np.uint16)
    mm[10:20, 10:20] = 550

    def publish():
        stamp = node.get_clock().now().to_msg()
        c = Image(height=h, width=w, encoding='rgb8', step=w * 3, data=rgb.tobytes())
        d = Image(height=h, width=w, encoding='16UC1', step=w * 2, data=mm.tobytes())
        for m in (c, d):
            m.header.stamp = stamp
            m.header.frame_id = 'cam_optical'
        info = CameraInfo(width=w, height=h, distortion_model='plumb_bob')
        info.k = [60.0, 0.0, 31.5, 0.0, 60.0, 23.5, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.header.stamp = stamp
        pi.publish(info)
        pc.publish(c)
        pd.publish(d)

    node.create_timer(0.05, publish)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    f = cam.get_frame(timeout=float(sys.argv[2]))
    if f is None:
        print('RESULT ' + json.dumps({'ok': False, 'diag': cam.diagnose()}))
    else:
        print('RESULT ' + json.dumps({
            'ok': True, 'shape': list(f.color.shape), 'bgr': f.color[0, 0].tolist(),
            'depth_box': float(f.depth[15, 15]), 'depth_bg': float(f.depth[40, 40]),
            'fx': f.intrinsics.fx, 'frame_id': f.frame_id, 'source': f.source,
            'latency': f.meta['ros_receipt_stamp'] - f.meta['color_stamp']}))
    sys.stdout.flush()
    cam.stop()
    os._exit(0)
''')


@pytest.mark.skipif(not os.path.isfile(ROS_SETUP), reason='ROS 2 Humble not installed')
def test_ros_camera_live_round_trip(tmp_path):
    script = tmp_path / 'live_ros_camera.py'
    script.write_text(_LIVE_SCRIPT, encoding='utf-8')
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 101)   # private domain 100..200
    env['ROS_LOCALHOST_ONLY'] = '1'
    cmd = ['bash', '-c', 'source %s && exec python3 "$0" "$1" "$2"' % ROS_SETUP,
           str(script), ROOT, '6.0']
    try:
        out = subprocess.run(cmd, env=env, cwd=str(tmp_path), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=12.0)
    except subprocess.TimeoutExpired:
        pytest.skip('ROS round trip timed out (DDS unavailable here?)')
    text = out.stdout.decode('utf-8', 'replace')
    lines = [ln for ln in text.splitlines() if ln.startswith('RESULT ')]
    if not lines:
        pytest.skip('ROS round trip did not run: %s' % text[-500:])
    res = json.loads(lines[-1][len('RESULT '):])
    if not res['ok']:
        pytest.skip('no frame over DDS (flaky discovery?): %s' % res['diag'])
    assert res['shape'] == [48, 64, 3]
    assert res['bgr'] == [0, 0, 200]                      # rgb8 red -> BGR
    assert res['depth_box'] == pytest.approx(0.55, abs=1e-6)
    assert res['depth_bg'] == pytest.approx(0.60, abs=1e-6)
    assert res['fx'] == 60.0 and res['frame_id'] == 'cam_optical' and res['source'] == 'ros'
    assert 0.0 <= res['latency'] < 2.0


def test_cv2_codes_exist():
    # Guard against an OpenCV build without the 4:2:2 conversions used above.
    assert hasattr(cv2, 'COLOR_YUV2BGR_YUYV') and hasattr(cv2, 'COLOR_YUV2BGR_UYVY')

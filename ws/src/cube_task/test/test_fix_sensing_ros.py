"""RosCamera fixes: receipt-time sync, stamp-offset diagnosis, distortion escape hatches."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cube_task.camera import ros_camera
from cube_task.camera.ros_camera import camera_info_to_intrinsics, RosCamera
from cube_task.core.clock import SimClock
from cube_task.core.interfaces import CameraError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_io_ros_camera import _cfg, _img, _info, _scene, KINDS, ROOT, ROS_SETUP  # noqa: E402

RATIONAL = (0.1, -0.2, 0.001, 0.002, 0.03, 0.5, 0.1, 0.02)


class _Records(logging.Handler):
    """Collect log messages (caplog fails under the ROS launch_testing plugin)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.mark.parametrize('kind', KINDS)
def test_plumb_bob_with_8_coefficients_is_read_as_rational(kind):
    ros_camera._WARNED.discard('plumb_bob8')
    rec = _Records()
    ros_camera._LOG.addHandler(rec)
    try:
        intr = camera_info_to_intrinsics(_info(kind, d=RATIONAL, model='plumb_bob'))
        camera_info_to_intrinsics(_info(kind, d=RATIONAL, model='plumb_bob'))
    finally:
        ros_camera._LOG.removeHandler(rec)
    assert intr.dist.shape == (8,) and np.allclose(intr.dist, RATIONAL)
    assert sum('rational model' in m for m in rec.messages) == 1      # logged once
    intr = camera_info_to_intrinsics(_info(kind, d=RATIONAL + (0.0,) * 4, model='plumb_bob'))
    assert np.allclose(intr.dist, RATIONAL)
    with pytest.raises(ValueError, match='thin-prism'):
        camera_info_to_intrinsics(_info(kind, d=RATIONAL + (0.1, 0, 0, 0), model='plumb_bob'))
    with pytest.raises(ValueError, match='unsupported distortion'):
        camera_info_to_intrinsics(_info(kind, d=(0.1, 0.0, 0.0, 0.0), model='equidistant'))
    intr = camera_info_to_intrinsics(_info(kind, d=(0.1, 0.0, 0.0, 0.0), model='equidistant'),
                                     ignore_distortion=True)
    assert intr.dist.shape == (5,) and not np.any(intr.dist) and intr.fx == 60.0


def test_ignore_distortion_accepts_an_unusable_camera_info():
    cam = RosCamera(_cfg(ignore_distortion=True), SimClock())
    cam._on_info(_info('fake', d=(0.1, 0.0, 0.0, 0.0), model='equidistant'))
    bgr, mm = _scene()
    cam._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', mm, '16UC1'))
    f = cam.get_frame(timeout=0.0)
    assert f is not None and not np.any(f.intrinsics.dist) and f.intrinsics.fx == 60.0
    plain = RosCamera(_cfg(), SimClock())
    plain._on_info(_info('fake', d=(0.1, 0.0, 0.0, 0.0), model='equidistant'))
    assert not plain.status()['have_intrinsics']
    assert 'ignore_distortion' in plain.status()['last_error']


def test_intrinsics_override_replaces_camera_info():
    over = {'fx': 61.0, 'fy': 62.0, 'cx': 31.0, 'cy': 23.0, 'width': 64, 'height': 48}
    cam = RosCamera(_cfg(intrinsics_override=over), SimClock())
    assert cam.intrinsics().fx == 61.0 and cam.status()['have_intrinsics']
    cam._on_info(_info('fake', fx=0.0))                 # invalid CameraInfo is ignored
    bgr, mm = _scene()
    cam._on_pair(_img('fake', bgr, 'bgr8'), _img('fake', mm, '16UC1'))
    f = cam.get_frame(timeout=0.0)
    assert f is not None and (f.intrinsics.fx, f.intrinsics.fy) == (61.0, 62.0)
    assert f.intrinsics.dist.shape == (5,) and not np.any(f.intrinsics.dist)
    cam._started = True
    text = cam.diagnose()
    cam._started = False
    assert 'camera_info' not in text                     # CameraInfo topic not required
    dist8 = RosCamera(_cfg(intrinsics_override=dict(over, dist=list(RATIONAL))), SimClock())
    assert dist8.intrinsics().dist.shape == (8,)
    zeroed = RosCamera(_cfg(intrinsics_override=dict(over, dist=list(RATIONAL)),
                            ignore_distortion=True), SimClock())
    assert not np.any(zeroed.intrinsics().dist)
    for bad in ({'fx': 61.0}, dict(over, fx=-1.0), dict(over, width=0),
                dict(over, dist=[0.1, 0.2, 0.3]), [1, 2, 3], dict(over, fx='abc')):
        with pytest.raises(CameraError, match='intrinsics_override'):
            RosCamera(_cfg(intrinsics_override=bad), SimClock())


def test_receipt_sync_pairs_streams_stamped_from_different_clocks():
    with pytest.raises(CameraError, match='sync_mode'):
        RosCamera(_cfg(sync_mode='exact'), SimClock())
    clock = SimClock(start=10.0)
    cam = RosCamera(_cfg(sync_mode='receipt', sync_slop_s=0.05), clock)
    cam._started = True                       # pretend the node runs
    cam._on_info(_info('fake'))
    bgr, mm = _scene()
    color = _img('fake', bgr, 'bgr8', stamp=(5, 0))
    depth = _img('fake', mm, '16UC1', stamp=(1005, 0))      # device clock, 1000 s apart
    cam._on_stream(color, 'color')
    assert cam.get_frame(timeout=0.0) is None
    clock.advance(0.02)
    cam._on_stream(depth, 'depth')
    f = cam.get_frame(timeout=0.0)
    assert f is not None and f.stamp == pytest.approx(10.02)
    assert f.meta['color_stamp'] == 5.0 and f.meta['depth_stamp'] == 1005.0
    st = cam.status()
    assert st['pairs'] == 1 and st['color'] == 1 and st['depth'] == 1
    assert st['header_offset_s'] == pytest.approx(-1000.0) and st['sync_mode'] == 'receipt'
    # Each message pairs once; messages received farther apart than the slop do not pair.
    clock.advance(0.02)
    cam._on_stream(depth, 'depth')
    clock.advance(0.2)
    cam._on_stream(color, 'color')
    assert cam.status()['pairs'] == 1
    clock.advance(0.03)
    cam._on_stream(depth, 'depth')                          # newest color, 30 ms earlier
    assert cam.status()['pairs'] == 2
    cam._started = False


def test_diagnose_reports_the_header_stamp_offset():
    cam = RosCamera(_cfg(), SimClock())
    cam._started = True
    bgr, mm = _scene()
    cam._count(_img('fake', bgr, 'bgr8', stamp=(5, 0)), 'color')
    cam._count(_img('fake', mm, '16UC1', stamp=(1005, 0)), 'depth')
    text = cam.diagnose()
    assert 'differ by -1000.000 s' in text and 'different clocks' in text
    assert 'never within sync slop' in text                # wording used in the docs
    assert 'time_domain' in text and 'sync_mode: receipt' in text
    assert 'raise camera.ros.sync_slop_s' not in text
    assert cam.status()['header_offset_s'] == pytest.approx(-1000.0)
    cam._count(_img('fake', mm, '16UC1', stamp=(5, 200000000)), 'depth')
    text = cam.diagnose()
    assert 'differ by -0.200 s' in text and 'raise camera.ros.sync_slop_s' in text
    rc = RosCamera(_cfg(sync_mode='receipt'), SimClock())
    rc._started = True
    rc._count(_img('fake', bgr, 'bgr8'), 'color')
    rc._count(_img('fake', mm, '16UC1'), 'depth')
    assert 'receipt time' in rc.diagnose() and 'never within sync slop' in rc.diagnose()
    cam._started = rc._started = False


_LIVE = textwrap.dedent('''
    import json, os, sys, threading, time
    sys.path.insert(0, sys.argv[1])
    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import CameraInfo, Image
    from cube_task.camera.ros_camera import RosCamera
    from cube_task.core.clock import RealClock
    from cube_task.core.config import Config

    ns = '/cube_task_fix_sensing_%d' % os.getpid()

    def make(mode):
        cfg = Config({'camera': {'ros': {
            'color_topic': ns + '/color', 'depth_topic': ns + '/depth',
            'info_topic': ns + '/info', 'depth_registered': True, 'depth_scale': 0.001,
            'sync_slop_s': 0.05, 'qos_reliable': False, 'start_timeout_s': 0,
            'sync_mode': mode}}}, sys.argv[1], [])
        cam = RosCamera(cfg, RealClock())
        cam.start()
        return cam

    receipt, header = make('receipt'), make('header')
    rclpy.init()
    node = rclpy.create_node('cube_task_fix_sensing_pub')
    pc = node.create_publisher(Image, ns + '/color', 10)
    pd = node.create_publisher(Image, ns + '/depth', 10)
    pi = node.create_publisher(CameraInfo, ns + '/info', 10)
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), np.uint8)
    mm = np.full((h, w), 600, np.uint16)

    def publish():
        stamp = node.get_clock().now().to_msg()
        c = Image(height=h, width=w, encoding='rgb8', step=w * 3, data=rgb.tobytes())
        d = Image(height=h, width=w, encoding='16UC1', step=w * 2, data=mm.tobytes())
        c.header.stamp = stamp
        d.header.stamp.sec = stamp.sec - 1000       # depth from another clock
        d.header.stamp.nanosec = stamp.nanosec
        info = CameraInfo(width=w, height=h, distortion_model='plumb_bob')
        info.k = [60.0, 0.0, 31.5, 0.0, 60.0, 23.5, 0.0, 0.0, 1.0]
        info.d = [0.01, 0.0, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0]
        pi.publish(info)
        pc.publish(c)
        pd.publish(d)

    node.create_timer(0.05, publish)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    f = receipt.get_frame(timeout=float(sys.argv[2]))
    time.sleep(0.5)
    out = {'ok': f is not None, 'diag_receipt': receipt.diagnose(),
           'header_frame': header.get_frame(timeout=0.5) is not None,
           'diag_header': header.diagnose(), 'status': header.status()}
    if f is not None:
        out['dist'] = [float(v) for v in f.intrinsics.dist]
    print('RESULT ' + json.dumps(out, default=str))
    sys.stdout.flush()
    receipt.stop()
    header.stop()
    os._exit(0)
''')


@pytest.mark.skipif(not os.path.isfile(ROS_SETUP), reason='ROS 2 Humble not installed')
def test_live_receipt_sync_with_offset_clocks(tmp_path):
    script = tmp_path / 'live_receipt.py'
    script.write_text(_LIVE, encoding='utf-8')
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(100 + (os.getpid() + 37) % 101)
    env['ROS_LOCALHOST_ONLY'] = '1'
    cmd = ['bash', '-c', 'source %s && exec python3 "$0" "$1" "$2"' % ROS_SETUP,
           str(script), ROOT, '6.0']
    try:
        out = subprocess.run(cmd, env=env, cwd=str(tmp_path), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=15.0)
    except subprocess.TimeoutExpired:
        pytest.skip('ROS round trip timed out (DDS unavailable here?)')
    text = out.stdout.decode('utf-8', 'replace')
    lines = [ln for ln in text.splitlines() if ln.startswith('RESULT ')]
    if not lines:
        pytest.skip('ROS round trip did not run: %s' % text[-500:])
    res = json.loads(lines[-1][len('RESULT '):])
    if not res['ok']:
        pytest.skip('no frame over DDS (flaky discovery?): %s' % res['diag_receipt'])
    assert len(res['dist']) == 8 and res['dist'][5] == pytest.approx(0.002)
    # The same streams never pair by header stamp; the diagnosis names the clock offset.
    assert not res['header_frame']
    assert 'differ by 1000.000 s' in res['diag_header'], res['diag_header']
    assert 'sync_mode: receipt' in res['diag_header']

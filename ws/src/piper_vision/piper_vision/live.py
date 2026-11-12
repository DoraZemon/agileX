"""
Live sources: a ROS 2 RGB-D subscriber and two read-only PiPER pose readers.

RosTcpPose subscribes to the agx_arm_ros topic /feedback/tcp_pose (the pose
source of the AgileX hand-eye tutorial). ArmPoseReader opens CAN directly with
pyAgxArm and works without the ROS driver.

All import their dependencies lazily (rclpy, pyAgxArm), so the rest of the
package works without ROS or the arm SDK. The arm reader never sends a motion
or mode command: it only opens CAN and reads the flange pose, so it is safe
to use while the arm is in drag-teach mode.

Neither class has been tested against the real camera or arm yet.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .camera import CameraConfig, Intrinsics
from .transforms import make_T, pose6_to_T, quat_to_rot

LOG = logging.getLogger('piper_vision.live')


class RosRgbd:
    """Subscribe to color, depth and camera_info; keep the latest pair."""

    def __init__(self, cfg: CameraConfig, sync_slop_s: float = 0.05):
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError('rclpy not importable (%s); source /opt/ros/humble/setup.bash'
                               % exc)
        self._rclpy = rclpy
        self.cfg = cfg
        self.slop = float(sync_slop_s)
        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._info: Optional[Intrinsics] = None
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('piper_vision_rgbd')
        qos = qos_profile_sensor_data
        self.node.create_subscription(Image, cfg.topics['color'], self._on_color, qos)
        self.node.create_subscription(Image, cfg.topics['depth'], self._on_depth, qos)
        self.node.create_subscription(CameraInfo, cfg.topics['info'], self._on_info, qos)
        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin.start()

    def _spin_loop(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001 - shutdown races end the thread quietly
            LOG.debug('spin ended: %s', exc)

    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_color(self, msg) -> None:
        img = _image_to_numpy(msg)
        with self._lock:
            self._color = (self._stamp(msg), img)

    def _on_depth(self, msg) -> None:
        img = _image_to_numpy(msg)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) * self.cfg.depth_scale
        with self._lock:
            self._depth = (self._stamp(msg), img.astype(np.float32))

    def _on_info(self, msg) -> None:
        with self._lock:
            self._info = Intrinsics.from_camera_info(msg)

    def get(self, timeout: float = 3.0) -> Tuple[np.ndarray, Optional[np.ndarray], Intrinsics]:
        """Return (color BGR, depth m or None, intrinsics); raise on timeout."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                c, d, info = self._color, self._depth, self._info
            intr = self.cfg.intrinsics or info
            if c is not None and intr is not None:
                # A depth frame must match this color frame: an older depth frame
                # (color updated first) would be a different moment in time.
                if d is not None and 0.0 <= c[0] - d[0] <= self.slop:
                    return c[1], d[1], intr
                if d is None and time.monotonic() > t_end - timeout / 2:
                    return c[1], None, intr
            time.sleep(0.02)
        raise RuntimeError('no color frame (and intrinsics) on %s within %.1f s'
                           % (self.cfg.topics['color'], timeout))

    def close(self) -> None:
        """Stop spinning, destroy the node and shut ROS down (avoids a crash at exit)."""
        self._executor.shutdown(timeout_sec=1.0)
        self._spin.join(timeout=2.0)
        self.node.destroy_node()
        self._rclpy.try_shutdown()


def _image_to_numpy(msg) -> np.ndarray:
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('bgr8', 'rgb8'):
        rows = buf.reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = rows.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == 'rgb8' else img.copy()
    if enc in ('16uc1', 'mono16'):
        arr = buf.view('>u2' if msg.is_bigendian else '<u2')
        return arr.reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.uint16)
    if enc == '32fc1':
        arr = buf.view('>f4' if msg.is_bigendian else '<f4')
        return arr.reshape(msg.height, msg.step // 4)[:, :msg.width].astype(np.float32)
    raise ValueError('unsupported image encoding %r' % msg.encoding)


class RosTcpPose:
    """
    Read T_base_tcp from a geometry_msgs/PoseStamped topic (agx_arm_ros /feedback/tcp_pose).

    agx_arm_ros publishes the flange pose moved by its ``tcp_offset`` launch
    argument; with the default (all zeros) this is the flange pose.
    """

    def __init__(self, topic: str = '/feedback/tcp_pose'):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.executors import SingleThreadedExecutor
        except ImportError as exc:
            raise RuntimeError('rclpy not importable (%s); source /opt/ros/humble/setup.bash'
                               % exc)
        self._rclpy = rclpy
        self.topic = topic
        self._lock = threading.Lock()
        self._last: Optional[Tuple[float, np.ndarray]] = None
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('piper_vision_tcp_pose')
        self.node.create_subscription(PoseStamped, topic, self._on_pose, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin.start()

    def _spin_loop(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001 - shutdown races end the thread quietly
            LOG.debug('spin ended: %s', exc)

    def _on_pose(self, msg) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        T = make_T(quat_to_rot([q.x, q.y, q.z, q.w]), (p.x, p.y, p.z))
        with self._lock:
            self._last = (time.monotonic(), T)

    def pose(self, timeout: float = 2.0) -> np.ndarray:
        """Return the latest pose received within 0.5 s; raise when none arrives."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                last = self._last
            if last is not None and time.monotonic() - last[0] < 0.5:
                return np.array(last[1])
            time.sleep(0.02)
        raise RuntimeError('no pose on %s within %.1f s; is agx_arm_ros running?'
                           % (self.topic, timeout))

    def averaged(self, n: int = 8, dt: float = 0.03) -> np.ndarray:
        """Average n readings (position mean, rotation of the first; arm must be still)."""
        Ts = []
        for _ in range(n):
            Ts.append(self.pose())
            time.sleep(dt)
        T = np.array(Ts[0])
        T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
        return T

    def close(self) -> None:
        """Stop spinning and destroy the node (ROS itself is shut down by RosRgbd.close)."""
        self._executor.shutdown(timeout_sec=1.0)
        self._spin.join(timeout=2.0)
        self.node.destroy_node()


class ArmPoseReader:
    """Read T_base_flange from a PiPER over CAN with pyAgxArm. Sends no commands."""

    def __init__(self, channel: str = 'can0', firmware: str = 'default',
                 interface: str = 'socketcan'):
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except ImportError as exc:
            raise RuntimeError('pyAgxArm not importable (%s)' % exc)
        # The SDK keyword is spelled 'firmeware_version'.
        conf = create_agx_arm_config(robot='piper', firmeware_version=firmware,
                                     interface=interface, channel=channel)
        self.robot = AgxArmFactory.create_arm(conf)
        self.robot.connect()

    def flange(self, timeout: float = 2.0) -> np.ndarray:
        """Return the latest T_base_flange; raise when no feedback arrives."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            m = self.robot.get_flange_pose()
            if m is not None and getattr(m, 'msg', None) is not None:
                return pose6_to_T([float(v) for v in m.msg])
            time.sleep(0.02)
        raise RuntimeError('no flange pose feedback from the arm within %.1f s' % timeout)

    def flange_averaged(self, n: int = 8, dt: float = 0.02) -> np.ndarray:
        """Average n readings (position mean, rotation of the first; arm must be still)."""
        Ts = []
        for _ in range(n):
            Ts.append(self.flange())
            time.sleep(dt)
        T = np.array(Ts[0])
        T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
        return T

    def close(self) -> None:
        """Close CAN (no disable / reset)."""
        fn = getattr(self.robot, 'disconnect', None)
        if fn is not None:
            fn()

"""
Bridge between MoveIt 2 (RViz drag-and-plan) and the arm behind SafeArm.

MoveIt plans and shows the arm in RViz; this node plays the role of the ros2_control
controllers it would execute on:

- publishes ``feedback/joint_states`` (joint1..joint6 + ``gripper`` = opening width, m),
  so move_group / robot_state_publisher follow the real (or simulated) arm;
- serves ``arm_controller/follow_joint_trajectory`` and
  ``gripper_controller/follow_joint_trajectory`` (control_msgs), the action names the
  agx_arm_moveit MoveIt controller manager config uses.

Every executed waypoint goes through ``SafeArm.move_joints`` (joint limits, table
clearance, joint path sweep, health), the gripper through ``SafeArm.gripper``. Nothing
bypasses SafeArm. Without ``execute`` a trajectory goal is refused (plan-only): RViz
planning and animation work, the arm does not move.

MoveIt's trajectory is resampled to a few waypoints (``waypoint_step`` rad apart in
joint space); the controller interpolates between them itself, so the executed path
is a polyline of the planned one, not the exact planned trajectory.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional, Sequence

import numpy as np

_LOG = logging.getLogger('cube_task.robot.moveit_bridge')

ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
GRIPPER_JOINT = 'gripper'
#: default joint-space distance (rad) between two executed waypoints.
WAYPOINT_STEP = 0.15
#: never send more waypoints than this for one trajectory.
MAX_WAYPOINTS = 24


def resample_waypoints(positions: Sequence[Sequence[float]], step: float = WAYPOINT_STEP,
                       max_points: int = MAX_WAYPOINTS) -> List[np.ndarray]:
    """
    Return waypoints of a joint trajectory: a point whenever any joint moved `step` rad.

    The last point is always kept; at most max_points are returned (the spacing
    grows when the trajectory is long). An empty trajectory gives [].
    """
    pts = [np.asarray(p, dtype=float).reshape(-1) for p in positions]
    if not pts:
        return []
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError('step must be > 0, got %r' % (step,))
    out = []
    last = pts[0]
    for p in pts[1:-1]:
        if float(np.max(np.abs(p - last))) >= step:
            out.append(p)
            last = p
    out.append(pts[-1])
    while len(out) > max_points:           # keep first/last, thin the middle evenly
        keep = np.linspace(0, len(out) - 1, max_points).round().astype(int)
        out = [out[i] for i in sorted(set(keep.tolist()))]
    return out


class MoveItBridge:
    """rclpy node object wrapping an ArmInterface (use SafeArm). Create, then spin()."""

    def __init__(self, arm, force: Optional[float] = None, execute: bool = False,
                 waypoint_step: float = WAYPOINT_STEP, rate_hz: float = 20.0,
                 max_waypoints: int = MAX_WAYPOINTS,
                 node_name: str = 'cube_task_moveit_bridge'):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from sensor_msgs.msg import JointState
        self._Action = FollowJointTrajectory
        self.arm = arm
        self.force = force
        self.execute = bool(execute)
        self.waypoint_step = float(waypoint_step)
        self.max_waypoints = int(max_waypoints)
        self.busy = threading.Lock()
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self._cb = ReentrantCallbackGroup()
        self._GoalResponse = GoalResponse
        self._JointState = JointState
        self._pub = self.node.create_publisher(JointState, 'feedback/joint_states', 10)
        self.node.create_timer(1.0 / float(rate_hz), self._publish, callback_group=self._cb)
        self._arm_server = ActionServer(
            self.node, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory',
            execute_callback=self._exec_arm, goal_callback=self._goal,
            cancel_callback=lambda _g: CancelResponse.ACCEPT, callback_group=self._cb)
        self._grip_server = ActionServer(
            self.node, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory',
            execute_callback=self._exec_gripper, goal_callback=self._goal,
            cancel_callback=lambda _g: CancelResponse.ACCEPT, callback_group=self._cb)
        self.node.get_logger().info(
            'MoveIt bridge up: %s (waypoint step %.2f rad)' % (
                'EXECUTING on the arm' if self.execute else 'PLAN-ONLY, trajectories refused',
                self.waypoint_step))

    # ------------------------------------------------------------ feedback
    def _publish(self) -> None:
        try:
            q = self.arm.get_joints()
            w = self.arm.gripper_width()
        except Exception:  # noqa: B902 - a feedback glitch must not kill the node
            _LOG.exception('reading arm state failed')
            return
        if q is None or q.value is None:
            return
        msg = self._JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(v) for v in np.asarray(q.value, dtype=float).reshape(-1)[:6]]
        if w is not None and w.value is not None and math.isfinite(float(w.value)):
            msg.name.append(GRIPPER_JOINT)
            msg.position.append(float(w.value))
        self._pub.publish(msg)

    # ------------------------------------------------------------- actions
    def _goal(self, _goal) -> object:
        if not self.execute:
            self.node.get_logger().warning(
                'trajectory goal REFUSED: bridge is plan-only (start with --execute)')
            return self._GoalResponse.REJECT
        return self._GoalResponse.ACCEPT

    def _result(self, code: int, text: str):
        res = self._Action.Result()
        res.error_code = code
        res.error_string = text
        return res

    def _exec_arm(self, handle):
        A = self._Action
        traj = handle.request.trajectory
        names = list(traj.joint_names)
        if sorted(names) != sorted(ARM_JOINTS):
            handle.abort()
            return self._result(A.Result.INVALID_JOINTS, 'joints %s' % names)
        order = [names.index(j) for j in ARM_JOINTS]
        pts = [[p.positions[i] for i in order] for p in traj.points]
        way = resample_waypoints(pts, self.waypoint_step, self.max_waypoints)
        if not way:
            handle.succeed()
            return self._result(A.Result.SUCCESSFUL, '')
        if not self.busy.acquire(blocking=False):
            handle.abort()
            return self._result(A.Result.INVALID_GOAL, 'another trajectory is executing')
        try:
            self.node.get_logger().info('executing %d waypoints' % len(way))
            for k, q in enumerate(way):
                if handle.is_cancel_requested:
                    self.arm.stop()
                    handle.canceled()
                    return self._result(A.Result.SUCCESSFUL, 'canceled')
                m = self.arm.move_joints(q)
                if not m.ok:
                    self.arm.stop()
                    handle.abort()
                    text = 'waypoint %d/%d: %s %s' % (k + 1, len(way), m.error, m.detail)
                    self.node.get_logger().error('trajectory aborted, ' + text)
                    return self._result(A.Result.PATH_TOLERANCE_VIOLATED, text)
            handle.succeed()
            return self._result(A.Result.SUCCESSFUL, '')
        finally:
            self.busy.release()

    def _exec_gripper(self, handle):
        A = self._Action
        traj = handle.request.trajectory
        names = list(traj.joint_names)
        if GRIPPER_JOINT not in names or not traj.points:
            handle.abort()
            return self._result(A.Result.INVALID_JOINTS, 'need joint %r' % GRIPPER_JOINT)
        width = float(traj.points[-1].positions[names.index(GRIPPER_JOINT)])
        self.node.get_logger().info('gripper goal: width %.4f m' % width)
        if not self.busy.acquire(blocking=False):
            handle.abort()
            return self._result(A.Result.INVALID_GOAL, 'another trajectory is executing')
        try:
            g = self.arm.gripper(width, force=self.force, wait=True)
            self.node.get_logger().info('gripper result: ok=%s width=%s error=%s %s'
                                        % (g.ok, g.width, g.error, g.detail))
            if g.ok or g.error == 'timeout':
                handle.succeed()
                return self._result(A.Result.SUCCESSFUL, '')
            handle.abort()
            return self._result(A.Result.PATH_TOLERANCE_VIOLATED,
                                'gripper %s %s' % (g.error, g.detail))
        finally:
            self.busy.release()

    # ---------------------------------------------------------------- spin
    def spin(self) -> None:
        """Spin until shutdown (multi-threaded so feedback continues during execution)."""
        from rclpy.executors import MultiThreadedExecutor
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(self.node)
        try:
            ex.spin()
        finally:
            ex.shutdown()

    def close(self) -> None:
        """Destroy the node (the arm is stopped by the caller, never disabled)."""
        try:
            self.node.destroy_node()
        except Exception:  # noqa: B902
            _LOG.exception('destroy_node failed')

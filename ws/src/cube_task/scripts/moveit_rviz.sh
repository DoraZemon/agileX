#!/usr/bin/env bash
# Start MoveIt (robot_state_publisher + move_group + RViz) following the arm that
# `python3 -m cube_task motion_ctl moveit` drives. No ros2_control: the bridge replaces it.
# Usage: scripts/moveit_rviz.sh [--no-rviz]      (Ctrl-C stops everything)
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../../.." && pwd)"
# conda's python breaks ROS tools: drop it from PATH
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -E 'conda' | paste -sd:)"
export PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export LC_NUMERIC=en_US.UTF-8
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

L="$WS/src/agx_arm_ros/src/agx_arm_moveit/launch"
# TCP 0.1425 m past the flange: keep equal to robot.tcp_offset z in the task config
ARGS=(arm_type:=piper effector_type:=agx_gripper follow:=true
      "tcp_offset:=[0.0, 0.0, 0.1425, 0.0, 0.0, 0.0]")

pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null' EXIT INT TERM
ros2 launch "$L/rsp.launch.py" "${ARGS[@]}" & pids+=($!)
ros2 launch "$L/move_group.launch.py" "${ARGS[@]}" & pids+=($!)
# MoveIt cancels an execution that runs longer than 1.2 x its plan + 0.5 s. The bridge
# moves at robot.speed_percent, usually slower than the plan, so switch that off.
(
    for _ in $(seq 1 30); do
        sleep 1
        if ros2 param set /move_group trajectory_execution.execution_duration_monitoring false \
                >/dev/null 2>&1; then
            echo "moveit_rviz.sh: MoveIt execution duration monitoring off"
            exit 0
        fi
    done
    echo "moveit_rviz.sh: WARNING could not switch off MoveIt execution duration monitoring;" \
         "long moves may be cancelled" >&2
) &
if [ "${1:-}" != "--no-rviz" ]; then
    sleep 3
    ros2 launch "$L/moveit_rviz.launch.py" "${ARGS[@]}" \
        rviz_config:="$WS/src/agx_arm_ros/src/agx_arm_moveit/config/moveit.rviz" & pids+=($!)
fi
wait

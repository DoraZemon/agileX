#!/usr/bin/env bash
# Shared setup for the on-site scripts (arm_check.sh, camera_check.sh, handeye.sh,
# capture_dataset.sh). Source it, do not run it:  source "$(dirname "$0")/_env.sh"
#
# Sets PKG (cube_task package root), WS (colcon workspace), PY (/usr/bin/python3) and
# LOG_DIR; drops conda from PATH; sources ROS 2 Humble and the workspace install.

set +u
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$PKG/../.." && pwd)"
PY=/usr/bin/python3
LOG_DIR="${CUBE_TASK_LOG_DIR:-$HOME/vision_runs/onsite}"
mkdir -p "$LOG_DIR"

# conda's python breaks rclpy and the ROS tools
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -E 'conda' | paste -sd:)"
export PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE PYTHONHOME
export LC_NUMERIC=en_US.UTF-8
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [ -f "$WS/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$WS/install/setup.bash"
fi
cd "$PKG" || exit 1

# Camera topics (Orbbec defaults; config/default.yaml camera.ros.* must match).
CAM_DRIVER="${CAM_DRIVER:-orbbec}"          # orbbec | realsense
if [ "$CAM_DRIVER" = realsense ]; then
    CAM_NS=/camera/camera
    COLOR_TOPIC=$CAM_NS/color/image_raw
    DEPTH_TOPIC=$CAM_NS/aligned_depth_to_color/image_raw
    INFO_TOPIC=$CAM_NS/color/camera_info
    CLOUD_TOPIC=$CAM_NS/depth/color/points
else
    CAM_NS=/camera
    COLOR_TOPIC=$CAM_NS/color/image_raw
    DEPTH_TOPIC=$CAM_NS/depth/image_raw
    INFO_TOPIC=$CAM_NS/color/camera_info
    CLOUD_TOPIC=$CAM_NS/depth_registered/points
fi

say()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }

# confirm "question" -> 0 when the answer is y / yes
confirm() {
    local ans
    read -r -p "$1 [y/N] " ans || return 1
    [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# Ask the operator to check the arm before anything powers or moves it.
arm_safety_notice() {
    warn 'Before power-on: the J1 zero scale mark must be aligned. Hand on the e-stop.'
    warn 'Wiring: aviation plug -> XT30 24 V adapter (base LED blinks green) +'
    warn '        USB-to-CAN module (yellow CAN_H / blue CAN_L) -> laptop USB.'
}

# can_ready IFACE -> 0 when the interface is UP at 1 Mbit/s (never uses sudo)
can_ready() {
    bash "$PKG/scripts/can_up.sh" --check "${1:-can0}" >/dev/null 2>&1
}

camera_running() {
    ros2 topic list 2>/dev/null | grep -qx "$COLOR_TOPIC"
}

# wait_topic TOPIC SECONDS -> 0 when one message arrives in time
wait_topic() {
    timeout "$2" ros2 topic echo --once --no-arr "$1" >/dev/null 2>&1
}

# Orbbec launch file for the plugged-in Dabai (USB vendor 2bc5); dabai.launch.py when unknown.
orbbec_launch_file() {
    local pid
    pid="$(lsusb 2>/dev/null | sed -n 's/.*ID 2bc5:\([0-9a-f]\{4\}\).*/\1/p' | head -n1)"
    case "$pid" in
        0659|0559) echo dabai_dcw.launch.py ;;
        065a) echo dabai_dw.launch.py ;;
        06a0|0561) echo dabai_dcw2.launch.py ;;
        069f) echo dabai_dw2.launch.py ;;
        069a) echo dabai_max.launch.py ;;
        069e|0560) echo dabai_max_pro.launch.py ;;
        0658) echo dabai_d1.launch.py ;;
        *) echo dabai.launch.py ;;
    esac
}

# Start the camera driver in the background (log in $LOG_DIR/camera_driver.log) unless
# its color topic already exists. Sets CAM_PID (empty when it was already running).
# CAM_LAUNCH overrides the Orbbec launch file.
CAM_PID=''
start_camera() {
    if camera_running; then
        ok "camera driver already running ($COLOR_TOPIC)"
        return 0
    fi
    local log="$LOG_DIR/camera_driver.log"
    if [ "$CAM_DRIVER" = realsense ]; then
        ros2 pkg prefix realsense2_camera >/dev/null 2>&1 || {
            fail 'realsense2_camera is not installed'; return 1; }
        say "starting realsense2_camera (log: $log)"
        ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
            pointcloud.enable:=true >"$log" 2>&1 &
    else
        if ! ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
            fail 'the orbbec_camera ROS 2 driver is not built; see: bash scripts/camera_check.sh --help'
            return 1
        fi
        local launch="${CAM_LAUNCH:-$(orbbec_launch_file)}"
        say "starting orbbec_camera $launch (log: $log)"
        ros2 launch orbbec_camera "$launch" depth_registration:=true enable_point_cloud:=true \
            enable_colored_point_cloud:=true >"$log" 2>&1 &
    fi
    CAM_PID=$!
    local t
    for t in $(seq 1 20); do
        sleep 1
        if ! kill -0 "$CAM_PID" 2>/dev/null; then
            fail "the camera driver exited; last lines of $log:"
            tail -n 15 "$log"
            CAM_PID=''
            return 1
        fi
        if camera_running && wait_topic "$COLOR_TOPIC" 3; then
            ok "camera streaming after ${t}s ($COLOR_TOPIC)"
            return 0
        fi
    done
    fail "no image on $COLOR_TOPIC after 20 s; last lines of $log:"
    tail -n 15 "$log"
    return 1
}

stop_camera() {
    if [ -n "$CAM_PID" ] && kill -0 "$CAM_PID" 2>/dev/null; then
        kill -INT "$CAM_PID" 2>/dev/null
        wait "$CAM_PID" 2>/dev/null
    fi
    CAM_PID=''
}

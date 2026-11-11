#!/usr/bin/env bash
# 2/4 Depth camera check: USB -> driver -> RGB-D frames through cube_task -> RViz.
#
# Usage: bash scripts/camera_check.sh [--launch FILE] [--realsense] [--no-rviz]
#   --launch FILE   Orbbec launch file (default: picked from the USB id, e.g.
#                   dabai_dcw.launch.py; dabai.launch.py when the model is unknown)
#   --realsense     use a RealSense (D435i) instead, for practice without the Dabai
#   --no-rviz       checks only, no window
#
# Steps:
#   1. USB: the camera is on the bus and on a USB 3 port (5000M)
#   2. udev rules for Orbbec (asks for sudo once to install them)
#   3. start the driver with depth registered to color and the colored point cloud
#   4. grab RGB-D frames through cube_task's RosCamera (the same code run_task uses):
#      resolution, intrinsics from camera_info, frame rate, valid depth, depth at center
#   5. RViz: RGB image, depth image, point cloud (config/rviz/camera.rviz)
# Ctrl-C (or closing RViz) stops the driver this script started.
#
# Driver: orbbec_camera (OrbbecSDK_ROS2, branch main = SDK v1, supports the Dabai
# series) is built in the workspace from ws/src/camera_drivers. To rebuild it:
#   cd ~/vision/ws && colcon build --symlink-install --packages-select image_publisher \
#     orbbec_camera_msgs orbbec_description orbbec_camera --cmake-args \
#     -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=-isystem $HOME/vision/ws/third_party_include"
# Sources (if ws/src/camera_drivers is missing), each into its own directory:
#   https://codeload.github.com/orbbec/OrbbecSDK_ROS2/tar.gz/refs/heads/main
#   https://codeload.github.com/ros-perception/image_pipeline/tar.gz/refs/heads/humble
#     (only image_publisher)
#   https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
#     (single_include/nlohmann/json.hpp -> ws/third_party_include/nlohmann/)
# The official image (技术附件) also contains the driver.
LAUNCH=''
RVIZ=1
while [ $# -gt 0 ]; do
    case "$1" in
        --launch) LAUNCH="$2"; shift ;;
        --realsense) export CAM_DRIVER=realsense ;;
        --no-rviz) RVIZ=0 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[ -n "$LAUNCH" ] && export CAM_LAUNCH="$LAUNCH"

say '1/5 USB'
if [ "$CAM_DRIVER" = realsense ]; then VENDOR=8086; else VENDOR=2bc5; fi
DEV="$(lsusb 2>/dev/null | grep -i "ID $VENDOR:" | head -n1)"
if [ -z "$DEV" ]; then
    fail "no camera with USB vendor $VENDOR found (lsusb). Plug it in (USB 3 port, short cable)."
    exit 1
fi
ok "$DEV"
for d in /sys/bus/usb/devices/*; do
    if [ "$(cat "$d/idVendor" 2>/dev/null)" = "$VENDOR" ]; then
        speed="$(cat "$d/speed" 2>/dev/null)"
        if [ "${speed:-0}" -ge 5000 ]; then
            ok "USB speed ${speed}M (USB 3)"
        else
            warn "USB speed ${speed:-?}M: not USB 3. Depth + color may drop frames; use a blue USB 3 port."
        fi
        break
    fi
done

if [ "$CAM_DRIVER" = orbbec ]; then
    say '2/5 udev rules'
    if ls /etc/udev/rules.d/*obsensor* >/dev/null 2>&1; then
        ok 'Orbbec udev rules installed'
    else
        RULES="$WS/src/camera_drivers/OrbbecSDK_ROS2/orbbec_camera/scripts/99-obsensor-libusb.rules"
        warn 'Orbbec udev rules missing: the driver cannot open the camera without root'
        if [ -f "$RULES" ] && confirm 'Install them now (sudo)?'; then
            sudo cp "$RULES" /etc/udev/rules.d/99-obsensor-libusb.rules &&
                sudo udevadm control --reload-rules && sudo udevadm trigger &&
                ok 'installed; unplug and replug the camera, then press Enter' && read -r _
        fi
    fi
else
    say '2/5 udev rules (RealSense: installed with librealsense, skipped)'
fi

say '3/5 camera driver'
trap stop_camera EXIT INT TERM
start_camera || exit 1

say '4/5 RGB-D frames through cube_task (RosCamera)'
SETS=()
if [ "$CAM_DRIVER" = realsense ]; then
    SETS=("camera.ros.color_topic=$COLOR_TOPIC" "camera.ros.depth_topic=$DEPTH_TOPIC"
          "camera.ros.info_topic=$INFO_TOPIC")
fi
if ! "$PY" - "${SETS[@]}" <<'EOF'
import sys
import time

import numpy as np

from cube_task.camera.factory import make_camera
from cube_task.core.clock import RealClock
from cube_task.core.config import load_config

cfg = load_config()
for kv in sys.argv[1:]:
    k, v = kv.split('=', 1)
    cfg.set(k, v)
cfg.set('camera.backend', 'ros')
cfg.set('camera.frames_to_average', 1)    # raw frame rate
clock = RealClock()
cam = make_camera(cfg, clock)
cam.start()
try:
    frames, t0 = [], time.monotonic()
    while time.monotonic() - t0 < 3.0:
        f = cam.get_frame(timeout=2.0)
        if f is not None and (not frames or f.stamp != frames[-1].stamp):
            frames.append(f)
    if not frames:
        print('  no RGB-D pair in 3 s: check camera.ros.* topics and depth registration')
        sys.exit(1)
    f = frames[-1]
    k = f.intrinsics
    h, w = f.color.shape[:2]
    print('  color %dx%d, %d RGB-D pairs in 3 s (%.1f fps)' % (w, h, len(frames),
                                                           len(frames) / 3.0))
    print('  intrinsics (camera_info): fx %.1f fy %.1f cx %.1f cy %.1f' % (k.fx, k.fy, k.cx,
                                                                          k.cy))
    if f.depth is None:
        print('  NO DEPTH in the frame')
        sys.exit(1)
    d = f.depth
    valid = np.isfinite(d) & (d > 0)
    c = d[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    cv = c[np.isfinite(c) & (c > 0)]
    print('  depth %dx%d aligned to color, %.0f%% valid, center %.3f m'
          % (d.shape[1], d.shape[0], 100.0 * valid.mean(),
             float(np.median(cv)) if cv.size else float('nan')))
    if valid.mean() < 0.3:
        print('  less than 30% valid depth: too close (< 0.15 m), lens cap, or IR blocked')
        sys.exit(1)
finally:
    cam.stop()
EOF
then
    fail 'cube_task could not read RGB-D frames (see above)'
    exit 1
fi
ok 'cube_task reads RGB-D frames'

if [ "$RVIZ" = 0 ]; then
    say 'done (--no-rviz)'
    exit 0
fi
say '5/5 RViz: RGB, depth, point cloud (close the window or Ctrl-C to stop)'
RVIZ_CFG="$LOG_DIR/camera.rviz"
sed -e "s#__COLOR__#$COLOR_TOPIC#" -e "s#__DEPTH__#$DEPTH_TOPIC#" \
    -e "s#__CLOUD__#$CLOUD_TOPIC#" -e "s#__FRAME__#camera_link#" \
    "$PKG/config/rviz/camera.rviz" >"$RVIZ_CFG"
rviz2 -d "$RVIZ_CFG" >"$LOG_DIR/rviz_camera.log" 2>&1

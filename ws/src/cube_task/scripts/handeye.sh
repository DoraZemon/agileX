#!/usr/bin/env bash
# 3/4 Hand-eye calibration of the wrist camera (eye_in_hand): T_flange_camera.
#
# Usage: bash scripts/handeye.sh [--target block|aruco] [--aruco-size M] [--points N]
#                                [--iface can0] [--resume PAIRS_JSON] [--verify-only]
#                                [--headless]
#   --target block   (default) one 50 mm L1 cube alone on the table; touch the CENTER OF
#                    ITS TOP FACE with the fingertip center
#   --target aruco   a printed ArUco marker (DICT_4X4_50) lying flat; touch its center;
#                    --aruco-size = side of the black square in m (default 0.04)
#   --points N       touch points (default 8; spread them over the whole pick area)
#   --resume FILE    continue an interrupted session (pairs.json printed by the tool)
#   --verify-only    skip the calibration, only run the hover check
#   --headless       no preview window: type Enter / s / u / q in this terminal. With the
#                    window, keys only work when the PREVIEW WINDOW has the focus (click it)
#
# The organizers give no calibration (Q&A Q8/Q9): calibrate on the shared arm in your
# debug slot. Method (tools/calib_points.py, the arm is never moved by the tool):
#   for each point: put the arm in drag-teach mode, move it so the camera sees the target,
#   press Enter (camera measures, the flange pose is stored), drag the arm until the closed
#   fingertips touch the target point, press Enter. Change the view between points
#   (height, tilt, position). It solves T_flange_camera, prints per-point residuals
#   (aim: RMS <= 3 mm) and writes config/site/extrinsics.yaml after a backup.
# Needs robot.tcp_offset = flange -> closed fingertip center (measure it first!).
# Then the hover check (tools/calib_verify.py, MOVES the arm, needs the observe pose from
# teach_poses): the fingertip hovers 3 cm above each detected cube; read the offset.
# Stop arm_check.sh first (only one program may drive the arm).
# --realsense: practice with a RealSense instead of the Dabai.

IFACE=can0
TARGET=block
ARUCO_SIZE=0.04
POINTS=8
RESUME=()
CALIB=1
UI=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="$2"; shift ;;
        --aruco-size) ARUCO_SIZE="$2"; shift ;;
        --points) POINTS="$2"; shift ;;
        --iface) IFACE="$2"; shift ;;
        --resume) RESUME=(--resume "$2"); shift ;;
        --verify-only) CALIB=0 ;;
        --headless) UI=(--headless) ;;
        --realsense) export CAM_DRIVER=realsense ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

arm_safety_notice
say '1/4 CAN and camera'
if ! can_ready "$IFACE"; then
    bash "$PKG/scripts/can_up.sh" "$IFACE" 1000000 || { fail "CAN $IFACE is not up"; exit 1; }
fi
ok "$IFACE up"
trap stop_camera EXIT INT TERM
start_camera || exit 1
SETS=()
if [ "$CAM_DRIVER" = realsense ]; then
    SETS=(--set "camera.ros.color_topic=$COLOR_TOPIC" --set "camera.ros.depth_topic=$DEPTH_TOPIC"
          --set "camera.ros.info_topic=$INFO_TOPIC")
fi

say '2/4 configuration'
read -r MOUNT TCP OBS < <("$PY" - <<'EOF'
from cube_task.core.config import load_config
cfg = load_config()
obs = cfg.get('robot.poses.observe_joints', None)
print(cfg.get('camera.mount'), cfg.get('robot.tcp_offset')[2], 'yes' if obs else 'no')
EOF
)
if [ "$MOUNT" != eye_in_hand ]; then
    fail "camera.mount is $MOUNT; the competition camera is on the wrist (eye_in_hand)"
    exit 1
fi
ok "camera.mount eye_in_hand, robot.tcp_offset z = $TCP m (flange -> closed fingertip center)"
warn 'If you have not measured the TCP yet: measure it, put it in config/site/local.yaml, rerun.'

if [ "$CALIB" = 1 ]; then
    say "3/4 calibration: $POINTS touch points on a $TARGET target (drag-teach mode)"
    ARGS=(--mode real --task L1 --mount eye_in_hand --target "$TARGET" --points "$POINTS")
    [ "$TARGET" = aruco ] && ARGS+=(--aruco-dict DICT_4X4_50 --aruco-size "$ARUCO_SIZE")
    if ! "$PY" -m cube_task calib_points "${ARGS[@]}" "${RESUME[@]}" "${UI[@]}" "${SETS[@]}"; then
        fail 'calibration not written (see above); --resume continues a session'
        exit 1
    fi
    ok 'extrinsics written to config/site/extrinsics.yaml'
fi

say '4/4 hover check (moves the arm)'
if [ "$OBS" != yes ]; then
    warn 'robot.poses.observe_joints is not taught: run teach_poses (key o) and then'
    warn '  bash scripts/handeye.sh --verify-only'
    exit 0
fi
warn 'Leave drag-teach mode. The arm goes to the observe pose and hovers 3 cm above each cube.'
if confirm 'Run the hover check now?'; then
    "$PY" -m cube_task calib_verify --mode real --task L1 --mount eye_in_hand --all \
        --hover 0.03 --speed 20 "${UI[@]}" "${SETS[@]}"
fi

#!/usr/bin/env bash
# 4/4 Dataset capture: RGB images (plus aligned depth and the arm pose) from the wrist camera.
#
# Usage: bash scripts/capture_dataset.sh SESSION [--interval S] [--count N] [--out DIR]
#                                        [--no-arm] [--iface can0] [--realsense]
#   SESSION        what is on the table, e.g. cucumber, l2_mixed, cubes_red (file-name safe;
#                  stored with every frame, used as the class by tools/yolo/make_dataset.py)
#   --interval S   auto-save period in seconds (key a toggles auto-save; default 1.0)
#   --count N      unattended: save N frames every --interval s, then exit
#   --out DIR      output directory (default ~/vision_data/<date>_<SESSION>)
#   --no-arm       do not read the arm pose (default: read it when CAN is up)
#
# Window keys: s / space save one frame, a toggle auto-save, q quit.
# Files per frame: <i>_color.png (RGB image, 8-bit BGR), <i>_depth.png (16-bit mm, aligned),
# <i>_meta.json (intrinsics, joints and flange pose when the arm is read).
# The arm is never moved: put it in drag-teach mode (or move it with arm_check.sh in another
# terminal) and vary height, angle and object placement between frames. Light, background
# and object poses should look like the competition table.

IFACE=can0
INTERVAL=1.0
EXTRA=()
NO_ARM=0
SESSION=''
while [ $# -gt 0 ]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift ;;
        --count) EXTRA+=(--count "$2"); shift ;;
        --out) EXTRA+=(--out "$2"); shift ;;
        --no-arm) NO_ARM=1 ;;
        --iface) IFACE="$2"; shift ;;
        --realsense) export CAM_DRIVER=realsense ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) SESSION="$1" ;;
    esac
    shift
done
if [ -z "$SESSION" ]; then
    echo 'give a SESSION name, e.g.: bash scripts/capture_dataset.sh cucumber' >&2
    exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

say '1/2 camera'
trap stop_camera EXIT INT TERM
start_camera || exit 1
SETS=()
if [ "$CAM_DRIVER" = realsense ]; then
    SETS=(--set "camera.ros.color_topic=$COLOR_TOPIC" --set "camera.ros.depth_topic=$DEPTH_TOPIC"
          --set "camera.ros.info_topic=$INFO_TOPIC")
fi

if [ "$NO_ARM" = 0 ] && ! can_ready "$IFACE"; then
    warn "CAN $IFACE is not up: capturing without arm poses (run arm_check.sh first to record them)"
    NO_ARM=1
fi
[ "$NO_ARM" = 1 ] && EXTRA+=(--no-arm)

say "2/2 capture session '$SESSION' (s save, a auto-save every ${INTERVAL}s, q quit)"
"$PY" -m cube_task capture --mode real --task L2 --session "$SESSION" \
    --interval "$INTERVAL" "${EXTRA[@]}" "${SETS[@]}"

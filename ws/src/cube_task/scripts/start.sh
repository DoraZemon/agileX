#!/usr/bin/env bash
# One-click competition start on the real robot.
#
# Usage: bash scripts/start.sh <L1|L2> [--force] [run_task options...]
#   e.g. bash scripts/start.sh L1 --targets red,blue
#        bash scripts/start.sh L2 --dry-run --speed 20
#
# Steps: cd to the workspace, source ROS 2 Humble and ws/install/setup.bash (if
# present), check CAN (scripts/can_up.sh --check), run check_setup for the task,
# then exec "run_task --task <T> --mode real <options>". A FAIL in the checks
# stops here unless --force is given. Nothing here moves the arm: run_task asks
# before the first motion (skip with --yes).
#
# Environment: PYTHON (default python3), ROS_SETUP (default /opt/ros/humble/setup.bash).
# The package is run from this source tree (python3 -m cube_task), so a stale
# colcon install cannot shadow it.
set -euo pipefail

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

TASK="${1:-}"
case "$TASK" in
    L1|L2) shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "start.sh: first argument must be L1 or L2" >&2; usage >&2; exit 2 ;;
esac

FORCE=0
RUN_ARGS=()
CHECK_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1 ;;
        --task|--task=*|--mode|--mode=*)
            echo "start.sh: do not pass $1 (the task is the first argument, mode is real)" >&2
            exit 2 ;;
        --config|--set)
            if [ $# -lt 2 ]; then echo "start.sh: $1 needs a value" >&2; exit 2; fi
            RUN_ARGS+=("$1" "$2"); CHECK_ARGS+=("$1" "$2"); shift ;;
        --config=*|--set=*|-v|--verbose)
            RUN_ARGS+=("$1"); CHECK_ARGS+=("$1") ;;
        *) RUN_ARGS+=("$1") ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(dirname "$SCRIPT_DIR")"
WS_ROOT="$(cd "$PKG_ROOT/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

cd "$WS_ROOT"
echo "== cube_task start: $TASK ($(date '+%Y-%m-%d %H:%M:%S')), workspace $WS_ROOT"

# ROS setup files read unset variables: relax nounset while sourcing them.
set +u
if [ -f "$ROS_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
else
    echo "WARN: $ROS_SETUP not found; the ROS camera will not work"
fi
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$WS_ROOT/install/setup.bash"
fi
set -u

export CUBE_TASK_ROOT="$PKG_ROOT"
export PYTHONPATH="$PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"

FAILED=0
CHANNEL="$("$PYTHON" -c 'from cube_task.core.config import load_config
print(load_config().get("robot.can_channel", "can0"))' 2>/dev/null || echo can0)"
echo "== CAN ($CHANNEL)"
if ! bash "$SCRIPT_DIR/can_up.sh" --check "$CHANNEL"; then
    FAILED=1
fi

echo "== check_setup --task $TASK"
if ! "$PYTHON" -m cube_task check_setup --task "$TASK" --mode real \
        ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}; then
    FAILED=1
fi

if [ "$FAILED" = 1 ]; then
    if [ "$FORCE" = 1 ]; then
        echo "WARN: checks failed, continuing because of --force"
    else
        echo "STOP: fix the FAIL items above (or rerun with --force to start anyway)."
        exit 1
    fi
fi

echo "== run_task --task $TASK --mode real ${RUN_ARGS[*]+${RUN_ARGS[*]}}"
exec "$PYTHON" -m cube_task run_task --task "$TASK" --mode real ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}

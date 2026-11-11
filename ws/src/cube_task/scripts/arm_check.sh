#!/usr/bin/env bash
# 1/4 Arm check: CAN link -> arm feedback -> read-only state -> drag the arm in RViz.
#
# Usage: bash scripts/arm_check.sh [--iface can0] [--speed PCT] [--plan-only] [--no-rviz]
#   --iface IFACE   SocketCAN interface of the USB-to-CAN module (default can0)
#   --speed PCT     arm speed percent for executed trajectories (default 20)
#   --plan-only     RViz plans and previews only; the arm never moves
#   --no-rviz       start move_group without the RViz window
#
# Steps (each must pass before the next):
#   1. bring up IFACE at 1 Mbit/s (scripts/can_up.sh, asks for sudo)
#   2. listen 1 s on the bus: the powered arm sends feedback frames continuously
#   3. motion_ctl status --mode real: joints, TCP pose, gripper width (read-only)
#   4. MoveIt bridge (motion_ctl moveit) + move_group + RViz (scripts/moveit_rviz.sh).
#      In RViz pick the planning group arm or gripper, drag the marker, Plan & Execute.
#      Every waypoint goes through SafeArm (joint limits, table clearance).
# Ctrl-C stops everything; the arm is stopped and holds its pose (never disabled).
# Logs: ~/vision_runs/onsite/moveit_bridge.log
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

IFACE=can0
SPEED=20
EXECUTE=1
RVIZ_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --iface) IFACE="$2"; shift ;;
        --speed) SPEED="$2"; shift ;;
        --plan-only) EXECUTE=0 ;;
        --no-rviz) RVIZ_ARGS=(--no-rviz) ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown option $1"; exit 2 ;;
    esac
    shift
done

arm_safety_notice

say "1/4 CAN interface $IFACE"
if ! bash "$PKG/scripts/can_up.sh" "$IFACE" 1000000; then
    fail "CAN interface $IFACE is not up (is the USB-to-CAN module plugged in?)"
    exit 1
fi
ok "$IFACE up at 1 Mbit/s"

say '2/4 arm feedback on the bus (1 s)'
if ! "$PY" - "$IFACE" <<'EOF'
import sys
import time

import can

bus = can.interface.Bus(channel=sys.argv[1], interface='socketcan')
n, ids, t0 = 0, set(), time.time()
while time.time() - t0 < 1.0:
    msg = bus.recv(0.2)
    if msg is not None:
        n += 1
        ids.add(msg.arbitration_id)
bus.shutdown()
print('  %d frames, ids %s' % (n, ' '.join('0x%X' % i for i in sorted(ids)[:16])))
sys.exit(0 if n else 1)
EOF
then
    fail 'no CAN frames: arm not powered (base LED?), CAN_H / CAN_L swapped, or wrong interface'
    exit 1
fi
ok 'the arm is talking'

say '3/4 arm state (read-only, motors not enabled)'
if ! "$PY" -m cube_task motion_ctl status --mode real; then
    fail 'could not read the arm state (see the message above)'
    exit 1
fi

say '4/4 MoveIt bridge + RViz'
BRIDGE_ARGS=(--mode real --speed "$SPEED" -y)
if [ "$EXECUTE" = 1 ]; then
    warn "RViz Plan & Execute will MOVE the arm at ${SPEED}% (motors get enabled now)."
    warn 'The arm must not be in drag-teach mode. Clear the table, hand on the e-stop.'
    if ! confirm 'Enable the motors and allow RViz to move the arm?'; then
        warn 'falling back to plan-only (RViz previews, the arm does not move)'
        EXECUTE=0
    fi
fi
[ "$EXECUTE" = 1 ] && BRIDGE_ARGS+=(--execute)
BRIDGE_LOG="$LOG_DIR/moveit_bridge.log"
"$PY" -m cube_task motion_ctl moveit "${BRIDGE_ARGS[@]}" >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
cleanup() {
    if kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -INT "$BRIDGE_PID" 2>/dev/null
        wait "$BRIDGE_PID" 2>/dev/null
    fi
    say "bridge stopped (arm holds its pose). Log: $BRIDGE_LOG"
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 20); do
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        fail "the MoveIt bridge exited; last lines of $BRIDGE_LOG:"
        tail -n 15 "$BRIDGE_LOG"
        exit 1
    fi
    grep -q 'bridge running' "$BRIDGE_LOG" && break
    sleep 1
done
if ! grep -q 'bridge running' "$BRIDGE_LOG"; then
    fail "the MoveIt bridge did not start in 20 s; see $BRIDGE_LOG"
    exit 1
fi
ok "bridge running ($([ "$EXECUTE" = 1 ] && echo EXECUTE || echo plan-only)); starting move_group and RViz"
bash "$PKG/scripts/moveit_rviz.sh" "${RVIZ_ARGS[@]}"

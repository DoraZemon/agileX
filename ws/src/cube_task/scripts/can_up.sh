#!/usr/bin/env bash
# Bring up the SocketCAN interface of the PiPER (default can0 at 1000000 bit/s).
#
# Usage: bash scripts/can_up.sh [--check] [IFACE] [BITRATE]
#   --check   only report; exit 0 when IFACE is UP at BITRATE, 1 otherwise. Never uses sudo.
#
# Without --check, an interface that is DOWN, at another bitrate or BUS-OFF is
# reconfigured with:
#   sudo ip link set IFACE down
#   sudo ip link set IFACE type can bitrate BITRATE
#   sudo ip link set IFACE up
# Do not run it while a program is controlling the arm.
#
# The official AgileX script does the same and can also pick / rename an adapter
# by its USB port when several are plugged in:
#   bash third_party/pyAgxArm/pyAgxArm/scripts/ubuntu/can_activate.sh can0 1000000 [usb-bus-info]
#   (bus-info of each adapter: bash .../find_all_can_port.sh)
set -euo pipefail

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
}

CHECK_ONLY=0
POS=()
for arg in "$@"; do
    case "$arg" in
        --check|-c) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "can_up.sh: unknown option $arg" >&2; usage >&2; exit 2 ;;
        *) POS+=("$arg") ;;
    esac
done
IFACE="${POS[0]:-can0}"
BITRATE="${POS[1]:-1000000}"

if [[ ! "$IFACE" =~ ^[A-Za-z0-9_.-]{1,15}$ ]]; then
    echo "can_up.sh: invalid interface name '$IFACE' (letters, digits, _ . -, at most 15)" >&2
    exit 2
fi
if [[ ! "$BITRATE" =~ ^[0-9]+$ ]]; then
    echo "can_up.sh: bitrate must be a number, got '$BITRATE'" >&2
    exit 2
fi
if ! command -v ip >/dev/null 2>&1; then
    echo "can_up.sh: the 'ip' command is missing: sudo apt install iproute2" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL="$(cd "$SCRIPT_DIR/../../../.." 2>/dev/null && pwd)/third_party/pyAgxArm/pyAgxArm/scripts/ubuntu/can_activate.sh"

# Prints "<up> <bitrate> <can state> <virtual>" for $IFACE (fields '-' when unknown).
link_state() {
    local details up rate state virt
    details="$(ip -details link show "$IFACE" 2>/dev/null)" || return 1
    up=no
    if head -n1 <<<"$details" | grep -Eq '<([^>]*,)?UP(,[^>]*)?>'; then up=yes; fi
    rate="$(sed -n 's/.*bitrate \([0-9][0-9]*\).*/\1/p' <<<"$details" | head -n1)"
    state="$(sed -n 's/.*can \(<[^>]*> \)\{0,1\}state \([A-Z-][A-Z-]*\).*/\2/p' <<<"$details" | head -n1)"
    virt=no
    if grep -Eq '^[[:space:]]+vcan\b' <<<"$details"; then virt=yes; fi
    echo "$up ${rate:--} ${state:--} $virt"
}

list_adapters() {
    echo "CAN interfaces present:"
    ip -br link show type can 2>/dev/null | sed 's/^/  /' || true
    if command -v lsusb >/dev/null 2>&1; then
        echo "USB devices that look like CAN adapters:"
        lsusb | grep -Ei 'can|gs_usb|candle|1d50:606f|geschwister|peak' | sed 's/^/  /' \
            || echo "  none found (is the USB-CAN adapter plugged in?)"
    fi
}

if ! STATE="$(link_state)"; then
    echo "CAN interface $IFACE does not exist."
    list_adapters
    echo "Plug in the USB-CAN adapter; if it shows up under another name, run:"
    echo "  bash $SCRIPT_DIR/can_up.sh <name> $BITRATE   (and set robot.can_channel: <name>)"
    if [ -f "$OFFICIAL" ]; then
        echo "With several adapters, the official script selects one by USB port:"
        echo "  bash $OFFICIAL $IFACE $BITRATE <usb-bus-info>"
    fi
    exit 1
fi
read -r UP RATE CSTATE VIRTUAL <<<"$STATE"

if [ "$VIRTUAL" = yes ]; then
    if [ "$UP" = yes ]; then
        echo "OK: $IFACE is a virtual CAN interface and UP (no real arm on it)."
        exit 0
    fi
    echo "$IFACE is a virtual CAN interface and DOWN: sudo ip link set $IFACE up"
    exit 1
fi

if [ "$UP" = yes ] && [ "$RATE" = "$BITRATE" ] && [ "$CSTATE" != BUS-OFF ]; then
    echo "OK: $IFACE is UP at $BITRATE bit/s (can state $CSTATE)."
    exit 0
fi

if [ "$UP" != yes ]; then
    WHY="$IFACE is DOWN"
elif [ "$RATE" != "$BITRATE" ]; then
    WHY="$IFACE is UP at bitrate $RATE, need $BITRATE"
else
    WHY="$IFACE is BUS-OFF (check the CAN cable and the arm power)"
fi

if [ "$CHECK_ONLY" = 1 ]; then
    echo "NOT READY: $WHY."
    echo "Fix: bash $SCRIPT_DIR/can_up.sh $IFACE $BITRATE"
    exit 1
fi

echo "$WHY; reconfiguring (needs sudo):"
for cmd in "ip link set $IFACE down" "ip link set $IFACE type can bitrate $BITRATE" \
           "ip link set $IFACE up"; do
    echo "  sudo $cmd"
    # shellcheck disable=SC2086
    sudo $cmd
done

if STATE="$(link_state)"; then
    read -r UP RATE CSTATE VIRTUAL <<<"$STATE"
    if [ "$UP" = yes ] && [ "$RATE" = "$BITRATE" ]; then
        echo "OK: $IFACE is UP at $BITRATE bit/s (can state $CSTATE)."
        exit 0
    fi
fi
echo "FAILED: $IFACE is still not UP at $BITRATE bit/s. Check: ip -details link show $IFACE"
if [ -f "$OFFICIAL" ]; then
    echo "Official script: bash $OFFICIAL $IFACE $BITRATE"
fi
exit 1

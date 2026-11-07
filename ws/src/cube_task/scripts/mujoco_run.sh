#!/usr/bin/env bash
# Run a cube_task tool with the MuJoCo physics simulation.
#
# Usage: bash scripts/mujoco_run.sh <tool> [options...]
#   e.g. bash scripts/mujoco_run.sh run_task --task L1 --mode mujoco --seed 0 \
#            --video ~/vision_runs/mujoco/L1_s0.mp4
#        bash scripts/mujoco_run.sh sim_bench --backend mujoco --tasks L1 --seeds 0-9 --jobs 6
#
# Uses the MuJoCo venv (MUJOCO_PYTHON, default ~/.venvs/piper-mujoco/bin/python;
# MuJoCo is not installed for the system python3), renders offscreen on the GPU
# (MUJOCO_GL=egl unless already set), drops the ROS PYTHONPATH so the venv's
# packages are used, and runs "python -m cube_task <tool> ..." from the package
# root (this source tree).
set -euo pipefail

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MUJOCO_PYTHON:-$HOME/.venvs/piper-mujoco/bin/python}"
if [ ! -x "$PY" ]; then
    echo "mujoco_run.sh: no MuJoCo python at $PY (set MUJOCO_PYTHON)" >&2
    exit 2
fi
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec env -u PYTHONPATH "$PY" -m cube_task "$@"

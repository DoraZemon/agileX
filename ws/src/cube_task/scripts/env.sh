# 上机环境脚本: 在任何终端里先执行  source scripts/env.sh   (必须 source, 不能 bash)
# 之后所有命令只依赖工程本身:  $PY -m cube_task <工具> ...   或   ct <工具> ...
# 它做的事: 去掉 conda、加载 ROS、找到虚拟环境、进入 cube_task 目录。
# 虚拟环境不在默认位置时:  PIPER_VENV=<目录> source scripts/env.sh
# 本脚本不使用 sudo, 不碰系统 Python, 不连接机械臂。

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "请用 source 运行:  source scripts/env.sh" >&2
    exit 1
fi

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUBE_TASK_ROOT="$(cd "$_here/.." && pwd)"
_repo="$(cd "$CUBE_TASK_ROOT/../.." && pwd)"

# 1. 去掉 conda
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -i conda | paste -sd:)"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export LC_NUMERIC=en_US.UTF-8

# 2. ROS 2 和工作空间 (有就加载)
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f "$_repo/install/setup.bash" ] && source "$_repo/install/setup.bash"

# 3. 虚拟环境: PIPER_VENV > <仓库>/../venvs/piper > ~/.venvs/piper-mujoco
_venv=""
for d in "${PIPER_VENV:-}" "$_repo/../../venvs/piper" "$HOME/.venvs/piper-mujoco"; do
    if [ -n "$d" ] && [ -x "$d/bin/python" ]; then _venv="$(cd "$d" && pwd)"; break; fi
done
if [ -z "$_venv" ]; then
    echo "env.sh: 找不到虚拟环境, 请设置 PIPER_VENV=<目录>" >&2
    return 1
fi
export PY="$_venv/bin/python"
export MUJOCO_PYTHON="$PY"
export PYTHONPATH="$CUBE_TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 4. 进入工作目录, 提供简写 ct
cd "$CUBE_TASK_ROOT" || return 1
ct() { "$PY" -m cube_task "$@"; }

echo "env ok: PY=$PY"
echo "        目录=$PWD"
"$PY" -c "import pyAgxArm" 2>/dev/null && echo "        pyAgxArm: 可用" || echo "        pyAgxArm: 不可用 (真机命令会失败)"
unset _here _repo _venv d

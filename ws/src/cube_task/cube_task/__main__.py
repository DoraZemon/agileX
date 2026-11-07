"""Run a tool without ROS: python3 -m cube_task <tool> [args...]."""
from __future__ import annotations

import importlib
import sys

TOOLS = ['run_task', 'check_setup', 'capture', 'view', 'calib_points', 'calib_verify',
         'teach_poses', 'teach_classes', 'color_calib', 'sim_bench']


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print('usage: python3 -m cube_task <tool> [args]\ntools: ' + ', '.join(TOOLS))
        return 2
    mod = importlib.import_module('cube_task.tools.' + sys.argv[1])
    sys.argv = ['cube_task ' + sys.argv[1]] + sys.argv[2:]
    return int(mod.main() or 0)


if __name__ == '__main__':
    sys.exit(main())

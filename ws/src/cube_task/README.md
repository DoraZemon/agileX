# cube_task

PiPER 六轴机械臂 + 平行夹爪 + 组委会安装的奥比中光“大白”RGB-D 相机，完成松灵杯三项任务：

| 任务 | 内容 | 限时 | 默认分类器 |
|---|---|---|---|
| L1 | 3 个彩色方块放到指定区域 | 180 s | `color`（HSV 颜色区间） |
| L2 | 6 个果蔬模型按类别放进容器 | 300 s | `knn`（也可 `onnx` / `onnx+knn`，见 docs/yolo.md） |
| L3（自愿） | 自然语言指令，如“把最左边的苹果放到蓝色托盘” | 300 s | `knn` |

开始信号之后程序全程自主运行。出故障时停下并报告，不自动重启（每轮只有一次经裁判同意的重启）。
现场相关的数值（相机话题、外参、TCP、区域位置、类别映射、速度）全部在 YAML 配置里，现场只填参数和
示教，不改代码。没有真机时，同一套代码在仿真里跑完整的 L1 / L2 / L3，并按真值打分。

## 结构

```
TaskRunner (task/runner.py): 限时循环  观察 -> 选目标 -> 抓放 -> 回观察位
  |
  |-- camera/        get_frame()      真机 RosCamera | 回放 FileCamera | 仿真 SimCamera
  |      | Frame: 彩色 + 对齐到彩色的深度 (m) + 内参
  |      v
  |-- perception/    Perception.process()   桌面平面 -> 深度分割 -> 几何 -> 分类
  |      | Detection: base 系中心、顶面高度、朝向、长宽、标签
  |      v
  |-- perception/grasp.py   GraspPlanner / PlacePlanner   抓取、放置的 TCP 位姿
  |   task/reach.py         高处位姿向外倾斜（joint5 限位）
  |      | GraspPlan / PlacePlan
  |      v
  |-- task/skills.py  PickPlaceSkill   张开 -> 预抓 -> 下降 -> 闭合 -> 抬起 -> 转运 -> 放下 -> 退回
  |      | move_tcp / move_joints / gripper
  |      v
  `-- robot/safety.py SafeArm   工作空间、桌面高度、姿态、逆解检查；不通过就不发指令
         |                               |
         v                               v
     PyAgxArm --CAN--> PiPER 真机     SimArm --> SimWorld (sim/) --> SimCamera 渲染 RGB-D
```

- 物体分割只用深度：高出桌面平面的点就是物体。位置、高度、朝向、长宽都来自深度，分类器只负责贴标签。
- 所有运动都经过 `SafeArm`（由 `robot/factory.make_arm` 创建）。
- 程序里的时间都来自注入的 `Clock`。仿真用 `SimClock`，L1 一轮（约 58 s 仿真时间）实际只要 2 s。

## 快速开始

编译（`--symlink-install` 让安装目录指向源码，改 Python 不用重新编译；只有改 `setup.py` 时要重编）：

```bash
cd ~/vision/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cube_task
source install/setup.bash
```

不要在 ~/vision/ws 里跑不带 `--symlink-install` 的 `colcon build`：那样 cube_task 被复制进 install/，
`ros2 run cube_task <工具>` 只读复制的 config（不读 config/site 和 models/），工具写的文件下次编译会丢掉
（docs/现场流程.md 第 2 节）。

每个工具有两种等价的启动方式：`ros2 run cube_task <工具> ...` 或 `python3 -m cube_task <工具> ...`
（后者不需要 ROS 环境）。跑一轮仿真：

```bash
cd ~/vision/ws/src/cube_task
python3 -m cube_task run_task --task L1 --mode sim
python3 -m cube_task run_task --task L2 --mode sim --seed 3 --mount eye_in_hand
python3 -m cube_task sim_bench --tasks L1,L2 --seeds 0-9 --jobs 8
```

成功时最后打印 `L1 COMPLETE: placed 3/3 ...`、`ground truth (sim): correct 3, wrong 0` 和
`exit code 0 (round complete)`。第一次跑 L2 / L3 会先用仿真数据训练一个 KNN 模型（每种相机安装约 12 s，
缓存在 `~/.cache/cube_task/`）。

跑测试（约 3 分钟，不需要硬件和网络）：

```bash
cd ~/vision/ws/src/cube_task
python3 -m pytest -q -p no:cacheprovider test/
```

## 三种模式（`--mode`）

| 模式 | 机械臂 | 相机 | 读哪些配置 | 用途 |
|---|---|---|---|---|
| `real`（默认） | PyAgxArm，经 CAN | RosCamera（ROS 2 话题） | default.yaml + config/site/*.yaml | 真机 |
| `sim` | SimArm（运动学） | SimCamera 渲染仿真世界 | default.yaml + 场景生成的配置；忽略 config/site（加 `--site` 才合并） | 全流程仿真，按真值打分 |
| `replay` | SimArm（运动学，没有物体） | FileCamera 读 `capture` 存的目录（`--replay-dir`） | default.yaml + config/site/*.yaml | 用现场图离线检查感知 |

## 工具

| 工具 | 动臂吗 | 作用 | 文档 |
|---|---|---|---|
| `run_task` | 是（先确认） | 比赛入口：跑一轮 L1 / L2 / L3，`--dry-run` 只悬停不闭爪 | 现场流程 |
| `check_setup` | 否 | 模块、CAN、ROS 话题、配置、可达性检查；`--live` 读臂和相机 | 现场流程 |
| `scripts/can_up.sh` | 否 | 把 CAN 口设成 UP、1000000 bit/s；`--check` 只检查 | 现场流程 |
| `scripts/start.sh` | 否（交给 run_task） | 比赛一键启动：CAN 检查 + check_setup + run_task | 现场流程 |
| `view` | 否 | 实时看分割、标签、桌面平面、耗时 | 现场流程 |
| `capture` | 否 | 存 RGB-D 帧（训练、KNN、回放） | 仿真与测试 |
| `teach_poses` | 否（拖动示教） | 示教观察位、区域、取物多边形 → config/site/poses.yaml | 现场流程 |
| `calib_verify` | 是（先确认） | 指尖悬停到检测到的物体上方，量偏差，可修正外参 | 现场流程 |
| `calib_points` | 否（拖动示教） | 指尖碰点，现场重新估计外参（备用） | 现场流程 |
| `color_calib` | 否（`--observe` 时动） | 现场灯光下标定方块 HSV → config/site/colors.yaml | 现场流程 |
| `teach_classes` | 否 | 录果蔬样本，训练 KNN → models/knn_model.npz | 现场流程 |
| `sim_bench` | 否 | 批量仿真：成功率、用时、放错 | 仿真与测试 |
| `tools/yolo/*` | 否 | YOLO 数据集、训练、导出 ONNX、检查 | docs/yolo.md |

所有工具都接受 `--task --mode --config --set --site --seed --mount --difficulty --extrinsics-error
--replay-dir --log-dir -v`，`-h` 看完整参数。

## 目录

```
cube_task/                   包根目录（环境变量 CUBE_TASK_ROOT 可改）
├── config/default.yaml      全部默认值和说明；现场不要改它
├── config/site/             现场文件，覆盖 default.yaml（docs/配置说明.md）
├── config/examples/         现场文件模板
├── models/                  knn_model.npz、detector.onnx（*.onnx 不进 git，要单独备份）
├── scripts/                 can_up.sh、start.sh
├── docs/                    文档
├── cube_task/core/          配置、坐标变换、外参、日志、时钟、数据类型
├── cube_task/camera/        ROS 相机、文件回放、录制、深度对齐
├── cube_task/perception/    桌面拟合、分割、分类（color / knn / onnx）、抓取规划
├── cube_task/robot/         PiPER（pyAgxArm）、仿真臂、运动学、SafeArm
├── cube_task/sim/           仿真世界、RGB-D 渲染、场景、仿真 KNN 训练
├── cube_task/task/          TaskRunner、抓放技能、自检、L3 指令解析、操作员按键
├── cube_task/tools/         命令行工具（一个文件一个工具）、yolo/
└── test/                    pytest
```

## 日志和输出在哪里

根目录是 `logging.dir`（默认 `~/vision_runs`，`--log-dir` 可改）。

| 内容 | 位置 |
|---|---|
| run_task 每一轮 | `~/vision_runs/<YYYYmmdd_HHMMSS>_<任务>/`：events.jsonl、obs_*.jpg、report.json、summary.json、config.yaml |
| sim_bench 结果 | `~/vision_runs/sim_bench/<时间>.json` |
| 无窗口模式的预览图 | `~/vision_runs/preview/<工具>/` |
| view 按 s 存的帧 | `~/vision_runs/view/<时间>/` |
| 仿真模式下工具写的现场文件 | `~/vision_runs/sim_site/`（不会覆盖 config/site） |
| capture 采集 | `~/vision_data/<日期>_<session>/` |
| 仿真 KNN 缓存 | `~/.cache/cube_task/knn_sim_*.npz`（可以删，用时会重训） |

日志的读法见 docs/仿真与测试.md 第 5 节。

## 文档

- [docs/现场流程.md](docs/现场流程.md)：11/12–13 上机和 11/14 比赛的操作清单，出问题时怎么办。**现场先看这个。**
- [docs/配置说明.md](docs/配置说明.md)：每个配置项、现场文件、外参文件格式。
- [docs/仿真与测试.md](docs/仿真与测试.md)：仿真验证了什么、没验证什么，测试、sim_bench、回放、读日志。
- [docs/需求分析.md](docs/需求分析.md)：赛题要求和设计取舍。
- [docs/yolo.md](docs/yolo.md)：L2 的 YOLO 检测器训练、导出、部署。
- [docs/CONTRACTS.md](docs/CONTRACTS.md)：给开发者的模块接口和约定。

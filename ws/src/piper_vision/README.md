# piper_vision

> 负责人：sanlan。本包全部内容，以及下文“接入 cube_task”一节列出的 cube_task 改动，都是 sanlan 的工作。

相机配置、手眼标定、位姿转换和 L1 识别。与 `cube_task` 完全独立：不导入它的任何代码，只依赖 numpy、OpenCV、PyYAML；ROS 相机（rclpy）和机械臂位姿读取（pyAgxArm）在用到时才导入。

## 文件

| 文件 | 内容 |
|---|---|
| `config/camera.example.yaml` | 相机参数模板：安装方式、话题、内参、外参、TCP 偏移、标定板，各种外参写法都有说明 |
| `config/camera.yaml` | 现场配置：腕端大白（eye_in_hand）、大白驱动话题、内参读 camera_info、还没有外参。已加入 `.gitignore`，不提交；丢了就从模板复制 |
| `config/l1.yaml` | L1 识别参数：HSV 颜色、方块 / 卡纸尺寸、高度阈值、桌面高度、ROI |
| `piper_vision/camera.py` | 读写 camera.yaml；外参支持多种格式（4x4、平移+四元数、xyz+rpy、反方向、相对 TCP、camera_link + T_link_optical）；`T_base_camera`、点的相机 / 基座互转 |
| `piper_vision/handeye.py` | ChArUco 检测（兼容 OpenCV 4.5 / 4.7+）、`cv2.calibrateHandEye` 求解（与松灵手眼标定教程的约定一致）、残差和姿态覆盖检查 |
| `piper_vision/l1.py` | L1 检测器 |
| `piper_vision/live.py` | ROS 2 RGB-D 订阅；末端位姿读取：订阅 agx_arm_ros 的 `/feedback/tcp_pose`，或用 pyAgxArm 经 CAN 只读（都不发运动或模式指令）。**未在真机上测过** |
| `piper_vision/record.py` | 从 ROS 相机采集训练帧（cube_task 录制格式）和录 rosbag2 |
| `piper_vision/cli.py` | 命令行 |

坐标约定与 pyAgxArm / cube_task 一致：`T_a_b` 把 b 中的点变到 a；相机系为彩色光学坐标系；eye_to_hand 存 `T_base_camera`，eye_in_hand 存 `T_flange_camera`。

## L1 识别方法

方块和卡纸都是红 / 绿 / 黄，单靠颜色分不开，所以用高度区分：深度转到基座系 → RANSAC 拟合桌面 → 每种颜色的 HSV 掩码分别与“高于桌面 20–90 mm”取交集得到方块，与“贴着桌面”取交集得到卡纸。方块取顶面点的最小外接矩形，得到中心、yaw（90° 对称，范围 [-45°, 45°)）和边长；同色方块挨在一起时按面积估计个数，用 k-means 拆开。边长或高度偏离 50 mm 超过 15 mm 时降分并给出警告。

没有深度时退化为：颜色连通域射线与 `table_z + 50 mm` 平面求交，必须在 l1.yaml 里填 `table_z`，侧面会让位置向远离相机的方向偏 1–2 cm。

输出的 `center_base` 是顶面中心；`grasp_center` 是方块中心。

## 用法

```bash
cd ws/src/piper_vision
python3 -m piper_vision show                                  # 打印当前相机配置（config/camera.yaml）
python3 -m piper_vision l1 --frames <capture目录> --out /tmp/l1  # 对录制帧做 L1 识别，叠加图写到 /tmp/l1
python3 -m piper_vision l1                                    # 直接从 ROS 话题取一帧
python3 -m piper_vision transform --pixel 320 240 0.85        # 像素 + 深度 -> 基座坐标
python3 -m piper_vision transform --point 0.1 0 0.8 --flange x y z r p yaw   # 腕部相机
```

录制帧可以用本包的 `record`（下一节），也可以用 cube_task 的 `capture`，两者格式相同。meta 里的 `T_base_flange` 用于腕部相机。

### 启动大白驱动

采集和在线标定都从这里开始。大白驱动（OrbbecSDK_ROS2 的 `orbbec_camera`）编译在 `~/orbbec_ws`。下面的命令每开一个新终端都要先执行前两行。conda 环境里的 python 会让 rclpy 出错，先 `conda deactivate`。

```bash
source /opt/ros/humble/setup.bash
source ~/orbbec_ws/install/setup.bash

# 1. 确认相机插在 USB 3 口上（蓝色口），记下 2bc5: 后面的 PID
lsusb | grep 2bc5

# 2. 按型号选 launch 文件（与 cube_task scripts/_env.sh 的对应表一致）：
#    0659/0559 dabai_dcw   065a dabai_dw    06a0/0561 dabai_dcw2   069f dabai_dw2
#    069a dabai_max        069e/0560 dabai_max_pro                0658 dabai_d1
#    认不出时用 dabai.launch.py
#    depth_registration:=true 必须加，否则深度不对齐彩色图
ros2 launch orbbec_camera dabai_dcw.launch.py depth_registration:=true
```

驱动报 “no device” 或打不开相机时，检查 udev 规则（`ls /etc/udev/rules.d/ | grep obsensor`），装好后要拔插一次相机。

另开终端确认三个话题都在出数据，与 `config/camera.yaml` 的 `topics` 一致：

```bash
ros2 topic list | grep /camera/
ros2 topic hz /camera/color/image_raw              # dabai_dcw 默认 10 Hz
ros2 topic hz /camera/depth/image_raw
ros2 topic echo --once /camera/color/camera_info   # k 里是内参，camera.yaml 的 intrinsics: null 就是读这里
```

话题名不同（例如改了 `camera_name`，或者用 RealSense 的 `/camera/camera/...`）时改 `camera.yaml` 的 `topics`。也可以直接用 cube_task 的检查脚本，它会启动驱动、检查帧率和深度并打开 RViz：`bash ws/src/cube_task/scripts/camera_check.sh`。

### 采集数据集和录包

`config/camera.yaml` 已经建好（腕端大白、话题同上、内参读 camera_info、还没有外参）。采集不需要外参。

```bash
# 终端 1：大白驱动（见上一节）
# 终端 2：
source /opt/ros/humble/setup.bash
cd ws/src/piper_vision
PY="env -u PYTHONPATH /usr/bin/python3 -s"
$PY -m piper_vision show            # 确认读到的是 camera.yaml：mount eye_in_hand

# 采集训练帧：一个目录只放一类，--session 就是类名
$PY -m piper_vision record --session banana                 # 预览窗口：s/空格 存一帧，a 开关自动存，q 退出
$PY -m piper_vision record --session banana --count 30 --interval 1   # 无窗口，每 1 s 存一帧，共 30 帧
$PY -m piper_vision record --session pear --arm             # 同时经 CAN 读法兰位姿（只读，不发运动指令）

# 录 rosbag2（彩色、深度、camera_info 三个话题），之后可以 ros2 bag play 回放再采帧
$PY -m piper_vision record --session l2_scene --bag --duration 60
```

- 预览窗口的按键只在窗口有焦点时生效，先点一下窗口。左上角显示已存帧数，右半边是深度图；深度大片发黑说明相机离桌面太近（< 0.2 m）或镜头保护膜没撕。
- 默认输出到 `~/vision_data/<日期>_<session>`，同名目录会接着编号往后存。
- 每帧存 `NNNNNN_color.png`、`_depth.png`（mm）、`_depth.npy`（m）、`_meta.json`。meta 里有内参，`extra.session` 是类名。
- `capture`、`session` 这类通用名不能当类名，命令会直接拒绝。
- `--arm` 需要 pyAgxArm 和已经 up 的 CAN（`bash ws/src/cube_task/scripts/can_up.sh can0`），位姿存在 meta 的 `extra.T_base_flange`；腕部相机的帧要做 L1 识别或离线标定时才需要它，训练 YOLO / KNN 不需要。
- 相机在手腕上：用示教模式把机械臂拖到观察位（和比赛时的观察姿态相近），然后在桌上摆物体。一个会话只放一类果蔬，3–6 个，每存几帧就挪位置、转方向，也可以换几个观察高度和角度。每类至少录 2 个会话（不同摆放、不同光照），训练集和验证集按会话划分。

录好的目录可以直接给 cube_task 用（在 `ws/src/cube_task` 下运行）：

```bash
# YOLO 数据集：按深度自动出框，人工检查 preview/ 后再训练（docs/yolo.md）
python3 -m cube_task.tools.yolo.make_dataset --out ~/vision_runs/yolo/datasets/own \
    --classes cucumber,pear,banana,potato,persimmon,bell_pepper \
    --session banana=~/vision_data/20261112_banana --session pear=~/vision_data/20261112_pear \
    --preview
# KNN：类名取自 meta 里的 session
python3 -m cube_task teach_classes --task L2 --from-dir ~/vision_data/20261112_banana
```

### 手眼标定

方法与松灵《ROS2机械臂手眼标定教程（眼在手上，眼在手外，nero可用）》V2.0 一致：机械臂驱动用 agx_arm_ros，从 `/feedback/tcp_pose` 读末端位姿，`cv2.calibrateHandEye` 解 AX=XB，眼在手外时把末端位姿取逆再送进求解器。区别是标定板：教程用 aruco_ros 识别单个 ArUco 码，本包直接识别 ChArUco 板（角点多，姿态更稳），不需要 aruco_ros。

两种安装方式下标定板放的位置不同：

| 安装方式 | 相机 | 标定板 | 求出 |
|---|---|---|---|
| eye_in_hand（比赛就是这种） | 固定在法兰上，随机械臂动 | **平放在桌上，全程不动** | `T_flange_camera` |
| eye_to_hand | 固定在支架上不动 | **固定在法兰（末端）上，随机械臂动** | `T_base_camera` |

比赛只有腕端大白，所以用第一行：标定板放桌上，只动机械臂。

1. 生成并打印标定板：`$PY -m piper_vision board --camera config/camera.yaml --out board.png`，按 200 dpi、100% 比例打印，贴在硬板上。用尺子量格子，把实测值写进 `camera.yaml` 的 `board.square_m` / `marker_m`。
2. 终端 1 启动大白驱动（见“启动大白驱动”）。
3. 终端 2 启动机械臂驱动（PiPER 上电、CAN 已 up，`auto_enable:=false` 不会使能电机）：

   ```bash
   source /opt/ros/humble/setup.bash
   source <agx_arm_ros 工作空间>/install/setup.bash
   ros2 launch agx_arm_ctrl start_single_agx_arm_rviz.launch.py can_port:=can0 arm_type:=piper \
       effector_type:=agx_gripper auto_enable:=false tcp_offset:='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
   ros2 topic echo --once /feedback/tcp_pose      # 另开终端确认有位姿
   ```

   `tcp_offset` 建议保持全 0，这时 `/feedback/tcp_pose` 就是法兰位姿。教程里填了夹爪的 `0.13`，那样结果是相对 TCP 的；如果照填，下一步要加同样的 `--tcp-offset 0 0 0.13 0 0 0`，程序会换算回法兰。
4. 终端 3 采集并求解：

   ```bash
   cd ws/src/piper_vision
   $PY -m piper_vision handeye --save-dir /tmp/he      # 默认 --pose-source ros，读 /feedback/tcp_pose
   ```

   机械臂切到示教模式，拖到一个姿态、停稳，回车采一组；`u` 撤销上一组，`q` 求解。每组都要：标定板完整在画面里、终端打印的重投影误差 < 1.5 px。姿态要有俯仰、横滚、偏航三个方向的转动，不要只平移（平均转角 < 15° 会警告）；标定板在画面里的位置、大小、倾角也要各不相同。教程建议 15–25 组。
5. 看输出：残差是“本应不动的东西”算出来的离散度，eye_in_hand 是标定板在基座系的位置，eye_to_hand 是标定板相对末端的位置。rms ≤ 5 mm 才允许加 `--write` 写回 `camera.yaml`（`--max-rms` 可调），同时打印 5 种求解方法的 rms 供对比。确认后重跑最后一步加 `--write`，或者用 `/tmp/he` 里保存的数据。

没有 ROS 驱动时可以 `--pose-source can` 直接用 pyAgxArm 读法兰位姿；也可以离线：先 `record --arm` 录帧（meta 里带 `T_base_flange`），再 `handeye --frames <目录>`。

已经用教程的 `handeye_calibration_ros` 标过时，可以把它输出的 JSON（`position` + `orientation` 四元数 xyzw）填进 `camera.yaml`，不用重标：

```yaml
extrinsics:
  T_tcp_camera:                     # 结果是 TCP -> 相机
    position: [x, y, z]
    orientation: [qx, qy, qz, qw]
  tcp_offset: [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]    # 标定时 agx_arm_ros 用的 tcp_offset
  child_frame: camera_color_optical_frame
  source: handeye_calibration_ros
```

那个工具的相机坐标系取决于 aruco_ros 的 `reference_frame`：填的是 `camera_color_optical_frame` 就按上面写；填的是 `camera_link` 就把 `child_frame` 改成 `camera_link`，并加上 `T_link_optical`（`ros2 run tf2_ros tf2_echo camera_link camera_color_optical_frame` 的结果）。

比赛不提供标定文件（答疑 Q5–Q9），在本队调试时段自己标。cube_task 用的是自己的 `calib_points`（`scripts/handeye.sh`），本包的结果可以用来核对它。

## 接入 cube_task（sanlan）

L1 新识别通过配置开关接入 cube_task，默认仍是原方法。改动只有以下 4 处，其中只有 `factory.py` 是已有文件：

| 文件 | 改动 |
|---|---|
| `cube_task/perception/factory.py` | 约 10 行：task 为 L1 且 `perception.backend: l1_new` 时返回 `L1NewPerception`，其他情况走原 `Perception` |
| `cube_task/perception/l1_new.py`（新增） | 把本包的 L1 结果转成 cube_task 的 `Detection`，补齐抓取规划要用的字段（`table_z`、顶面网格点 `_points_base` / `_points_xy`、`cap_center_xy`、邻物 `_neighbor_xy`）；落在放置区内的方块标为无效，避免被重新抓取；`process()` 不抛异常，错误写进 warnings |
| `test/test_l1_new.py`（新增） | 开关行为、两种安装方式下的定位和字段、异常帧不崩 |
| `config/site/local.yaml`（新增） | 打开开关：`perception.backend: l1_new`，`perception.l1_block_size: 0.050` |

HSV 阈值读 cube_task 的 `perception.classifier.color.classes`（红 / 绿 / 黄），外参读 cube_task 自己的外参配置，不读本包的 `camera.yaml`。

切回原方法：把 `local.yaml` 里的 `backend` 改成 `pipeline`，或删掉这一行。

运行（在 `ws/src/cube_task` 下）：

```bash
python3 -m cube_task run_task --task L1 --mode sim --site   # 仿真，必须加 --site 才读 local.yaml
source /opt/ros/humble/setup.bash
python3 -m cube_task run_task --task L1 --mode real         # 真机，直接读 local.yaml
```

仿真对比（cube_task L1，20 个种子 × 两种安装 × normal / hard × 有无抓取失败，共 160 局）：

| 方法 | 成功率 | 平均用时 | 放错 |
|---|---|---|---|
| 原方法 | 86% | 57.8 s | 0 |
| 新方法 | 100% | 59.5 s | 0 |

原方法失败主要因为 `zone_exclude_margin: 0.02`，离卡纸 2 cm 内的方块被丢弃；改成 0 后差距会缩小，但没有重跑整套对比。新方法每帧约 300 ms（原方法约 40 ms），整局多约 2 s。

### 待确认 / 未验证

- 现场灯光下的 HSV 阈值需要用 cube_task 的 `color_calib` 重标。
- 外参要换成组委会的标定结果；TCP 长度 0.140 与 0.1425 需要确认，并在 cube_task 和本包 `camera.yaml` 两处同步。
- `live.py` 的 ROS 取图、`/feedback/tcp_pose` 订阅和 pyAgxArm 只读位姿没有在真机上验证。真机空跑（`--dry-run`）已走到等待相机画面一步。`record` 只用一个假的 ROS 相机节点测过（640×480，深度 16UC1），没有接过大白。
- L2 还没有切到新方法，开关目前只作用于 L1。

## 测试

```bash
cd ws/src/piper_vision
env -u PYTHONPATH /usr/bin/python3 -m pytest -q test
```

测试用一个光线投射渲染器生成桌面、卡纸和方块的 RGB-D 图，对照真值检查识别结果；手眼部分用合成位姿和透视变换后的标定板图像。不需要硬件。

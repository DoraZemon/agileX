# piper_vision

> 负责人：sanlan。本包全部内容，以及下文“接入 cube_task”一节列出的 cube_task 改动，都是 sanlan 的工作。

相机配置、手眼标定、位姿转换和 L1 识别。与 `cube_task` 完全独立：不导入它的任何代码，只依赖 numpy、OpenCV、PyYAML；ROS 相机（rclpy）和机械臂位姿读取（pyAgxArm）在用到时才导入。

## 文件

| 文件 | 内容 |
|---|---|
| `config/camera.example.yaml` | 相机参数模板：安装方式、话题、内参、外参、TCP 偏移、标定板。现场 `cp config/camera.example.yaml config/camera.yaml` 后填真实值；`camera.yaml` 已加入 `.gitignore`，不提交 |
| `config/l1.yaml` | L1 识别参数：HSV 颜色、方块 / 卡纸尺寸、高度阈值、桌面高度、ROI |
| `piper_vision/camera.py` | 读写 camera.yaml；外参支持多种格式（4x4、平移+四元数、xyz+rpy、反方向、相对 TCP、camera_link + T_link_optical）；`T_base_camera`、点的相机 / 基座互转 |
| `piper_vision/handeye.py` | ChArUco 检测（兼容 OpenCV 4.5 / 4.7+）、`cv2.calibrateHandEye` 求解、残差和姿态覆盖检查 |
| `piper_vision/l1.py` | L1 检测器 |
| `piper_vision/live.py` | ROS 2 RGB-D 订阅；pyAgxArm 只读位姿（不发任何运动或模式指令）。**未在真机上测过** |
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
python3 -m piper_vision show                                  # 打印当前相机配置
python3 -m piper_vision l1 --frames <capture目录> --out /tmp/l1  # 对录制帧做 L1 识别，叠加图写到 /tmp/l1
python3 -m piper_vision l1                                    # 直接从 ROS 话题取一帧
python3 -m piper_vision transform --pixel 320 240 0.85        # 像素 + 深度 -> 基座坐标
python3 -m piper_vision transform --point 0.1 0 0.8 --flange x y z r p yaw   # 腕部相机
```

录制帧可以用本包的 `record`（下一节），也可以用 cube_task 的 `capture`，两者格式相同。meta 里的 `T_base_flange` 用于腕部相机。

### 采集数据集和录包

先 `source /opt/ros/humble/setup.bash`，并确认大白的驱动已经在发 `camera.yaml` 里的三个话题。

```bash
# 采集训练帧：一个目录只放一类，--session 就是类名
python3 -m piper_vision record --session banana                 # 预览窗口：s/空格 存一帧，a 开关自动存，q 退出
python3 -m piper_vision record --session banana --count 30 --interval 1   # 无窗口，每 1 s 存一帧，共 30 帧
python3 -m piper_vision record --session pear --arm             # 同时读法兰位姿（只读，不发运动指令）

# 录 rosbag2（彩色、深度、camera_info 三个话题），之后可以 ros2 bag play 回放再采帧
python3 -m piper_vision record --session l2_scene --bag --duration 60
```

- 默认输出到 `~/vision_data/<日期>_<session>`，同名目录会接着编号往后存。
- 每帧存 `NNNNNN_color.png`、`_depth.png`（mm）、`_depth.npy`（m）、`_meta.json`。meta 里有内参，`extra.session` 是类名。
- `capture`、`session` 这类通用名不能当类名，命令会直接拒绝。
- 一个会话只放一类果蔬，3–6 个，每存几帧就挪位置、转方向，机械臂停在视野外。每类至少录 2 个会话（不同摆放、不同光照），训练集和验证集按会话划分。

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

1. `python3 -m piper_vision board --out board.png`，按 200 dpi、100% 比例打印，用尺子量格子，把实测值写进 camera.yaml 的 `board.square_m` / `marker_m`。
2. eye_in_hand：标定板平放在桌上不动，拖动机械臂从 10–15 个不同角度拍。eye_to_hand：标定板固定在法兰上，在相机视野里换 10–15 个姿态。姿态之间要有明显的转动（平均 >15°，否则会警告）。
3. 二选一：
   - 在线：`python3 -m piper_vision handeye --can can0 --save-dir /tmp/he`，回车采一组，`u` 撤销，`q` 求解；
   - 离线：用 cube_task `capture` 录好帧，`python3 -m piper_vision handeye --frames <目录>`。
4. 看输出的残差（标定板在基座系里的位置离散度），rms ≤ 5 mm 才允许 `--write` 写回 camera.yaml（`--max-rms` 可调）。同时会打印 5 种求解方法的 rms 供对比。

规则里相机由组委会安装和标定。拿到官方结果后直接填进 camera.yaml 的 `extrinsics` 段即可；自己的标定只用来核对官方结果，或者在官方允许时替换。

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
- `live.py` 的 ROS 取图和 pyAgxArm 只读位姿没有在真机上验证。真机空跑（`--dry-run`）已走到等待相机画面一步。`record` 只用一个假的 ROS 相机节点测过（640×480，深度 16UC1），没有接过大白。
- L2 还没有切到新方法，开关目前只作用于 L1。

## 测试

```bash
cd ws/src/piper_vision
env -u PYTHONPATH /usr/bin/python3 -m pytest -q test
```

测试用一个光线投射渲染器生成桌面、卡纸和方块的 RGB-D 图，对照真值检查识别结果；手眼部分用合成位姿和透视变换后的标定板图像。不需要硬件。

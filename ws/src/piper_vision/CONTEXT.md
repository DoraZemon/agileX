# 工作上下文：相机 + L1 识别（sanlan）

> 作者：sanlan。记录截至 2026-11-12 的状态，供之后继续这部分工作时快速接上。用法和方法细节见同目录 [README.md](README.md)。

## 目标

项目是识别 + PIPER 机械臂抓取。sanlan 负责识别部分，要求与已有代码解耦（cube_task 的其他文件不是 sanlan 负责的）。

1. 相机：一个 YAML 管内参和外参，支持手眼标定和位姿转换。**已完成**
2. L1 方块识别。**已完成**，已接入 cube_task，仿真验证通过
3. L2 果蔬检测。**未开始**

比赛物料（L1 / L2 / VLA 共用）：

- 小方块：≥12 个，红 / 绿 / 黄，哑光泡棉，50×50×50 mm（±2 mm）
- 果蔬模型：≥12 个，梨 / 香蕉 / 柿子 / 彩椒 / 黄瓜 / 土豆，仿真塑料 / EVA
- 目标区卡纸：每个工位 3 张，绿 / 黄 / 红，200×200 mm
- 物料由裁判摆放并锁定；相机由组委会安装和标定

## 做了什么

### 新包 `ws/src/piper_vision/`

不导入 cube_task，只依赖 numpy、OpenCV、PyYAML；rclpy 和 pyAgxArm 用到时才导入。

| 文件 | 作用 |
|---|---|
| `config/camera.example.yaml` | 相机参数模板。现场复制成 `config/camera.yaml`（已 gitignore） |
| `config/l1.yaml` | 命令行用的 L1 参数 |
| `camera.py` | 读写相机配置；外参支持多种写法；相机 / 基座坐标互转 |
| `transforms.py` | 4x4、rpy、四元数等位姿工具 |
| `handeye.py` | ChArUco 检测（兼容 OpenCV 4.5 / 4.7+）、手眼求解、残差和姿态覆盖检查 |
| `l1.py` | L1 检测器 |
| `frames.py` / `live.py` | 读 cube_task 录制帧 / ROS 2 相机 + pyAgxArm 只读位姿 |
| `cli.py` | `python3 -m piper_vision {board,handeye,l1,transform,show}` |
| `test/` | 25 个测试，合成 RGB-D 图像，不需要硬件 |

L1 的思路：卡纸和方块同色，所以用高度区分。深度转到基座系，用 RANSAC 拟合桌面，HSV 掩码与“高于桌面 20–90 mm”取交集得到方块，与“贴着桌面”取交集得到卡纸。同色方块粘在一起时，按面积估计个数，再用 k-means 拆开。

### cube_task 的改动（只有 4 处）

| 文件 | 改动 |
|---|---|
| `cube_task/perception/factory.py` | 唯一修改的已有文件，约 10 行。L1 且 `perception.backend: l1_new` 时用新方法 |
| `cube_task/perception/l1_new.py` | 新增适配器，把 piper_vision 的结果转成 cube_task 的 `Detection` |
| `test/test_l1_new.py` | 新增，4 个测试 |
| `config/site/local.yaml` | 新增，打开开关，方块边长 0.050 |

没有开关时，cube_task 的行为和原来完全一样。

## 关键决定和原因

- **用开关接入，默认走原方法**：不改别人的代码，出问题时改一行就能回退。
- **适配器补齐抓取规划要用的字段**：`extra['table_z']`、`_points_base`、`_points_xy`、`cap_center_xy`、`_neighbor_xy`。第一版足迹只给 4 个角点，抓取规划看不到相邻方块，hard 场景成功率只有 35%；改成覆盖顶面的 5 mm 网格点后恢复到 100%。
- **落在放置区里的方块标为无效**：防止把已经放好的方块再抓起来。
- **`process()` 不抛异常**：错误写进 warnings，主循环不会因为单帧出错而停下。
- **新方法在 cube_task 里读 cube_task 自己的配置**：HSV 阈值来自 `perception.classifier.color.classes`，外参来自 cube_task 的外参文件。piper_vision 的 `camera.yaml` 和 `l1.yaml` 只供它的命令行使用。
- **删掉了 `camera.yaml`，只留模板**：避免把示例外参（`source: example_only`）提交上去被当成真值。

## 验证结果

- 仿真对比：cube_task L1，20 个种子 × 2 种安装 × normal / hard × 有无抓取失败，共 160 局。
  - 原方法：成功 86%，平均 57.8 s，放错 0。
  - 新方法：成功 100%，平均 59.5 s，放错 0。
  - 原方法失败的主要原因是 `zone_exclude_margin: 0.02`。这个值改成 0 以后没有重跑整套对比。
- 速度：新方法每帧约 300 ms，原方法约 40 ms，整局多约 2 s。
- `run_task --task L1 --mode sim --site`：4 种组合都以退出码 0 结束，hard 局 3/3 放对。
- 真机空跑 `--mode real --dry-run`：配置、感知和自检都通过，因为本机没接相机，停在等待画面那一步。
- 测试：piper_vision 25 个通过（OpenCV 4.5.4 和 4.12 都测过），`test_l1_new.py` 4 个通过。ament_flake8（行宽 99）和 ament_pep257 没有报错。
- cube_task 原有的感知测试有 17 个失败，在干净的 HEAD 上也能复现，与这次改动无关。

## 环境注意事项

- 测试和运行用 `env -u PYTHONPATH /usr/bin/python3 -s ...`，避免混进其他 Python 环境的包。
- 真机模式先 `source /opt/ros/humble/setup.bash`，否则会报 `No module named 'message_filters'`。
- 仿真工具默认不读 `config/site/`，要加 `--site`。
- 使用 piper_vision 命令行前先 `cp config/camera.example.yaml config/camera.yaml`。缺这个文件时会给出提示。

## 待办

- [ ] **提交**：改动都还没提交（当前在 `motion` 分支）。提交信息要以 TODO 编号开头（例如 `P0-3:`），编号还没定，计划一次提交一项。
- [ ] 现场灯光下用 cube_task 的 `color_calib` 重标 HSV。
- [ ] 换成组委会给的外参；确认 TCP 长度是 0.140 还是 0.1425，并在 cube_task 和 `camera.yaml` 两处同步。
- [ ] 真机验证 `live.py` 的 ROS 取图和 pyAgxArm 位姿读取。
- [ ] 确认 `config/site/local.yaml` 要不要进仓库（目前没有被 gitignore）。
- [ ] L2 果蔬识别：还没做，开关目前只对 L1 生效。
- [ ] 可选：把 `zone_exclude_margin` 改成 0 后重跑一遍对比。

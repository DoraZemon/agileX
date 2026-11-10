# 赛前收尾 TODO（PiPER 智能操作挑战 L1/L2）

> 读者：后续落实这些任务的模型（以及项目负责人本人）。
> 工程位置：`~/vision/ws/src/cube_task`（ROS2 Humble，ament_python）。
> 执行顺序：P0 → P1 → P2 → P3 → P4。**P0 是硬门槛**：P0 没有全部验收通过，不得做后面任何一项。P1 的指标如果在 P1-6 修一轮后仍未达到，记录原因后继续 P2，不能拖到 11/12。时间不够时按最后一节「砍项和保底」处理。
> 每一项都写了「验收」，验收数字必须是实际跑出来的，不得估计或编造。

## 0. 给执行模型的规则（动手前读完第 0、1、2 节）

### 0.1 背景

- 比赛：松灵杯 PiPER 智能操作挑战，2026-11-14。
  - L1：3 个 50 mm 泡沫方块，限时 180 s，按颜色放到同色卡纸区。
  - L2：6 个果蔬模型，限时 300 s，按类别放到对应卡纸区。
  - L3（大模型探索赛）本次不做，不要改 L3 相关代码，只保证它的测试不坏。
- 机械臂（PiPER）和相机（奥比中光“大白”）由现场提供；相机由组委会安装和标定，队伍不能动它；队伍只带电脑。
- 日程：11/10 线上培训；11/12–13 现场适配，真机时间很少；11/14 比赛。
- 夹爪是 70 mm 还是 100 mm 行程目前未知。它直接决定 L2 能抓几类，见 2.2 节。

### 0.2 环境

| 项目 | 值 |
|---|---|
| 工程目录 | `~/vision/ws/src/cube_task`（git 仓库根目录是 `~/vision`） |
| 系统 Python | `/usr/bin/python3`，3.10，numpy 1.21.5，OpenCV 4.5.4，跑普通测试和真机 |
| MuJoCo | 只装在 `~/.venvs/piper-mujoco`；用 `bash scripts/mujoco_run.sh <tool> ...` 运行；测试用 `MUJOCO_GL=egl env -u PYTHONPATH ~/.venvs/piper-mujoco/bin/python -m pytest ...` |
| ROS | `source /opt/ros/humble/setup.bash`（跑 lint 和全量测试前必须先 source） |
| 编译 | 只能用 `colcon build --symlink-install`；普通 `colcon build` 会让工具找不到配置 |
| 日志 | 真机和仿真运行日志在 `~/vision_runs/`；MuJoCo 视频在 `~/vision_runs/mujoco/` |

### 0.3 必须遵守

1. 一次只做一个 TODO 项，只改该项列出的文件。做完单独提交一次 git，提交信息以 TODO 编号开头，例如 `P0-3: ...`。
2. 不得为了通过测试或验收而：
   - 删除测试，或给失败的测试加 skip / xfail；
   - 放宽“放错数为 0”之类的断言；
   - 修改 `robot/safety.py` 里的工作空间、桌面高度、倾斜角限制；
   - 改大 `robot.speed_percent` 默认值，或改小 `grasp.finger_clearance`（0.005）；
   - 修改基准测试的场景、种子或本文件里的验收阈值。
3. 真机相关代码里不得调用 `disable()` / `reset()`，不得绕开 `SafeArm` 发运动指令。
4. 不得往系统 Python 装任何包。MuJoCo 测试文件必须用 try-import + `pytestmark = skipif` 的写法（照抄 `test/test_mujoco_scene.py`）；不能在模块顶层写 `pytest.importorskip`，否则整个 `test/` 会收集 0 个测试却显示成功。
5. 代码保持 Python 3.8 兼容（每个模块开头 `from __future__ import annotations`）；`ament_flake8`（行宽 99）和 `ament_pep257`（D213 风格）必须通过。
6. 不改 `third_party/` 子模块；不改 `config/site/` 里由现场工具写入的文件（`poses.yaml`、`colors.yaml`、`extrinsics.yaml`）。
7. 保留用户已有的改动，不要回退：
   - 固定相机放完物体后在原地拍照、不回观察位（`task/runner.py` 的 `_capture_in_place` / `_retry_from_observe`，配置 `runner.return_to_observe_after_pick: false`）；
   - `scripts/mujoco_run.sh` 在桌面开 `--viewer` 时用 GLFW，否则用 EGL。
8. 汇报只写实际跑出来的数字，附命令和日志路径。验收没通过就写“未通过”和原因，不要改验收标准，也不要跳到下一级。
10. **同一时间只允许一个执行者改这个仓库。** 上一次就是几个代理同时改文件，留下了半成品和红测试。开始前先 `git status`，有不属于本项的未提交改动就停下来汇报。
11. 不确定的事（硬件行为、比赛规则）不要猜，写进汇报的“遗留问题”，由负责人去问。
9. 每一项完成后都要跑下面的统一检查，结果写进汇报：

```bash
cd ~/vision/ws/src/cube_task && source /opt/ros/humble/setup.bash
python3 -m pytest -q -p no:cacheprovider          # 0 failed、0 error；收集数不少于 1161
MUJOCO_GL=egl env -u PYTHONPATH ~/.venvs/piper-mujoco/bin/python -m pytest -q -p no:cacheprovider test/test_mujoco_*.py   # 0 failed
```

全量测试里已经包含 `test_flake8.py` 和 `test_pep257.py`，所以 lint 也在其中。

### 0.4 交给执行模型的提示词模板

```text
你在 ~/vision/ws/src/cube_task 工作。先完整阅读 ~/vision/schedule/赛前收尾TODO.md 的第 0、1、2 节，
然后只完成 TODO 项 <编号>。严格按该项的“步骤”和“验收”执行，验收数字必须实际跑出来。
不要修改该项没有列出的文件。完成后 git commit（信息以 <编号> 开头），
并按该项“汇报”的要求回复；没通过的验收如实写出。
```

### 0.5 汇报格式（每项都用）

```text
编号：P?-?          commit：<hash>
改动文件：...
验收：逐条写“通过 / 未通过”，附实测数字
统一检查：系统测试 N passed / N failed；MuJoCo 测试 N passed / N failed
命令和日志：...
遗留问题：...
```

### 0.6 术语

| 术语 | 含义 |
|---|---|
| 轮成功 | 本轮要求的物体全部按仿真真值放对、放错 0 个、在限时内（`sim_bench` 的 success） |
| 放错 | 物体停在不属于它类别的区域里（比不放更糟，可能扣分） |
| 掉落 | 物体最后不在桌上也不在任何区域（MuJoCo 的 dropped） |
| 单类放对率 | 某一类物体在所有轮里最终被放对的比例 |
| 首次成功率 | 第一次尝试就放对的比例 |
| 运动学仿真 | `--backend sim`，没有物理，验证识别、规划和流程逻辑 |
| MuJoCo | `--backend mujoco`，有接触物理，验证打滑、滚动、掉落 |

## 1. 当前状态（截至 2026-11-09 22:55）

- **git**：仓库 `~/vision`，分支 `main`，最后一次提交 `7a38132 demo`。之后约 25 个文件有未提交的改动，另有新测试 `test/test_mujoco_official.py`、`test/test_plan_official_*.py`。
- **后台工作流** `wf_27b0de55-f94`（官方物料适配）已经**停止，不再重启**，剩余工作全部按本文件做。停止时各部分的状态：

| 部分 | 文件 | 状态 |
|---|---|---|
| MuJoCo 场景 | `sim/mujoco/scene.py`、`world.py` | 完成：组合形状、真实质量和摩擦、20 cm 卡纸、70/100 mm 两种夹爪 |
| 抓放规划 | `perception/grasp.py`、`segment.py`、`task/reach.py`、`skills.py`、`robot/safety.py` | 完成：按高度选夹持截面、梨夹颈部、高物体降低接近高度、长物体沿卡纸对角线、平铺卡纸低位释放、L2 夹持力 2.5 N |
| 运动学仿真的官方物料 | `sim/scenarios.py`、`sim/world.py`、`sim/render.py` | **未完成**，20:30 仍在修改 |
| 基准测试、修复、视频 | — | 未开始 |
| 文档 | — | 未开始 |

- **测试**（22:55 抽查）：`test_sim_scenarios.py` 有失败，`test_e2e_l1.py` 4 个失败（例如 `test_targets_pick_only_the_given_labels_in_order`），`test_e2e_l2.py` 之前也有失败。**代码现在不是全绿状态，不能直接上真机。** 全量结果在 P0-1 重新跑。
- **工具缺选项**：`sim_bench` 和 `run_task` 还没有 `--materials` / `--layout` / `--gripper-width`，没法按夹爪宽度跑基准（P0-4 补）。
- **最后一个可用的旧版本**：提交 `7a38132`（旧物料，全量测试当时全绿）。如果 P0 在 11/11 晚上仍未完成，按 CUT 节的保底方案处理。
- **默认配置还是旧物料**：`perception.l1_block_size` 是 0.035（应为 0.05）；`tasks.L1.labels` 是 `[red, green, blue]`（应含 yellow）；代码读取的 11 个新键在 `config/default.yaml` 里不存在（清单见 P0-3）。
- `~/vision_runs/mujoco/` 里现有的 5 个视频都是旧物料（小方块、苹果、托盘），不能代表官方物料。

## 2. 已确定的事实和已测得的结果

### 2.1 官方物料（组委会公布，视为真值）

| 项目 | 内容 |
|---|---|
| 放置区 | 3 张平铺硬卡纸，每张 200 × 200 mm，没有围边：绿色、黄色、红色。在桌上的位置未知 |
| L1 方块 | EVA 泡沫，约 10 g，50 × 50 × 50 mm（±2 mm）。目标色：红、绿、黄。官方照片里还有粉、橙、蓝、紫，可能作为干扰块出现 |
| L1 映射 | red → zone_red，green → zone_green，yellow → zone_yellow |
| 绿区（L2） | 黄瓜：深绿，长 260 mm，直径 40 mm，略弯，43 g；梨：黄绿，立放，高 125 mm（含短柄），下部最宽 85 mm，上部收窄成颈，100 g |
| 黄区（L2） | 香蕉：黄，弯，长 180 mm，直径 40 mm，50 g；土豆：黄，椭圆，100 × 70 mm，厚度未知（按 55 mm 估），57 g |
| 红区（L2） | 柿子：橙，扁圆，宽 70 mm，高 50 mm，顶上有平的绿萼片，100 g；彩椒：红，高 90 mm（含绿柄），宽 70 mm，82 g |
| 标签名 | `cucumber`、`pear`、`banana`、`potato`、`persimmon`、`bell_pepper`；区域名 `zone_green`、`zone_yellow`、`zone_red` |

### 2.2 两种夹爪能抓什么（MuJoCo 实测 + 规划器计算）

MuJoCo 试验方法：顶部抓取，抬升 15 cm，搬运约 4 s。规划器的“可用开口”= 夹爪行程 − 2 × 5 mm 手指余量。

| 类别 | 70 mm 夹爪 | 100 mm 夹爪 | 备注 |
|---|---|---|---|
| 50 mm 方块 | 能，1.5 N 稳 | 能 | |
| 黄瓜 | 能，1.5 N 稳 | 能 | |
| 香蕉 | 能，1.5 N 稳 | 能 | |
| 梨 | 只能夹颈部，且需 ≥ 2.5 N | 同左 | 顶下 30 mm 处夹必掉；顶下 40 mm 和 50 mm 处 2.5 N 能稳；夹住后梨会倾斜 20–25° |
| 土豆 | **不能**（截面约 72 mm） | 能 | |
| 柿子 | **不能**（约 71 mm） | 能 | |
| 彩椒 | **不能**（67–70 mm） | 能 | |

结论：70 mm 夹爪时 L2 每轮最多放对 3 个（黄瓜、香蕉、梨）；100 mm 时 6 个都能抓。L2 夹持力用 2.5 N，3.0 N 在仿真里没有更多余量；L1 泡沫方块用 1.5 N。

**风险**：规划器给梨选的夹持高度是离桌面 86 mm，也就是顶下约 39 mm，正好在“30 mm 必掉、40 mm 能稳”的边界附近。P1-4 必须专门验证。

### 2.3 放置（MuJoCo 实测）

- 在平铺卡纸上，从 1.5 cm 高、靠近中心松手：所有类别都留在卡纸上，漂移 ≤ 1.4 mm。
- 从 6 cm 高、离中心 70 mm、带 0.3 m/s 横向速度松手：所有类别都滑出卡纸。所以平铺卡纸必须低位（`place.release_height_flat` 0.015）、靠中心释放。
- 黄瓜 260 mm 比卡纸 200 mm 长，只能沿对角线放，两端余量约 2 mm。物体是否允许超出卡纸要在培训时问清（见 Q 节）。
- 绿区要放黄瓜和梨：先放黄瓜（对角线），梨放在角上，规划间隙约 1 mm。如果用 `tasks.L2.order` 强制梨先放，放黄瓜时手指会压到梨约 11 mm。**所以 L2 不要设置 `order`。**
- 仿真里塑料模型的弹跳性是假设的。用“欠阻尼”接触做敏感性测试时，土豆从 6 cm 会滚出卡纸，梨会倒。

### 2.4 可达性

- PiPER 第 5 关节只有 ±70°。TCP 0.14 m 时，竖直向下的工具最高只能到 TCP z ≈ 0.10 m（半径 0.2–0.3 m）、0.065 m（0.40 m）、0.02 m（0.45 m）。更高的位姿由程序自动外倾：抓和放的位姿最多 30°，空中位姿最多 40°（`robot.workspace.max_tilt_free_deg`）。
- 梨高 125 mm：接近位姿会自动降到顶上 2 cm（`grasp.min_approach_clearance`）。
- 100 mm 夹爪如果更长，TCP 会变长，可达范围会再缩小。现场必须量 TCP。

### 2.5 仍然未知（11/10 培训时问，见 Q 节）

夹爪型号和行程、TCP 长度、3 张卡纸和拾取区的位置、相机安装方式和标定文件格式、L1 是否有干扰块、“指定目标”怎么给、物体能否超出卡纸、果蔬模型的材质。

## 时间表

| 级别 | 内容 | 截止 |
|---|---|---|
| Q | 11/10 培训时问清的问题（负责人本人做） | 11/10 培训结束 |
| P0 | 代码恢复全绿、补齐配置和工具选项 | 11/10 22:00 |
| P1 | 官方物料下的仿真指标（运动学 + MuJoCo） | 11/11 14:00 |
| P2 | 赛前离线准备：配置模板、文档、离线包、备份 | 11/11 22:00 |
| P3 | 现场适配（11/12–13，负责人在真机上做，模型只协助） | 11/13 适配结束 |
| P4 | 比赛日流程 | 11/14 |

## P0 代码恢复全绿（硬门槛）

### P0-1 快照和基线

- **目的**：先把现在的半成品存档，保证随时能回退。
- **步骤**：
  1. `cd ~/vision && git tag legacy-green 7a38132`（最后一个全绿的旧物料版本）。
  2. `git add -A && git commit -m "P0-1: snapshot of the unfinished official-materials work (tests red)"`。
  3. 跑一次全量测试，把结果存档：`python3 -m pytest -q -p no:cacheprovider > ~/vision_runs/p0_baseline.txt 2>&1`；MuJoCo 测试同样存到 `~/vision_runs/p0_baseline_mujoco.txt`。
  4. 把失败的测试按文件归类，写进汇报。
- **改动文件**：无代码改动。
- **验收**：tag `legacy-green` 存在；快照 commit 存在；两份基线文件存在，失败数已写入汇报。

### P0-2 完成运动学仿真的官方物料

- **文件**：`cube_task/sim/scenarios.py`、`cube_task/sim/world.py`、`cube_task/sim/render.py`、`cube_task/sim/teach.py`、`test/test_sim_*.py`。
- **要求**：
  1. `make_scenario(task, seed, mount, difficulty, materials='official', layout='A', gripper_width=None)`：`materials` 默认 `official`，`legacy` 必须和旧版本完全一样。
  2. 官方物料按 2.1 节建模，用 `compound` 组合形状；L1 是 50 mm 方块；`hard` 难度加 1–2 个粉/橙/蓝/紫干扰块，并让两个方块紧贴。
  3. 放置区是 3 张 0.2 × 0.2 m 平铺卡纸（`wall_height` 0），名字是 `zone_green` / `zone_yellow` / `zone_red`。
  4. 布局 A、B 都要可达：每张卡纸的 4 个角和释放位姿在外倾 ≤ 30° 时有逆解；拾取区和卡纸不重叠（留 2 cm）。
  5. 两种相机安装都能看到整个拾取区和 3 张卡纸。
  6. `scenario_config_overlay` 要写入：L1/L2 的 labels 和 class_to_zone、zones、`perception.l1_block_size` 0.05、拾取区多边形；给了 `gripper_width` 时还要写 `robot.gripper.max_width`。
  7. `sim/teach.py` 的仿真 KNN 用官方物料、两种安装一起训练，并输出每类的留一法准确率和跨区混淆。
- **验收**：
  - `python3 -m pytest -q -p no:cacheprovider test/test_sim_*.py` 0 失败。
  - 下面 320 个组合全部能生成，且同一组参数两次生成的结果完全一致：L1/L2 × 种子 0–19 × 两种安装 × 布局 A/B × 夹爪 0.07/0.10。
  - 每个场景里，夹爪能抓的物体在 `try_grasp` 下确实能抓。
  - 仿真 KNN 留一法每类准确率 ≥ 95%，跨区混淆 0（梨↔土豆/香蕉、柿子↔土豆尤其要看）。
  - `materials='legacy'` 的旧测试全部照常通过。
- **汇报**：两个布局的卡纸坐标和可达余量；70 mm 夹爪下哪些类别不可抓。

### P0-3 默认配置补齐

- **文件**：`config/default.yaml`、`config/examples/tasks.example.yaml`。
- **要求**：
  1. 改值：`perception.l1_block_size` 0.05；`tasks.L1.labels` `[red, green, yellow]`。
  2. 代码读取但 `default.yaml` 里缺的键全部补上，带注释，值用代码里的默认值：

| 键 | 默认值 |
|---|---|
| `grasp.section_profile` | true |
| `grasp.finger_clearance` | 0.005 |
| `grasp.pad_half` | 0.015 |
| `grasp.min_grasp_height` | 0.008 |
| `grasp.min_approach_clearance` | 0.02 |
| `grasp.transit_object_clearance` | 0.03 |
| `place.release_height_flat` | 0.015 |
| `robot.workspace.max_tilt_free_deg` | 40 |
| `tasks.L1.gripper_force` | 1.5 |
| `tasks.L2.gripper_force` | 2.5 |
| `sim.mujoco.gripper_max_width` | null |
| `sim.mujoco.materials` | null |
| `sim.mujoco.card_collision` | true |
| `sim.mujoco.grip_force_scale` | 2.0 |
| `sim.mujoco.gripper_kp` | 1500 |
| `sim.mujoco.gripper_kv` | 15 |
| `sim.mujoco.gripper_speed` | 0.10 |

  3. `robot.gripper.max_width` 保持 0.07，注释写明“SITE：100 mm 夹爪改成 0.10”。
  4. `tasks.example.yaml` 改成官方的 L1/L2 标签和映射（2.1 节）。
- **验收**：
  - 写一个检查脚本（放 `test/test_config_keys.py`，作为测试保留）：扫描 `cube_task/` 下所有 `cfg.get('...')` / `_float(cfg, '...')` / `_required_float(cfg, '...')` 的完整键路径，每个都必须在 `default.yaml` 里存在。结果 0 缺失。
  - `python3 -c "import yaml; yaml.safe_load(open('config/default.yaml'))"` 不报错。

### P0-4 工具加上物料、布局、夹爪宽度选项

- **文件**：`cube_task/tools/common.py`、`cube_task/tools/sim_bench.py`、`cube_task/sim/mujoco/run.py`，以及相应测试。
- **要求**：
  1. `run_task` / `view` 等工具在 `--mode sim` 和 `--mode mujoco` 下支持 `--materials official|legacy`（默认 official）、`--layout A|B`、`--gripper-width 0.07|0.10`。
  2. `sim_bench` 支持 `--materials`、`--layouts A,B`、`--gripper-widths 0.07,0.10`，对两种 `--backend` 都有效。
  3. 夹爪宽度要同时设 `robot.gripper.max_width`，MuJoCo 下还要设 `sim.mujoco.gripper_max_width`。
  4. `sim_bench` 的输出和 JSON 增加**按类别统计**：每类的出现次数、尝试次数、放对、放错、掉落、因 `too_wide` 跳过的次数、失败原因。
- **验收**：
  - 这两条命令都能跑完，并打印按类别的统计表：
    - `python3 -m cube_task sim_bench --tasks L2 --seeds 0-1 --layouts A,B --gripper-widths 0.07,0.10 --jobs 4`
    - `bash scripts/mujoco_run.sh sim_bench --backend mujoco --tasks L1 --seeds 0 --gripper-widths 0.10`
  - `test/test_tools_runtime_sim_bench.py` 覆盖新选项。

### P0-5 修好所有失败的测试

- **文件**：失败测试涉及的代码和测试。
- **要求**：
  - 测试失败是因为默认物料从旧目录换成了官方物料：在测试里显式写 `materials='legacy'`，或按官方物料改期望值。
  - 测试失败是因为代码有 bug：修代码。
  - 特别注意 `test_e2e_l1.py::test_targets_pick_only_the_given_labels_in_order`。它对应 L1 的“裁判指定目标”功能，必须真的修好，不能把测试改松。
  - 新增官方物料的端到端测试 `test/test_e2e_official.py`（运动学仿真）：
    - L1：种子 0、1，两种安装，3/3 放对，放错 0；
    - L2 + 0.10 夹爪：种子 0，两种安装，6/6 放对，放错 0；
    - L2 + 0.07 夹爪：种子 0，黄瓜、香蕉、梨放对，其余 3 类一次都不尝试抓取，放错 0。
- **验收**（P0 的总门槛）：
  - 系统 Python 全量测试 0 failed、0 error，收集数 ≥ P0-1 基线；
  - MuJoCo 测试 0 failed；
  - `colcon build --symlink-install --packages-select cube_task` 成功；
  - `git tag p0-green`。

## P1 官方物料下的仿真指标

运行时 `robot.speed_percent` 保持 30，不得为了时间指标调高速度。每个基准都要用 `--json` 存结果到 `~/vision_runs/p1/`，并用 `--save-logs` 保留失败轮次的日志。

### P1-1 运动学仿真基准

- **命令**：

```bash
python3 -m cube_task sim_bench --tasks L1,L2 --seeds 0-19 \
  --mounts eye_to_hand,eye_in_hand --layouts A,B --gripper-widths 0.07,0.10 \
  --jobs 12 --json ~/vision_runs/p1/kin_main.json --save-logs
```

- **指标**：

| 场景 | 指标 |
|---|---|
| 所有组合 | 放错 = 0 |
| L1（两种夹爪） | 轮成功率 ≥ 95%；平均用时 ≤ 90 s，最长 ≤ 120 s |
| L2 + 0.10 | 轮成功率 ≥ 90%；平均用时 ≤ 200 s，最长 ≤ 260 s |
| L2 + 0.07 | 黄瓜、香蕉、梨的单类放对率都 ≥ 90%；土豆/柿子/彩椒的抓取尝试次数 = 0；三个可抓物体放完后 ≤ 30 s 内结束本轮 |

### P1-2 MuJoCo 物理基准

- **命令**：`bash scripts/mujoco_run.sh sim_bench --backend mujoco --tasks L1,L2 --seeds 0-9 --mounts eye_to_hand,eye_in_hand --layouts A --gripper-widths 0.07,0.10 --jobs 6 --json ~/vision_runs/p1/mj_main.json --save-logs`
- **指标**：

| 场景 | 指标 |
|---|---|
| 所有组合 | 放错 = 0；掉落物体数 ≤ 物体总数的 2% |
| L1 | 轮成功率 ≥ 90%；首次成功率 ≥ 85% |
| L2 + 0.10 | 轮成功率 ≥ 80%；每一类的单类放对率 ≥ 85% |
| L2 + 0.07 | 黄瓜、香蕉单类放对率 ≥ 90%，梨 ≥ 80% |
| 用时 | L1 平均 ≤ 90 s；L2（0.10）平均 ≤ 200 s，最长 ≤ 260 s |

### P1-3 鲁棒性

- **命令**：在 P1-2 的命令上分别加：
  - `--extrinsics-error 0.003 0.5`
  - `--extrinsics-error 0.005 1.0`
  - `--difficulty hard`（不加外参误差）
- **指标**：
  - 所有轮次放错 = 0。
  - 3 mm / 0.5°：L1 轮成功率 ≥ 85%，L2（0.10）≥ 75%。
  - 5 mm / 1°：只记录结果，不设门槛。如果固定相机下 L2 低于 50%，在汇报里单独标出，作为现场必须重新核对外参的依据。
  - `hard`：干扰块被抓取的次数 = 0；紧贴方块的 L1 轮成功率 ≥ 85%。

### P1-4 梨的专项验证

- **原因**：规划器给梨选的夹持点在顶下约 39 mm，正好在“30 mm 必掉、40 mm 能稳”的边界附近（见 2.2 节）。
- **步骤**：用完整流程（识别 → 规划 → MujocoArm）跑 20 个只有一个梨的场景，梨的位置和朝向随机，两种夹爪各跑一遍；夹持力 2.5 N。
- **指标**：每种夹爪 ≥ 18/20 放进绿区；梨倒下不算失败，只要在区域内。
- **如果不达标**：只允许在 `perception/grasp.py` 里调颈部夹持高度规则（例如要求夹持点在顶下 ≥ 45 mm），并加回归测试。

### P1-5 L1 干扰色专项

- **原因**：粉色和红色接近，橙色在红黄之间。把干扰块放进目标区就是放错。
- **步骤**：
  1. 在 `config/default.yaml` 的颜色类别里补上 `pink`。
  2. 把粉、橙、蓝、紫加进 `tasks.L1.labels`，但不放进 `class_to_zone`。这样它们会被识别成自己的颜色，并且永远不会被抓。
  3. 在运动学和 MuJoCo 渲染里各统计 ≥ 200 个干扰块实例。
- **指标**：干扰块被识别成 red / green / yellow 的次数 = 0；红、绿、黄目标块被识别成其他颜色的比例 ≤ 2%。
- **注意**：自检对“有标签没区域”只报 WARN，这是预期行为，不要改成 FAIL。

### P1-6 修一轮

- 按 P1-1 到 P1-5 的失败原因排序，修排在前 3 的、能在代码里修的问题。“70 mm 夹爪抓不了”这类硬件限制不算。
- 每个修复都要加回归测试，并重跑受影响的那部分基准，汇报修复前后的对比。
- 修复后统一检查必须仍然全绿。

### P1-7 演示视频

- 用官方物料录 3 段视频，放到 `~/vision_runs/mujoco/`：
  - `official_L1_s0.mp4`
  - `official_L2_s0_w100.mp4`
  - `official_L2_s0_w70.mp4`
- 命令示例：`bash scripts/mujoco_run.sh run_task --task L2 --mode mujoco --seed 0 --gripper-width 0.10 --video ~/vision_runs/mujoco/official_L2_s0_w100.mp4`
- **验收**：每段视频抽 3 帧检查：物体外形和颜色像真物（黄瓜细长深绿、梨立着、彩椒红身绿柄、柿子橙色带萼片）；3 张卡纸都能看见；画面里的结果和打印的物理真值一致。

## P2 赛前离线准备（11/11）

### P2-1 培训答案落到配置

- **输入**：负责人填好的 Q 节答案。
- **文件**：`config/examples/*.yaml`、`docs/需求分析.md`。
- **要求**：每个已知答案都写进对应的配置模板，仍未知的标成 `SITE: 待现场确认`；不得猜测取值。
- **验收**：Q 节每一条都有“已写入 X”或“待现场确认”的记录。

### P2-2 两套夹爪配置，现场只需复制

- **文件**：新建 `config/examples/gripper70.yaml`、`config/examples/gripper100.yaml`，更新 `config/examples/tasks.example.yaml`，并在 `config/default.yaml` 的 `site_overlays` 里加一行 `config/site/gripper.yaml`。
- **内容**：
  - `gripper70.yaml`：`robot.gripper.max_width: 0.070`，`tasks.L2.gripper_force: 2.5`，注释写明“土豆、柿子、彩椒会被跳过”。
  - `gripper100.yaml`：`robot.gripper.max_width: 0.100`，`tasks.L2.gripper_force: 2.5`，并提醒必须在现场量 `robot.tcp_offset`（100 mm 夹爪可能更长）。
- **用法**（写进文件头注释）：现场 `cp config/examples/gripper100.yaml config/site/gripper.yaml`，并把 `config/site/gripper.yaml` 加到 `default.yaml` 的 `site_overlays` 列表里（P2-2 顺手加上，文件不存在时自动忽略）。
- **验收**：对两个文件分别跑 `python3 -m cube_task check_setup --mode sim --task L2 --config config/examples/gripperXX.yaml`，0 个 FAIL；各跑一轮 `run_task --task L2 --mode sim --no-wait --config ...`，结果与 P1-1 一致。

### P2-3 文档更新

- **文件**：`docs/现场流程.md`、`docs/配置说明.md`、`docs/仿真与测试.md`、`docs/需求分析.md`、`README.md`。
- **要求**：
  - 现场流程按本文件 P3 的顺序重写：示教 3 张 0.2 × 0.2 卡纸（`wall 0`）、`perception.l1_block_size 0.05`、标定包括干扰色在内的所有颜色、学习 6 类果蔬、量 TCP、第一天先确认夹爪行程。
  - 需求分析加“官方物料”一节，内容用本文件 2.2–2.4 节和 P1 的实测数字。
  - 仿真文档写清 `--materials` / `--layout(s)` / `--gripper-width(s)` 和 MuJoCo 命令。
- **验收**：用脚本把文档里所有 `python3 -m cube_task ...` 和 `scripts/*.sh` 命令行交给对应工具的 argparse 解析，0 个错误；文档里的数字和 P1 汇报一致。

### P2-4 离线包和备份

- **步骤**：
  1. `cd ~/vision && git bundle create ~/vision_backup_$(date +%m%d).bundle --all`。
  2. 把 bundle、`~/.cache/cube_task/`（仿真 KNN 和 MuJoCo 模型）、`ws/src/cube_task/models/` 拷到 U 盘。
  3. 在一个空目录里从 bundle 克隆，带子模块，`colcon build --symlink-install`，跑全量测试。
  4. 断开网络后跑一轮 `run_task --task L1 --mode sim --no-wait`，确认没有任何联网下载。
- **验收**：克隆目录里全量测试 0 failed；断网仿真轮跑完；U 盘内容列表写进汇报。

### P2-5 真机通路的空转检查（无硬件）

- **步骤**：
  - `bash scripts/start.sh --help`；
  - `python3 -m cube_task check_setup --task L1`；
  - `python3 -m cube_task check_setup --task L2 --live`；
  - `ros2 run cube_task run_task --help`。
- **验收**：没有硬件时，每个 FAIL 下面都有一行 `fix:` 提示，没有 Python traceback；`ros2 run` 能找到配置（说明是用 `--symlink-install` 编的）。

### P2-6（可选）用自己的 D435i 验证真实深度

- 只在有时间、手上有类似物体时做，不影响其他项。
- 启动 `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true`，用 `capture --session test` 录 20 帧，再用 `view --mode replay --replay-dir <目录> --task L1` 看分割效果。
- **验收**：桌面拟合 rms ≤ 3 mm；50 mm 的物体被分割成单独实例，尺寸误差 ≤ 5 mm。不达标只记录，不改代码。

## P3 现场适配（11/12–13，负责人在真机上做）

> 模型在这一级只能帮忙解释日志、改配置。任何让机械臂运动的命令都由人执行，且人的手放在急停上。
> 每一步没过就不进入下一步。按下表顺序，时间是估计值。

| # | 做什么 | 命令 / 方法 | 通过标准 | 估时 |
|---|---|---|---|---|
| 1 | 接线、CAN、自检 | `bash scripts/can_up.sh can0`；`python3 -m cube_task check_setup --task L1 --live` | CAN UP、码率 1000000；能读到关节、TCP、夹爪宽度；相机有帧 | 15 min |
| 2 | **确认夹爪行程** | 示教器或官方示例把夹爪全开，读 `check_setup --live` 里的宽度 | 得到 0.07 或 0.10；复制对应的 `gripper70/100.yaml` 到 `config/site/` | 5 min |
| 3 | 量 TCP | 尺子量法兰面到指尖中心 | 写进 `config/site/local.yaml` 的 `robot.tcp_offset` 第 3 个数 | 5 min |
| 4 | 相机和外参文件 | 问清话题名和外参文件；`view --task L1` | 画面里拾取区和 3 张卡纸的投影位置与实物对得上；桌面 rms ≤ 3 mm | 20 min |
| 5 | 示教观察位、拾取区、卡纸 | `python3 -m cube_task teach_poses`：`o` 观察位；`p` 触碰拾取区 4 角；`z` 触碰每张卡纸中心，名字 `zone_green/zone_yellow/zone_red`，尺寸 `0.2 0.2`，围边 `0`；`w` 写入 | 写入后每张卡纸的可达性检查全部 OK；`perception.table.expected_z` 已写入 | 30 min |
| 6 | 核对外参 | `python3 -m cube_task calib_verify --task L1 --all --hover 0.03 --speed 20` | 至少 3 个物体上方悬停，指尖中心与物体中心偏差 ≤ 5 mm；超过时先问工作人员，经同意后才用 `--apply` 或 `calib_points` | 20 min |
| 7 | 颜色标定 | `python3 -m cube_task color_calib --task L1` | 现场所有颜色（包括干扰色）都标定；重叠告警为 0；预览时每个方块的颜色都对 | 20 min |
| 8 | 果蔬学习 | `python3 -m cube_task teach_classes --task L2` | 6 类每类 ≥ 10 个样本，来自 ≥ 3 个不同摆放；留一法 ≥ 95%；跨区混淆 0 | 30 min |
| 9 | 空跑 | `python3 -m cube_task run_task --task L1 --dry-run --speed 20` | 每个目标都悬停到位，夹爪不合；路径不碰任何东西 | 10 min |
| 10 | 单物体真抓 | 每类只放 1 个在桌上跑 `run_task --speed 20`，每类 5 次 | 每类 ≥ 4/5 放对，且没有放错 | 40 min |
| 11 | 整轮 | L1 跑 3 轮，L2 跑 2 轮，速度 30 | 连续 3 轮都成功后，才可以试速度 50；记录每轮用时 | 40 min |
| 12 | 冻结 | 在 `~/vision` 提交 `config/site/` 和模型文件；重做 P2-4 备份 | 有 commit 和新 bundle | 10 min |

现场遇到问题先查 `docs/现场流程.md` 的故障表。这几条不要在现场改：安全检查的阈值、夹持高度规则、识别算法。现场只调配置。

## P4 比赛日（11/14）

1. **开赛前**：
   - `bash scripts/can_up.sh can0`
   - `python3 -m cube_task check_setup --task L1 --live` 和 `--task L2 --live`，都必须是 0 FAIL。
   - 确认 `config/site/` 是 11/13 冻结时的版本：`git status` 干净。
2. **L1 每轮**：
   - `bash scripts/start.sh L1 --targets <裁判指定的颜色，按顺序>`。裁判没有指定时不加 `--targets`。
   - 如果指定的目标少于 3 个，加 `--set tasks.L1.required=N`。
   - 安全开关确认后按 Enter。听到开始信号再按 Enter 开始计时。
3. **L2 每轮**：`bash scripts/start.sh L2`。L2 不要设置 `order`。
4. **运行中**：
   - 看到不对，按 `p` 停住保持；需要急停，按 `e` 或硬件急停。
   - 程序不会自动重启。每轮只有一次经裁判同意的重启，最长 3 分钟，计入本轮时间。
   - 夹爪夹着东西停下时，重启前先把物体拿走。程序会拒绝在夹着物体时启动。
5. **每轮结束后**：保留 `~/vision_runs/` 里本轮的日志目录，用于申诉。成绩公示后 10 分钟内可以提一次书面申诉。
6. **两轮之间**：只改有实测依据的配置，例如速度、夹持力、颜色范围，不改代码。

## Q 11/10 培训时要问清的问题（负责人本人做）

> 按顺序问，前 4 条最要紧。答案当天填进右边一列，P2-1 再据此写进配置。问不到的写“未答复”，不要猜。

| # | 问题 | 为什么重要 | 答案写到哪里 | 答案 |
|---|---|---|---|---|
| 1 | 夹爪是哪一款？行程 70 mm 还是 100 mm？法兰面到指尖中心多长？ | 70 mm 时土豆、柿子、彩椒抓不了（2.2 节）；TCP 长度决定可达范围和下探深度 | `robot.gripper.max_width`、`robot.tcp_offset` | |
| 2 | 相机装在哪里（固定支架还是手腕）？用什么驱动、ROS 话题名是什么？深度是否已对齐到彩色图？ | 决定坐标链和取图方式 | `camera.mount`、`camera.ros.*` | |
| 3 | 官方给的标定结果是什么格式？矩阵方向（相机到基座还是相反）？父坐标系是基座、法兰还是 TCP？子坐标系是不是 `*_optical_frame`？如果相对 TCP，标定时驱动的 `tcp_offset` 是多少？ | 方向或坐标系弄错，抓取会偏几厘米甚至几十厘米；程序会拒绝非 optical 坐标系和缺 `tcp_offset` 的 TCP 外参 | `camera.extrinsics_file` | |
| 4 | 3 张卡纸和拾取区在桌上的位置固定吗？每轮会不会变？卡纸是否贴在桌上？ | 决定要不要每轮重新示教区域 | `teach_poses` 示教 | |
| 5 | L1 的“指定目标”怎么给（口头、卡片、颜色顺序）？每轮是不是 3 个都要放？会不会有粉、橙、蓝、紫干扰块？ | 决定 `--targets` 的用法和颜色标定范围 | `run_task --targets`、`tasks.L1.labels` | |
| 6 | L2 每轮是不是 6 类各一个？初始怎么摆（梨是否立放、物体会不会挨在一起）？ | 影响抓取规划和成功率预期 | `tasks.L2.*` | |
| 7 | 怎样算“放置正确”：重心在卡纸内？整体在卡纸内？可以超出边缘吗？黄瓜 26 cm 比卡纸长怎么判？物体倒下算不算？ | 决定释放位置和是否必须沿对角线放 | `place.*` | |
| 8 | 计时从什么时候开始？计时前能否先把机械臂移到观察位？什么时候算结束？ | 决定启动流程（`runner.start_on_enter`） | 启动流程 | |
| 9 | 11/12–13 每队能用真机多久？能否在现场采集图片？ | 决定 P3 的步骤能做完几项 | P3 时间安排 | |
| 10 | 电脑怎么接：USB-CAN 模块是现场提供还是自带？有没有网线和电源？ | 决定携带清单 | 携带清单 | |
| 11 | 允许调速度吗？有哪些安全要求（急停、围栏、操作员位置）？ | 现场流程和速度设置 | `robot.speed_percent` | |
| 12 | 果蔬模型的材质和重量偏差多大？能否提前借一套样品拍照？ | 影响夹持力和识别训练 | `tasks.L2.gripper_force`、`teach_classes` | |

## 现场携带清单（11/12 出发前核对）

| 物品 | 用途 | 核对 |
|---|---|---|
| 笔记本电脑 + 电源适配器 + 排插 | 运行程序；电池不能撑一整天 | |
| U 盘：P2-4 的 git bundle、`~/.cache/cube_task/`、`models/` | 电脑出问题时恢复 | |
| USB-CAN 模块和线（若需自带，见 Q-10） | 连 PiPER | |
| 网线、USB 延长线、转接头 | 连相机或网络 | |
| 钢尺（30 cm）和卷尺 | 量 TCP、卡纸、方块尺寸 | |
| 美纹纸胶带、记号笔 | 标记位置 | |
| 打印好的 ArUco 标记（DICT_4X4_50，40 mm） | `calib_points --target aruco` 的备用方案 | |
| 本文件和 `docs/现场流程.md` 的打印版 | 断网也能看 | |
| 下面的现场记录表（打印几份） | 记录每次运行，用于排查和申诉 | |

出发前在电脑上确认：已关闭系统自动更新；断网跑过一轮仿真；`git status` 干净；电量充满。

## 现场记录表（每次真机运行填一行）

| 时间 | 任务 | 速度 | 夹爪 | 结果（放对/要求） | 用时 | 失败原因 | 日志目录 | 改了什么配置 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

日志目录在 `~/vision_runs/` 下，名字形如 `20261112_143000_L1`。每次改配置都要记一行，否则无法知道哪次改动有效。

## 砍项和保底（CUT）

### 时间不够时按这个顺序砍

1. 先砍：P2-6（D435i）、P1-7（演示视频）、P1-3 里 5 mm / 1° 的那组。
2. 再砍：P1-3 的 `hard` 组、P1-5 的 MuJoCo 部分（保留运动学部分）。
3. 不能砍：P0 全部、P1-1、P1-2、P1-4（梨）、P2-2、P2-4、P3 第 1–9 步。

### 保底方案：P0 到 11/11 12:00 仍然没有全绿

1. 停止修官方物料的代码。P0-1 的快照 commit 保留，以后还能接着做。
2. 新建分支，从 tag `legacy-green`（`7a38132`，旧物料，全量测试全绿）出发：`git switch -c race legacy-green`。
3. 只靠配置适配官方物料，写在 `config/site/local.yaml` 和 `config/site/tasks.yaml`：
   - `perception.l1_block_size: 0.05`
   - `tasks.L1.labels: [red, green, yellow]`，`class_to_zone` 按 2.1 节；
   - `tasks.L2.labels` 和 `class_to_zone` 按 2.1 节；
   - `place.release_height: 0.015`（旧版本里这是物体底面离区域地面的高度，对平铺卡纸就是低位释放）；
   - `robot.gripper.force: 2.5`（旧版本没有按任务区分夹持力，L1 也会用 2.5 N，泡沫方块可以承受）；
   - `robot.gripper.max_width` 按 Q-1 的答案；
   - 100 mm 夹爪时加 `grasp.max_grasp_depth: 0.045`，让梨的夹持点落到顶下 45 mm 的颈部（MuJoCo 实测 2.5 N 能稳）。其他物体的夹持深度是高度的一半，不受影响。
4. 旧版本的已知限制，要提前接受：
   - 旧版本按底面最大尺寸判断能不能抓。用 70 mm 夹爪时，梨（底面 85 mm）、土豆、柿子、彩椒都会被判 `too_wide` 跳过，L2 只剩黄瓜和香蕉。
   - 不加上面的 `max_grasp_depth` 时，梨会在顶下 30 mm 处被夹，MuJoCo 实测这样**必掉**。
   - 没有长物体沿对角线释放，黄瓜可能伸出卡纸。
   - 没有官方物料的仿真，只能用旧物料仿真验证流程。
5. 用旧物料仿真跑一遍 P2-4 和 P2-5 的检查，然后直接进入 P3。

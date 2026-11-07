# L2 果蔬识别：YOLO 检测器（训练、导出、部署）

本文对应代码：`cube_task/perception/detector_onnx.py`（机器人运行时）和
`cube_task/tools/yolo/`（训练、导出、自动标注、检查）。所有数字都是 2026-11-06 在本机
（i9-12900H 20 线程，RTX 3060 Laptop 6 GB，Ubuntu 22.04）实测的。

## 1. 选哪个分类后端

后端由 `tasks.<任务>.classifier` 决定（没写时用 `perception.classifier.backend`）。
几何（位置、高度、朝向）始终来自深度分割；后端只负责给每个物体实例贴标签。

| 后端 | 原理 | 适用 | 准备工作 |
|---|---|---|---|
| `color` | HSV 颜色区间 | L1 方块 | `color_calib` 现场调颜色 |
| `knn` | 颜色+尺寸特征做 KNN | L2 的快速方案 | `teach_classes` 录样本，几分钟 |
| `onnx` | YOLO 检测框和深度实例按掩码重叠匹配（`onnx.min_mask_overlap`） | L2，模型训练好之后 | 本文 |
| `onnx+knn` | 先用检测器，检测器没框到的实例再交给 KNN | **L2 正式比赛推荐** | 两者都要 |

`onnx` 后端下，检测器没框到的实例标签是 `unknown`，不会被抓；`onnx+knn` 会用 KNN 补上。
颜色相近的类（橙子/橘子、红苹果/番茄）KNN 容易混，检测器在这类情况更可靠。

## 2. 零样本基线：COCO 模型

`models/yolo11n_coco.onnx`（COCO 80 类）里有 5 个果蔬类：banana、apple、orange、
broccoli、carrot。不训练就能试：

```yaml
# config/site/local.yaml
tasks:
  L2:
    classifier: onnx+knn
perception:
  classifier:
    onnx:
      model_file: models/yolo11n_coco.onnx
      labels_file: models/yolo11n_coco_labels.txt
      label_map: {'*': '', apple: apple, orange: orange, banana: banana,
                  broccoli: broccoli, carrot: carrot}
```

`label_map` 的规则：键是模型类名，值是任务类名，值为 `''` 的类被丢弃，`'*'` 匹配所有没列出的类
（上例 `'*': ''` 表示只保留这 5 类）。改名只能用 `label_map`，不要改标签文件。
不允许互换或链式映射（`a: b, b: a`），因为 pipeline 还会再套用一次 `label_map`。

局限（用前一定要在真实物料上实测）：

- COCO 是真实水果的照片。塑料模型的颜色、反光、纹理都不同，俯视角度下物体也更小，
  召回率可能很低。
- 容易混：orange 和 sports ball，apple 和 orange/donut。
- 葡萄、辣椒、茄子、番茄、玉米、土豆等不在 COCO 里，零样本做不了。

所以它主要用来检查整条链路是否通，正式比赛要用自己训练的模型。

## 3. 训练数据

### 3.1 组委会样本集

期望的 YOLO 格式：

```
dataset/
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt    # 每行: 类号 cx cy w h（0-1 归一化）
  data.yaml            # path, train, val, names: {0: apple, 1: banana, ...}
```

如果拿到的是其他格式：

- **COCO json / VOC xml**：用 ultralytics 自带转换（`ultralytics.data.converter.convert_coco`），
  或用 X-AnyLabeling / CVAT 导入后再导出成 YOLO 格式。
- **按类分文件夹、没有框**（`apple/*.jpg`, `banana/*.jpg`）：每个文件夹就是"单一类别的一个
  会话"。先用自己的采集训一个初版模型，再用它自动出框（框和类别无关），类别取文件夹名：

  ```bash
  python3 -m cube_task.tools.yolo.make_dataset --out ~/vision_runs/yolo/datasets/org \
      --classes apple,banana,carrot --boxes model \
      --model models/detector.onnx --labels models/detector_labels.txt \
      --session apple=/path/org/apple --session banana=/path/org/banana --preview
  ```

  然后检查 `preview/`，删掉错的帧。也可以直接用 X-AnyLabeling 手工补框。
- 类别顺序必须在所有数据集之间一致（类号 = 行号）。`data.yaml` 的 `train:` 可以写成列表，
  同时用多个数据集，不用合并文件。

### 3.2 自动标注自己的采集（推荐，最贴近比赛现场）

1. 用采集工具录会话（`camera/recorder.py` 格式：`000000_color.png/_depth.png/_meta.json`）。
   **一个会话只放一种果蔬**，放 3~6 个，每帧之间挪动位置、转方向；机械臂停在视野外。
   每类至少 2 个会话（不同摆放和光照），这样才能按会话划分训练集和验证集。
2. 生成数据集（系统 python3，在包根目录执行）：

   ```bash
   python3 -m cube_task.tools.yolo.make_dataset --out ~/vision_runs/yolo/datasets/own \
       --classes apple,banana,carrot \
       --session apple=~/captures/apple_1 --session apple=~/captures/apple_2 \
       --session banana=~/captures/banana_1 --session banana=~/captures/banana_2 \
       --session carrot=~/captures/carrot_1 --session carrot=~/captures/carrot_2 \
       --roi 80,40,600,460 --preview
   ```

   - `--boxes plane`（默认）：在相机坐标系下 RANSAC 拟合主平面（桌面），高出桌面
     8 mm~25 cm 的连通块就是物体。不需要外参，也不需要机械臂。
   - `--roi x0,y0,x1,y1`：只看这个像素矩形，把托盘、机械臂、桌边排除在外。
   - 默认丢掉碰到图像边缘的块（`--keep-border` 可以保留）；`--max-boxes N` 跳过框数异常的帧。
   - `--boxes perception`：改用 `Perception` 的实例（会用拾取多边形和区域排除），需要外参。
   - 划分按会话进行（同一会话的帧几乎重复）：每类约 20 % 的会话进 val，也可以用
     `--val-session` 指定。每类只有一个会话时会警告，val 退回用 train（mAP 会偏乐观）。
3. **逐张看 `preview/*.jpg`**。错框的帧把 `images/` 和 `labels/` 里同名文件一起删掉。
   `manifest.json` 记录了每个会话用了多少帧、跳过了哪些帧、跳过原因。

## 4. 训练、导出、部署

训练环境是独立的 venv（不碰系统 Python）。ROS 的 `PYTHONPATH` 会漏进 venv，所以运行前用
`env -u PYTHONPATH`：

```bash
cd ~/vision/ws/src/cube_task
env -u PYTHONPATH ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.train_yolo \
    --data ~/vision_runs/yolo/datasets/own/data.yaml --model yolo11n.pt \
    --epochs 100 --batch 16        # 默认有 GPU 就用 GPU；--device cpu 强制用 CPU
```

`train_yolo.py` 依次做：训练（输出到 `~/vision_runs/yolo/<run>`）→ 用验证过的设置导出 ONNX 并打补丁
→ 用**系统 python3 + OpenCV 4.5.4**（和机器人完全相同的代码）在几张 val 图上跑，和 ultralytics
的结果比对（同类且 IoU > 0.9）→ 通过后复制为 `models/detector.onnx` 和
`models/detector_labels.txt`（旧文件改名加时间戳保留）→ 打印要写进 `config/site/local.yaml`
的配置。比对不通过就不部署。

其他参数：`--model yolo11s.pt` 更准但更慢；`--arg fliplr=0.5` 可以传任意 ultralytics 训练参数；
`--no-deploy` 只训练不部署。

单独导出或检查一个模型：

```bash
env -u PYTHONPATH ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.export_onnx best.pt \
    --out models/detector.onnx --verify --images 某几张有目标的图.png
python3 -m cube_task.tools.yolo.check_onnx --model models/detector.onnx \
    --labels models/detector_labels.txt --image 现场图.png --repeat 20 --save-vis /tmp/vis
```

部署配置（`config/site/local.yaml`）：

```yaml
tasks:
  L2:
    classifier: onnx+knn
    labels: [apple, banana, carrot]
perception:
  classifier:
    onnx:
      model_file: models/detector.onnx
      labels_file: models/detector_labels.txt
      input_size: 640           # 必须和导出时一致；null = 从模型读
      conf_threshold: 0.35
      nms_threshold: 0.45
      backend: auto             # auto: 能 import onnxruntime 就用它，否则用 opencv
      label_map: {}
      min_mask_overlap: 0.3     # pipeline 用：实例掩码落在框内的比例
```

构造检测器时会先检查所有配置，任何一项不对都会直接报错，不会在比赛中途才出问题。检查项：
模型和标签文件是否存在；类别数和类名是否与模型里内嵌的 names 一致；input_size 是否与模型一致；
是否检测模型（非 seg/pose）；是否没有内置 NMS；opencv 后端下是否打过补丁。最后跑一次预热推理。

## 5. GPU 训练环境（已建好：`~/.venvs/piper-yolo`）

版本：Python 3.10.12，torch 2.6.0+cu124（cuDNN 9.1），torchvision 0.21.0+cu124，
ultralytics 8.4.202，onnx 1.23.2，onnxslim 0.1.98，onnxruntime 1.23.2，numpy 2.2.6，
opencv-python 5.0.0。驱动 595.99（CUDA 13.2）可以跑 cu124 版本的轮子。

重建步骤（经代理下载 CUDA 轮子约 2.8 GB，实测 506 s；已缓存到 `~/.venvs/wheels-cu124`）：

```bash
python3 -m venv ~/.venvs/piper-yolo
P=~/.venvs/piper-yolo/bin/python
$P -m pip install --upgrade pip
$P -m pip download -d ~/.venvs/wheels-cu124 torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124          # 可以断点重试
$P -m pip install --no-index --find-links ~/.venvs/wheels-cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124
printf 'torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\n' > /tmp/torch-pin.txt
$P -m pip install -c /tmp/torch-pin.txt ultralytics==8.4.202 onnx onnxslim onnxruntime pyyaml
env -u PYTHONPATH $P -c "import torch; print(torch.cuda.is_available())"
```

- 只要 CPU 版：把 `cu124` 换成 `cpu`，下载约 1 分钟。CPU 可以训练，但慢很多。
- 安装 ultralytics 时**必须**带 `-c torch-pin.txt`，否则 pip 会把 torch 升级成 CUDA 13 的新版
  （会再下载好几 GB）。
- 实测：12 张图、40 epoch、640、batch 16 在 GPU 上 22 s，显存 1.8 GB；yolo11n/s 在 6 GB 显存上
  batch 16 都够用。

## 6. 实测延迟（640 输入，单张图，包括前后处理）

| 后端 | 环境 | 中位延迟 |
|---|---|---|
| opencv（cv2.dnn CPU） | 系统 OpenCV 4.5.4，20 线程 | 42-43 ms（推理 40 ms，前处理 1.7 ms，后处理 1.4 ms） |
| opencv，`cv2.setNumThreads(4)` | 同上 | 53.5 ms |
| onnxruntime 1.12.1 CPU | 系统 numpy 1.21.5 | 22 ms（推理 19 ms） |
| opencv，320 输入 | 同上 | 12-13 ms |
| 构造时预热 | opencv | 约 48 ms |

cv2.dnn 用不了 CUDA：Ubuntu 的 OpenCV 4.5.4 编译时没开 CUDA（`getBuildInformation()` 里没有
NVIDIA CUDA，`cv2.cuda.getCudaEnabledDeviceCount()` 返回 0）。这个延迟对 300 s 内抓 6 个物体
完全够用。

### onnxruntime 能否装进系统 Python（只检查了，没有安装）

用 `/usr/bin/python3 -I -m pip download --no-deps --only-binary=:all: onnxruntime==<v>` 下载 cp310
轮子，读取 `onnxruntime-<v>.dist-info/METADATA` 里的 Requires-Dist（检查了 1.12.0 和 1.12~1.23
每个小版本的最新补丁版）：

| 版本 | numpy 要求 | 和 numpy 1.21.5 兼容？ |
|---|---|---|
| 1.12.0, 1.12.1 | `>=1.21.0` | 兼容 |
| 1.13.1, 1.14.1, 1.15.1, 1.16.3, 1.17.3 | `>=1.21.6` | 不兼容 |
| 1.18.1 | `<2.0,>=1.21.6` | 不兼容 |
| 1.19.2, 1.20.1, 1.21.1, 1.22.1, 1.23.2 | `>=1.21.6` | 不兼容 |

用 `pip install --dry-run -c ~/constraints.txt numpy==1.21.5 onnxruntime==<v>`（venv 里的 pip 26.3，
`--python-version 310`）复核：1.12.1 能解析；1.13.1、1.16.3、1.23.2 报 `conflicting dependencies`。

1.12.1 已经实际验证：把轮子 `--no-deps` 解到临时目录，用系统 python3 + numpy 1.21.5 可以 import，
跑出来的框和 opencv 后端完全一样（0 px 差），未打补丁的原始导出也能跑。要用的话（由你决定，
会装到 `~/.local`）：

```bash
python3 -m pip install --user --no-deps -c ~/constraints.txt onnxruntime==1.12.1
python3 -c "import numpy, onnxruntime; print(numpy.__version__, onnxruntime.__version__)"  # 必须是 1.21.5
```

必须加 `--no-deps`：pip 解析（用 `--dry-run` 实测）会顺带安装 protobuf 7.36、packaging 26.3、
sympy 1.14 到 `~/.local`，覆盖 ROS 依赖的系统 protobuf 3.12.4。推理只需要 onnxruntime 本体，
`--no-deps` 实测可以正常运行。

不装也没关系，opencv 后端就够用；`backend: auto` 找不到 onnxruntime 时自动用 opencv。

## 7. 验证过的导出设置（以及 OpenCV 4.5.4 的两个 bug）

`export_onnx.EXPORT_ARGS`：`format=onnx, imgsz=640, opset=12, simplify=True, dynamic=False,
batch=1, device=cpu`，fp32，不设 `nms`（外部 NMS，输出 `[1, 4+nc, 8400]`）。

用 `export_onnx.py yolo11n.pt --sweep` 实测（系统 python3，在 bus.jpg / zidane.jpg 上和
ultralytics 对比）：

| 设置 | 不打补丁 | 打补丁 |
|---|---|---|
| opset 12, simplify | 前向报错 `shape_utils.hpp:170 ... in function 'total'` | **通过**：.pt 最小 IoU 0.939 / 0.975，onnx 0.995 / 0.998，类别全部一致 |
| opset 11, simplify | 同上 | 通过（结果同上） |
| opset 12 / 11, 不 simplify | 加载失败（`/model.23/Add` 解析错误） | 加载失败 |

在 OpenCV 4.5.4 里，最后检测头里的两个算子算错或者跑不起来。`patch_for_opencv45()` 会改写它们，
改写后用 onnxruntime 比对，结果和原图完全相同（最大差 0）：

1. `Sub(常量, x)`（dist2bbox 里的 `anchor - lt`）被算成 `x - 常量`，框会**完全错位**，
   而且不报错。补丁把它改成 `Add(Mul(x, -1), 常量)`。
2. 形状为 `(1, 8400)` 的步长常量乘 `(1, 4, 8400)` 张量时前向报错。补丁把常量展开成
   `(1, 4, 8400)`。

opencv 后端加载模型时会检查第 1 种情况，发现就拒绝加载，并提示用 `export_onnx.py` 重新导出。
所以忘了打补丁的模型不会悄悄输出错误的框。

COCO 模型重新生成：

```bash
env -u PYTHONPATH ~/.venvs/piper-yolo/bin/python -m cube_task.tools.yolo.export_onnx \
    ~/vision_runs/yolo/verify/yolo11n.pt --out models/yolo11n_coco.onnx --verify
```

## 8. 常见报错

| 报错 | 处理 |
|---|---|
| `ONNX model not found` / `labels file not found` | 检查 `model_file` / `labels_file`（相对包根目录）。标签可以从模型导出：`check_onnx --write-labels` |
| `does not match the class names stored in` | 标签文件不是这个模型的，用导出时一起生成的 `*_labels.txt` |
| `model has N classes` | 类别数对不上，同上 |
| `constant first operand` | 模型没打补丁，用 `export_onnx.py` 重新导出，或改用 `backend: onnxruntime` |
| `input_size is ... exported for` | `input_size` 和导出尺寸不一致，改成一致或写 `null` |
| `end-to-end export` | 模型里内置了 NMS（YOLO26/YOLOv10），换 YOLO11/YOLOv8 并且不要传 `nms=True` |

# models/

运行时用到的模型文件。路径在配置里写成相对包根目录，例如 `models/detector.onnx`。

| 文件 | 来源 | 用途 |
|---|---|---|
| `detector.onnx` | `tools/yolo/train_yolo.py` 或 `export_onnx.py` | L2 果蔬检测器（`perception.classifier.onnx.model_file`） |
| `detector_labels.txt` | 同上，和模型一起生成 | 类名，每行一个，行号 = 类号 |
| `detector.<时间戳>.onnx`, `detector_labels.<时间戳>.txt` | `train_yolo.py` 部署时 | 上一个版本，需要回退时改回原名 |
| `yolo11n_coco.onnx`, `yolo11n_coco_labels.txt` | `export_onnx.py yolo11n.pt --out models/yolo11n_coco.onnx` | COCO 80 类零样本基线（docs/yolo.md 第 2 节） |
| `*.raw.onnx` | `export_onnx.py` | 未打补丁的原始导出，只用于对比，opencv 后端会拒绝加载 |
| `knn_model.npz` | `teach_classes` | KNN 分类器（`perception.classifier.knn.model_file`） |

## Git

仓库根目录的 `.gitignore` 忽略 `*.onnx`、`*.pt`、`*.pth`、`*.engine`，所以模型权重**不进 git**，
要单独备份（U 盘或网盘），比赛前确认参赛电脑上有这些文件。`*_labels.txt` 和 `knn_model.npz` 不会被
忽略，可以提交；标签文件必须和它对应的 `.onnx` 一起更新。

检查一个模型能不能在机器人环境里用（系统 python3，OpenCV 4.5.4）：

```bash
python3 -m cube_task.tools.yolo.check_onnx --model models/detector.onnx \
    --labels models/detector_labels.txt --image 现场图.png
```

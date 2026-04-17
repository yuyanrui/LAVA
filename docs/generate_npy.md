# 使用 YOLOv8 生成轨迹聚类 NPY 文件

## 快速使用

```bash
python scripts/generate_tracks.py \
    --video /path/to/video.mp4 \
    --output /path/to/fixed_files/preprocessed \
    --name amsterdam \
    --yolo_weight /path/to/yolov8x.pt
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 视频文件或图像目录路径 | 必填 |
| `--output` | NPY 文件输出目录 | `/path/to/fixed_files/preprocessed` |
| `--name` | 视频名称（用于命名输出文件） | 必填 |
| `--yolo_weight` | YOLOv8 模型权重路径 | `/path/to/yolov8x.pt` |
| `--min_len` | 最小轨迹长度（帧数） | 35 |
| `--max_len` | 最大轨迹长度（帧数） | 1350 |
| `--min_disp` | 最小位移距离（像素） | 30 |
| `--max_tracks` | 最大轨迹数量 | 2000 |
| `--k_min` | 最小聚类数 | 8 |
| `--k_max` | 最大聚类数 | 32 |
| `--device` | 运行设备 | `cuda:0` |
| `--classes` | 跟踪类别ID（逗号分隔） | `1,2,3,5,7` |

## 输出文件

运行后会在 `{output}/{name}/` 目录生成以下文件：

| 文件 | 内容 |
|------|------|
| `{name}_0_{max_tracks}_tracks_clustered.npy` | 聚类后的代表性轨迹 |
| `{name}_0_{max_tracks}_tracks_filtered.npy` | 过滤后的轨迹 |
| `{name}_0_{max_tracks}_tracks_origin.npy` | 原始轨迹 |
| `{name}.npy` | 简化版本（仅聚类结果） |

## 跟踪类别说明

COCO 数据集类别 ID：

| ID | 类别 |
|----|------|
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

## 使用现有标注文件

如果已有 YOLOv8 跟踪生成的标注文件，可以跳过跟踪步骤：

```bash
python scripts/generate_tracks.py \
    --skip_tracking \
    --label_file /path/to/labels.txt \
    --output /path/to/fixed_files/preprocessed \
    --name amsterdam
```

## 加载 NPY 文件示例

```python
import numpy as np

# 加载聚类轨迹
tracks = np.load("/path/to/fixed_files/preprocessed/amsterdam/amsterdam.npy", allow_pickle=True)

print(f"轨迹数量: {len(tracks)}")
print(f"第一条轨迹长度: {len(tracks[0])}")
print(f"第一条轨迹第一个bbox: {tracks[0][0]}")  # [x_min, y_min, x_max, y_max]
```

## 流程概述

```
视频 → YOLOv8检测跟踪 → 轨迹标注(.txt) → 过滤筛选 → KMedoids聚类 → NPY文件
```

## 目录结构

```
LAVA/
├── traj_cluster/
│   ├── cluster/
│   │   ├── Kmediods.py    # KMedoids 聚类
│   │   └── FCM.py         # FCM 聚类
│   └── distance/
│       └── __init__.py    # Fréchet 距离计算
├── evaluate.py            # DBI 聚类评估
├── scripts/
│   └── generate_tracks.py # 生成脚本
└── fixed_files/
    └── preprocessed/
        └── {video_name}/
            ├── {name}.npy
            └── ...
```
"""
使用 YOLOv8 生成轨迹标注并聚类为 NPY 文件
完整流程脚本

Usage:
    python scripts/generate_tracks.py --video /path/to/video.mp4 --output /path/to/output --name video_name
"""
import os
import sys
import cv2
import math
import argparse
import numpy as np
from collections import Counter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

from traj_cluster.cluster.Kmediods import Kmedoids
from traj_cluster.distance import distance, pairwise_distance
from evaluate import DBI_Vec


# ========== 默认配置参数 ==========
DEFAULT_YOLO_WEIGHT = "/path/to/yolov8x.pt"
DEFAULT_OUTPUT_DIR = "/path/to/fixed_files/preprocessed"
TRACK_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

# 过滤参数
MIN_TRAJ_LEN = 35
MAX_TRAJ_LEN = 1350
MIN_DISPLACEMENT = 30
MAX_TRACKS = 2000

# 聚类参数
K_MIN = 8
K_MAX = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectory NPY files from video using YOLOv8")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or image directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for NPY files")
    parser.add_argument("--name", type=str, required=True, help="Video name for output files")
    parser.add_argument("--yolo_weight", type=str, default=DEFAULT_YOLO_WEIGHT, help="Path to YOLOv8 weight")
    parser.add_argument("--min_len", type=int, default=MIN_TRAJ_LEN, help="Minimum trajectory length")
    parser.add_argument("--max_len", type=int, default=MAX_TRAJ_LEN, help="Maximum trajectory length")
    parser.add_argument("--min_disp", type=int, default=MIN_DISPLACEMENT, help="Minimum displacement")
    parser.add_argument("--max_tracks", type=int, default=MAX_TRACKS, help="Maximum number of tracks")
    parser.add_argument("--k_min", type=int, default=K_MIN, help="Minimum cluster number")
    parser.add_argument("--k_max", type=int, default=K_MAX, help="Maximum cluster number")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for YOLOv8")
    parser.add_argument("--classes", type=str, default="1,2,3,5,7", help="Track class IDs (comma separated)")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip tracking, use existing labels")
    parser.add_argument("--label_file", type=str, default=None, help="Existing label file path")
    return parser.parse_args()


def run_yolov8_tracking(video_path, yolo_weight, classes, device, save_dir):
    """运行 YOLOv8 跟踪"""
    print(f"Running YOLOv8 tracking on: {video_path}")

    model = YOLO(yolo_weight)
    class_ids = [int(c) for c in classes.split(",")]

    results = model.track(
        video_path,
        save_txt=True,
        save=True,
        persist=True,
        classes=class_ids,
        device=device,
        project=save_dir,
        name="tracks"
    )

    # YOLOv8 保存的标注文件路径
    label_file = os.path.join(save_dir, "tracks", "labels.txt")

    return results, label_file


def parse_labels(label_file):
    """解析 YOLOv8 跟踪输出的标注文件"""
    print(f"Parsing labels from: {label_file}")

    with open(label_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        print("Warning: Empty label file")
        return {}

    tuple_dict = {}

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue

        frame_id = int(parts[0])
        track_id = int(parts[1])
        x_center = float(parts[2])
        y_center = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        obj_class = int(parts[7])

        # 转换为 xmin, ymin, xmax, ymax
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        if track_id not in tuple_dict:
            tuple_dict[track_id] = [frame_id, frame_id, [[x_min, y_min, x_max, y_max]], []]
        else:
            tuple_dict[track_id][1] = frame_id
            tuple_dict[track_id][2].append([x_min, y_min, x_max, y_max])
        tuple_dict[track_id][3].append(obj_class)

    return tuple_dict


def filter_trajectories(tuple_dict, min_len, max_len, min_disp, max_tracks):
    """过滤轨迹"""
    print(f"Filtering trajectories (min_len={min_len}, max_len={max_len}, min_disp={min_disp})")

    filtered = {}
    tracks_list = []
    origin_tracks = []

    for tid, info in tuple_dict.items():
        start, end, bboxes, classes = info
        length = abs(end - start)

        # 长度过滤
        if length < min_len or length > max_len:
            continue

        # 位移检查
        s_bbox = bboxes[0]
        e_bbox = bboxes[-1]
        s_center = ((s_bbox[0] + s_bbox[2]) / 2, (s_bbox[1] + s_bbox[3]) / 2)
        e_center = ((e_bbox[0] + e_bbox[2]) / 2, (e_bbox[1] + e_bbox[3]) / 2)
        displacement = math.sqrt((s_center[0] - e_center[0])**2 + (s_center[1] - e_center[1])**2)

        if displacement < min_disp:
            continue

        # 保留主要类别
        main_class = Counter(classes).most_common(1)[0][0]
        filtered[tid] = [start, end, bboxes, main_class]
        origin_tracks.append(bboxes)

    print(f"Filtered: {len(filtered)} trajectories from {len(tuple_dict)}")

    # 限制数量
    if len(tracks_list) > max_tracks:
        origin_tracks = origin_tracks[:max_tracks]
        print(f"Limited to {max_tracks} tracks")

    return filtered, origin_tracks


def convert_to_center_points(tracks):
    """将轨迹转换为中心点序列"""
    traj_dict = {}
    for idx, track in enumerate(tracks):
        traj_dict[idx] = [
            [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            for bbox in track
        ]
    return traj_dict


def cluster_trajectories(tracks, k_min, k_max):
    """使用 KMedoids 对轨迹进行聚类"""
    print(f"Clustering trajectories (k_min={k_min}, k_max={k_max})")

    if len(tracks) < k_min:
        print(f"Warning: Not enough tracks ({len(tracks)}) for clustering")
        return tracks, 0

    # 转换为中心点序列
    traj_dict = convert_to_center_points(tracks)

    # 计算距离矩阵
    print("Computing distance matrix...")
    distances = distance.cdist(traj_dict, traj_dict, type="frechet")

    # 转换为矩阵形式
    n = len(traj_dict)
    distance_matrix = np.zeros((n, n))
    for (i, j), dist in distances.items():
        distance_matrix[i][j] = dist

    # 寻找最优 k
    best_score = float('inf')
    best_centers = None
    best_k = k_min

    for k in range(k_min, min(k_max, n), 2):
        labels, centers = Kmedoids(k, distance_matrix)

        # 构建聚类字典
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(idx)

        score = DBI_Vec(cluster_dict, distance_matrix)

        print(f"  k={k}, DBI={score:.4f}")

        if score < best_score:
            best_score = score
            best_centers = [tracks[c] for c in centers]
            best_k = k

    print(f"Best k={best_k} with DBI={best_score:.4f}")
    return best_centers, best_score


def save_npy_files(video_name, clustered, filtered, origin, output_dir, max_tracks):
    """保存 NPY 文件"""
    save_path = os.path.join(output_dir, video_name)
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f"{video_name}_0_{max_tracks}_tracks_clustered.npy"), clustered)
    np.save(os.path.join(save_path, f"{video_name}_0_{max_tracks}_tracks_filtered.npy"), filtered)
    np.save(os.path.join(save_path, f"{video_name}_0_{max_tracks}_tracks_origin.npy"), origin)
    np.save(os.path.join(save_path, f"{video_name}.npy"), clustered)

    print(f"Saved NPY files to: {save_path}")
    print(f"  - {video_name}_0_{max_tracks}_tracks_clustered.npy ({len(clustered)} tracks)")
    print(f"  - {video_name}_0_{max_tracks}_tracks_filtered.npy ({len(filtered)} tracks)")
    print(f"  - {video_name}_0_{max_tracks}_tracks_origin.npy ({len(origin)} tracks)")
    print(f"  - {video_name}.npy (simplified version)")


def main():
    args = parse_args()

    print("=" * 50)
    print("YOLOv8 + Tracking NPY Generator")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Name: {args.name}")

    # Step 1: 运行跟踪或加载现有标注
    if args.skip_tracking and args.label_file:
        print("Skipping tracking, using existing labels...")
        label_file = args.label_file
    else:
        results, label_file = run_yolov8_tracking(
            args.video,
            args.yolo_weight,
            args.classes,
            args.device,
            args.output
        )

    # Step 2: 解析标注
    tuple_dict = parse_labels(label_file)
    if len(tuple_dict) == 0:
        print("Error: No trajectories found")
        return

    # Step 3: 过滤轨迹
    filtered_dict, origin_tracks = filter_trajectories(
        tuple_dict,
        args.min_len,
        args.max_len,
        args.min_disp,
        args.max_tracks
    )

    if len(origin_tracks) == 0:
        print("Error: No valid trajectories after filtering")
        return

    # Step 4: 聚类轨迹
    clustered_tracks, score = cluster_trajectories(origin_tracks, args.k_min, args.k_max)

    # Step 5: 保存结果
    save_npy_files(
        args.name,
        clustered_tracks,
        origin_tracks,  # filtered 就是 origin_tracks
        origin_tracks,
        args.output,
        args.max_tracks
    )

    print("=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
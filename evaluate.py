"""
聚类评估指标
"""

import numpy as np


def DBI_Vec(cluster_dict, distance_matrix):
    """
    Davies-Bouldin Index (DBI) 聚类质量评估

    Args:
        cluster_dict: {cluster_id: [point_indices]}
        distance_matrix: 点之间的距离矩阵

    Returns:
        DBI 值 (越小越好)
    """
    cluster_ids = list(cluster_dict.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        return 0

    # 计算每个簇的类内平均距离
    intra_dists = {}
    centroids = {}

    for cid in cluster_ids:
        points = cluster_dict[cid]
        if len(points) == 0:
            intra_dists[cid] = 0
            centroids[cid] = 0
        elif len(points) == 1:
            intra_dists[cid] = 0
            centroids[cid] = points[0]
        else:
            total_dist = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    total_dist += distance_matrix[points[i]][points[j]]
            intra_dists[cid] = total_dist / (len(points) * (len(points) - 1) / 2)

            # 计算簇中心 (距离其他点总和最小的点)
            min_total = float('inf')
            centroid = points[0]
            for p in points:
                total = sum(distance_matrix[p][other] for other in points if other != p)
                if total < min_total:
                    min_total = total
                    centroid = p
            centroids[cid] = centroid

    # 计算 DBI
    dbi_sum = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i == j:
                continue

            c_i = cluster_ids[i]
            c_j = cluster_ids[j]

            inter_dist = distance_matrix[centroids[c_i]][centroids[c_j]]
            if inter_dist == 0:
                inter_dist = 1e-10

            ratio = (intra_dists[c_i] + intra_dists[c_j]) / inter_dist
            if ratio > max_ratio:
                max_ratio = ratio

        dbi_sum += max_ratio

    return dbi_sum / n_clusters


def silhouette_score(cluster_dict, distance_matrix):
    """
    Silhouette 轮廓系数评估

    Args:
        cluster_dict: {cluster_id: [point_indices]}
        distance_matrix: 点之间的距离矩阵

    Returns:
        平均 silhouette 值 (范围 -1 到 1，越大越好)
    """
    all_points = []
    for cid, points in cluster_dict.items():
        for p in points:
            all_points.append((p, cid))

    if len(all_points) < 2:
        return 0

    scores = []
    for point, cid in all_points:
        # a: 到同簇其他点的平均距离
        same_cluster = [p for p in cluster_dict[cid] if p != point]
        if len(same_cluster) == 0:
            a = 0
        else:
            a = sum(distance_matrix[point][p] for p in same_cluster) / len(same_cluster)

        # b: 到最近其他簇的平均距离
        min_b = float('inf')
        for other_cid in cluster_dict:
            if other_cid == cid:
                continue
            other_cluster = cluster_dict[other_cid]
            if len(other_cluster) == 0:
                continue
            b = sum(distance_matrix[point][p] for p in other_cluster) / len(other_cluster)
            if b < min_b:
                min_b = b

        if min_b == float('inf'):
            scores.append(0)
        else:
            scores.append((min_b - a) / max(a, min_b))

    return np.mean(scores)
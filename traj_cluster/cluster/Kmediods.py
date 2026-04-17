"""
KMedoids 聚类算法实现 (基于距离矩阵)
"""
import numpy as np
from random import randint


def random_generate_k_centers(k, seq_id_list):
    """随机选择 k 个初始中心"""
    centers = []
    for _ in range(k):
        centers.append(seq_id_list[randint(0, len(seq_id_list) - 1)])
    return centers


def compute_cluster_cost(distance_matrix, centers, seq_id_list):
    """计算当前聚类的总成本"""
    total_cost = 0
    labels = []

    for i in range(len(seq_id_list)):
        min_dist = float('inf')
        min_center = 0

        for j, center in enumerate(centers):
            idx_i = seq_id_list.index(i)
            idx_c = seq_id_list.index(center)
            dist = distance_matrix[idx_i][idx_c]

            if dist < min_dist:
                min_dist = dist
                min_center = j

        total_cost += min_dist
        labels.append(min_center)

    return total_cost, labels


def Kmedoids(n_clusters, distance_matrix, max_iter=100):
    """
    KMedoids 聚类算法

    Args:
        n_clusters: 聚类数量
        distance_matrix: 预计算的距离矩阵
        max_iter: 最大迭代次数

    Returns:
        labels: 每个点的聚类标签
        centers: 聚类中心索引列表
    """
    assert n_clusters > 0, "n_clusters must be > 0"

    seq_id_list = list(range(len(distance_matrix)))
    centers = random_generate_k_centers(n_clusters, seq_id_list)

    best_cost, best_labels = compute_cluster_cost(distance_matrix, centers, seq_id_list)

    for _ in range(max_iter):
        improved = False
        for i in range(len(seq_id_list)):
            if i in centers:
                continue

            for j, center in enumerate(centers):
                new_centers = centers.copy()
                new_centers[j] = i

                cost, labels = compute_cluster_cost(distance_matrix, new_centers, seq_id_list)

                if cost < best_cost:
                    best_cost = cost
                    best_labels = labels
                    centers = new_centers
                    improved = True

        if not improved:
            break

    return np.array(best_labels), centers
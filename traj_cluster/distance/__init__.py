"""
轨迹距离计算模块
支持 Fréchet 距离、欧氏距离等
"""
import numpy as np
from typing import Dict, List, Tuple, Union


def euclidean_distance(p1: List, p2: List) -> float:
    """计算两点之间的欧氏距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def frechet_distance(traj1: List, traj2: List) -> float:
    """
    计算两条轨迹之间的离散 Fréchet 距离

    Args:
        traj1: 轨迹1的点列表 [[x,y], ...]
        traj2: 轨迹2的点列表 [[x,y], ...]

    Returns:
        Fréchet 距离值
    """
    n, m = len(traj1), len(traj2)

    if n == 0 or m == 0:
        return float('inf')

    # 距离矩阵
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = euclidean_distance(traj1[i], traj2[j])

    # 动态规划计算 Fréchet 距离
    ca = np.zeros((n, m))
    ca[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], dist_matrix[i, 0])

    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], dist_matrix[0, j])

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                dist_matrix[i, j]
            )

    return ca[n-1, m-1]


def dtw_distance(traj1: List, traj2: List) -> float:
    """
    计算两条轨迹之间的 DTW (Dynamic Time Warping) 距离

    Args:
        traj1: 轨迹1的点列表
        traj2: 轨迹2的点列表

    Returns:
        DTW 距离值
    """
    n, m = len(traj1), len(traj2)

    if n == 0 or m == 0:
        return float('inf')

    # DTW 距离矩阵
    dtw = np.zeros((n + 1, m + 1))
    dtw[0, :] = float('inf')
    dtw[:, 0] = float('inf')
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(traj1[i-1], traj2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return dtw[n, m]


class distance:
    """距离计算类，兼容原项目接口"""

    @staticmethod
    def cdist(traj_dict1: Dict, traj_dict2: Dict = None, type: str = "frechet") -> Dict:
        """
        计算两组轨迹之间的距离

        Args:
            traj_dict1: {id: [[x,y], ...]}
            traj_dict2: {id: [[x,y], ...]}，若为None则计算traj_dict1内部距离
            type: 距离类型 ("frechet", "euclidean", "dtw")

        Returns:
            {(id1, id2): distance_value}
        """
        if traj_dict2 is None:
            traj_dict2 = traj_dict1

        distances = {}
        distance_func = {
            "frechet": frechet_distance,
            "euclidean": lambda t1, t2: sum(euclidean_distance(t1[i], t2[i]) for i in range(min(len(t1), len(t2)))) / max(len(t1), len(t2)) if min(len(t1), len(t2)) > 0 else float('inf'),
            "dtw": dtw_distance
        }

        func = distance_func.get(type, frechet_distance)

        for id1, traj1 in traj_dict1.items():
            for id2, traj2 in traj_dict2.items():
                distances[(id1, id2)] = func(traj1, traj2)

        return distances


def pairwise_distance(trajs: List, type: str = "frechet") -> np.ndarray:
    """
    计算轨迹列表中所有轨迹之间的 pairwise 距离矩阵

    Args:
        trajs: 轨迹列表 [[[x,y], ...], ...]
        type: 距离类型

    Returns:
        distance_matrix: 距离矩阵 (n x n)
    """
    n = len(trajs)
    distance_matrix = np.zeros((n, n))

    distance_func = {
        "frechet": frechet_distance,
        "dtw": dtw_distance
    }
    func = distance_func.get(type, frechet_distance)

    for i in range(n):
        for j in range(i, n):
            dist = func(trajs[i], trajs[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix
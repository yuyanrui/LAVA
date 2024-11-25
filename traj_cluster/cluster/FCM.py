import numpy as np
from random import randint

def random_generate_k_centers(k, seq_id_list):
    centers = []
    for i in range(k):
        centers.append(seq_id_list[randint(0, len(seq_id_list)-1)])
    return centers

def update_membership(distance_matrix, centers, seq_id_list, m=2):
    """
    更新隶属度矩阵U, U[i][j]表示数据点i对聚类中心j的隶属度
    """
    num_points = len(seq_id_list)
    num_centers = len(centers)
    U = np.zeros((num_points, num_centers))
    
    for i in range(num_points):
        for j in range(num_centers):
            index_0 = seq_id_list.index(i)
            index_1 = seq_id_list.index(centers[j])
            distance_value = distance_matrix[index_0][index_1]
            if distance_value == 0:  # 避免除零错误
                U[i][j] = 1
                continue
            # 计算隶属度
            sum_d = 0
            for k in range(num_centers):
                index_2 = seq_id_list.index(centers[k])
                distance_value_k = distance_matrix[index_0][index_2]
                if distance_value_k == 0:  # 避免除零错误
                    continue
                sum_d += (distance_value / distance_value_k) ** (2 / (m - 1))
            U[i][j] = 1 / sum_d
    return U

def update_center_fcm(U, distance_matrix, seq_id_list):
    """
    根据隶属度矩阵U更新聚类中心
    """
    new_centers = []
    num_centers = U.shape[1]
    
    for j in range(num_centers):
        weighted_distances = []
        for i in range(len(seq_id_list)):
            slave_index = seq_id_list.index(i)
            weighted_distance_sum = 0
            for k in range(len(seq_id_list)):
                slave__index = seq_id_list.index(k)
                distance_value = distance_matrix[slave_index][slave__index]
                weighted_distance_sum += U[k][j] ** 2 * distance_value
            weighted_distances.append(weighted_distance_sum)
        new_center = np.argmin(weighted_distances)
        new_centers.append(new_center)
    
    return  new_centers

def FCM(n_clusters, distance_matrix, m=2, max_iter=100, epsilon=1e-5):
    """
    距离矩阵版本的模糊C均值聚类算法(FCM)
    """
    assert n_clusters > 0, "n_clusters must be > 0"
    
    seq_id_list = [i for i in range(len(distance_matrix))]
    centers = random_generate_k_centers(n_clusters, seq_id_list)
    centers.sort()

    iteration_num = 0
    while iteration_num < max_iter:
        # Step 1: 计算隶属度矩阵
        U = update_membership(distance_matrix, centers, seq_id_list, m)

        # Step 2: 更新聚类中心
        new_centers = update_center_fcm(U, distance_matrix, seq_id_list)
        new_centers.sort()

        # Step 3: 检查收敛条件
        if max(np.abs(np.array(new_centers) - np.array(centers))) < epsilon:
            break

        centers = new_centers
        iteration_num += 1

    # 输出每个点的隶属度最高的簇作为硬标签
    labels = np.argmax(U, axis=1)
    
    # 返回软标签和聚类中心
    return labels, list(centers), U
    
    # return labels, list(centers)

# Example usage (replace with actual distance matrix):
# labels, centers = FCM(n_clusters=3, distance_matrix=your_distance_matrix)

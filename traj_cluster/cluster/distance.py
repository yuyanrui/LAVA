import numpy as np

def feature_cosine_distance(feature0, feature1):
    num = float(np.dot(feature0, feature1))
    denom = np.linalg.norm(feature0) * np.linalg.norm(feature1)
    dist = (1 - num / denom)
    return dist


def feature_l2_distance(feature0, feature1):
    dist = np.linalg.norm(np.array(feature0) - np.array(feature1))
    return dist
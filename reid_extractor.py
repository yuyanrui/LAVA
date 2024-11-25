from vehicle_reid.fastreid.config import get_cfg
import argparse
import numpy as np
import torch.nn.functional as F
import cv2
import os
import json
import itertools
import random
from collections import defaultdict
from vehicle_reid.predictor import FeatureExtractionDemo

def compute_cosine_similarity(vec1,vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def setup_cfg(args):
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg

class Parser(object):
    def __init__(self):

        self.FILE = "vehicle_reid/configs/warsaw/bagtricks_R50-ibn.yml"
        self.config_file = "vehicle_reid/configs/warsaw/bagtricks_R50-ibn.yml"
        self.parallel = False
        self.opts = []

    def change_yml_config(self,config_file):
        self.config_file = config_file


def get_parser():

    parser = Parser()    
    return parser

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features
        

class feature_extractor(object):
    
    def __init__(self,width,height):
        self.args = []
        self.extractor = []
        # self.reid_weight = reid_weight
        self.width = width
        self.height = height
        
    def init_extractor(self,config_file="",):
        
        # args = get_parser().parse_args()
        args = get_parser()
        self.args = args
        if config_file != "":
            args.config_file = config_file
        cfg = setup_cfg(args)
        self.extractor = FeatureExtractionDemo(cfg, width=self.width, height=self.height)
        
    def inference_pic(self,img):
        score = self.extractor.run_on_images(img)
        # feat = postprocess(feat)
        return score

    def inference_feature(self,img):
        feature = self.extractor.run_on_image(img)
        # feat = postprocess(feat)
        return feature
    
    def similarity(self, img1, img2):
        vec1 = self.inference_feature(img1)[0]
        vec2 = self.inference_feature(img2)[0]
        return compute_cosine_similarity(vec1, vec2)

def parse_filename(filename):
    """
    解析文件名，提取 carid 和其他信息
    """
    parts = filename.split('_')
    carid = parts[1]
    return carid

def list_images_by_carid(directory, extensions=['.jpg', '.jpeg', '.png']):
    """
    列出目录下所有图片，并根据 carid 分组
    
    :param directory: 图片所在目录
    :param extensions: 支持的图片文件扩展名列表
    :return: 按 carid 分组的图片字典
    """
    carid_dict = defaultdict(list)
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            filepath = os.path.join(directory, filename)
            carid = parse_filename(filename)
            carid_dict[carid].append(filepath)
    
    return carid_dict

def construct_true_pairs(carid_dict):
    """
    构造分类为 true 的图片对 (同一 carid 下的图片对)
    
    :param carid_dict: 按 carid 分组的图片字典
    :return: 分类为 true 的图片对列表
    """
    true_pairs = []
    for images in carid_dict.values():
        true_pairs.extend(itertools.combinations(images, 2))
    return true_pairs

def construct_false_pairs(carid_dict):
    """
    构造分类为 false 的图片对 (不同 carid 下的图片对)
    
    :param carid_dict: 按 carid 分组的图片字典
    :return: 分类为 false 的图片对列表
    """
    false_pairs = []
    carids = list(carid_dict.keys())
    for carid1, carid2 in itertools.combinations(carids, 2):
        images1 = carid_dict[carid1]
        images2 = carid_dict[carid2]
        false_pairs.extend(itertools.product(images1, images2))
    return false_pairs

def sample_pairs(pairs, sample_size):
    """
    从图片对中随机采样指定数量的图片对
    
    :param pairs: 图片对列表
    :param sample_size: 需要采样的图片对数量
    :return: 随机采样的图片对列表
    """
    if sample_size > len(pairs):
        raise ValueError("采样数量大于图片对的总数量")
    return random.sample(pairs, sample_size)

def evaluation(path, extractor):
    # 测试一个数据集的识别准确率，穷举所有的图片对，计算相似度，如果相似度大于阈值，认为是同一个车辆
    images = os.listdir(path)
    # image_pairs = list(itertools.combinations(images,2))
    true_image_pairs = construct_true_pairs(list_images_by_carid(path))
    false_image_pairs = construct_false_pairs(list_images_by_carid(path))
    false_image_pairs = sample_pairs(false_image_pairs, len(true_image_pairs))
    
    true_similarity = []
    for pair in true_image_pairs:
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        similarity = float(extractor.similarity(img1, img2).cpu().data)
        true_similarity.append(similarity)
        print("True pair similarity: ", similarity)
    
    false_similarity = []
    for pair in false_image_pairs:
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        similarity = float(extractor.similarity(img1, img2).cpu().data)
        false_similarity.append(similarity)
        print("False pair similarity: ", similarity)
    
    similarity_dict = {}
    similarity_dict["true"] = true_similarity
    similarity_dict["false"] = false_similarity
    
    with open("detrac_redcar_reid_similarity.txt", "w") as f:
        json.dump(similarity_dict, f)

    similarity_threshold = 0.5
    true_positive = sum(1 for s in true_similarity if s > similarity_threshold)
    false_positive = sum(1 for s in false_similarity if s > similarity_threshold)
    true_negative = len(false_similarity) - false_positive
    false_negative = len(true_similarity) - true_positive
    print("True positive: ", true_positive)
    print("False positive: ", false_positive)
    print("True negative: ", true_negative)
    print("False negative: ", false_negative)
    # 画出ROC曲线
    # 计算TPR和FPR
    TPR = true_positive / (true_positive + false_negative)
    FPR = false_positive / (false_positive + true_negative)
    print("TPR: ", TPR)
    print("FPR: ", FPR)

def detrac_test(path):
    images = os.listdir(path)
    images.sort()
    image_pairs = list(itertools.combinations(images,2))
    true_similarity = {}
    for pair in image_pairs:
        img1 = cv2.imread(os.path.join(path, pair[0]))
        img2 = cv2.imread(os.path.join(path, pair[1]))
        similarity = float(extractor.similarity(img1, img2).cpu().data)
        true_similarity[pair] = similarity
        print("True pair similarity: ", similarity)
    true_similarity = np.array(true_similarity)

if __name__ == "__main__":
    
    extractor = feature_extractor(960, 540)
    extractor.init_extractor()
    # pic1 = cv2.imread("/home/yyr/project/LEAP/datasets/DETRAC/detrac_original/image_train/1_1_truck_0_99_144_215_20.jpg")
    # pic2 = cv2.imread("/home/yyr/project/LEAP/datasets/DETRAC/detrac_original/image_train/1_2_car_700_125_804_182_20.jpg")
    # # vec1 = extractor.inference_pic(pic1)[0]
    # # vec2 = extractor.inference_pic(pic2)[0]
    # # vec1 = extractor.inference_feature(pic1)[0]
    # # vec2 = extractor.inference_feature(pic2)[0]
    # # print(compute_cosine_similarity(vec1,vec2))
    # print(extractor.similarity(pic1,pic2))
    # evaluation("/home/yyr/project/LEAP/datasets/DETRAC/redcar/image_train", extractor)
    detrac_test("outputs/detrac_first_car")        

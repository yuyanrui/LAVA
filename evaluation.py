import os
import sys
import cv2
import json
import yaml
import numpy as np
import argparse
import time
from ultralytics import YOLO
from loguru import logger
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from settings.settings import video_details
from tools.frame_difference import frame_difference_score
from reid_extractor import feature_extractor
from match_object import get_point_update
from inference import allocate_tracks
from tools.utils import get_sample_gap_n, justify_intersect
from evaluate import evaluation
from pipline.sample_query import make_parser, getYaml

def covert_entity_to_frame(video_details):
    frame_level_result = {}
    for car_id in video_details.per_car_sample.keys():
        interval = video_details.per_car_sample[car_id][1]
        for frame in range(interval[0], interval[1]+1):
            if frame in frame_level_result.keys():
                frame_level_result[frame].append(car_id)
            else:
                frame_level_result[frame] = [car_id]

    for frame in frame_level_result.keys():
        frame_level_result[frame] = [frame_level_result[frame], len(frame_level_result[frame])]
    
    return frame_level_result

def convert_label2int(s):
    start_index = len(s) - 1

    # 检查字符串中的每个字符，直到找到一个非数字字符
    while start_index >= 0 and s[start_index].isdigit():
        start_index -= 1

    # 计算数字部分的起始位置
    start_index += 1

    # 提取数字部分并转换为整数
    number = int(s[start_index:])
    return number

def convert_json_to_label(json_file):
    with open(json_file, "r") as f:
        input_label = json.load(f)
    frame_level_result = {}
    
    for key in input_label.keys():
        frame = convert_label2int(key)
        frame_level_result[frame] = input_label[key]
    return frame_level_result
if __name__ == "__main__":
    # 使用pickle保存video_details
    parser = make_parser()
    args = parser.parse_args()
    cfg = getYaml(args.config)
    
    import pickle
    # with open("video_details_warsaw_reid_threshold_0.25.pkl", "rb") as f:
        # video_details = pickle.load(f)
    
    evaluate_object = evaluation("red_car_label", cfg)
    with open("/home/yyr/yu88/CLIP_Surgery/background_subtraction/red_car_label_count.json","r") as f:
        input_label = json.load(f)
    # frame_level_result = covert_entity_to_frame(video_details)
    frame_level_result = convert_json_to_label("/home/yyr/yu88/CLIP_Surgery/warsaw1_red_car.json")
    evaluate_object.preprocess(input_label, frame_level_result)
    F1,recall,precision,accuracy = evaluate_object.selection_query_1()
    print("Selection_query: F1:%5f, recall:%5f, precision:%5f, accuracy:%5f"%(F1,recall,precision,accuracy))
    MAE,ACC = evaluate_object.aggregation_query_1()
    print("Aggregation_query: MAE:%5f, ACC:%5f"%(MAE,ACC))
    GT_COUNT,PRED_COUNT = evaluate_object.aggregation_query_3()
    print("Aggregation_query: GT_COUNT:%5f, PRED_COUNT:%5f"%(GT_COUNT,PRED_COUNT))
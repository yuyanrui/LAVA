import os
import sys
import cv2
import json
import clip
import yaml
import numpy as np
import argparse
import time
import torch
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
from train_prompt import reset_cfg, setup_cfg, extend_cfg
import trainers.maple
from dassl.engine import build_trainer
from dassl.data import ReidDataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# 首先要加载轨迹聚类的结果
# 随后根据聚类的结果匹配当前轨迹的长度、给每辆车分配一个 id，对于每辆车，根据匹配的轨迹的长度，平均采样 n 帧，做车辆的 reid 匹配

# 使用 np 读取 npy 文件
# 轨迹聚类的结果
# 读取聚类结果
def getYaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def make_parser():

    parser = argparse.ArgumentParser()
    #LEAP config
    parser.add_argument('--config', type=str, default='./configs/warsaw.yaml')
    parser.add_argument('--weight', type=str, default='./weights/best.pt')
    parser.add_argument("-rw","--reid_weight",default=\
                            "vehicle_reid/logs/warsaw_history/bagtricks_R50-ibn/model_final.pth")

    parser.add_argument("--use_cluster",action="store_false")   
    parser.add_argument("--use_filter",action="store_false")  
    parser.add_argument("--use_hand_label",action="store_false")
    parser.add_argument("--type",type=str,default="test")
    parser.add_argument("--adaptive_sample",action="store_false")
    parser.add_argument("--use_external_pattern",action="store_true")
    parser.add_argument("--cross_roads",action="store_true") 
    parser.add_argument("--yolo_classes", type=str, default="tram,bus,van")
    parser.add_argument("-vis","--visualize",action='store_false')  
    parser.add_argument("--active_log",action='store_true')
    parser.add_argument("-sr","--save_result",action="store_false",help="Save parsed result")
    parser.add_argument("-ld","--load",action='store_false',help="load file from dict")
    parser.add_argument("--use_label",action='store_true',help="Use preprocesed label instead of bytetracker")
    parser.add_argument("--use_mask",action='store_true',help="Use mask while parsing")
    parser.add_argument("--device",type=str,default='1')
    parser.add_argument("--yolo_weight",type=str,default="/home/yyr/yu88/Grounded-Segment-Anything/yolo_weights/yolov8x-worldv2.pt")
    parser.add_argument('--split', type=str, default='train')
    # maple_config
    parser.add_argument("--root", type=str, default="/data/usrs/yyr/datasets/lava_dataset", help="path to dataset")
    parser.add_argument("--dataset_name", type=str, default="warsaw", help="path to dataset config file")
    parser.add_argument("--query", type=str, default="bus with multi-section design", help="query")
    parser.add_argument("--full", action="store_true", help="full dataset")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--train_num", type=int, default=2000, help="number of training samples")
    parser.add_argument("--output-dir", type=str, default="output", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=3, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml", help="path to config file"
    )
    parser.add_argument("--trainer", type=str, default="MaPLe", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="output/MaPLe/full/warsaw/bus with multi-section design/seed1",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, default=10, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    return parser

def read_cluster_result(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data

def get_next_frame(current_frame, skip_flag = False, skip_frames = None):
    # 获取per_car_sample中所有的采样帧，找到大于current_frame的最小的一个，作为下一帧
    sample_all_frame_list = video_details.sample_all_frame_list
    for frame_list in video_details.per_car_sample.values():
        sample_all_frame_list.extend(frame_list[0])
    sample_all_frame_list = list(set(sample_all_frame_list))
    sample_all_frame_list = sorted(sample_all_frame_list)
    video_details.sample_all_frame_list = sample_all_frame_list
    for frame in sample_all_frame_list:
        if frame > current_frame:
            return frame
    if skip_flag:
        skip_frame = current_frame + skip_frames
        if skip_frame in video_details.chunk_frames:
            pass
        return current_frame + skip_frames
    # 如果后续没有了，那么在chunk_frames中招第一个比当前帧大的帧
    for frame in video_details.chunk_frames:
        if frame > current_frame:
            sample_all_frame_list.append(current_frame + 1)
            video_details.sample_all_frame_list = sample_all_frame_list
            return frame
    return False

def process_detect_results(current_frame, current_image, results, tracks, trainer, cfg):
    
    n = cfg['sample_num']
    
    # 给定每个 bbox，匹配最近的轨迹，确定需要采样的后续 n 帧序列
    if current_frame == cfg['start_frame']:# 这里也要用 clip_scores进行错检筛选
        for box in results[0].boxes:
            apply_id = max(video_details.resolved_tuple.keys()) + 1 \
            if len(video_details.resolved_tuple) != 0 else 1
            # 从轨迹中找到最近的轨迹
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            car_image = current_image[y1:y2, x1:x2, :]
            # cv2.imwrite(f"outputs/detrac_cut_red_car/{current_frame}_{x1}_{x2}_{y1}_{y2}.jpg", car_image)
            
            match_traj_dict = get_point_update([x1, y1, x2, y2], tracks, cfg)
            video_details.per_car_sample[apply_id] = get_sample_gap_n(current_frame, tracks, match_traj_dict, n-1, cfg, video_details) # return sample_frame_list & best_interval 
            
            video_details.resolved_tuple[apply_id] = [[[x1, y1, x2, y2, current_frame, current_image]], match_traj_dict, [car_image], {}] # 要不要存储当前的间隔 
    else:
        # 遍历per_car_sample, 如果采样到了当前的帧，就做一个匹配，并修正轨迹
        per_car_sample = video_details.per_car_sample.copy()
        for box in results[0].boxes: 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            car_image = current_image[y1:y2, x1:x2, :]
            clip_score = clip_scoring(trainer, car_image)
            if clip_score == 0:
                continue
            # cv2.imwrite(f"outputs/detrac_cut_red_car/{current_frame}_{x1}_{x2}_{y1}_{y2}.jpg", car_image)
            # 进行轨迹匹配
            match_traj_dict = get_point_update([x1, y1, x2, y2], tracks, cfg)
            match_flag = False
            for car_id, frame_list_and_interval in video_details.per_car_sample.items():
                if current_frame in frame_list_and_interval[0]:
                    # 进行 reid 匹配，需要对car_id下的前一帧进行 reid 匹配
                    if len(video_details.resolved_tuple[car_id][0]) != 0:
                        # reid_score = extractor.similarity(car_image, video_details.resolved_tuple[car_id][2][-1])
                        # clip_score = clip_scoring(trainer, car_image)
                        reid_score = clip_reid(trainer, car_image, video_details.resolved_tuple[car_id][2][-1])
                        # 如果匹配上了，则加入到resolved_tuple中
                        if reid_score > cfg['reid_thresh']:
                            video_details.resolved_tuple[car_id][0].append([x1, y1, x2, y2, current_frame, current_image])
                            video_details.resolved_tuple[car_id][2].append(car_image)
                            if current_frame in video_details.resolved_tuple[car_id][3].keys():
                                if reid_score > video_details.resolved_tuple[car_id][3][current_frame]:
                                    video_details.resolved_tuple[car_id][3][current_frame] = reid_score
                                else:
                                    continue
                            else:
                                video_details.resolved_tuple[car_id][3][current_frame] = reid_score                                
                            # 修正轨迹, 更新resolved_tuple[car_id][1] 即 match_traj_dict,
                            match_intersect = justify_intersect(video_details.resolved_tuple[car_id][1], match_traj_dict)# 这里需要根据n个轨迹的结果来做
                            if match_intersect:
                                # video_details.resolved_tuple[car_id][1] = {match_intersect[0]:match_traj_dict[match_intersect[0]]}
                                video_details.resolved_tuple[car_id][1] = match_intersect
                                # 更新per_car_sample
                                exist_num = len(video_details.resolved_tuple[car_id][0])
                                per_car_sample[car_id] = get_sample_gap_n(current_frame, tracks, match_intersect, n-exist_num, cfg, video_details) # 不能在遍历的时候改变字典的大小，保留备份，最后再更新。
                            match_flag = True
                            break
                # 确认是否是新的车出现, 逻辑错误，这里只是当前carid没有匹配上，并不是所有carid没有匹配上
                elif  frame_list_and_interval[1][0]< current_frame <frame_list_and_interval[1][1]:
                    # 这里是匹配不上的，才算是新的, 新车直接添加到resolved_tuple中，旧车就先pass，后续可以对旧车进行处理，辅助识别的效果
                    # reid_score = extractor.similarity(car_image, video_details.resolved_tuple[car_id][2][-1])
                    reid_score = clip_reid(trainer, car_image, video_details.resolved_tuple[car_id][2][-1])
                    # clip_score = clip_scoring(trainer, car_image)
                    if reid_score > cfg['reid_thresh']:
                        match_flag = True
                        break
                
                #     # apply_id += 1
                #     if reid_score < cfg['reid_thresh']:# 这个判断是为了筛选出新车
                #         apply_id = max(video_details.resolved_tuple.keys()) + 1 \
                #         if len(video_details.resolved_tuple) != 0 else 1
                #         per_car_sample[apply_id] = get_sample_gap_n(current_frame, tracks, match_traj_dict, n-1, cfg)
                #         video_details.resolved_tuple[apply_id] = [[[x1, y1, x2, y2, current_frame, current_image]], match_traj_dict, [car_image], []]
                #     break
            if not match_flag:
                apply_id = max(video_details.resolved_tuple.keys()) + 1 \
                if len(video_details.resolved_tuple) != 0 else 1
                per_car_sample[apply_id] = get_sample_gap_n(current_frame, tracks, match_traj_dict, n-1, cfg, video_details)
                video_details.resolved_tuple[apply_id] = [[[x1, y1, x2, y2, current_frame, current_image]], match_traj_dict, [car_image], {}]
        
        video_details.per_car_sample = per_car_sample    
    next_frame = get_next_frame(current_frame)
    # if next_frame == 154:
    #     print("debug")
    return next_frame

def main(args, video_path, cluster_result_path, detect_object, trainer, cfg):
    
    video_details.skip_frames = cfg['skip_frames']
    video_details.adaptive_skip = cfg['skip_frames']
    
    # detect_object.set_classes([cfg['classes']])
    detect_object.set_classes(args.yolo_classes.split(","))
    tracks = read_cluster_result(cluster_result_path)
    
    stop_region = cfg['stop_area']
    allocate_tracks(tracks, stop_region)
    
    # extractor = feature_extractor(cfg['w'], cfg['h'])
    # extractor.init_extractor()
    
    # 从头开始读取视频
    # videoCapture = cv2.VideoCapture(video_path)
    images = os.listdir(video_path)
    images.sort()
    images = [os.path.join(video_path, image_name) for image_name in images]
    current_frame = cfg['start_frame']
    skip_frames = video_details.skip_frames
    cfg['end_frame'] = len(images)
    if args.use_mask:
        logger.info("use mask")
        mask_image = cv2.imread("./masks/"+cfg["video_name"]+"_mask.jpg", cv2.IMREAD_GRAYSCALE)
    
    while current_frame < cfg['end_frame']:
    # while current_frame < 300:
        decode_time = time.time()
        # videoCapture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        # success, current_image = videoCapture.read()
        current_image = cv2.imread(images[current_frame])
        # original_image = current_image.copy()
        video_details.background_img = current_image
        video_details.decode_time += time.time() - decode_time
        
        if args.use_mask:
            current_image = cv2.add(current_image, np.zeros(np.shape(current_image), dtype=np.uint8), mask=mask_image)
        
        if current_image is None:
            print("Error occured")
            print("Current Frame is %d"%current_frame)
            exit()

        if args.use_filter:
            if video_details.blank_frame: #这里的逻辑完善一下，如果为空，那就跳过很多帧，加速处理
                differ_time = time.time()
                difference_score = frame_difference_score(current_image, video_details.history_frame, cfg["differ_abs_thresh"])
                video_details.frame_differencer_time += time.time() - differ_time

                if args.active_log:
                    logger.info("Similarity is : %5f"%difference_score)

                if difference_score < cfg["difference_thresh"]:
                    current_frame = get_next_frame(current_frame, True, skip_frames)
                    if args.active_log:
                        logger.info("Filtered one")
                    video_details.differencor += 1
                    continue
        
        detect_time = time.time()
        # yolo world或者 grounding dino
        results = detect_object.predict(current_image)
        video_details.detector_time += time.time() - detect_time
        print("-------frame________:", current_frame)
        if len(results[0].boxes) == 0:
            current_frame  = get_next_frame(current_frame, True, skip_frames)
            video_details.history_frame = current_image
            video_details.frame_sampled.append(current_frame)
            video_details.blank_frame = True
            continue
        
        video_details.frame_sampled.append(current_frame)
        video_details.blank_frame = False
        match_time = time.time()
        # next_frame = process_detect_results(current_frame, current_image, results, tracks, extractor, cfg)
        next_frame = process_detect_results(current_frame, current_image, results, tracks, trainer, cfg)
        if not next_frame:
            break
        video_details.match_time += time.time() - match_time
        video_details.history_frame = current_frame
        current_frame = next_frame
    
    return video_details.frame_sampled, video_details.resolved_tuple
        # 这里进行匹配，决定采样的帧数
        # 1. 给当前检测结果（每个 bbox）都匹配最接近的轨迹
        # 2. 对于每个轨迹，根据匹配的轨迹的长度，平均采样 n 帧
        # 3. 对于每个采样的帧，进行 reid 匹配 怎么才能个 bbox 采样 n 帧？
        # 4. 对于每个 bbox，匹配最近的轨迹，确定需要采样的后续n 帧序列，依次处理 n 帧，进行目标检测，reid 匹配，处理完当前帧的所有 bbox 之后，选择所有 bbox 中轨迹最短的一个，确定下一帧的采样位置

def covert_entity_to_frame():
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

def make_trainer(args):
    cfg = setup_cfg(args)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    
    return trainer, cfg
    # 测试clip 是否可以用于 reid，如果可以，那省好多事儿啊， 根据 track_id 选择相同的和不同的，测试

def clip_reid(trainer, np_img1, np_img2):
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    img1 = Image.fromarray(cv2.cvtColor(np_img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(np_img2, cv2.COLOR_BGR2RGB))
    img1 = transform(img1)
    img2 = transform(img2)
    similarity = trainer.model.reid_similarity(img1.to(trainer.device), img2.to(trainer.device))
    
    return float(similarity.cpu())

def clip_scoring(trainer, img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    preds = trainer.judge_pic(img)
    
    return int(preds)

def test_clip_reid(args, cfg1):
    trainer, cfg = make_trainer(args)
    # clip
    # transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    # reid
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    Dataset = ReidDataset(cfg, transform)
    dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True)
    similarities = []
    labels = []
    extractor = feature_extractor(cfg1['w'], cfg1['h'])
    extractor.init_extractor()
    for batch in tqdm(dataloader):
        img1, img2, label = batch
        # clip
        # similarity = trainer.model.reid_similarity(img1.to(trainer.device), img2.to(trainer.device))
        # reid 
        similarity = extractor.similarity(img1, img2)
        similarities.append(float(similarity.cpu()))
        labels.append(int(label))
        del img1
        del img2
        del label
        del similarity
        # 尝试释放显存
        torch.cuda.empty_cache()
    
    print("Similarities: ", similarities)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    cfg = getYaml(args.config)
    cfg['classes'] = args.query
    # # # test_clip_reid(args, cfg1)
    trainer, cfg_dassl = make_trainer(args)
    
    video_path = os.path.join(args.root, args.dataset_name, args.split, 'image')
    # cluster_result_path = "fixed_files/preprocessed/caldot1/caldot1_0_0_tracks_clustered.npy"
    cluster_result_path = f'fixed_files/preprocessed/{args.dataset_name}/{args.dataset_name}.npy'
    # 从 config 中读取，存到一个固定的地方
    chunk_path = os.path.join('sample_output', args.dataset_name, args.query, args.split, 'chunk.json')
    model = YOLO(args.yolo_weight)
    video_details.read_chunks_frame(chunk_path)
    
    frame_sampled, resolved_tuple = main(args, video_path, cluster_result_path, model, trainer, cfg)
    # 使用pickle保存video_details
    save_path = os.path.join('query_output', args.dataset_name, args.query, args.split)
    import pickle
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'query_result.pkl'), "wb") as f:
        pickle.dump(video_details, f)
    with open(os.path.join(save_path, 'query_result.pkl'), "rb") as f:
        video_details = pickle.load(f)
    evaluate_object = evaluation("language_label", cfg)
    label_path = os.path.join(args.root, args.dataset_name, args.split, 'final_label.json')
    
    with open(label_path,"r") as f:
        input_label = json.load(f)
    frame_level_result = covert_entity_to_frame()
    evaluate_object.preprocess(input_label, frame_level_result)
    save_dict = {}
    F1,recall,precision,accuracy = evaluate_object.selection_query_1()
    print("Selection_query: F1:%5f, recall:%5f, precision:%5f, accuracy:%5f"%(F1,recall,precision,accuracy))
    save_dict['Selection_query'] = [F1,recall,precision,accuracy]
    MAE,ACC = evaluate_object.aggregation_query_1()
    save_dict['Aggregation_query'] = [MAE,ACC]
    print("Aggregation_query: MAE:%5f, ACC:%5f"%(MAE,ACC))
    overlap_rate = evaluate_object.top_k_query_1()
    save_dict['Topk query'] = [overlap_rate]
    print("Topk query: overlap_rate:%5f"%(overlap_rate))
    # GT_COUNT,PRED_COUNT = evaluate_object.aggregation_query_3()
    # print("Aggregation_query: GT_COUNT:%5f, PRED_COUNT:%5f"%(GT_COUNT,PRED_COUNT))
    with open(os.path.join('query_output', args.dataset_name, args.query, args.split, 'query_result.json'), "w") as f:
        json.dump(save_dict, f)
    # 为什么还是[11 61]?
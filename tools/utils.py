import cv2
import os
import tqdm
import numpy as np
from settings import settings
import copy
import cv2
import os
from settings.settings import video_details
from collections import Counter
import math
import time
from scipy.spatial import KDTree

def concat_videos(image_list_path,output_path,video_name,fps=25):
    
    image_list = os.listdir(image_list_path)
    image_list.sort(key = lambda x:int(x[-9:-4]))

    img = cv2.imread(os.path.join(image_list_path,image_list[0]))
    (height, width, _) = img.shape
    videoWriter = cv2.VideoWriter(os.path.join(output_path,video_name),\
        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))

    for i in tqdm.trange(len(image_list)):
        img_path = os.path.join(image_list_path, image_list[i])
        img = cv2.imread(img_path)
        videoWriter.write(img)
    videoWriter.release()

def concat_video_slices(video_list_path,output_path,video_name):
    
    video_list = os.listdir(video_list_path)
    video_list.sort(key = lambda x:int(x[:-4]))
    video_list = video_list[:720]

    cap_settings = cv2.VideoCapture(os.path.join(video_list_path,video_list[0]))
    fps = cap_settings.get(cv2.CAP_PROP_FPS)
    width = int(cap_settings.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_settings.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_settings.release()
    videoWriter = cv2.VideoWriter(os.path.join(output_path,video_name),\
        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))

    for i in tqdm.trange(len(video_list)):
        cap = cv2.VideoCapture(os.path.join(video_list_path,video_list[i]))
        while cap.isOpened():  
            ret, frame = cap.read()
            if ret:
                videoWriter.write(frame)
            else:
                break
        cap.release()
    videoWriter.release()


def calculate_IOU(rec1,rec2):
    """
    calculate IoU
    """
    rec1 = [int(x) for x in rec1]
    rec2 = [int(x) for x in rec2]
    left_max = max(rec1[0],rec2[0])
    top_max = max(rec1[1],rec2[1])
    right_min = min(rec1[2],rec2[2])
    bottom_min = min(rec1[3],rec2[3])
    
    if (left_max < right_min and bottom_min > top_max):
        rect1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rect2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        area_cross = (bottom_min - top_max) * (right_min - left_max)
        return area_cross / (rect1_area + rect2_area - area_cross)

    return 0

def calculate_intersection(rec1,rec2):
    """_summary_
    Args:
        rec1 (bbox_list): detection
        rec2 (bbox_list): background_area
    """
    left_max = max(rec1[0],rec2[0])
    top_max = max(rec1[1],rec2[1])
    right_min = min(rec1[2],rec2[2])
    bottom_min = min(rec1[3],rec2[3])
    if (left_max < right_min and bottom_min > top_max):
        rect1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rect2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        area_cross = (bottom_min - top_max) * (right_min - left_max)
        return area_cross / (rect1_area)
    return 0

def cosine_similarity(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def pic_difference(vector_a, vector_b, pixel_num):
    diff = 0
    for i in range(256):
        diff += abs((vector_a[i][0]/pixel_num) - (vector_b[i][0]/pixel_num))
    return diff


def generate_video(video_path, video_save_path, video_name,fps=25):
    file_list = os.listdir(video_path)
    file_list.sort(key=lambda x: int(x[-9:-4]))
    img_0 = cv2.imread(os.path.join(video_path,file_list[0]))
    
    size = (img_0.shape[1],img_0.shape[0])
    video = cv2.VideoWriter(os.path.join(video_save_path,video_name+".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for file in file_list:
        image_path = os.path.join(video_path,file)
        img = cv2.imread(image_path)
        video.write(img)
        
    video.release()
    cv2.destroyAllWindows()

def convert_detection(detect):
    # input:  x1, y1, x2, y2, c
    # output: x_mid, y_mid, w, h, c
    p_w = video_details.patch_width
    x_mid = int((detect[0] + detect[2])/(2*p_w))
    y_mid = int((detect[1] + detect[3])/(2*p_w))
    
    width = int(detect[2] - detect[0])
    height = int(detect[3] - detect[1])

    return [x_mid, y_mid, width, height, float(detect[4])]

def matrix_fill(matrix2fill):
    for col in range(len(matrix2fill)):
        for row in range(len(matrix2fill[col])):
            if matrix2fill[col][row] == 0:
                matrix2fill[col][row] = [1,1]
    return matrix2fill

def get_img(image_base,frame_num):

    image_list = os.listdir(image_base)
    image_list.sort(key=lambda x: int(x[-9:-4]))
    image_name = os.path.join(image_base,image_list[frame_num])
    image = cv2.imread(image_name)
    return image

def associate_cars(detect1,detect2):

    detect_1 = detect1[1:]
    detect_2 =  detect2[:-1]
    iou = calculate_IOU(detect_1,detect_2)
    return iou

def match_last_car_with_gt(last_car,gt_detects,current_frame,cfg):

    last_car = last_car[:-1]
    iou_max = 0
    id_match = -1
    IoU_min = settings.IoU_min
    
    for gt_detect in gt_detects:
        gt_detect = gt_detect[:-1]
        gt_detect = [float(x) for x in gt_detect]
        last_car = [float(x) for x in last_car]
        iou = associate_cars(gt_detect,last_car)
        car_id = int(gt_detect[0])
        if iou > iou_max:
            if iou > IoU_min:
                id_match = car_id
            iou_max = iou
        
    if iou_max <= IoU_min:
        for i in range(max(cfg["start_frame"],current_frame-30), min(cfg["end_frame"],current_frame+30)):
            for gt_detect in video_details.gt_labels[i]:
                gt_detect = gt_detect[:-1]
                gt_detect = [float(x) for x in gt_detect]
                last_car = [float(x) for x in last_car]
                iou = associate_cars(gt_detect,last_car)
                car_id = int(gt_detect[0])
                if iou > iou_max:
                    if iou > IoU_min:
                        id_match = car_id
                    iou_max = iou

    return id_match

def paint_image(img,gt_detects,detects):
    for gt_detect in gt_detects:
        car_id = int(gt_detect[0])
        cv2.putText(img,str(car_id)+" "+gt_detect[-1],(int(gt_detect[1])-10,int(gt_detect[2])-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.rectangle(img,(int(gt_detect[1]),int(gt_detect[2])),(int(gt_detect[3]),int(gt_detect[4])),(0,255,0),2)
    for detect in detects:
        class_num = int(detect[-1]) if int(detect[-1]) in settings.coco_names_invert.keys() else 0
        cv2.rectangle(img,(int(detect[0]),int(detect[1])),(int(detect[2]),int(detect[3])),(0,0,255),2)
        cv2.putText(img,settings.coco_names_invert[class_num],(int(detect[0])+20,int(detect[1])+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return img


def locate_background_color(bg_color,clustered_color,perc):

    bg_color = np.array(bg_color)
    clustered_color = np.array(clustered_color)
    distance = 1e6
    filtered_color = 0
    index = -1

    for i in range(3):
        if  perc[i] > 0.1:
            dis = sum((bg_color-clustered_color[i])**2)
            if dis < distance:
                distance = dis
                index = i
        
    index_2 = -1
    distance = 1e6

    for i in range(3):

        if perc[i] < 0.12:
            index_2 = i
            break
        if i != index:
            dis = sum((clustered_color[i]-np.array([0,0,0]))**2)
            if dis<distance:
                distance = dis
                index_2 = i
    for i in range(3):
        if i!=index and i!=index_2:
            filtered_color = clustered_color[i]
    return filtered_color

def get_background_color(img):

    img_shape = img.shape
    background_block = img[int(0.02*img_shape[0]):int(0.12*img_shape[0]),int(0.02*img_shape[1]):int(0.12*img_shape[1]),:]
    background_block = cv2.mean(background_block)[:3]
    background_block = [int(x) for x in background_block]
    return background_block

def palette_perc(k_cluster):

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items(),reverse=False))

    colors = []
    for i in perc.keys():
        colors.append(list(k_cluster.cluster_centers_[i]))

    return colors,perc

def filter_tracks(tracks):

    thresh_len = video_details.fps * 25
    min_len = video_details.fps * 2
    min_dis = video_details.v_height * 0.2
    track_filtered_by_length = [[],[]] 

    for track in tracks:
        if len(track)<thresh_len and len(track)>min_len:
            if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:   
                track_start = [0.5*track[0][0]+0.5*track[0][2],0.5*track[0][1]+0.5*track[0][3]]
                track_end = [0.5*track[-1][0]+0.5*track[-1][2],0.5*track[-1][1]+0.5*track[-1][3]]
                flag = filter_by_region(track_start,track_end)       
                if flag:      
                    track_filtered_by_length[0].append(track)       
        
    return track_filtered_by_length

def judge_traffic_light(current_frame_id):
    traffic_light = settings.traffic_light[video_details.video_name]
    current_time = (current_frame_id - traffic_light[0][0]) % (traffic_light[1][0]+traffic_light[1][1])
    if current_time > traffic_light[1][0]:
        state = "red"
    else:
        state = "green"
    return state

def filter_by_region(start,end):

    video_name = video_details.video_name
    in_out_region = settings.in_out_region[video_name]
    start_reg_id = -1
    end_reg_id = -1
    for id, item in enumerate(in_out_region):
        # id, [x1,y1,x2,y2] middle_x middle_y
        if (start[0]>item[0] and start[0]<item[2]) and \
            (start[1]>item[1] and start[1]<item[3]):
                start_reg_id = id
        if (end[0]>item[0] and end[0]<item[2]) and \
            (end[1]>item[1] and end[1]<item[3]):
                end_reg_id = id
      
    if start_reg_id != end_reg_id and start_reg_id!=-1 and end_reg_id!=-1:
        return True
    
    else:
        return False

def get_sample_gap(current_frame,tracks,match_traj_dict):
    
    interval = []
    dis_min = 1e4
    best_interval = []
    for key, value in match_traj_dict.items():
        track_id, nearest_point, distance = value
        estimated_starttime = current_frame - nearest_point
        estimated_endtime = current_frame + (len(tracks[track_id]) \
            - nearest_point)
        interval.append([estimated_starttime,estimated_endtime])
        if distance < dis_min: #对于所有候选轨迹，都会计算一个interval，对于所有interval，找到最小的distance
            best_interval = [estimated_starttime,estimated_endtime]
            dis_min = distance

    return interval,best_interval

def get_sample_gap_n(current_frame,tracks,match_traj_dict, n, cfg, video_details):
    interval = []
    dis_min = 1e4
    best_interval = []
    for key, value in match_traj_dict.items():
        track_id, nearest_point, distance = value
        estimated_starttime = max(cfg['start_frame'], current_frame - nearest_point)
        estimated_endtime = current_frame + (len(tracks[track_id]) \
            - nearest_point)
        estimated_endtime = min(cfg['end_frame'], estimated_endtime)
        interval.append([estimated_starttime,estimated_endtime])
        if distance < dis_min:
            best_interval = [estimated_starttime,estimated_endtime]
            dis_min = distance
    best_gap = [current_frame, best_interval[1]]
    
    if n==0:
        return [[], best_interval]
    # 在best_gap区间均匀采样n个点
    gap = (best_gap[1]-best_gap[0])//n
    sample_points = []
    # 如果best_gap区间长度小于n，则直接返回从current_frame到best_interval[1]的所有帧
    # 不能返回所有，返回一个，否则会采样点过多
    # 加入之前得进行一个有无最近点的判断（<5），有的话，给这辆车加上这个点
    # if gap == 0:
    #     for i in range(best_gap[1]-current_frame):
    #         sample_points.append(current_frame+i)
    #     return [sample_points, best_interval]
    if gap == 0:
        # 在best_gap[1]和current_frame之间随机采样一个点
        sample_points.append(np.random.randint(current_frame, best_gap[1]))
    
    for i in range(n):
        sample_points.append(best_gap[0]+(i+1)*gap) 
    
    video_points = np.array(video_details.sample_all_frame_list)
    tree = KDTree(video_points.reshape(-1, 1))  # 创建KD - Tree，将点重塑为列向量
    selected_points = []
    # video_details.sample_all_frame_list
    for sample_point in sample_points:
    # 查询距离当前采样点小于5的点
        distances, indices = tree.query([[sample_point]], k=1, distance_upper_bound=5)
        if distances[0] < 5:
            selected_points.append(video_points[indices[0]])
        else:
            selected_points.append(sample_point)
    
    video_details.sample_all_frame_list.extend(selected_points)
    video_details.sample_all_frame_list = list(set(video_details.sample_all_frame_list))
    video_details.sample_all_frame_list.sort()
    
    return [selected_points, best_interval]

def justify_intersect(history_cache, curr_detect):

    # apply_traj = history_cache[-2]
    apply_traj = history_cache
    route_matched = {}
    closest_intersect = False

    for key_1 in apply_traj.keys():
        for key_2 in curr_detect.keys():
            if key_1 == key_2:
                # route_matched[key_1] = [apply_traj[key_1],curr_detect[key_2]]
                route_matched[key_1] = curr_detect[key_2]
    # # 返回route_matched是不是就可以了？ 返回closest和全部的
    # dis = 1e4
    # for key, value in route_matched.items():
    #     if value[0][-1] + value[1][-1] < dis:
    #         dis = value[0][-1] + value[1][-1]
    #         closest_intersect = [key,value]

    return route_matched #, closest_intersect
        
def justify_intersect_rate(interval_1,interval_2): # 公式7

    left_min = min(interval_1[0],interval_2[0])
    right_max = max(interval_1[1],interval_2[1])
    left_max = max(interval_1[0],interval_2[0])
    right_min = min(interval_1[1],interval_2[1])
    if right_min > left_max:        
        interset_ratio = abs(right_min-left_max)/abs(right_max-left_min)
        return interset_ratio
    else:
        return 0
    
def convert_float(results):
    tmp = []
    for result in results:
        result = [float(x) for x in result]
        tmp.append(result) 
    return tmp


if __name__ == "__main__":
    concat_video_slices("XXX",\
        "XXX","XXX")
from settings import settings
from settings.settings import video_details
import cv2
import os
from match_object import match_cars_main_updated
from tools.timer import Timer
from tools.utils import match_last_car_with_gt,paint_image,convert_float
from tools.frame_difference import frame_difference_score
from reid_extractor import feature_extractor
import numpy as np
from tools.utils import calculate_IOU,calculate_intersection
import time

def is_point_in_rectangle(point, rectangle):

    x, y = point
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom

def allocate_tracks(tracks,stop_area):
    
    str2list = {}
    allocated_id = {}
    count = 0
    for track in tracks:
        traj_pair = []
        start_point = ((track[0][0]+track[0][2])/2,(track[0][1]+track[0][3])/2)
        end_point = ((track[-1][0]+track[-1][2])/2,(track[-1][1]+track[-1][3])/2)
        
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(start_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
            
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(end_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
            
        if len(traj_pair) < 2:
            traj_pair = [-1,-1]

        if str(traj_pair) not in str2list.keys():
            str2list[str(traj_pair)] = [traj_pair]
        if str(traj_pair) not in allocated_id.keys():
            allocated_id[str(traj_pair)] = [count]
        else:
            allocated_id[str(traj_pair)].append(count)
            
        count += 1

    video_details.allocate_id = allocated_id
    video_details.traj_type_dict = str2list           
        

def filter_by_status(detects):

    stop_cars = video_details.stop_cars
    if len(stop_cars.keys()) == 0:
        return detects
    return_detects = []
    for detect in detects:
        match_flag = False
        for bbox in stop_cars.keys():
            if calculate_IOU(list(bbox),detect[:4])>settings.stop_iou_thresh:
                video_details.stop_cars[bbox] += 2
                if stop_cars[bbox]>1: 
                    match_flag = True
                    break
        if not match_flag:
            return_detects.append(detect)
            
    for key in video_details.stop_cars.keys():
        video_details.stop_cars[key]-=1
        
    return return_detects 

def filter_related_detections(detects):

    history_car_detection = video_details.last_frame_detections

    if len(history_car_detection)==0:    
        return detects
    
    for detect in detects:
        for h_detect in history_car_detection:
            match_stop_region = False
            if calculate_IOU(detect[:4],h_detect[:4])>settings.stop_iou_thresh:
                for stop_region in video_details.stop_cars.keys():
                    if calculate_IOU(list(stop_region),h_detect[:4])>settings.stop_iou_thresh:
                        match_stop_region = True
                        break

                if not match_stop_region:
                    video_details.stop_cars[tuple(h_detect[:4])] = 1
                break
    return detects

def filter_by_predefined_area(detects,cfg):
    
    return_detects = []
    filter_area = cfg["ignore_region"]
    if len(filter_area) == 0:
        return detects
    for detect in detects:
        iou_flag = False
        for bbox in filter_area:
            if calculate_intersection(detect,bbox) > 0.65:
                iou_flag = True
                break
        if not iou_flag:
            return_detects.append(detect)
            
    return return_detects
                
def save_sampled_pics(frame_sampled,frame_list,video_image_path,video_image_path_list):
    
    for frame in frame_sampled:
        img = cv2.imread(os.path.join(video_image_path,video_image_path_list[frame]))
        for record in frame_list[frame]:
            cv2.rectangle(img,(int(record[1]),int(record[2])),(int(record[3]),int(record[4])),(0,255,0),2)
            cv2.putText(img,str(record[0]),(int(record[1]),int(record[2])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imwrite('%s.jpg'%(str(frame)),img)

def get_final_tuple_from_detector(tracks,reid_weight,video_path,detect_object,cfg,args,logger):
    
    video_details.skip_frames = cfg["skip_frames"]
    video_details.adaptive_skip = cfg["skip_frames"]
    
    if len(tracks) == 2:
        tracks = tracks[0]
    
    stop_region = cfg["stop_area"]

    allocate_tracks(tracks, stop_region)

    extractor = feature_extractor(reid_weight,cfg["w"],cfg["h"])
    extractor.init_extractor()

    videoCapture = cv2.VideoCapture(video_path)

    current_frame = cfg["start_frame"]
        
    skip_frames = video_details.skip_frames

    if args.use_mask:
        logger.info("mask")
        mask_image = cv2.imread("./masks/"+cfg["video_name"]+"_mask.jpg"\
            ,cv2.IMREAD_GRAYSCALE)
        
    video_details.start_time = time.time()
    while current_frame < cfg["end_frame"]:

        if args.active_log:
            logger.info("Sampled Frame: %d"%(current_frame))
        decode_time = time.time()
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, current_image = videoCapture.read() 
        original_image = current_image.copy()
        video_details.background_img = current_image
        video_details.decode_time += time.time() - decode_time
        if args.use_mask:
            current_image = cv2.add(current_image,np.zeros(np.shape(current_image),dtype=np.uint8),mask=mask_image)  
        
        # if not success:
        if current_image is None:
            print("Error occured")
            print("Current Frame is %d"%current_frame)
            exit()     

        if args.use_filter:
            if video_details.blank_frame: # 如果上一帧是空白帧，这一帧跟上一帧之间的差异很小，就跳过
                differ_time = time.time()
                difference_score = frame_difference_score(current_image,video_details.history_frame,cfg["differ_abs_thresh"])
                video_details.frame_differencer_time += time.time() - differ_time

                if args.active_log:
                    logger.info("Similarity is : %5f"%difference_score)
                    
                if difference_score < cfg["difference_thresh"]: # 差异值小于阈值就跳过
                    current_frame += skip_frames
                    if args.active_log:
                        logger.info("Filtered one")
                    video_details.differencor += 1
                    continue

        detect_time = time.time()                
        if args.use_distill:                 
            _, results = detect_object.detect(current_image)
        else:
            results,flag = detect_object(current_image)
            if len(results)>0:
                results = results.cpu().numpy()
        video_details.detector_time += time.time()-detect_time   

        image_selected = current_image 
        
        results = filter_by_predefined_area(results, cfg)
        
        if len(results)==0:
            if args.visualize:
                cv2.imwrite("./outputs/selected_frames/%d.jpg"%current_frame,image_selected)
            
            video_details.history_frame = current_image
            video_details.frame_sampled.append(current_frame)
            video_details.blank_frame = True 
            current_frame += skip_frames            
            continue
        
        video_details.frame_sampled.append(current_frame)
        video_details.blank_frame = False
        results = convert_float(results)


        if args.visualize:
            cv2.imwrite("./outputs/selected_frames_origin/%d.jpg"%current_frame, original_image)
            image_selected = paint_image(image_selected,video_details.gt_labels[current_frame],results)
            cv2.imwrite("./outputs/selected_frames/%d.jpg"%current_frame, image_selected)

        match_time = time.time()
        sample_gap = match_cars_main_updated(current_frame,current_image,\
            results,extractor,tracks,cfg,args,logger)
        video_details.match_time += time.time()-match_time
        sample_gap_selected = max(sample_gap)
        
        if sample_gap_selected < cfg["fps"] * cfg["min_gap_time"]: # 处理采样间隔过小的情况
            current_frame = current_frame + cfg["fps"] * cfg["min_gap_time"]
            
        elif sample_gap_selected > cfg["fps"] * cfg["max_gap_time"]: # 处理采样间隔较大的情况

            if args.adaptive_sample: # 如果开启自适应采样，就在sample_gap中选择一个小于最大间隔的最大值

                gap_max = cfg["fps"] * cfg["min_gap_time"]
                for gap in sample_gap:
                    if gap < cfg["fps"] * cfg["max_gap_time"] and gap > gap_max:
                        gap_max = gap
                current_frame += gap_max
            else:
                current_frame += cfg["fps"] * cfg["max_gap_time"] # 否则直接选择最大间隔
            
        else:           
            current_frame = sample_gap_selected + current_frame
        
        if args.active_log:
            logger.info("Sample gap selected: %d"%(current_frame-video_details.frame_sampled[-1]))

        video_details.history_frame = current_image

    return video_details.frame_sampled, video_details.resolved_tuple
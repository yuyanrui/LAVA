from settings import settings
from tools.utils import calculate_IOU
from tools.utils import match_last_car_with_gt
from settings.settings import video_details
import cv2
import time

def get_point_update(detect,tracks,cfg):

    dis_thresh = cfg["dis_thresh"]
    match_dict = {}
    allocate_id = video_details.allocate_id
    detect_mid_x = (detect[0]+detect[2])/2
    detect_mid_y = (detect[1]+detect[3])/2
    
    match_flag = False
    for key in allocate_id.keys(): # 一个区域对只挑最近的一个轨迹
        distance = 1e10
        candidate_traj_id = -1
        nearest_point_location = -1
        for id in allocate_id[key]:
            pos_count = 0            
            for point in tracks[id]:
                x_mid = (point[0]+point[2])/2
                y_mid = (point[1]+point[3])/2
                dis = (detect_mid_x-x_mid)**2 + (detect_mid_y-y_mid)**2
                if dis < distance:
                    distance = dis
                    candidate_traj_id = id
                    nearest_point_location = pos_count
                pos_count += 1
        # if distance < dis_thresh and key!="[-1, -1]":
        if distance < dis_thresh:
            match_dict[key] = [candidate_traj_id,nearest_point_location,distance]
            match_flag = True            
    
    if not match_flag:
        match_dict["[66, 66]"] = [5,60,cfg["dis_thresh"]]
    return match_dict

def get_candidate_traj(detect, tracks, cfg):
    dis_thresh = cfg["dis_thresh"]
    match_dict = []
    
    detect_mid_x = (detect[0]+detect[2])/2
    detect_mid_y = (detect[1]+detect[3])/2
    
    match_flag = False
    
    for track in tracks:
        pass
    

def sort_detections(detctions,direction="vertical",reverse=True):

    detections_sorted = []
    if direction=="vertical":
        detections_sorted = sorted(detctions,key=lambda x:x[1],reverse=reverse)
    if direction=="horizon":
        detections_sorted = sorted(detctions,key=lambda x:x[0],reverse=reverse)
    return detections_sorted

def match_cars_main_updated(current_frame,curr_frame_img,records,\
    extractor,tracks,cfg,args,logger):

    reverse = video_details.reverse
    direction = video_details.direction

    if cfg["video_name"] in ["m30","m30hd"]:
        records = sort_detections(records, direction, reverse)

    stop_region = cfg["stop_area"]

    sample_gap = []
    reid_count = 0
    reid_count = 0
    
    if current_frame == cfg["start_frame"] or len(video_details.frame_sampled)==1:

        if args.active_log:
            logger.info("No history detection, all cars are new!")

        for record in records:
            x1, y1, x2, y2, conf, class_id = record
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            car_image = curr_frame_img[y1:y2,x1:x2,:]
            match_traj_dict = get_point_update(record,tracks,cfg)

            apply_id = max(video_details.resolved_tuple.keys())+1 \
                if len(video_details.resolved_tuple.keys())!=0 else 1
                
            video_details.history_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                car_image])

            video_details.resolved_tuple[apply_id]=[[[x1,y1,x2,y2,class_id,current_frame]],match_traj_dict,\
                [car_image],[]]

            from tools.utils import get_sample_gap
            intervals, closest_interval = get_sample_gap(current_frame,tracks,match_traj_dict)
            video_details.resolved_tuple[apply_id][-1] = [closest_interval] # 轨迹距离最近的被选为预测轨迹，选取其 interval 作为当前 detect object 的 interval

            for interval in intervals:
                sample_gap.append(interval[1]-current_frame)

            id_match = match_last_car_with_gt(record,video_details.gt_labels[current_frame],current_frame,cfg) # 这一步目的是什么？
            video_details.match_dict[apply_id] = id_match
            video_details.object_type[apply_id] = class_id

        return sample_gap
    
    else:
        exclude_list = [] 
        current_cache = []
        for record in records:
            x1,y1,x2,y2,conf,class_id = record
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            car_image = curr_frame_img[y1:y2,x1:x2,:]
            match_traj_dict = get_point_update(record,tracks,cfg)
            selection_gap = int(current_frame-video_details.frame_sampled[-2])
            match_flag = False
            
            for history_obj in video_details.history_cache: 
                bbox_hist = history_obj[1:5]
                match_iou = calculate_IOU([x1,y1,x2,y2],bbox_hist)
                apply_id = history_obj[0] 
                from tools.utils import justify_intersect
                match_intersect = justify_intersect(history_obj,match_traj_dict)
                
                if match_iou > cfg["stop_iou_thresh"] and apply_id not in exclude_list: #对应5.2节最后

                    current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                    video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])
                    
                    interval_new = []
                    for interval in video_details.resolved_tuple[apply_id][-1]:
                        interval_new.append([interval[0],interval[1]+selection_gap])
                        
                    video_details.resolved_tuple[apply_id][-1] = interval_new
                    video_details.resolved_tuple[apply_id][2].append(car_image)
                    
                    exclude_list.append(apply_id)
                    match_flag = True
                    sample_gap.append(cfg["skip_frames"])
                    current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                    break
                    
                elif (match_intersect) and (apply_id not in exclude_list):

                    last_frame_id = video_details.frame_sampled[-2]
                    traj_hist = match_intersect[1][0]
                    traj_curr = match_intersect[1][1]
                    if traj_hist[0] == traj_curr[0] and traj_hist[1] > traj_curr[1]:
                        hist_interval = [last_frame_id-(len(tracks[traj_hist[0]]) - traj_hist[1]),last_frame_id + \
                                        traj_hist[1]]
                        curr_interval = [current_frame-(len(tracks[traj_hist[0]]) - traj_hist[1]),current_frame + \
                                        traj_hist[1]]
                    else:
                        hist_interval = [last_frame_id-traj_hist[1],last_frame_id + \
                                        (len(tracks[traj_hist[0]]) - traj_hist[1])]
                        curr_interval = [current_frame-traj_curr[1],current_frame + \
                                        (len(tracks[traj_curr[0]]) - traj_curr[1])]
                        
                    from tools.utils import justify_intersect_rate
                    MAPE_GAP = justify_intersect_rate(hist_interval,curr_interval)
                    if MAPE_GAP > cfg["intersect_overlap_thresh"]:
                        
                        apply_id = history_obj[0]                                                        
                        current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                    car_image])
                        video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])
                        video_details.resolved_tuple[apply_id][1] = {match_intersect[0]:match_traj_dict[match_intersect[0]]}
                        video_details.resolved_tuple[apply_id][2].append(car_image)
                        exclude_list.append(apply_id)
                        match_flag = True
                        sample_gap.append(min(hist_interval[1],curr_interval[1])-current_frame)

                        if traj_hist[0] == traj_curr[0]:
                            error_rate = abs(traj_hist[1]-traj_curr[1])/selection_gap
                            curr_interval = [min(hist_interval[0],curr_interval[0]),max(hist_interval[1],curr_interval[1])]
                        video_details.resolved_tuple[apply_id][-1] = [curr_interval]
                        break
                    
                    else:

                        curr_frame = video_details.frame_sampled[-1]
                        last_frame = video_details.frame_sampled[-2]
                        if cfg["dataset_group"] == "blazeit":
                            curr_bbox = [x1,y1,x2,y2]
                            last_bbox = history_obj[1:5]
                            history_img = history_obj[-1]

                            reid_time = time.time()
                            image_score = float(extractor.inference_pic([car_image,history_img,\
                                curr_bbox,last_bbox,curr_frame,last_frame]).item()) # 这里调用的匹配函数
                            video_details.reid_time += time.time()-reid_time
                            
                        else:
                            image_score = 0.1

                        if args.visualize:
                            reid_count+=1
                            cv2.imwrite("./outputs/reid/%s/%d_%d_%s.jpg"%(cfg["video_name"],current_frame,reid_count,image_score),car_image)
                            reid_count+=1
                            cv2.imwrite("./outputs/reid/%s/%d_%d.jpg"%(cfg["video_name"],current_frame,reid_count),history_img)  
                                
                        if args.visualize:

                            curr_label = video_details.gt_labels[curr_frame]
                            last_label = video_details.gt_labels[last_frame]
                            curr_id = -1
                            last_id = -2
                            for label in curr_label:
                                thresh = calculate_IOU(curr_bbox,label[1:5])
                                if thresh>0.8:
                                    curr_id = label[0]
                                    break
                            for label in last_label:
                                thresh = calculate_IOU(last_bbox,label[1:5])
                                if thresh>0.8:
                                    last_id = label[0]
                                    break
                            if curr_id==last_id and image_score > cfg["image_thresh_score"]:
                                video_details.reid_acc[0]+=1
                            elif curr_id!=last_id and image_score < cfg["image_thresh_score"]:
                                video_details.reid_acc[0]+=1
                            else:
                                video_details.reid_acc[1]+=1                            
                            
                        if image_score > cfg["image_thresh_score"]:
                            
                            apply_id = history_obj[0]
                            current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                        car_image])
                            video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])

                            video_details.resolved_tuple[apply_id][1] = {match_intersect[0]:match_traj_dict[match_intersect[0]]}
                            video_details.resolved_tuple[apply_id][2].append(car_image)

                            exclude_list.append(apply_id)
                            match_flag = True
                            traj_curr = match_traj_dict[match_intersect[0]] # 前后两个目标轨迹的交点（相同/最接近的轨迹）
                            curr_interval = [current_frame-traj_curr[1],current_frame + \
                                     (len(tracks[traj_curr[0]]) - traj_curr[1])] #traj_curr[1]是[candidate_traj_id,nearest_point_location,distance] 中的 nearest_point_location，当前目标最近的轨迹点的位置（帧数）
                            sample_gap.append(curr_interval[1] - current_frame)
                            break # 此record找到了就不再继续找了，换下一个 record 继续匹配 
                        else:
                            pass
                        
            if not match_flag:

                apply_id = max(video_details.resolved_tuple.keys())+1 \
                    if len(video_details.resolved_tuple.keys())!=0 else 1
                    
                current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                video_details.resolved_tuple[apply_id]=[[[x1,y1,x2,y2,class_id,current_frame]],match_traj_dict,\
                                            [car_image],[]]

                from tools.utils import get_sample_gap
                intervals, closest_interval = get_sample_gap(current_frame,tracks,match_traj_dict)
                video_details.resolved_tuple[apply_id][-1] = [closest_interval]
                
                for interval in intervals:
                    sample_gap.append(interval[1]-current_frame)
                
                id_match = match_last_car_with_gt(record,video_details.gt_labels[current_frame],current_frame,cfg)
                
                video_details.match_dict[apply_id] = id_match
                video_details.object_type[apply_id] = class_id
    
    video_details.history_cache = current_cache
    return sample_gap


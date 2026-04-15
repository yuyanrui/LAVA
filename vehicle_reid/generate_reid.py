import cv2
import numpy as np
import os
import random

object_type = {0:"red_car", 2:'car',5:'bus',7:'truck'}
video_base = "/path/to/video/dataset/"
label_base = "/path/to/label/dataset/"
save_base = "/path/to/reid_dataset"
mask_base = "/path/to/mask/dataset/"

def get_labels(video_name,d_type):
    # label_path = os.path.join(label_base, "%s%s.txt"%(video_name,d_type))
    label_path = "/path/to/label/dataset/detrac_red_car.txt"
    with open(label_path,"r") as f:
        contents = f.readlines()
    parsed_result = [[] for i in range(50000)]
    parsed_tuple = {}

    for content in contents:
        content = content.split(",")
        frame_id = int(content[0])
        car_id = int(content[1])
        x_min = max(0,int(float(content[2])))
        y_min = max(0,int(float(content[3])))
        x_max = int(float(content[4]))
        y_max = int(float(content[5]))
        conf = float(content[7])
        object_class = object_type[int(float(content[6]))]
        parsed_result[frame_id].append([car_id,x_min,y_min,x_max,y_max,conf,object_class])
        if car_id not in parsed_tuple.keys():
            parsed_tuple[car_id] = [frame_id,frame_id]
        else:
            parsed_tuple[car_id][1] = frame_id
    return parsed_result,parsed_tuple #parsed_result 存储的是每一帧包含的车辆信息，parsed_tuple存储的是每一辆车的起始帧和结束帧

def generate_train_dataset(video_name, skip_frame, parsed_result, parsed_tuple,cam_id="9"):

    # image_mask = cv2.imread(os.path.join(mask_base,video_name+"_mask.jpg"),cv2.IMREAD_GRAYSCALE)
    save_path = os.path.join(save_base,video_name,"image_train")
    # video_path = os.path.join(video_base,video_name+".mp4")
    video_path = "/path/to/video/dataset/detrac_original.mp4"
    video_object = cv2.VideoCapture(video_path)
    print("Parsed")
    frame_count = 0
    os.makedirs(save_path,exist_ok=True)
    while True:
        ret,frame = video_object.read()
        if ret:
            # frame_masked = cv2.add(frame,np.zeros(np.shape(frame),dtype=np.uint8),mask=image_mask)
            frame_masked = frame
            labels = parsed_result[frame_count]
            if frame_count%skip_frame==0:
                for label in labels:
                    car_id,x_min,y_min,x_max,y_max,\
                        conf,object_class = label
                    
                    duration_time = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                    if duration_time < 900 and duration_time > 15:
                        pic = frame_masked[y_min:y_max,x_min:x_max,:]
                        cv2.imwrite(os.path.join(save_path,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,car_id,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
            frame_count += 1
        else:
            break

def generate_test_query_dataset(video_name, skip_frame, parsed_result, parsed_tuple,cam_id="9"):

    # image_mask = cv2.imread(os.path.join(mask_base,video_name+"_mask.jpg"),cv2.IMREAD_GRAYSCALE)
    save_path_test = os.path.join(save_base,video_name,"image_test")
    save_path_query = os.path.join(save_base,video_name,"image_query")
    video_path = os.path.join(video_base,video_name+"1.mp4")
    video_object = cv2.VideoCapture(video_path)
    print("Parsed")
    frame_count = 0

    test_dict = {}
    os.makedirs(save_path_test,exist_ok=True)
    os.makedirs(save_path_query,exist_ok=True)
    while True:
        ret,frame = video_object.read()
        if ret:
            # frame_masked = cv2.add(frame,np.zeros(np.shape(frame),dtype=np.uint8),mask=image_mask)
            frame_masked = frame
            labels = parsed_result[frame_count]
            if frame_count%skip_frame==0:
                for label in labels:
                    car_id,x_min,y_min,x_max,y_max,\
                        conf,object_class = label
                    duration_time = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                    if duration_time<900 and duration_time>15:
                        pic = frame_masked[y_min:y_max,x_min:x_max,:]
                        if car_id not in test_dict.keys():
                            test_dict[car_id] = [[pic,object_class,x_min,y_min,x_max,y_max,frame_count]]
                        else:
                            test_dict[car_id].append([pic,object_class,x_min,y_min,x_max,y_max,frame_count])
            frame_count += 1
        else:
            for key in test_dict.keys():
                query_id = random.randint(0,len(test_dict[key])-1)
                for i in range(len(test_dict[key])):
                    if i != query_id:
                        pic,object_class,x_min,y_min,x_max,y_max,frame_count = test_dict[key][i]
                        cv2.imwrite(os.path.join(save_path_test,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,key,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
                    else:
                        pic,object_class,x_min,y_min,x_max,y_max,frame_count = test_dict[key][i]
                        cv2.imwrite(os.path.join(save_path_query,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,key,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
            break

if __name__ == "__main__":

    parsed_result,parsed_tuple = get_labels("DETRAC",d_type="0")
    generate_train_dataset("DETRAC",5,parsed_result,parsed_tuple,cam_id="1")
    # parsed_result,parsed_tuple = get_labels("warsaw",d_type="1")
    # generate_test_query_dataset("warsaw",5,parsed_result,parsed_tuple,cam_id="1")
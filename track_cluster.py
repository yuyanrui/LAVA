import sklearn
import os
import shutil
import multiprocessing
from traj_cluster.distance import distance
import numpy as np
import math
from traj_cluster.cluster.Kmediods import Kmediods
from traj_cluster.cluster.FCM import FCM 
from evaluate import DBI_Vec
import xml.dom.minidom
from collections import Counter

def get_blazeit_labels(video_name,d_type="train"):
    
    coco_names_invert = {7:"truck",2:"car",5:"bus", 0:"red car"}
    label_base = "XXX"
    
    # label_file = os.path.join(label_base,video_name,d_type,\
    #             "label_%s_%s.txt"%(video_name,d_type))
    label_file = "generate-datasets/amsterdam.txt"
    with open(label_file,"r") as f:
        pseudo_label = f.readlines()
    f.close()

    label_parsed = [[] for i in range(111300)]
    label_tuple_origin = {}
    tuple_dict = {}

    for label in pseudo_label:
        
        label = label.strip()
        label = label.split(",")
        frame_id = int(label[0])
        car_id = int(label[1])
        x_min = int(float(label[2]))
        y_min = int(float(label[3]))
        x_max = int(float(label[4]))
        y_max = int(float(label[5]))
        obj_class = 0
        # obj_class = int(float(label[7]))
        # obj_class = int(float(label[6]))
        # object_name = coco_names_invert[obj_class]
        object_name = label[6]
        label_parsed[frame_id].append([car_id,x_min,y_min,x_max,y_max,object_name])
        if car_id not in tuple_dict.keys():
            tuple_dict[car_id] = [frame_id,frame_id,[[x_min,y_min,x_max,y_max]],[]]
            label_tuple_origin[car_id] = [frame_id,frame_id,[[x_min,y_min,x_max,y_max]],[]]
        else:
            tuple_dict[car_id][1] = frame_id
            label_tuple_origin[car_id][1] = frame_id
            tuple_dict[car_id][2].append([x_min, y_min, x_max, y_max])
            label_tuple_origin[car_id][2].append([x_min, y_min, x_max, y_max])
        tuple_dict[car_id][-1].append(obj_class)
        label_tuple_origin[car_id][-1].append(obj_class)
            
    car_ids = list(tuple_dict.keys())
    for car_id in car_ids:

        # 删掉时间过长的轨迹和时间过短的轨迹，删掉距离过短的轨迹
        if abs(tuple_dict[car_id][1] - tuple_dict[car_id][0]) < 35:
            del tuple_dict[car_id]
            continue
        
        if abs(tuple_dict[car_id][1] - tuple_dict[car_id][0]) > 1350:
            del tuple_dict[car_id]
            continue
            
        start_xmin, start_ymin, start_xmax, start_ymax = tuple_dict[car_id][2][0]
        end_xmin,   end_ymin,   end_xmax,   end_ymax   = tuple_dict[car_id][2][-1]
        start_xcenter, start_ycenter = int((start_xmin + start_xmax)/2.0), int((start_ymin + start_ymax)/2.0)
        end_xcenter,   end_ycenter = int((end_xmin + end_xmax)/2.0), int((end_ymin + end_ymax)/2.0)
        
        if ((start_xcenter - end_xcenter)**2+(start_ycenter - end_ycenter)**2)**0.5 < 30:
        # 过短的轨迹不予考虑
            del tuple_dict[car_id]
            continue
        
        class_number_dict = {}

        for class_id in tuple_dict[car_id][-1]:
            if class_id not in class_number_dict:
                class_number_dict[class_id] = 0
            class_number_dict[class_id] += 1
        all_class_numbers = []
        for class_id, class_number in class_number_dict.items():
            all_class_numbers.append([class_id, class_number])
        tuple_dict[car_id][-1] = max(all_class_numbers, key=lambda x:x[1])[0]            

    for car_id in label_tuple_origin.keys():
        label_tuple_origin[car_id][-1] = Counter(label_tuple_origin[car_id][-1]).most_common(2)[0][0]
        
    return  label_parsed, tuple_dict, label_tuple_origin

def get_m30_labels(video_name,K):
    coco_names = {"car":2,"bus":5,"truck":7,0:"others","van":7,"big-truck":7}
    label_files = os.listdir(os.path.join("XXX",video_name,"xml"))
    label_files.sort(key=lambda x:int(x.split(".")[0]))
    label_files = label_files[:K]
    frame_num_all = K
    label_parsed = [[] for i in range(frame_num_all)]
    tuple_dict = {}

    for file in label_files:
        frame_num = int(file.split(".")[0])
        dom = xml.dom.minidom.parse(os.path.join("XXX",video_name,"xml",file))
        root = dom.documentElement
        objects = root.getElementsByTagName("object")

        for object in objects:
            object_class = object.getElementsByTagName("class")[0].childNodes[0].nodeValue

            if object_class == "car" or object_class == "truck" or object_class == "big-truck" or object_class == "van":
                car_id = int(object.getElementsByTagName("ID")[0].childNodes[0].nodeValue)   
                bnd_box = object.getElementsByTagName("bndbox")[0]
                xmin = int(bnd_box.getElementsByTagName("xmin")[0].childNodes[0].nodeValue) 
                ymin = int(bnd_box.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
                xmax = int(bnd_box.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
                ymax = int(bnd_box.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)   
                label_parsed[frame_num].append([car_id,xmin,ymin,xmax,ymax,object_class])   # Frame number need attention 

                if car_id not in tuple_dict.keys():
                    tuple_dict[car_id] = [frame_num,frame_num,[[xmin,ymin,xmax,ymax]],coco_names[object_class]]
                else:
                    tuple_dict[car_id][1] = frame_num
                    tuple_dict[car_id][2].append([xmin, ymin, xmax, ymax])

    return  label_parsed, tuple_dict

def compute_distance(tr_id,trajs_dict,distance_type="frechet"):
    print("Begin Compute: ",tr_id)
    try:        
        trs_matrix = distance.cdist(trajs_dict,{tr_id:trajs_dict[tr_id]},type=distance_type)  
    except Exception as e:
        print(e)
    np.save('./outputs/traj_compute_tmp/'+str(tr_id)+".npy",trs_matrix)
    print("Done! ",tr_id)

def compute_distance_single(tr_id,trajs_dict,distance_type="frechet"):
    try:
        trs_matrix = distance.cdist(trajs_dict,{tr_id:trajs_dict[tr_id]},type=distance_type)  
    except Exception as e:
        print(e)
    return trs_matrix
    

def compute_distance_all(traj_dict,num_thread=48):

    shutil.rmtree('./outputs/traj_compute_tmp')
    os.mkdir('./outputs/traj_compute_tmp')
    
    pool = multiprocessing.Pool(processes=num_thread)
    for tr_id in traj_dict.keys():
        pool.apply_async(compute_distance, (tr_id,traj_dict))
    pool.close()
    pool.join()
    
def compute_distance_all_single(traj_dict):
    distances = {}
    for tr_id in traj_dict.keys():
        trs_matrix = compute_distance_single(tr_id,traj_dict)
        for id in trs_matrix:
            distances[id] = trs_matrix[id]
    return distances

def load_computed_distance():

    files = os.listdir("./outputs/traj_compute_tmp/")
    
    distances = {}
    for file in files:
        distance_dict = np.load(os.path.join("./outputs/traj_compute_tmp",file),allow_pickle=True)
        distance_dict = distance_dict.item()
        for id in distance_dict:
            distances[id] = distance_dict[id]
            id2 = tuple((id[1],id[0]))# 对称的位置添加
            if id2 not in distances.keys():
                distances[id2] = distance_dict[id]

    return distances

def convert2dict(tracks,split_ratio=2):# 隔一个点处理, 存储的内容是每个bbox的中心点
    traj_dict = {}
    count = 0
    for track in tracks:
        traj_dict[count] = []
        point_count = 0
        for detect in track:
            if point_count%split_ratio==0:
                traj_dict[count].append([int(0.5*detect[0]+0.5*detect[2]),int(0.5*detect[1]+0.5*detect[3])])
            point_count+=1
        count += 1
    return traj_dict

def convert2matrix(distance_dict):
    numbers = int(math.sqrt(len(distance_dict.keys())))
    distance_matrix = np.zeros((numbers,numbers))
    for key in distance_dict.keys():
        distance_matrix[key[0]][key[1]] = distance_dict[key]
    return distance_matrix

def get_traj_cluster(tracks,k_min=8,k_max=32):

    tracks_origin = tracks.copy()
    traj_dict = convert2dict(tracks)
    compute_distance_all(traj_dict)
    distances = load_computed_distance()
    distance_matrix = convert2matrix(distances)
    score_min = 1000
    center_result = 0
    for k_num in range(k_min,int(k_max),2):
        labels, centers = Kmediods(k_num, distance_matrix)

        cluster_dict = {}
        for num in range(len(labels)):
            if labels[num] in cluster_dict.keys():
                cluster_dict[labels[num]].append(num)
            else:
                cluster_dict[labels[num]]=[num]

        selected_centers = []
        for center in centers:
            selected_centers.append(tracks_origin[center])

        score = DBI_Vec(cluster_dict,distance_matrix)

        if score < score_min:

            score_min = score
            center_result = selected_centers

    return center_result, score_min

def get_unique_traj(centers, member_ship):
    centers = np.array(centers)
    unique_centers, indices = np.unique(centers, axis=0, return_index=True)
    unique_member_ship = member_ship[:, indices]
    return unique_centers, unique_member_ship

def get_traj_cluster_fcm(tracks,k_min=8,k_max=32):

    tracks_origin = tracks.copy()
    traj_dict = convert2dict(tracks)
    compute_distance_all(traj_dict)
    distances = load_computed_distance()
    distance_matrix = convert2matrix(distances)
    score_min = 1000
    center_result = 0
    m=2.00
    epsilon=1e-5
    
    for k_num in range(k_min,int(k_max),2):
        labels, centers, member_ship = FCM(k_num, distance_matrix, m=m, max_iter=100, epsilon=epsilon)
        centers, member_ship = get_unique_traj(centers, member_ship)
        cluster_dict = {}
        for num in range(len(labels)):
            if labels[num] in cluster_dict.keys():
                cluster_dict[labels[num]].append(num)
            else:
                cluster_dict[labels[num]]=[num]

        selected_centers = []
        for center in centers:
            selected_centers.append(tracks_origin[center])

        score = DBI_Vec(cluster_dict,distance_matrix)

        if score < score_min:

            score_min = score
            center_result = selected_centers
        # distance unique, member_ship unique
    return center_result, member_ship, score_min

if __name__ == "__main__":
    pass
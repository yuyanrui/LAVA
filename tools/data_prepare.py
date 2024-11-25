#coding=utf-8
import cv2
# import cfg
import os
import numpy as np
import xml.dom.minidom
import json
import pandas as pd
import pickle
from settings import settings
from collections import Counter

def data_preprocess(frame_list):
    convert_label = {}  # key: car_id  label: center_x  center_y  width height
    label_details = frame_list

    for frame in label_details:
        for curr_label in frame:
            car_id = int(curr_label[0])
            x1 = float(curr_label[1])
            y1 = float(curr_label[2])
            x2 = float(curr_label[3])
            y2 = float(curr_label[4])
            width = x2 - x1
            height = y2 - y1
            center_x = round((x1 + x2)/2,3)
            center_y = round((y1 + y2)/2,3)
            if car_id in convert_label.keys():
                convert_label[car_id].append([center_x,center_y,width,height])
            else:
                convert_label[car_id]=[[center_x,center_y,width,height]]

    return convert_label

# Standard Dataset Loader

class get_label_details(object):
    
    def __init__(self,cfg,args,logger):

        self.dataset_group = cfg["dataset_group"]
        self.frame_num = 0
        self.tuple_dict = {} 
        self.label_parsed = []
        self.d_type = args.type
        self.video_id = cfg["video_name"]
        self.label_base = settings.label_path[self.dataset_group]
        self.cfg = cfg
        self.args = args
        self.logger = logger

    def get_label(self):

        if self.dataset_group.lower() == "m30":

            label_files = os.listdir(os.path.join(self.label_base,self.video_id,"xml"))
            label_files.sort(key=lambda x:int(x.split(".")[0]))
            self.frame_num = len(label_files)
            self.label_parsed = [[] for i in range(self.frame_num)]

            for file in label_files:
                frame_num = int(file.split(".")[0])
                dom = xml.dom.minidom.parse(os.path.join(self.label_base,self.video_id,"xml",file))
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
                        self.label_parsed[frame_num].append([car_id,xmin,ymin,xmax,ymax,object_class])   # Frame number need attention 

                        if car_id not in self.tuple_dict.keys():
                            self.tuple_dict[car_id] = [frame_num,frame_num,settings.coco_names[object_class]]
                        else:
                            self.tuple_dict[car_id][1] = frame_num
            return  self.label_parsed, self.tuple_dict, "Perfect Label"

        elif self.dataset_group.lower() == "detrac":

            label_file = open(os.path.join(self.label_base,self.video_id,self.d_type,\
                "label_%s_%s.txt"%(self.video_id,self.d_type)), "r")

            contents = label_file.readlines()
            self.frame_num = int(contents[-1].split(" ")[1])
            self.label_parsed = [[] for i in range(self.frame_num)]
            
            for content in contents:
                
                content = content.split(" ")
                frame_num = int(content[1])
                object_id = int(content[8])
                object_class = str(content[3])
                speed = float(content[9])
                orient = float(content[10])
                [x_min,y_min,w,h] = [float(content[4]),\
                    float(content[5]),float(content[6]),float(content[7])]
                self.label_parsed[frame_num-1].append([object_id,x_min,y_min,x_min+w,y_min+h,object_class])

                if object_id not in self.tuple_dict.keys():
                    self.tuple_dict[object_id] = [frame_num,frame_num]
                else:
                    self.tuple_dict[object_id][1] = frame_num

        elif self.dataset_group == "blazeit":
            
            label_file = os.path.join(self.label_base,self.video_id,self.d_type,\
                "label_%s_%s.txt"%(self.video_id,self.d_type))

            with open(label_file,"r") as f:
                pseudo_label = f.readlines()
            f.close()
            self.label_parsed = [[] for i in range(111300)]
            label_tuple_origin = {}

            for label in pseudo_label:

                label = label.split(",")
                frame_id = int(label[0])
                car_id = int(label[1])
                x_min = int(float(label[2]))
                y_min = int(float(label[3]))
                x_max = x_min+int(float(label[4]))
                y_max = y_min+int(float(label[5]))
                obj_class = int(float(label[7]))
                object_name = settings.coco_names_invert[obj_class]
                self.label_parsed[frame_id].append([car_id,x_min,y_min,x_max,y_max,object_name])
                if car_id not in self.tuple_dict.keys():
                    self.tuple_dict[car_id] = [frame_id,frame_id,[x_min,y_min,x_max,y_max],[x_min, y_min, x_max, y_max],[]]
                    label_tuple_origin[car_id] = [frame_id,frame_id,[x_min,y_min,x_max,y_max],[x_min, y_min, x_max, y_max],[]]
                else:
                    self.tuple_dict[car_id][1] = frame_id
                    label_tuple_origin[car_id][1] = frame_id
                    self.tuple_dict[car_id][3] = [x_min, y_min, x_max, y_max]
                    label_tuple_origin[car_id][3] = [x_min, y_min, x_max, y_max]
                self.tuple_dict[car_id][-1].append(obj_class)
                label_tuple_origin[car_id][-1].append(obj_class)
            
            car_ids = list(self.tuple_dict.keys())
            for car_id in car_ids:
                if abs(self.tuple_dict[car_id][1] - self.tuple_dict[car_id][0]) < self.cfg["fps"] * 1: # * 2
                    del self.tuple_dict[car_id]
                    continue
                
                start_xmin, start_ymin, start_xmax, start_ymax = self.tuple_dict[car_id][2]
                end_xmin,   end_ymin,   end_xmax,   end_ymax   = self.tuple_dict[car_id][3]
                start_xcenter, start_ycenter = int((start_xmin + start_xmax)/2.0), int((start_ymin + start_ymax)/2.0)
                end_xcenter,   end_ycenter = int((end_xmin + end_xmax)/2.0), int((end_ymin + end_ymax)/2.0)
                
                if ((start_xcenter - end_xcenter)**2+(start_ycenter - end_ycenter)**2)**0.5 < self.cfg["traj_dist_min"]:
                    del self.tuple_dict[car_id]
                    continue
                
                class_number_dict = {}
                for class_id in self.tuple_dict[car_id][-1]:
                    if class_id not in class_number_dict:
                        class_number_dict[class_id] = 0
                    class_number_dict[class_id] += 1
                all_class_numbers = []
                for class_id, class_number in class_number_dict.items():
                    all_class_numbers.append([class_id, class_number])
                self.tuple_dict[car_id][-1] = max(all_class_numbers, key=lambda x:x[1])[0]            

            for car_id in label_tuple_origin.keys():
                label_tuple_origin[car_id][-1] = Counter(label_tuple_origin[car_id][-1]).most_common(2)[0][0]
            return  self.label_parsed, self.tuple_dict, label_tuple_origin
            
        else:
            max_frame = 109000
            frame_num = max_frame

            label_file = os.path.join(self.label_base,self.video_id,self.video_id+"-"+self.date+".csv")

            labels = pd.read_csv(label_file)
            
            self.label_parsed = [[] for i in range(max_frame)]
            for index, row in labels.iterrows():
                object_name = row["object_name"]
                frame_id = int(row["frame"])
                if frame_id>=max_frame:
                    continue
                x1 = int(row["xmin"])
                y1 = int(row["ymin"])
                x2 = int(row["xmax"])
                y2 = int(row["ymax"])
                car_id = int(row["ind"])
                if object_name in ["car","bus","truck","van"]:
                    self.label_parsed[frame_id].append([car_id,x1,y1,x2,y2,object_name])
                    
                    if car_id not in self.tuple_dict.keys():
                        self.tuple_dict[car_id] = [frame_id,frame_id]
                    else:
                        self.tuple_dict[car_id][1] = frame_id
            
                
        return  self.label_parsed, self.tuple_dict

if __name__ == "__main__":

    pass
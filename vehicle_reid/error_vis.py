import cv2
import os
from reid_extractor import feature_extractor 

class errorVis:
    def __init__(self, extractor, label_file, video_name, image_path):
        self.extractor = extractor
        self.label_file = label_file
        self.track_dict = {}
        self.frame_dict = {}
        self.video_name = video_name
        self.image_path = image_path
        self.get_label(label_file)
        
        
    def get_label(self, label_file):
        with open(label_file,"r") as f:
            pseudo_label = f.readlines()
        f.close()
        for label in pseudo_label: 
            label = label.strip()
            label = label.split(",")
            frame_id = int(label[0])
            self.car_id = int(label[1])
            x_min = int(float(label[2]))
            y_min = int(float(label[3]))
            x_max = int(float(label[4]))
            y_max = int(float(label[5]))
            # obj_class = int(float(label[7]))
            obj_class = int(float(label[6]))
            if frame_id in self.frame_dict.keys():
                self.frame_dict[frame_id].append([x_min, y_min, x_max, y_max, 0, 0])
            else:
                self.frame_dict[frame_id] = [[x_min, y_min, x_max, y_max,0, 0]] #[4]表示是否已经匹配过，[5]表示匹配的车辆id
        # 将frame_dict中的frame_id按照从小到大的顺序排列
        self.frame_id_set = list(self.frame_dict.keys())
        self.frame_id_set.sort()

    def reid_init(self):
        self.car_id = 0
        # 初始化
        frame_id = self.frame_id_set[0]
        for i, bbox in enumerate(self.frame_dict[frame_id]):
            self.car_id += 1
            self.track_dict[self.car_id] = [frame_id, frame_id, [bbox[0:4]]]
            self.frame_dict[frame_id][i][5] = self.car_id

    def dis_reid(self):
        # 根据前后两帧的 bbox 位置进行匹配
        frame_prev = self.frame_id_set[0]
        for frame_id in self.frame_id_set[1:]:
            if frame_id - frame_prev > 20:# 间隔太大，直接认为是新的车辆
                for i, bbox in enumerate(self.frame_dict[frame_id]):
                    self.car_id += 1
                    self.track_dict[self.car_id] = [frame_id, frame_id, [bbox[0:4]]]
                    self.frame_dict[frame_id][i][5] = self.car_id
            else:
                for j, bbox1 in enumerate(self.frame_dict[frame_id]):
                    dis_min = 100000
                    match_id = 9999
                    for i, bbox2 in enumerate(self.frame_dict[frame_prev]):
                        if bbox2[4] == 1:
                            continue
                        mid1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
                        mid2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
                        dis =  ((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)**0.5
                        if dis < dis_min and dis < 50:
                            dis_min = dis
                            match_id = i
                    if match_id != 9999:
                        self.frame_dict[frame_prev][match_id][4] = 1
                        self.track_dict[self.frame_dict[frame_prev][match_id][5]][1] = frame_id
                        self.track_dict[self.frame_dict[frame_prev][match_id][5]][2].append(bbox1[0:4])
                        self.frame_dict[frame_id][j][5] = self.frame_dict[frame_prev][match_id][5]
                    else:
                        self.car_id += 1
                        self.track_dict[self.car_id] = [frame_id, frame_id, [bbox1[0:4]]]
                        self.frame_dict[frame_id][j][5] = self.car_id
    
            frame_prev = frame_id
    
    def reid_sim(self):
        # 根据前后两帧的 bbox 相似度进行匹配
        image_files = os.listdir(self.image_path)
        image_files.sort()
        frame_prev = self.frame_id_set[0]
        for frame_id in self.frame_id_set[1:]:
            if frame_id - frame_prev > 20:# 间隔太大，直接认为是新的车辆
                for i, bbox in enumerate(self.frame_dict[frame_id]):
                    self.car_id += 1
                    self.track_dict[self.car_id] = [frame_id, frame_id, [bbox[0:4]]]
                    self.frame_dict[frame_id][i][5] = self.car_id
            else:
                img_new = cv2.imread(os.path.join(self.image_path, image_files[frame_id-1]))
                img_prev = cv2.imread(os.path.join(self.image_path, image_files[frame_prev-1]))
                for j, bbox1 in enumerate(self.frame_dict[frame_id]):
                    match_id = 9999
                    min_similarity = 0.1
                    img_bbox1 = img_new[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2]]
                    for i, bbox2 in enumerate(self.frame_dict[frame_prev]):
                        if bbox2[4] == 1:
                            continue
                        img_bbox2 = img_prev[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2]]
                        similarity = self.extractor.similarity(img_bbox1,img_bbox2)
                        if similarity > 0.5 and similarity > min_similarity:
                            match_id = i
                            min_similarity = similarity
                    if match_id != 9999:
                        self.frame_dict[frame_prev][match_id][4] = 1
                        self.track_dict[self.frame_dict[frame_prev][match_id][5]][1] = frame_id
                        self.track_dict[self.frame_dict[frame_prev][match_id][5]][2].append(bbox1[0:4])
                        self.frame_dict[frame_id][j][5] = self.frame_dict[frame_prev][match_id][5]
                    else:
                        self.car_id += 1
                        self.track_dict[self.car_id] = [frame_id, frame_id, [bbox1[0:4]]]
                        self.frame_dict[frame_id][j][5] = self.car_id
            frame_prev = frame_id
    
    def save_vis(self):
        # video_name = "caldot1-2"
        video_name = self.video_name
        image_background = cv2.imread("/path/to/images/%s.jpg"%(video_name))

        colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
        img_origin = image_background.copy()
        count = 0
        for tra in self.track_dict.values():
            track = tra[2]
            count += 1
            for i in range(len(track) - 1):  # 遍历每一对连续的点
                pt1 = (int(0.5*(track[i][0]+track[i][2])), int(0.5*(track[i][1]+track[i][3])))
                pt2 = (int(0.5*(track[i+1][0]+track[i+1][2])), int(0.5*(track[i+1][1]+track[i+1][3])))
                cv2.line(img_origin, pt1, pt2, colors[count%8], 2)
        cv2.imwrite("./%s_trajs_origin_error.jpg"%(video_name),img_origin)

        print(len(self.frame_id_set))

def main():
    extractor = feature_extractor(960, 540)
    extractor.init_extractor()
    label_file = "/home/yyr/yu88/CLIP_Surgery/runs/caldot1-2_error.txt"
    video_name = "caldot1-2"
    image_path = "/data/usrs/yyr/datasets/caldot/caldot1-2"
    visual = errorVis(extractor, label_file, video_name, image_path)
    visual.reid_init()
    visual.reid_sim()
    visual.save_vis()

if __name__ == "__main__":
    main()




            
            


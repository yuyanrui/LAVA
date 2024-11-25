import numpy as np
import os
from collections import Counter
import cv2
import math
from track_cluster import get_traj_cluster, get_traj_cluster_fcm, get_blazeit_labels, get_m30_labels 

class cluster_traj(object):

    def __init__(self,video_name,background_img,origin_tuples,min_traj_num,thresh_num=100):

        self.video_name = video_name
        self.track_filtered = []
        self.origin_tracks = []
        self.track_clustered = []
        self.background = background_img
        self.origin_tuples = origin_tuples
        self.min_traj_num = min_traj_num
        self.thresh_num = thresh_num

    def filter_traj(self):

        tracks = []
        for key,value in self.origin_tuples.items():
            tracks.append(value[2])
        self.origin_tracks = tracks

        min_len = 20
        max_len = 600
        min_dis = 100

        for track in tracks:

            if len(track) < max_len and len(track) > min_len:

                if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                    (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:                
                    self.track_filtered.append(track)

        self.track_filtered = self.track_filtered[:self.thresh_num]

    def get_cluster(self, line=True):
        tracks_clustered, _ = get_traj_cluster(self.track_filtered,min(\
            self.min_traj_num,len(self.track_filtered)),\
                max(0.5*len(self.track_filtered),min(self.min_traj_num, len(self.track_filtered))))
        # tracks_clustered, _ = get_traj_cluster_fcm(self.track_filtered,min(\
        #     self.min_traj_num,len(self.track_filtered)),\
        #         max(0.5*len(self.track_filtered),min(self.min_traj_num, len(self.track_filtered))))
                # int(0.5*len(self.track_filtered)))
        self.track_clustered = tracks_clustered
        print(len(tracks_clustered))
        if True:
            colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
            img_origin = self.background.copy()
            count = 0
            if line:
                for track in self.origin_tracks:
                    count += 1
                    for i in range(len(track) - 1):  # 遍历每一对连续的点
                        pt1 = (int(0.5*(track[i][0]+track[i][2])), int(0.5*(track[i][1]+track[i][3])))
                        pt2 = (int(0.5*(track[i+1][0]+track[i+1][2])), int(0.5*(track[i+1][1]+track[i+1][3])))
                        cv2.line(img_origin, pt1, pt2, colors[count%8], 2)
                cv2.imwrite("./%s_trajs_origin_line.jpg"%(self.video_name),img_origin)

                img_clustered = self.background.copy()
                count = 0
                for track in self.track_clustered:
                    count += 1
                    for i in range(len(track) - 1):  # 同样的逻辑应用于聚类后的轨迹
                        pt1 = (int(0.5*(track[i][0]+track[i][2])), int(0.5*(track[i][1]+track[i][3])))
                        pt2 = (int(0.5*(track[i+1][0]+track[i+1][2])), int(0.5*(track[i+1][1]+track[i+1][3])))
                        cv2.line(img_clustered, pt1, pt2, colors[count%8], 2)
                cv2.imwrite("./%s_trajs_clustered_s_line.jpg"%(self.video_name),img_clustered)
            else:
                for track in self.origin_tracks:
                    count += 1
                    for point in track:
                        cv2.circle(img_origin,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
                cv2.imwrite("./%s_trajs_origin.jpg"%(self.video_name),img_origin)

                img_clustered = self.background.copy()
                count = 0
                for track in self.track_clustered:
                    count += 1
                    for point in track:
                        cv2.circle(img_clustered,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
            
                cv2.imwrite("./%s_trajs_clustered_s.jpg"%(self.video_name),img_clustered)

def track_cluster_from_labels(video_name,v_type="blazeit",cluster_min_num=16,cluster_tracks_num=96,k=2000,save_result=True):
    
    if v_type == "blazeit":
        image_background = cv2.imread("./masks/background/%s.jpg"%(video_name))
        label_parsed, tuple_dict, label_tuple_origin = get_blazeit_labels(video_name)# 这里先把过长和过短的轨迹筛选掉了
        sorted_keys = sorted(tuple_dict, key=tuple_dict.get)[:k] # 保留前k辆车
        result_by_key = {key: tuple_dict[key] for key in sorted_keys}
        tuple_dict = result_by_key

    else:
        image_background = cv2.imread("XXX.jpg"%(video_name))
        label_parsed, tuple_dict = get_m30_labels(video_name,k)

    cluster = cluster_traj(video_name,image_background,tuple_dict,cluster_min_num,cluster_tracks_num)
    cluster.filter_traj() 
    cluster.get_cluster()

    if save_result:
        os.makedirs("./fixed_files/preprocessed/%s/"%(video_name),exist_ok=True)
        np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_clustered.npy"%(k),cluster.track_clustered)
        if v_type == "blazeit":
            np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_filtered.npy"%(k),cluster.track_filtered)
        np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_origin.npy"%(k),cluster.origin_tracks)

if __name__ == "__main__":

    video_name = "amsterdam"
    track_cluster_from_labels(video_name,"blazeit",16,100,2000,save_result=True)


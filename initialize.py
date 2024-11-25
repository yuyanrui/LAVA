import cv2
from labeler import *
import time
from settings import settings
import math
from track_cluster import get_traj_cluster
import numpy as np
from settings.settings import video_details

class pesudo_labeler(object):
    def __init__(self,cfg,args,logger):

        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.video_name = cfg["video_name"]
        self.type = args.type

    def get_pesudo_label(self):

        thresh_len = self.cfg["thresh_len"]
        mask_name = self.cfg["video_name"]
        mask_image = cv2.imread(settings.mask_path+mask_name+"_mask.jpg"\
        ,cv2.IMREAD_GRAYSCALE)

        from video_labeling.video_tracker import get_k_frame_label

        video_inference_path = os.path.join(settings.video_path,"standard_split", self.cfg["full_name"],"concat"\
            ,self.type,"%s_%s.mp4"%(self.video_name,self.type))
        pseudo_label,background_img = get_k_frame_label(video_inference_path,\
                                                    mask_image, k=self.cfg["start_frame"])
        if self.args.save_result:
            np.save("./outputs/traj_saved/"+self.cfg["video_name"]+"_0_"+\
                str(self.cfg["start_frame"])+".npy",pseudo_label)

        result_tuple = {} 
        frame_list = [[] for i in range(self.cfg["start_frame"])]
        tracks = []

        for i in pseudo_label.keys():
            descriptor = pseudo_label[i]
            [starttime, endtime] = descriptor[0]
            result_tuple[i] = [starttime, endtime]
            trajectory = descriptor[1]
            for record in trajectory:
                frame_id = record[4]
                frame_list[frame_id].append([i,record[0],record[1],record[2],record[3]])
            if len(descriptor[1])> self.cfg["thresh_len"]:
                tracks.append(descriptor[1])
        return tracks, result_tuple, frame_list, background_img

    def load_label_from_disc(self):
        
        gt_labels = video_details.gt_labels
        result_tuple = {}
        tracks_tmp = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        start_frame = self.cfg["start_frame"]
        frame_list = gt_labels[:start_frame]
        for i in range(start_frame):
            contents = gt_labels[i]
            for content in contents:
                car_id = content[0]
                x_min = content[1]
                y_min = content[2]
                x_max = content[3]
                y_max = content[4]
                object_class = content[5]
                if car_id not in result_tuple.keys():
                    result_tuple[car_id] = [i,i]
                    tracks_tmp[car_id] = [[x_min,y_min,x_max,y_max,i]]
                else:
                    result_tuple[car_id][1] = i
                tracks_tmp[car_id].append([x_min,y_min,x_max,y_max,i])
        tracks = [tracks_tmp[key] for key in tracks_tmp.keys()]
        return tracks, result_tuple, frame_list

class cluster_traj(object):

    def __init__(self,cfg,args,logger):

        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.video_name = cfg["video_name"]
        self.type = args.type
        self.track_filtered = []
        self.origin_tracks = []
        self.track_clustered = []

    def filter_traj(self,tracks):
        self.origin_tracks = tracks
        min_len = self.cfg["thresh_len"]
        max_len = self.cfg["max_len"]
        min_dis = self.cfg["traj_dist_min"]

        for track in tracks:

            if len(track) < max_len and len(track) > min_len:

                if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                    (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:                
                    self.track_filtered.append(track)

    def get_cluster(self, img_background = np.zeros((640,640,3))):
        tracks_clustered, _ = get_traj_cluster(self.track_filtered,min(\
            settings.cluster_min_traj_num,len(self.track_filtered)),\
                len(self.track_filtered))
        self.track_clustered = tracks_clustered

        if self.args.visualize:
            colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
            img_origin = img_background.copy()
            count = 0
            for track in self.origin_tracks:
                count += 1
                for point in track:
                    cv2.circle(img_origin,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
            cv2.imwrite("./outputs/cluster_pic/%s_trajs_origin.jpg"%self.cfg["video_name"],img_origin)

            img_clustered = img_background.copy()
            count = 0
            for track in self.track_clustered:
                for point in track:
                    cv2.circle(img_clustered,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
        
            cv2.imwrite("./outputs/cluster_pic/%s_trajs_clustered_s.jpg"%self.cfg["video_name"],img_clustered)
            self.logger.info("Clustered traj image saved ---")



def initialize(cfg,args,logger):

    start_time = time.time()
    frame_sampled = cfg["start_frame"]
    labeler = pesudo_labeler(cfg,args,logger)
    if args.use_label:
        tracks, result_tuple, frame_list = labeler.load_label_from_disc()
        background_img = np.zeros((1080,1920,3))
    else:
        tracks, result_tuple, frame_list, background_img = labeler.get_pesudo_label()

    cluster = cluster_traj(cfg,args,logger)
    cluster.filter_traj(tracks)
    cluster.get_cluster(background_img)

    if args.save_result:
        np.save("./outputs/fixed_files/preprocessed/"+cfg['video_name']+"_0_%d_tuple_dict.npy"%(frame_sampled),result_tuple)
        np.save("./outputs/fixed_files/preprocessed/"+cfg['video_name']+"_0_%d_tracks_clustered.npy"%(frame_sampled),cluster.track_clustered)
        np.save("./outputs/fixed_files/preprocessed/"+cfg['video_name']+"_0_%d_tracks_filtered.npy"%(frame_sampled),cluster.track_filtered)
        np.save("./outputs/fixed_files/preprocessed/"+cfg['video_name']+"_0_%d_tracks_origin.npy"%(frame_sampled),tracks)

    if args.use_cluster:
        return cluster.track_clustered, result_tuple

    else:
        return cluster.track_filtered, result_tuple
    


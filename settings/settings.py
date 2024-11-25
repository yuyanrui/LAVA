import json
video_path = "XXX"
label_path = {"blazeit":"XXX","m30":"XXX"}
mask_path = "XXX/masks/"

map_dict = {"adventure":"Adventure_Rentals","flat":"Flat_Creek_Inn",\
                "gather":"Gather","square":"Square_Northeast","jackson":"Jackson_Town",\
                    "taipei":"Taipei_Hires"}

coco_names_invert = {7:"truck",2:"car",5:"bus"}
coco_names = {"car":2,"bus":5,"truck":7,0:"others","van":2,"big-truck":7}

cluster_min_traj_num = 12

IoU_min = 0.3

class Video_info(object):
    def __init__(self):
        
        self.background_img = 0   
        self.reverse = True           
        self.direction = "vertical"   
        self.skip_frames = 50         
        self.adaptive_skip = 90       

        self.traj_dict = {}           
        self.allocate_id = {}         
        self.traj_type_dict = {}      
        self.gt_labels = []           
        self.gt_tuple = {}            
        self.gt_tuple_origin = {}
        self.match_dict = {}         
        self.object_type = {}         
        self.return_tuple = {}        
        self.stop_cars = {}           
        
        self.history_cache = []       
        self.history_frame = 0        
        self.blank_frame = False     
        self.resolved_tuple = {}      
        self.frame_sampled = []       
        self.bbox_rate = {}           
        self.stop_area = []

        self.reid_acc = [0,0]
        self.differencor = 0
        self.start_time = 0
        self.detector_time = 0
        self.reid_time = 0
        self.frame_differencer_time = 0
        self.match_time = 0
        self.decode_time = 0
        
        self.visualize = True
        self.load = False
        self.use_mask = True
        
        self.per_car_sample = {}
        self.sample_all_frame_list = []
        
    def read_chunks_frame(self, chunk_path):
        self.chunk_frames = json.load(open(chunk_path, 'r'))['sampled_frames']

video_details = Video_info()
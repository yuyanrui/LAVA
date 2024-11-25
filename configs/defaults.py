from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# using cuda or cpu for code environment
_C.MODEL.DEVICE = "cuda"
# If using conda, which device whodule you use
_C.MODEL.DEVICE_ID = '4'
# If using half float
_C.MODEL.HALF = False
# Name of backbone detaction module
_C.MODEL.DETACT_NAME = 'yolov5'
# Name of backbone embedding module
_C.MODEL.EMBEDDING_NAME = 'reid'
# Name of Model Baseline or ours
_C.MODEL.NAME = "baseline"
# Yolo-v5 release 
_C.MODEL.YOLO_WEIGHT = "./weights/yolov5x.pt"
# ReID weight 
_C.MODEL.EMB_WEIGHT = ""
# If Yolo Using DNN
_C.MODEL.YOLO_DNN = False
# Dataset Rooting
_C.MODEL_YOLO_CLASSSET = ""
# Detection Confidence Threshold
_C.MODEL.DETECT_CONF_THRE = 0.5
# Detection Iou Confidence Threshold
_C.MODEL.DETECT_IOU_THRE = 0.5
# Detection Filter By Class
_C.MODEL.DETECT_CLASSES = [2,5,7] # 0 # 2, 5, 7 
# Detection If Using Class-Agnostic NMS
_C.MODEL.DETECT_AGNOSTIC_NMS = False
# Detection Maximum Detections Per Image
_C.MODEL.DETECT_MAX_DETECT = 1000
# Delta for fast video processing
_C.MODEL.FVP_DELTA = 1.3

_C.SOLVER = CN()
# Model Training seed
_C.SOLVER.SEED = 1234

# Dataset detail
_C.DATASET = CN()
_C.DATASET.NAME = "blazeit"
_C.DATASET.GT_THRE = 25.0

# DataLoader detail
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

_C.INPUT = CN()
# Some Transform detail
_C.INPUT.SIZE = 640
_C.INPUT.EMB_SIZE = [384, 384]
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.OUTPUT = CN()
# Output detection visual result
_C.OUTPUT.VISUAL_DETECT = ""
# Output detection embedding result
_C.OUTPUT.EMBED_DETECT = ""
# Output detection ground truth embedding
_C.OUTPUT.GT_EMBED_DETECT = ""
# Output gt detection visual result
_C.OUTPUT.GT_DETECT = ""

_C.TEST = CN()
# Our video sampling fps rate
_C.TEST.SAMPLING_RATE = 25
# inferencing batch
_C.TEST.IMS_PER_BATCH = 128
# Extract detect embedding
_C.TEST.EMBED_STEP = 64
# Topk Number
_C.TEST.TOPK = 10

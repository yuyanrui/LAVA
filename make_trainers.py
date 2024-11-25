import argparse
import torch
from train_prompt import reset_cfg, setup_cfg, extend_cfg
import trainers.maple
from dassl.engine import build_trainer
from torchvision import transforms
from PIL import Image

def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/usrs/yyr/datasets/lava_dataset", help="path to dataset")
    parser.add_argument("--dataset_name", type=str, default="warsaw", help="path to dataset config file")
    parser.add_argument("--query", type=str, default="bus with single-section design", help="query")
    parser.add_argument("--full", action="store_false", help="full dataset")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--train_num", type=int, default=2000, help="number of training samples")
    parser.add_argument("--output-dir", type=str, default="output", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=3, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml", help="path to config file"
    )
    parser.add_argument("--trainer", type=str, default="MaPLe", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument(
        "--model-dir",
        type=str,
        default="output/MaPLe/full/warsaw/seed1",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--chunks_num", type=int, default=1000, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=2000, help="only positive value enables a fixed seed"
    )
    # args.chunks_num
    parser.add_argument(
        "--load-epoch", type=int, default=10, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    return args

def make_trainer():
    args = parsers()
    cfg = setup_cfg(args)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    
    return trainer, args


# trainer, _ = make_trainer()
# image = Image.open("/data/usrs/yyr/datasets/lava_dataset/warsaw/test/image/frame_000001.jpg")
# preds = trainer.judge_pic(image)
# print("make_trainer() is OK")
# 写一个脚本，根据数据集，seed，自动确定 model_dir,参考 test.sh如何写
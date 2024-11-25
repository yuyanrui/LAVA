import os
import cv2
import json
import random
import numpy as np

from PIL import Image
from make_trainers import make_trainer
from tqdm import tqdm


class Sample:
    def __init__(self, image_path, chunks, discrim, args, max_epoch, threshold):
        self.image_path = image_path
        self.chunks = chunks
        self.discrim = discrim
        self.max_epoch = max_epoch
        self.threshold = threshold
        self.N1 = [0] * len(chunks)
        self.n = [0] * len(chunks)
        self.args = args

    def run(self):
        images = os.listdir(self.image_path)
        images.sort()
        images = [os.path.join(self.image_path, image_name) for image_name in images]

        for i in tqdm(range(self.max_epoch)):
            # 1) 选择块和帧
            R = []
            for j in range(len(self.chunks)):
                Rj = self.sample_Rj(j)
                R.append(float(Rj))

            R = np.array(R)
            j_star = np.argmax(R)
            frame_id = self.chunks.sample_frame_from_chunk(j_star)

            rgb_frame = Image.open(images[frame_id])

            result = int(self.discrim.judge_pic(rgb_frame))

            # 3) 更新状态
            self.N1[j_star] += result
            self.n[j_star] += 1
        
        self.final_scores()
        self.final_sampled_frames()
        self.save_sampled_frames()
        return self.sampled_frames
        

    def sample_Rj(self, j):
        N1_j = self.N1[j] + 0.1  # 使用文章中提到的固定值
        n_j = self.n[j] + 1

        # 使用Gamma分布进行Thompson采样
        alpha = N1_j
        beta = n_j
        Rj = np.random.gamma(shape=alpha, scale=1 / beta)
        return Rj

    def final_scores(self):
        self.scores = []
        for alpha, beta in zip(self.N1, self.n):
            self.scores.append(alpha * beta)
    
    def final_sampled_frames(self):
        self.sampled_frames = []
        for score, chunk in zip(self.scores, self.chunks.chunks):
            if score > self.threshold:
                for i in range(chunk[0], chunk[1]):
                    self.sampled_frames.append(i)
        self.sampled_frames = list(set(self.sampled_frames))
        self.sampled_frames.sort()
    
    def save_sampled_frames(self):
        save_path = os.path.join('sample_output', self.args.dataset_name, self.args.query, self.args.split)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_dict = {}
        self.save_dict['sampled_frames'] = self.sampled_frames
        self.save_dict['scores'] = self.scores
        with open(os.path.join(save_path, 'chunk.json'), 'w') as f:
            json.dump(self.save_dict, f)
          
class Chunks:
    def __init__(self, image_path, chunks_num):
        # 模拟视频分块
        # 根据视频长度和块数，将视频分成若干块，每个块用一个长度区间来表示
        self.chunks = []
        self.image_path = image_path
        self.chunks_num = chunks_num
        self.sampled_frames = []
        self.split_video()

    def split_video(self):
        # 读取视频信息
        images = os.listdir(self.image_path)
        images.sort()
        images = [os.path.join(self.image_path, image_name) for image_name in images]

        # 获取视频帧数
        frame_num = len(images)

        # 计算每个块的长度
        chunk_len = frame_num // self.chunks_num

        # 分割视频
        start = 0
        for i in range(self.chunks_num - 1):
            end = start + chunk_len
            self.chunks.append([start, end])
            start = end

        # 最后一个块
        self.chunks.append([start, frame_num])
        for i in range(1, self.chunks_num):
            self.chunks[i][0] += 1
    
    def __len__(self):
        return self.chunks_num    
    
    # 从指定的第i块中采样一帧，但是要标记已经采样过的帧，避免重复采样
    def sample_frame_from_chunk(self, i):
        start, end = self.chunks[i]

        # 从尚未采样的帧中随机选择一帧
        available_frames = [frame_id for frame_id in range(start, end) if frame_id not in self.sampled_frames]
        if not available_frames:
            print("All frames in this chunk have been sampled.")
            return None

        sampled_frame_id = np.random.choice(available_frames)
        self.sampled_frames.append(sampled_frame_id)

        return sampled_frame_id

class Discriminator:
    def __init__(self, trainer):
        self.trainer = trainer
    def judge_pic(self, img):
        pred = self.trainer.judge_pic(img).cpu()
        return pred
        
def main():
    trainer, args = make_trainer()
    chunks_num  = args.chunks_num
    max_epoch = args.max_epoch
    image_path = os.path.join(args.root, args.dataset_name, args.split, 'image')  # 替换为你的视频路径
    chunks = Chunks(image_path, chunks_num)
    discrim = Discriminator(trainer)

    Sample = Sample(image_path, chunks, discrim, args, max_epoch=max_epoch, threshold=1)
    Sample.run()

if __name__ == "__main__":
    main()
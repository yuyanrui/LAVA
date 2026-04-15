# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Taipei(ImageDataset):
    """Taipei.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: -.
        - images: -.
    """
    dataset_dir = "taipei"
    dataset_name = "taipei"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(Taipei, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        data = []
        for img_path in img_paths:
            camid, carid, _, x_min, y_min, x_max, y_max, frame_id = img_path.split("/")[-1].rstrip(".jpg").split("_")
            
            camid = int(camid)
            carid = int(carid)
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            frame_id = int(frame_id)

            if carid == -1: continue
            assert 0 <= carid <= 9446
            assert 2<= camid < 3
            camid -= 1  # index starts from 0
            if is_train:
                carid = self.dataset_name + "_" + str(carid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, carid, camid, x_min / 1280, y_min / 720, x_max / 1280, y_max / 720, frame_id))

        return data
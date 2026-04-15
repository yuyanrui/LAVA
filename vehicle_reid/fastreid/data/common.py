# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        frameid = img_item[3]
        
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "frameids": frameid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

class JackDataset(Dataset):
    """Image Vehicle ReID Dataset on the Jackson"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        cid_set = set()
        cam_set = set()
        for i in img_items:
            cid_set.add(i[1])
            cam_set.add(i[2])

        self.cids = sorted(list(cid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.cid_dict = dict([(p, i) for i, p in enumerate(self.cids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        
        cid = img_item[1]
        camid = img_item[2]
        location_x1s = img_item[3]
        location_y1s = img_item[4]
        location_x2s = img_item[5]
        location_y2s = img_item[6]
        
        frameid = img_item[7]
        
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            cid = self.cid_dict[cid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": cid,
            "camids": camid,
            "frameids": frameid,
            "location_x1s": location_x1s,
            "location_y1s": location_y1s,
            "location_x2s": location_x2s,
            "location_y2s": location_y2s,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.cids)

    @property
    def num_cameras(self):
        return len(self.cams)
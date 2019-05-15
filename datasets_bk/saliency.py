import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random
import scipy.stats
import cv2


class Folder(data.Dataset):
    def __init__(self, root, crop=None, flip=False, transform=None,
                 mean=None, std=None):
        super(Folder, self).__init__()
        self.mean, self.std = mean, std
        self.flip = flip
        self.transform, self.crop = transform, crop
        gt_dir = os.path.join(root, 'masks')
        img_dir = os.path.join(root, 'images')
        names = [name[:-4] for name in os.listdir(gt_dir)]
        self.img_filenames = [os.path.join(img_dir, name+'.jpg') for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name+'.png') for name in names]
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        gt = Image.open(gt_file)
        if self.crop is not None:
            # random crop size of crop
            w, h = img.size
            th, tw = int(self.crop*h), int(self.crop*w)
            if w == tw and h == th:
                return 0, 0, h, w
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            img = img.crop((j, i, j + tw, i + th))
            gt = gt.crop((j, i, j + tw, i + th))
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        gt = np.array(gt, dtype=np.uint8)
        if self.flip and random.randint(0, 1):
            gt = gt[:, ::-1].copy()
            img = img[:, ::-1].copy()
        gt[gt != 0] = 1
        img = img.astype(np.float64) / 255
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()
        return img, gt, name
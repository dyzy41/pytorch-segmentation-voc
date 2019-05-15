import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
from torchvision import transforms
import pdb
import random
import sys
import matplotlib.pyplot as plt


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ['background', 'object']


class VOC(data.Dataset):
    def __init__(self, root, split='train', crop=None, transform=None,
                 mean=None, std=None):
        super(VOC, self).__init__()
        self.mean, self.std = mean, std
        self.split = split
        self.transform, self.crop = transform, crop
        gt_dir = os.path.join(root, 'gt')
        img_dir = os.path.join(root, 'img')
        names = open('{}/{}.txt'.format(root, split)).read().split('\n')[:-1]
        gt_filenames = ['{}/{}.png'.format(gt_dir, name) for name in names]
        img_filenames = ['{}/{}.jpg'.format(img_dir, name) for name in names]
        self.img_filenames = img_filenames
        self.gt_filenames = gt_filenames
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        gt = Image.open(gt_file).convert('RGB')
        if self.crop is not None:
            # random crop
            w, h = img.size
            th, tw = self.crop
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
        gt = np.array(gt)
        lbl = -1*np.ones((gt.shape[0], gt.shape[1]), dtype=np.int64)
        for i, color in enumerate(index2color):
            lbl[np.where(np.all(gt == color, axis=2))] = i

        img = img.astype(np.float64) / 255
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl)
        return img, lbl, name


if __name__ == "__main__":
    sb = VOC('/home/kawhi/Documents/VOC/VOCdevkit/VOC2012',
             transform=transforms.Compose([transforms.Resize((512, 512))]))
    b1, b2, name = sb.__getitem__(0)
    plt.imshow(b2)
    print(name)
    plt.show()

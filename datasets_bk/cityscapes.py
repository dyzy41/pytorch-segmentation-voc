import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random
import sys
from torchvision import transforms

sys.path.append('cityscapesScripts')
from cityscapesscripts.helpers.csHelpers import getCsFileInfo
from cityscapesscripts.helpers.labels import labels


class Cityscape(data.Dataset):
    def __init__(self, root, split='train', crop=(1024, 1024), transform=None,
                 mean=None, std=None):
        super(Cityscape, self).__init__()
        self.crop = crop
        self.mean, self.std = mean, std
        self.split = split
        self.transform = transform
        gt_dir = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', split)
        img_dir = os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split)
        gt_filenames = []
        gt_names = []
        names = []
        for path, _, _names in os.walk(gt_dir):
            gt_filenames += [os.path.join(path, name) for name in _names if name.endswith('color.png')]
            gt_names += [name for name in _names if name.endswith('color.png')]
        img_filenames = []
        for gt_name in gt_names:
            csFile = getCsFileInfo(gt_name)
            img_filenames.append(
                '{}/{}/{}_{}_{}_leftImg8bit.png'.format(img_dir, csFile.city, csFile.city , csFile.sequenceNb , csFile.frameNb)
            )
            names.append('{}_{}_{}'.format(csFile.city , csFile.sequenceNb , csFile.frameNb))
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
        gt = Image.open(gt_file)
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
        gt = np.array(gt, dtype=np.uint8)
        if gt.shape[2] > 3:
            gt = gt[:, :, :3]
        lbl = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.int64)
        for label in labels:
            lbl[np.where(np.all(gt == label.color, axis=2))] = label.trainId
        lbl[lbl==255] = -1

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
    sb = Cityscape('/home/zeng/data/datasets/segmentation_Dataset/Cityscapes',
                   transform=transforms.Compose([transforms.Resize(256)]))
    b1, b2, name = sb.__getitem__(0)
    pdb.set_trace()

from PIL import Image
import os
import numpy as np
p = '/home/weikai/new_sdb/jk/rs_data/AerialImageDataset/train/new_data/val_gt_255/'
tgt = '/home/weikai/new_sdb/jk/rs_data/AerialImageDataset/train/new_data/val_gt/'
files = os.listdir(p)
for i in range(len(files)):
    print(files[i])
    img = Image.open(p + files[i])
    arr = np.asarray(img)
    arr_ = np.where(arr == 255, 1, 0).astype(np.uint8)
    img_ = Image.fromarray(arr_)
    img_.save(tgt + files[i])
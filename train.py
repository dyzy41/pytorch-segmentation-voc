from __future__ import division
import torch
import re
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import os
from dataloaders import custom_transforms as tr
from dataloaders import pascal

from tensorboardX import SummaryWriter
from my_deeplab3_mit import DeepLabv3
import argparse
import utils_1 as utils
from torchvision.utils import make_grid
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_list = [0]

parser = argparse.ArgumentParser()

parser.add_argument('--b', type=int, default=2)
opt = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_size = 512
epoches = 240000
base_lr = 0.0001
lr = base_lr
weight_decay = 2e-5
momentum = 0.9
power = 0.99
decay_epoch=13
num_class = 21
data_file = './data/'
composed_transforms_tr = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        tr.RandomResizedCrop(img_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

composed_transforms_ts = standard_transforms.Compose([
    tr.RandomResizedCrop(img_size),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])

voc_train = pascal.VOCSegmentation(base_dir= data_file, split='train', transform=composed_transforms_tr)
trainloader = DataLoader(voc_train, batch_size=opt.b, shuffle=True, num_workers=1)
model_dir = './pth/'

def find_new_file(dir):
    if os.path.exists(dir) is False:
        os.mkdir(model_dir)
        dir = model_dir

    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
                    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

if torch.cuda.is_available():
    frame_work = DeepLabv3(num_class)
    model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)
    model_id = 0
    if find_new_file(model_dir) is not None:
        model.load_state_dict(torch.load(find_new_file(model_dir)))
        print('load the model %s' % find_new_file(model_dir))
        model_id = re.findall(r'\d+', find_new_file(model_dir))
        model_id = int(model_id[0])
    model.cuda()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(base_lr, i_iter, epoches, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
writer = SummaryWriter()
model.train()
for i, data in enumerate(trainloader):
    lr = adjust_learning_rate(optimizer, i + model_id)

    images, labels, ids = data['image'], data['gt'], data['id']
    labels = labels.view(images.size()[0], img_size, img_size).long()

    if torch.cuda.is_available():
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

    outputs = model(images)
    loss = criterion(outputs, labels)
    if i % 2 == 0:
        print('epoch is %d, loss is %f' % (i + model_id, float(loss)))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if i % 500 == 0:
    #     grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    #     writer.add_image('image', grid_image)
    #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()),
    #                                3,
    #                                normalize=False,
    #                                range=(0, 255))
    #     writer.add_image('Predicted label', grid_image)
    #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()),
    #                                3,
    #                                normalize=False, range=(0, 255))
    #     writer.add_image('Groundtruth label', grid_image)

    writer.add_scalar('loss', loss, i + model_id)
    writer.add_scalar('learning_rate', lr, i + model_id)
    if i % 1000 == 0:
        torch.save(model.state_dict(), "./pth/%d.pth" % (model_id + i + 1))
    if i + model_id == epoches:
        print('train over')
        exit()

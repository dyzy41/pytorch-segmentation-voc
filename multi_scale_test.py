import os
import torch
import numpy as np
from skimage.transform import resize
from torchvision.transforms import Scale
import torch.nn as nn
from PIL import Image
from my_deeplab3_mit import DeepLabv3
from torch.autograd import Variable
from collections import OrderedDict
from ptsemseg import get_loader
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_list = [0]
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



def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
                    if not os.path.isdir(dir + fn) else 0)
    file = os.path.join(dir, file_lists[-1])
    return file

def readfile(data_path, img_name):

    img_path = data_path + img_name + '.jpg'
    # lbl_path = data_path + 'seg_mask' + img_name + '.png'

    img = Image.open(img_path).convert('RGB')
    # lbl = Image.open(lbl_path).convert('P')

    return img

def test(frame_work):

    data_loader = get_loader('pascal')
    root = './data/'
    data_path = "./data/img/"
    output_dir = './results_out'
    gt_dir = './data/gt/'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print('delete sucessed')
    else:
        os.mkdir(output_dir)

    if os.path.exists(output_dir+'_vis'):
        shutil.rmtree(output_dir+'_vis')
        os.mkdir(output_dir+'_vis')
        print('delete sucessed')
    else:
        os.mkdir(output_dir + '_vis')

    testdata = data_loader(root, split="val", is_transform=False, img_size=(512, 512))

    n_classes = testdata.n_classes
    eps = 1e-10

    # (TODO): Choose the scale according to dataset requirements
    # scales = [0.5, 0.75, 1.0, 1.25, 1.50, 2.0]
    scales = [1.0]
    base_size = min(testdata.img_size)
    crop_size = (512, 512)
    stride = [0, 0]
    stride[0] = int(np.ceil(float(crop_size[0]) * 2/3))
    stride[1] = int(np.ceil(float(crop_size[1]) * 2/3))
    size_transform_img = [Scale(int(base_size*i)) for i in scales]

    model_dir = './pth/'
    state_dict = torch.load(find_new_file(model_dir))
    #state_dict = torch.load('../weikai_train/pth/fcn-deconv-86.pth')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    print(find_new_file(model_dir))

    model = frame_work
    model.load_state_dict(new_state_dict)
    model.cuda()
    model = nn.DataParallel(model, device_ids=gpu_list)
    model.eval()

    soft = nn.Softmax2d()
    if torch.cuda.is_available():
        soft.cuda()

    for f_no,line in enumerate(testdata.files):
        imgr = readfile(data_path, line)
        origw, origh = imgr.size

        # Maintain final prediction array for each image
        pred = np.zeros((n_classes, origh, origw), dtype=np.float32)

        # Loop over all scales for single image
        for i in range(len(scales)):
            img = size_transform_img[i](imgr)
            imsw, imsh = img.size

            imwstart, imhstart = 0, 0
            imw, imh = imsw, imsh
            # Zero padding if any size if smaller than crop_size
            if imsw < crop_size[1] or imsh < crop_size[0]:
                padw, padh = max(crop_size[1] - imsw, 0), max(crop_size[0] - imsh, 0)
                imw += padw
                imh += padh
                im = Image.new(img.mode, (imw, imh), tuple(testdata.filler))
                im.paste(img, (int(padw / 2), int(padh / 2)))
                imwstart += int(padw / 2)
                imhstart += int(padh / 2)
                img = im

            # Now tile image - each of crop_size and loop over them
            h_grid = int(np.ceil(float(imh - crop_size[0]) / stride[0])) + 1
            w_grid = int(np.ceil(float(imw - crop_size[1]) / stride[1])) + 1

            # maintain prediction probability for each pixel
            datascale = torch.zeros(n_classes, imh, imw).cuda()
            countscale = torch.zeros(n_classes, imh, imw).cuda()
            for w in range(w_grid):
                for h in range(h_grid):
                    # crop portion from image - crop_size
                    x1, y1 = w * stride[1], h * stride[0]
                    x2, y2 = int(min(x1 + crop_size[1], imw)), int(min(y1 + crop_size[0], imh))
                    x1, y1 = x2 - crop_size[1], y2 - crop_size[0]
                    img_cropped = img.crop((x1, y1, x2, y2))

                    # Input image as well its flipped version
                    img1 = testdata.image_transform(img_cropped)
                    img2 = testdata.image_transform(img_cropped.transpose(Image.FLIP_LEFT_RIGHT))
                    images = torch.stack((img1, img2), dim=0)

                    if torch.cuda.is_available():
                        images = Variable(images.cuda(), volatile=True)
                    else:
                        images = Variable(images, volatile=True)

                    # Output prediction for image and its flip version
                    outputs = model(images)

                    # Sum prediction from image and its flip and then normalize
                    prob = outputs[0] + outputs[1][:, :, getattr(torch.arange(outputs.size(3)-1, -1, -1), 'cuda')().long()]
                    prob = soft(prob.view(-1, *prob.size()))

                    # Place the score in the proper position
                    datascale[:, y1:y2, x1:x2] += torch.squeeze(prob.data)
                    countscale[:, y1:y2, x1:x2] += 1
            # After looping over all tiles of image, normalize the scores and bilinear interpolation to orignal image size
            datascale /= (countscale + eps)
            datascale = datascale[:, imhstart:imhstart+imsh, imwstart:imwstart+imsw]
            datascale = datascale.cpu().numpy()
            datascale = np.transpose(datascale, (1, 2, 0))
            datascale = resize(datascale, (origh, origw), order=1, preserve_range=True, mode='symmetric', clip=False)
            datascale = np.transpose(datascale, (2, 0, 1))

            # Sum up all the scores for all scales
            pred += (datascale / (np.sum(datascale, axis=0) + eps))

        pred = pred / len(scales)
        pred = pred.argmax(0)
        print(max(pred.flatten()))
        sz = pred.shape[0]
        sx = pred.shape[1]
        # file_path = '/home/kawhi/Documents/pytorch/pytorch-segmentation-master_bk/test_multiscale/'
        output_img = np.zeros((sz, sx, 3), dtype=np.uint8)
        for i, color in enumerate(index2color):
            output_img[pred == i, :] = color
        output_img = Image.fromarray(output_img)
        output_img.save('{}/{}.png'.format(output_dir + '_vis', line), 'PNG')
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save('{}/{}.png'.format(output_dir, line), 'PNG')
    miou, pacc = evaluate(output_dir, gt_dir, 21)
    return miou, pacc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def evaluate(pred_dir, gt_dir, num_class):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    names = os.listdir(pred_dir)
    for name in names:
        pred = Image.open('{}/{}'.format(pred_dir, name))
        gt = Image.open('{}/{}'.format(gt_dir, name)).convert('P')
        pred = pred.resize(gt.size)
        pred = np.array(pred, dtype=np.int64)
        gt = np.array(gt, dtype=np.int64)
        gt[gt==255] = -1
        acc, pix = accuracy(pred, gt)
        intersection, union = intersectionAndUnion(pred, gt, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    return iou.mean(), acc_meter.average()

def main():
    index2color = labelcolormap(21)
    framework = DeepLabv3(21)
    miou, pacc = test(framework)
    print(miou, pacc)

if __name__ == '__main__':
    main()

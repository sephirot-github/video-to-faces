import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..utils.download import prep_weights_file
from ..utils import bbox

# Source 1: https://github.com/biubug6/Pytorch_Retinaface
# Source 2: https://github.com/barisbatuhan/FaceDetector
# Paper: https://arxiv.org/pdf/1905.00641.pdf


class FPN(nn.Module):
    def __init__(self, cins, cout, relu):
        super(FPN, self).__init__()
        self.outputs = nn.ModuleList([ConvUnit(cin, cout, 1, 1, 0, relu) for cin in cins])
        self.merges = nn.ModuleList([ConvUnit(cout, cout, 3, 1, 1, relu) for _ in cins[1:]])

    def forward(self, xs):
        n = len(xs)
        xs = [self.outputs[i](xs[i]) for i in range(n)]
        for i in range(0, n - 1)[::-1]:
            xs[i] += F.interpolate(xs[i + 1], size=xs[i].shape[2:], mode='nearest')
            xs[i] = self.merges[i](xs[i])
        return xs


class SSH(nn.Module):

    def __init__(self, cin, cout, relu):
        super(SSH, self).__init__()
        self.conv1 = ConvUnit(cin, cout//2, 3, 1, 1, relu_type=None)
        self.conv2 = ConvUnit(cin, cout//4, 3, 1, 1, relu_type=relu)
        self.conv3 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=None)
        self.conv4 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=relu)
        self.conv5 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=None)

    def forward(self, x):
        y1 = self.conv1(x)
        t = self.conv2(x)
        y2 = self.conv3(t)
        y3 = self.conv5(self.conv4(t))
        out = torch.cat([y1, y2, y3], dim=1)
        out = F.relu(out)
        return out


class Head(nn.Module):

    def __init__(self, cin, num_anchors, task_len):
        super(Head, self).__init__()
        self.task_len = task_len
        self.conv = nn.Conv2d(cin, num_anchors * task_len, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class RetinaFace(nn.Module):

    def __init__(self, backbone='mobilenet'):
        super(RetinaFace,self).__init__()
        if backbone == 'mobilenet':
            self.body = MobileNetV1(0.25, relu_type='lrelu_0.1', return_inter=[5, 11])
            cins, cout, relu = [64, 128, 256], 64, 'lrelu_0.1'
        elif backbone == 'resnet':
            # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
            import torchvision.models
            import torchvision.models._utils as _utils
            self.body = _utils.IntermediateLayerGetter(torchvision.models.resnet50(), {'layer2': 1, 'layer3': 2, 'layer4': 3})
            cins, cout, relu = [512, 1024, 2048], 256, 'plain'
        else:
            raise ValueError('Unknown backbone')

        self.feature_pyramid = FPN(cins, cout, relu)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in cins])
        num_anchors = 2
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in cins])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in cins])
        self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in cins])

    def forward(self, imgs):
        x = cv2.dnn.blobFromImages(imgs, mean=(104, 117, 123), swapRB=False)
        x = torch.from_numpy(x).to(next(self.parameters()).device)

        xs = self.body(x)

        from collections import OrderedDict
        if isinstance(xs, OrderedDict): xs = list(xs.values())
        
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        box_reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        classif = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
        ldm_reg = torch.cat([self.heads_ldmks[i](xs[i]) for i in range(len(xs))], dim=1)
        scores = F.softmax(classif, dim=-1)[:, :, 1]
     
        bases = [(8, [(16, 16), (32, 32)]), (16, [(64, 64), (128, 128)]), (32, [(256, 256), (512, 512)])]
        priors = get_priors(x.shape[2:], bases).to(x.device)
        boxes = decode_boxes(box_reg, priors)

        #k = torch.arange(boxes.shape[0]).repeat_interleave(boxes.shape[1])
        #b, s = boxes.reshape(-1, 4), scores.flatten()
        #idx = s > 0.02
        #k, b, s = k[idx], b[idx], s[idx]
        #keep = bbox.nms_batch(b, s, k, 0.4)
        #k, b, s = k[keep], b[keep], s[keep]
        #r = torch.hstack([b, s.unsqueeze(1)])
        #l = torch.split(r, list(torch.bincount(k)))
        #return [a.detach().cpu().numpy() for a in l]

        l = []
        for i in range(len(imgs)):
            b, s = boxes[i], scores[i]
            idx = s > 0.02
            b, s = b[idx], s[idx]
            r = torch.hstack([b, s.unsqueeze(1)]).detach().cpu().numpy()
            keep = bbox.nms(r, 0.4)
            l.append(r[keep])
        return l


def get_priors(img_size, bases):
    """For every (stride, anchors) pair in ``bases`` list, walk through every stride-sized
    square patch of ``img_size`` canvas left-right, top-bottom and return anchors-sized boxes
    drawn around each patch's center in a form of (center_x, center_y, width, height).
    
    Example: get_priors((90, 64), [(32, [(8, 4), (25, 15)])])
    Output: shape = (12, 4)
    [[16, 16, 8, 4], [16, 16, 25, 15], [48, 16, 8, 4], [48, 16, 25, 15],
     [16, 48, 8, 4], [16, 48, 25, 15], [48, 48, 8, 4], [48, 48, 25, 15],
     [16, 80, 8, 4], [16, 80, 25, 15], [48, 80, 8, 4], [48, 80, 25, 15]]
    """
    p = []
    h, w = img_size
    for stride, anchors in bases:
        nx = math.ceil(w / stride)
        ny = math.ceil(h / stride)
        xs = torch.arange(nx) * stride + stride // 2
        ys = torch.arange(ny) * stride + stride // 2
        c = torch.dstack(torch.meshgrid(xs, ys, indexing='xy')).reshape(-1, 2)
        # could replace line above by "torch.cartesian_prod(xs, ys)" but that'd be for indexing='ij'
        c = c.repeat_interleave(len(anchors), dim=0)
        s = torch.Tensor(anchors).repeat(nx*ny, 1)
        p.append(torch.hstack([c, s]))
    return torch.cat(p)


def decode_boxes(pred, priors):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    xys = priors[:, 2:] * 0.1 * pred[:, :, :2] + priors[:, :2]
    whs = priors[:, 2:] * torch.exp(0.2 * pred[:, :, 2:])
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


class RetinaFaceDetector():

    def __init__(self, device, backbone='mobilenet'):
        if backbone == 'mobilenet':
            print('Initializing RetinaFace model (with MobileNetV1 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1', 'mobilenet0.25_Final.pth', gdrive=True)
        elif backbone == 'resnet':
            print('Initializing RetinaFace model (with ResNet50 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW', 'Resnet50_Final.pth', gdrive=True)
        
        self.model = RetinaFace(backbone).to(device)
        #for w in self.model.state_dict(): print(w, '\t', self.model.state_dict()[w].shape)
        wd_src = torch.load(wf, map_location=torch.device(device))
        wd_dst = {}
        names = list(wd_src)
        for i, w in enumerate(list(self.model.state_dict())):
            wd_dst[w] = wd_src[names[i]]
        self.model.load_state_dict(wd_dst)
        self.model.eval()

    def __call__(self, imgs):
        with torch.no_grad():
            res = self.model(imgs)
        return res
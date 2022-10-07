import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.ops import batched_nms

from itertools import product as product
from math import ceil

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..utils.download import prep_weights_file

# https://github.com/biubug6/Pytorch_Retinaface
# https://github.com/barisbatuhan/FaceDetector
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

    def __init__(self, bbone='mobilenet'):
        super(RetinaFace,self).__init__()
        if bbone == 'mobilenet':
            self.body = MobileNetV1(0.25, relu_type='lrelu_0.1', return_inter=[5, 11])
            cins, cout, relu = [64, 128, 256], 64, 'lrelu_0.1'
        else:
            #import torchvision.models as models
            #backbone = models.resnet50()
            #self.module.body = _utils.IntermediateLayerGetter(backbone, {'layer2': 1, 'layer3': 2, 'layer4': 3})
            cins, cout, relu = [512, 1024, 2048], 256, 'plain'
        
        self.feature_pyramid = FPN(cins, cout, relu)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in cins])
        num_anchors = 2
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in cins])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in cins])
        self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in cins])

    def forward(self, x):
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        box_reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        classif = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
        ldm_reg = torch.cat([self.heads_ldmks[i](xs[i]) for i in range(len(xs))], dim=1)
        return (box_reg, F.softmax(classif, dim=-1), ldm_reg)


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class RetinaFaceDetector():

    def __init__(self, device, bbone='mobilenet'):
        if bbone == 'mobilenet':
            print('Initializing RetinaFace model (with MobileNetV1 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1', 'mobilenet0.25_Final.pth', gdrive=True)
        else:
            print('Initializing RetinaFace model (with ResNet50 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW', 'Resnet50_Final.pth', gdrive=True)
        
        self.model = RetinaFace(bbone).to(device)
        #for w in self.model.state_dict(): print(w, '\t', self.model.state_dict()[w].shape)
        wd_src = torch.load(wf, map_location=torch.device(device))
        wd_dst = {}
        names = list(wd_src)
        for i, w in enumerate(list(self.model.state_dict())):
            wd_dst[w] = wd_src[names[i]]
        self.model.load_state_dict(wd_dst)
        self.model.eval()
    
    def __call__(self, img):
        img = np.float32(img)
        h, w, _ = img.shape
        img -= (104, 117, 123)
        scl = torch.Tensor([w, h, w, h])
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        loc, conf, landms = self.model(img)
    
        priorbox = PriorBox(image_size=(h, w))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
        boxes = boxes * scl
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        dets[:, :4] = np.floor(dets[:, :4])
        return dets


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
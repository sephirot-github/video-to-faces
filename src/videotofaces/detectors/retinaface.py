# https://github.com/biubug6/Pytorch_Retinaface
# https://github.com/barisbatuhan/FaceDetector

import cv2
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F

import torch
from itertools import product as product
import numpy as np
from math import ceil

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..utils.download import prep_weights_file


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class SSH(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class ClassHead(nn.Module):

    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous() 
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)





class FPN(nn.Module):
    def __init__(self, cins, cout, relu):
        super(FPN,self).__init__()
        self.output1 = ConvUnit(cins[0], cout, 1, 1, 0, relu)
        self.output2 = ConvUnit(cins[1], cout, 1, 1, 0, relu)
        self.output3 = ConvUnit(cins[2], cout, 1, 1, 0, relu)
        self.merge1 = ConvUnit(cout, cout, 3, 1, 1, relu)
        self.merge2 = ConvUnit(cout, cout, 3, 1, 1, relu)

    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


# https://arxiv.org/pdf/1905.00641.pdf

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
        
        out_channels = cout
        self.fpn = FPN(cins, cout, relu)
        self.ssh1 = SSH(cout, cout)
        self.ssh2 = SSH(cout, cout)
        self.ssh3 = SSH(cout, cout)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cout)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cout)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cout)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, x):
        xs = self.body(x)
        fpn = self.fpn(xs)
        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        return (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)


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


class RetinaFaceDetector():

    def __init__(self, device, bbone='mobilenet'):
        if bbone == 'mobilenet':
            print('Initializing RetinaFace model (with MobileNetV1 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1', 'mobilenet0.25_Final.pth', gdrive=True)
        else:
            print('Initializing RetinaFace model (with ResNet50 backbone) for face detection')
            wf = prep_weights_file('https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW', 'Resnet50_Final.pth', gdrive=True)
        
        self.model = RetinaFace(bbone).to(device)
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import prep, post
from .components.fpn import FPN
from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from ..utils.weights import load_weights


class Head(nn.Module):

    def __init__(self, c, num_anchors):
        super().__init__()
        self.conv = ConvUnit(c, c, 3, 1, 1, relu_type='plain', bn=None)
        self.cls = nn.Conv2d(c, num_anchors, 1, 1)
        self.reg = nn.Conv2d(c, num_anchors * 4, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        a = self.reg(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)
        b = self.cls(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 1)
        return a, b


class FasterRCNN(nn.Module):

    links = {
        '1': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    }

    def __init__(self, pretrained=True, device='cpu'):
        super().__init__()
        backbone = ResNet50(bn_eps=0.0)
        cins = [256, 512, 1024, 2048]
        cout = 256
        #anchors_per_level = 9
        #self.num_classes = 91
        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu=None, bn=None, pool=True, smoothP5=True)
        self.head = Head(cout, 3)
        
        self.to(device)
        if pretrained:
            load_weights(self, self.links['1'], 'resnet50_coco', device, self.conversion, add_num_batches=True)
    
    def conversion(self, wd):
        for n in list(wd)[-8:]:
            wd.pop(n)
        return wd

    def forward(self, imgs, post_impl='vectorized'):
        dv = next(self.parameters()).device
        ts = prep.normalize(imgs, dv, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
        ts, sz_orig, sz_used = prep.resize(ts, resize_min=800, resize_max=1333)
        x = prep.batch(ts, mult=32)
        
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        reg, log = map(list, zip(*[self.head(lvl) for lvl in xs]))
        lsz = [lvl.shape[1] for lvl in log]
        reg = torch.cat(reg, axis=1)
        obj = torch.cat(log, axis=1).sigmoid_()

        priors = post.get_priors(x.shape[2:], self.get_bases(), dv, loc='corner', patches='fit')
        b, _, _ = post.get_results(
            reg, obj, priors, score_thr=0.00, lvtop=1000, lvsizes=lsz,
            decode=(1, 1, math.log(1000 / 16)),
            clamp=True, min_size=1e-3, sizes_used=sz_used,
            iou_thr=0.7, imtop=1000, nms_per_level=True,
            implementation=post_impl)
        return b

    def get_bases(self):
        strides = [4, 8, 16, 32, 64]
        anchors = post.make_anchors_rounded([32, 64, 128, 256, 512], [1], [2, 1, 0.5])
        return list(zip(strides, anchors))
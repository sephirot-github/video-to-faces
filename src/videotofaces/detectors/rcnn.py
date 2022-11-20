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


class TwoMLPHead(nn.Module):
    def __init__(self, cin, cmid):
        super().__init__()
        self.fc6 = nn.Linear(cin, cmid)
        self.fc7 = nn.Linear(cmid, cmid)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self, cin, num_classes):
        super().__init__()
        self.cls = nn.Linear(cin, num_classes)
        self.reg = nn.Linear(cin, num_classes * 4)
    def forward(self, x):
        a = self.reg(x)
        b = self.cls(x)
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
        #anchors_per_level = 3
        self.num_classes = 91
        self.strides = [4, 8, 16, 32, 64]
        anchors = post.make_anchors_rounded([32, 64, 128, 256, 512], [1], [2, 1, 0.5])
        self.bases = list(zip(self.strides, anchors))
        self.decode_set = (1, 1, math.log(1000 / 16))

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu=None, bn=None, pool=True, smoothP5=True)
        self.head = Head(cout, 3)
        self.roi_head1 = TwoMLPHead(cout * 7**2, 1024)
        self.roi_head2 = FastRCNNPredictor(1024, self.num_classes)
        
        self.to(device)
        if pretrained:
            load_weights(self, self.links['1'], 'resnet50_coco', device, add_num_batches=True)

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
        
        # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py#L241
        priors = post.get_priors(x.shape[2:], self.bases, dv, loc='corner', patches='fit')
        proposals, _, _ = post.get_results(
            reg, obj, priors, score_thr=0.00, lvtop=1000, lvsizes=lsz, decode=self.decode_set,
            clamp=True, min_size=1e-3, sizes_used=sz_used,
            iou_thr=0.7, imtop=1000, nms_per_level=True,
            implementation=post_impl)
        
        x = roi_align_fpn(proposals, xs[:-1], self.strides[:-1])
        x = self.roi_head1(x)
        reg, log = self.roi_head2(x)

        lbl = torch.arange(self.num_classes, device=dv).view(1, -1).expand_as(log)[:, 1:]
        rrr = reg.reshape(reg.shape[0], -1, 4)[:, 1:, :]
        scr = F.softmax(log, dim=-1)[:, 1:]
        # torch.Size([2000, 90, 4]) torch.Size([2000, 90])

        #imlen = [len(p) for p in proposals]



        # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L668
        #b, s, c = post.get_results(
        #    reg, log, proposals, score_thr=0.05, decode=self.decode_set,
        #    clamp=True, min_size=1e-2, sizes_used=sz_used
        #    iou_thr=0.5, imtop=100,
        #    implementation='loop'
        #)

        return reg, log


def assign_fpn_levels(boxes, strides):
    """FPN Paper, Eq.1 https://arxiv.org/pdf/1612.03144.pdf"""
    kmin = math.log2(strides[0])
    kmax = math.log2(strides[-1])
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    k = 4 + torch.log2(torch.sqrt(ws * hs) / 224)
    k = torch.clamp(k, min=kmin, max=kmax)
    mapidx = (k - kmin).to(torch.int64)
    return mapidx


def roi_align_fpn(boxes_list, fmaps, strides):
    # https://arxiv.org/pdf/1703.06870.pdf
    # https://github.com/pytorch/vision/issues/4935
    # https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
    # https://chao-ji.github.io/jekyll/update/2018/07/20/ROIAlign.html
    import torchvision.ops
    imlen = [len(p) for p in boxes_list]
    boxes = torch.cat(boxes_list)
    mapidx = assign_fpn_levels(boxes, strides)
    imidx = torch.arange(len(imlen)).repeat_interleave(torch.tensor(imlen)).to(boxes.device)
    imboxes = torch.hstack([imidx.unsqueeze(-1), boxes])
    roi_maps = torch.zeros((len(mapidx), fmaps[0].shape[1], 7, 7))
    for level in range(len(strides)):
        scale = 1 / strides[level]
        idx = torch.nonzero(mapidx == level).squeeze()
        roi = torchvision.ops.roi_align(fmaps[level], imboxes[idx], (7, 7), scale, 2, False)
        roi_maps[idx] = roi.to(roi_maps.dtype)
    return roi_maps
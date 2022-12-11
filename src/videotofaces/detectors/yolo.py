import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from .operations import prep, post
from ..backbones.basic import ConvUnit
from ..utils import prep_weights_file
from ..utils.weights import load_weights

# adapted from: https://github.com/open-mmlab/mmdetection
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/yolo_head.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py


def conv_unit(cin, cout, k, s=1):
    return ConvUnit(cin, cout, k, s, p=(k-1)//2, activ='lrelu_0.1')


class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_unit(c, c // 2, k=1)
        self.conv2 = conv_unit(c // 2, c, k=3)
  
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x


class Darknet53(nn.Module):

    def __init__(self, ):
        super().__init__()
        L, C = [1, 2, 8, 8, 4], [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
        self.conv1 = conv_unit(3, 32, k=3)
        for i in range(len(L)):
            block = nn.Sequential()
            block.add_module('conv', conv_unit(C[i][0], C[i][1], k=3, s=2))
            for j in range(L[i]):
                block.add_module(f'res{j}', ResBlock(C[i][1]))
            self.add_module(f'conv_res_block{i + 1}', block)
  
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv_res_block1(x)
        x2 = self.conv_res_block2(x1)
        x3 = self.conv_res_block3(x2)
        x4 = self.conv_res_block4(x3)
        x5 = self.conv_res_block5(x4)
        return (x3, x4, x5)
        

class DetectionBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = conv_unit(cin, cout, k=1)
        self.conv2 = conv_unit(cout, cout*2, k=3)
        self.conv3 = conv_unit(cout*2, cout, k=1)
        self.conv4 = conv_unit(cout, cout*2, k=3)
        self.conv5 = conv_unit(cout*2, cout, k=1)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
        

class YOLOv3Neck(nn.Module):

    def __init__(self):
        super().__init__()
        self.detect1 = DetectionBlock(1024, 512)
        self.conv1 = conv_unit(512, 256, k=1)
        self.detect2 = DetectionBlock(768, 256)
        self.conv2 = conv_unit(256, 128, k=1)
        self.detect3 = DetectionBlock(384, 128)
  
    def forward(self, x):
        (x1, x2, x3) = x
        y3 = self.detect1(x3)
        t = self.conv1(y3)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x2), 1)
        y2 = self.detect2(t)
        t = self.conv2(y2)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x1), 1)
        y1 = self.detect3(t)
        return (y3, y2, y1)
        

class YOLOv3Head(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.convs_bridge = nn.ModuleList([
            conv_unit(512, 1024, k=3),
            conv_unit(256, 512, k=3),
            conv_unit(128, 256, k=3)
        ])
        cout = (num_classes + 5) * 3
        self.convs_pred = nn.ModuleList([
            nn.Conv2d(1024, cout, 1),
            nn.Conv2d(512, cout, 1),
            nn.Conv2d(256, cout, 1)
        ])
  
    def forward(self, x):
        maps = []
        for i in range(3):
            y = x[i]
            y = self.convs_bridge[i](y)
            y = self.convs_pred[i](y)
            maps.append(y)
        return tuple(maps)


class YOLOv3(nn.Module):

    # https://becominghuman.ai/understanding-anchors-backbone-of-object-detection-using-yolo-54962f00fbbb

    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/'
    links = {
        'anime': 'https://github.com/hysts/anime-face-detector/'\
                 'releases/download/v0.0.1/mmdet_anime-face_yolov3.pth',
        'wider': '1pjg1_IeAuzgRzZiY92r71uzd_amfcegu',
        'coco': mmhub + 'yolov3_d53_mstrain-608_273e_coco/'\
                        'yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth',
        'coco_mobile_416': mmhub + 'yolov3_mobilenetv2_mstrain-416_300e_coco/'\
                           'yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
    }

    def __init__(self, pretrained=None, device='cpu', canvas_size=608, num_classes=1): # 416, 320
        super().__init__()
        self.canvas_size = canvas_size
        self.num_classes = num_classes
        self.bases = [
            (32, [(116, 90), (156, 198), (373, 326)]),
            (16, [(30, 61), (62, 45), (59, 119)]),
            (8,  [(10, 13), (16, 30), (33, 23)])
        ]
        self.backbone = Darknet53()
        self.neck = YOLOv3Neck()
        self.bbox_head = YOLOv3Head(num_classes)
        if pretrained:
            sub = None if pretrained == 'wider' else 'state_dict'
            load_weights(self, self.links[pretrained], pretrained, device, sub=sub)
  
    def forward(self, imgs):
        dv = next(self.parameters()).device
        x, szo, sz = prep.full(imgs, dv, self.canvas_size, 'cv2', norm=None)
        xs = self.backbone(x)
        xs = self.neck(xs)
        xs = self.bbox_head(xs)
        priors = post.get_priors(x.shape[-2:], self.bases, 'cpu', 'center')
        b, s, c = self.postprocess(xs, priors, self.num_classes)
        b = post.scale_back(b, szo, sz)
        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
        return b, s, c

    def postprocess(self, pred_maps, priors, num_classes):
        maps = [m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, num_classes + 5) for m in pred_maps]
        map_sizes = [m.shape[1] for m in maps]
        maps = torch.cat(maps, dim=1)
        reg = maps[..., :4]
        obj = maps[..., 4].sigmoid()
        scr = maps[..., 5:].sigmoid()
      
        n, dim, num_classes = scr.shape
        obj = obj.unsqueeze(-1).expand(-1, -1, num_classes).flatten()
        reg, scr = reg.reshape(-1, 4), scr.flatten()
        fidx = torch.nonzero((obj >= 0.005) * (scr > 0.05)).squeeze()
        s = scr[fidx] * obj[fidx]
        c = fidx % num_classes
        idx = torch.div(fidx, num_classes, rounding_mode='floor')
        imidx = idx.div(dim, rounding_mode='floor')

        strides = [bs[0] for bs in self.bases]
        lvidx = post.get_lvidx(idx % dim, map_sizes)
        stidx = torch.tensor(strides)[lvidx].unsqueeze(-1)
        b = post.decode_boxes(reg[idx], priors[idx % dim], mode='yolo', strides=stidx)

        res = []
        for i in range(n):
            bi, si, ci = [x[imidx == i] for x in [b, s, c]]
            keep = torchvision.ops.batched_nms(bi, si, ci, 0.45)[:100]
            res.append((bi[keep], si[keep], ci[keep]))
        b, s, c = map(list, zip(*res))
        #k = imidx * 1000 + c
        #keep = torchvision.ops.batched_nms(b, s, k, 0.45)
        #b, s, c, imidx = [x[keep] for x in [b, s, c, imidx]]
        #b, s, c = [[x[imidx == i] for i in range(n)] for x in [b, s, c]]
        return b, s, c
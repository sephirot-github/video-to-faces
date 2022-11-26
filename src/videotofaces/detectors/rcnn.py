import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops    

from .operations import prep, post
from .components.fpn import FPN
from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from ..utils.weights import load_weights


class RegionProposalNetwork(nn.Module):

    def __init__(self, c, num_anchors, decode_settings):
        super().__init__()
        self.conv = ConvUnit(c, c, 3, 1, 1, relu_type='plain', bn=None)
        self.log = nn.Conv2d(c, num_anchors, 1, 1)
        self.reg = nn.Conv2d(c, num_anchors * 4, 1, 1)
        self.dec = decode_settings

    def head(self, x, priors, lvtop):
        n = x.shape[0]
        x = self.conv(x)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(n, -1, 4)
        log = self.log(x).permute(0, 2, 3, 1).reshape(n, -1, 1)
        log, top = log.topk(min(lvtop, log.shape[1]), dim=1)
        reg = reg.gather(1, top.expand(-1, -1, 4))
        pri = priors.expand(n, -1, -1).gather(1, top.expand(-1, -1, 4))
        boxes = post.decode_boxes(reg, pri, settings=self.dec)
        return boxes, log, log.shape[1]

    def forward(self, fmaps, priors, imsizes, lvtop, imtop, score_thr, iou_thr, min_size):
        tups = [self.head(x, p, lvtop) for x, p in zip(fmaps, priors)]
        boxes, logits, lvlen = map(list, zip(*tups))
        boxes = torch.cat(boxes, axis=1)
        obj = torch.cat(logits, axis=1).sigmoid()

        n, dim = boxes.shape[:2]
        boxes, obj = boxes.reshape(-1, 4), obj.flatten()
        idx = torch.nonzero(obj >= score_thr).squeeze()
        boxes, obj = boxes[idx], obj[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        #boxes.clamp_(min=torch.tensor(0), max=imsizes[imidx, :])
        boxes = post.clamp_to_canvas_vect(boxes, imsizes, imidx)
        boxes, obj, idx, imidx = post.remove_small(boxes, min_size, obj, idx, imidx)
        groups = imidx * 10 + post.get_lvidx(idx % dim, lvlen)
        keep = torchvision.ops.batched_nms(boxes, obj, groups, iou_thr)
        keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
        return boxes[keep], imidx[keep]


class RoIProcessingNetwork(nn.Module):

    def __init__(self, cin, cmid, num_classes, decode_settings):
        super().__init__()
        self.fc6 = nn.Linear(cin, cmid)
        self.fc7 = nn.Linear(cmid, cmid)
        self.cls = nn.Linear(cmid, num_classes)
        self.reg = nn.Linear(cmid, num_classes * 4)
        self.dec = decode_settings

    def heads(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        a = self.reg(x)
        b = self.cls(x)
        return a, b
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes, score_thr, iou_thr, imtop, min_size):
        roi_maps = post.roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides)
        reg, log = self.heads(roi_maps)
       
        reg = reg.reshape(reg.shape[0], -1, 4)[:, 1:, :]
        scr = F.softmax(log, dim=-1)[:, 1:]
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)[:, 1:]

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > score_thr).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = post.convert_to_cwh(proposals)
        boxes = post.decode_boxes(reg, proposals, settings=self.dec)
        boxes = post.clamp_to_canvas_vect(boxes, imsizes, imidx)
        boxes, scr, cls, imidx = post.remove_small(boxes, min_size, scr, cls, imidx)
        
        res = []
        for i in range(n):
            bi, si, ci = [x[imidx == i] for x in [boxes, scr, cls]]
            keep = torchvision.ops.batched_nms(bi, si, ci, iou_thr)[:imtop]
            res.append((bi[keep], si[keep], ci[keep]))
        return map(list, zip(*res))
        #groups = imidx * 1000 + cls
        #keep = torchvision.ops.batched_nms(boxes, scr, groups, iou_thr)
        #keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
        #boxes, scr, cls, imidx = [x[keep] for x in [boxes, scr, cls, imidx]]
        #boxes, scr, cls = [[x[imidx == i] for i in range(n)] for x in [boxes, scr, cls]]
        #return boxes, scr, cls
        

class FasterRCNN(nn.Module):

    links = {
        '1': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        '2': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
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
        self.rpn = RegionProposalNetwork(cout, 3, self.decode_set)
        self.roi = RoIProcessingNetwork(cout * 7**2, 1024, self.num_classes, (0.1, 0.2, math.log(1000 / 16)))
                
        self.to(device)
        if pretrained:
            load_weights(self, self.links['1'], 'resnet50_coco', device, add_num_batches=True)

    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = prep.normalize(imgs, dv, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
        ts, sz_orig, sz_used = prep.resize(ts, resize_min=800, resize_max=1333)
        x = prep.batch(ts, mult=32)
        
        priors = post.get_priors(x.shape[2:], self.bases, dv, 'corner', 'fit', concat=False)
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        proposals, imidx = self.rpn(xs, priors, sz_used, lvtop=1000, imtop=1000,
                                    score_thr=0.0, iou_thr=0.7, min_size=1e-3)
        boxes, scores, classes = self.roi(proposals, imidx, xs[:-1], self.strides[:-1], sz_used,
                                          score_thr=0.05, iou_thr=0.5, imtop=100, min_size=1e-2)
        
        scalebacks = torch.tensor(sz_orig) / torch.tensor(sz_used)
        scalebacks = scalebacks.flip(1).repeat(1, 2)
        boxes = [boxes[i] * scalebacks[i] for i in range(len(imgs))]

        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [boxes, scores, classes]]
        return b, s, c
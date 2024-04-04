import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, convert_to_cwh, decode_boxes, remove_small, scale_boxes
from .operations.post import get_lvidx, final_nms
from .operations.prep import preprocess
from .operations.roi import roi_align_multilevel
from ..utils.weights import load_weights


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, cins, cout, activ=None, bn=None):
        super().__init__()
        self.conv_laterals = nn.ModuleList([ConvUnit(cin, cout, 1, 1, 0, activ, bn) for cin in cins])
        self.conv_smooths = nn.ModuleList([ConvUnit(cout, cout, 3, 1, 1, activ, bn) for _ in range(len(cins))])

    def forward(self, C):
        n = len(C)
        P = [self.conv_laterals[i](C[i]) for i in range(n)]     
        for i in range(n - 1)[::-1]:
            P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
        for i in range(len(self.conv_smooths)):
            P[i] = self.conv_smooths[i](P[i])
        P.append(F.max_pool2d(P[-1], 1, stride=2))
        return P


class RegionProposalNetwork(nn.Module):

    def __init__(self, c, num_anchors):
        super().__init__()
        self.conv = ConvUnit(c, c, 3, 1, 1, 'relu', None)
        self.log = nn.Conv2d(c, num_anchors, 1, 1)
        self.reg = nn.Conv2d(c, num_anchors * 4, 1, 1)

    def head(self, x):
        n = x.shape[0]
        x = self.conv(x)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(n, -1, 4)
        log = self.log(x).permute(0, 2, 3, 1).reshape(n, -1, 1)
        return reg, log

    def filt_dec(self, regs, logs, priors, lvtop):
        res = []
        n = regs[0].shape[0]
        for reg, log, p in zip(regs, logs, priors):
            log, top = log.topk(min(lvtop, log.shape[1]), dim=1)
            reg = reg.gather(1, top.expand(-1, -1, 4))
            pri = p.expand(n, -1, -1).gather(1, top.expand(-1, -1, 4))
            boxes = decode_boxes(reg, pri, 1, 1)
            res.append((boxes, log, log.shape[1]))
        return map(list, zip(*res))

    def forward(self, fmaps, priors, imsizes):
        tuples = [self.head(x) for x in fmaps]
        regs, logs = map(list, zip(*tuples))
        
        dregs = [x.detach() for x in regs]
        dlogs = [x.detach() for x in logs]
        boxes, logits, lvlen = self.filt_dec(dregs, dlogs, priors, 1000)
        boxes = torch.cat(boxes, axis=1)
        obj = torch.cat(logits, axis=1).sigmoid()

        n, dim = boxes.shape[:2]
        boxes, obj = boxes.reshape(-1, 4), obj.flatten()
        idx = torch.nonzero(obj >= 0).squeeze()
        boxes, obj = boxes[idx], obj[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, obj, idx, imidx = remove_small(boxes, 0, obj, idx, imidx)
        groups = imidx * 10 + get_lvidx(idx % dim, lvlen)
        keep = torchvision.ops.batched_nms(boxes, obj, groups, 0.7)
        keep = torch.cat([keep[imidx[keep] == i][:1000] for i in range(n)])
        boxes, imidx = boxes[keep], imidx[keep]
        return boxes, imidx


class RoIProcessingNetwork(nn.Module):

    def __init__(self, c, roi_map_size, clin, num_classes):
        super().__init__()
        c1 = c * roi_map_size ** 2
        self.fc = nn.ModuleList([nn.Linear(c1 if i == 0 else clin, clin) for i in range(2)])
        self.cls = nn.Linear(clin, 1 + num_classes)
        self.reg = nn.Linear(clin, num_classes * 4)

    def heads(self, x):
        x = x.flatten(start_dim=1)
        if self.fc:
            for mlp in self.fc:
                x = F.relu(mlp(x))
        a = self.reg(x)
        b = self.cls(x)
        return a, b
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes):
        roi_maps = roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides, (0, True))
        reg, log = self.heads(roi_maps)
        reg = reg.reshape(reg.shape[0], -1, 4)
       
        scr = F.softmax(log, dim=-1)[:, :-1]
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)[:, :-1]

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > 0.05).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = convert_to_cwh(proposals, in_place=True)
        boxes = decode_boxes(reg, proposals, 0.1, 0.2)
        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, scr, cls, imidx = remove_small(boxes, 0, scr, cls, imidx)
        b, s, c = final_nms(boxes, scr, cls, imidx, n, 0.5, 100)
        return b, s, c


class FasterRCNN(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.body = ResNet50(bn=1e-5, num_freeze=2)
        cins, self.strides = [256, 512, 1024, 2048], [4, 8, 16, 32, 64]
        anchors = make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5])
        self.bases = list(zip(self.strides, anchors))
        self.fpn = FeaturePyramidNetwork(cins, 256)
        self.rpn = RegionProposalNetwork(256, len(anchors[0]))
        self.roi = RoIProcessingNetwork(256, 7, 1024, num_classes=1)
        self.to(device)

    def forward(self, imgs):
        dv = next(self.parameters()).device
        x, sz_orig, sz_used = preprocess(imgs, dv, (800, 1333), 'cv2')
        priors = get_priors(x.shape[2:], self.bases, dv, 'corner', 'as_is', concat=False)
        xs = self.body(x)
        xs = self.fpn(xs)
        p, imidx = self.rpn(xs, priors, sz_used)
        b, s, c = self.roi(p, imidx, xs[:-1], self.strides[:-1], sz_used)
        b = scale_boxes(b, sz_orig, sz_used)
        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
        return b, s, c


class AnimeFRCNN():

    link = 'https://github.com/hysts/anime-face-detector/'\
           'releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth'

    def wconv(self, wd):
        # in MMDet weights for RoI head, representation FC and final reg/log FCs are switched over
        wl = list(wd.items())
        els = [wl.pop(-1) for _ in range(8)][::-1] # last 8 entries
        for el in els[4:] + els[:4]:
            wl.append(el)
        wd = dict(wl)
        return wd

    def __init__(self, device=None):
        print('Initializing FasterRCNN model for anime face detection')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FasterRCNN(device)
        load_weights(self.model, self.link, 'frcnn_anime', self.wconv, 'state_dict')
        self.model.eval()

    def __call__(self, imgs):
        with torch.inference_mode():
            return self.model(imgs)
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from .components.fpn import FeaturePyramidNetwork
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, convert_to_cwh, decode_boxes, remove_small, scale_boxes
from .operations.post import get_lvidx, final_nms
from .operations.prep import preprocess, prep_targets
from .operations.roi import roi_align_multilevel
from ..utils.weights import load_weights


class RegionProposalNetwork(nn.Module):

    def __init__(self, c, num_anchors, conv_depth):
        super().__init__()
        self.conv = nn.Sequential(*[ConvUnit(c, c, 3, 1, 1, 'relu', None) for _ in range(conv_depth)])
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
            boxes = decode_boxes(reg, pri, mults=(1, 1), clamp=True)
            res.append((boxes, log, log.shape[1]))
        return map(list, zip(*res))

    def forward(self, fmaps, priors, imsizes, settings, gtboxes=None):
        score_thr, iou_thr, imtop_infer, min_size, lvtop_infer, imtop_train, lvtop_train = settings
        lvtop = lvtop_infer if not self.training else lvtop_train
        imtop = imtop_infer if not self.training else imtop_train

        tuples = [self.head(x) for x in fmaps]
        regs, logs = map(list, zip(*tuples))
        
        dregs = [x.detach() for x in regs]
        dlogs = [x.detach() for x in logs]
        boxes, logits, lvlen = self.filt_dec(dregs, dlogs, priors, lvtop)
        boxes = torch.cat(boxes, axis=1)
        obj = torch.cat(logits, axis=1).sigmoid()

        n, dim = boxes.shape[:2]
        boxes, obj = boxes.reshape(-1, 4), obj.flatten()
        idx = torch.nonzero(obj >= score_thr).squeeze()
        boxes, obj = boxes[idx], obj[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, obj, idx, imidx = remove_small(boxes, min_size, obj, idx, imidx)
        groups = imidx * 10 + get_lvidx(idx % dim, lvlen)
        keep = torchvision.ops.batched_nms(boxes, obj, groups, iou_thr)
        keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
        boxes, imidx = boxes[keep], imidx[keep]
        return boxes, imidx


class RoIProcessingNetwork(nn.Module):

    def __init__(self, c, roi_map_size, clin, roi_convdepth, roi_mlp_depth,
                 num_classes, bckg_class_first, roialign_settings):
        super().__init__()
        self.conv = nn.ModuleList([ConvUnit(c, c, 3, 1, 1, 'relu') for _ in range(roi_convdepth)])
        c1 = c * roi_map_size ** 2
        self.fc = nn.ModuleList([nn.Linear(c1 if i == 0 else clin, clin) for i in range(roi_mlp_depth)])
        self.cls = nn.Linear(clin, 1 + num_classes)
        self.reg = nn.Linear(clin, num_classes * 4)
        self.ralign_set = roialign_settings
        self.bckg_first = bckg_class_first

    def heads(self, x):
        if self.conv:
            for layer in self.conv:
                x = layer(x)
        x = x.flatten(start_dim=1)
        if self.fc:
            for mlp in self.fc:
                x = F.relu(mlp(x))
        a = self.reg(x)
        b = self.cls(x)
        return a, b
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes, settings, gtb=None, gtl=None):
        roi_maps = roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides, self.ralign_set)
        reg, log = self.heads(roi_maps)
        reg = reg.reshape(reg.shape[0], -1, 4)
       
        scr = F.softmax(log, dim=-1)
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)
        scr = scr[:, :-1] if not self.bckg_first else scr[:, 1:]
        cls = cls[:, :-1] if not self.bckg_first else cls[:, 1:]

        score_thr, iou_thr, imtop, min_size = settings

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > score_thr).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = convert_to_cwh(proposals, in_place=True)
        boxes = decode_boxes(reg, proposals, mults=(0.1, 0.2), clamp=True)
        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, scr, cls, imidx = remove_small(boxes, min_size, scr, cls, imidx)
        b, s, c = final_nms(boxes, scr, cls, imidx, n, iou_thr, imtop)
        return b, s, c


class FasterRCNN(nn.Module):

    def __init__(self, device='cpu',
                 # 1) architectural settings for FPN and RPN
                 fpn_batchnorm=None, rpn_convdepth=1, roi_convdepth=0, roi_mlp_depth=2,
                 # 2) classes info and ROI algo details
                 num_classes=80, bckg_class_first=False, roialign_settings=(0, True),
                 # 3) settings for preprocessing, anchors and priors
                 prep_resize='cv2', priors_patches='as_is', round_anchors=False,
                 resize_min=800, resize_max=1333,
                 # 4) settings for filtering during RPN and ROI forwards
                 score_thr1=0.0, iou_thr1=0.7, imtop1=1000, min_size1=0, lvtop=1000,
                 score_thr2=0.05, iou_thr2=0.5, imtop2=100, min_size2=0,
                 imtop1_train=2000, lvtop_train=2000):
        super().__init__()

        self.body = ResNet50(bn=1e-5, num_freeze=2)
        cins, self.strides = [256, 512, 1024, 2048], [4, 8, 16, 32, 64]
        anchors = make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5], round_anchors)

        self.bases = list(zip(self.strides, anchors))
        self.resize = (resize_min, resize_max)
        self.resize_with = prep_resize
        self.priors_patches = priors_patches
        self.bxset1 = (score_thr1, iou_thr1, imtop1, min_size1, lvtop, imtop1_train, lvtop_train)
        self.bxset2 = (score_thr2, iou_thr2, imtop2, min_size2)

        self.fpn = FeaturePyramidNetwork(cins, 256, None, fpn_batchnorm, pool=True, smoothP5=True)
        self.rpn = RegionProposalNetwork(256, len(anchors[0]), rpn_convdepth)
        self.roi = RoIProcessingNetwork(256, 7, 1024, roi_convdepth, roi_mlp_depth,
                                        num_classes, bckg_class_first, roialign_settings)
        self.to(device)

    def forward(self, imgs):
        dv = next(self.parameters()).device
        x, sz_orig, sz_used = preprocess(imgs, dv, self.resize, self.resize_with)
        priors = get_priors(x.shape[2:], self.bases, dv, 'corner', self.priors_patches, concat=False)
        xs = self.body(x)
        xs = self.fpn(xs)
        p, imidx = self.rpn(xs, priors, sz_used, self.bxset1)
        b, s, c = self.roi(p, imidx, xs[:-1], self.strides[:-1], sz_used, self.bxset2)
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
        self.model = FasterRCNN(num_classes=1)
        load_weights(self.model, self.link, 'frcnn_anime', self.wconv, 'state_dict')
        self.model.eval()

    def __call__(self, imgs):
        with torch.inference_mode():
            return self.model(imgs)
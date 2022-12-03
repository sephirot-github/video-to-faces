import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from .operations import prep, post
from .components.fpn import FeaturePyramidNetwork
from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from ..backbones.mobilenet import MobileNetV3L
from ..utils.weights import load_weights


class RegionProposalNetwork(nn.Module):

    def __init__(self, c, num_anchors, conv_depth):
        super().__init__()
        self.conv = nn.Sequential(*[ConvUnit(c, c, 3, 1, 1, 'relu', None) for _ in range(conv_depth)])
        self.log = nn.Conv2d(c, num_anchors, 1, 1)
        self.reg = nn.Conv2d(c, num_anchors * 4, 1, 1)

    def head(self, x, priors, lvtop):
        n = x.shape[0]
        x = self.conv(x)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(n, -1, 4)
        log = self.log(x).permute(0, 2, 3, 1).reshape(n, -1, 1)
        log, top = log.topk(min(lvtop, log.shape[1]), dim=1)
        reg = reg.gather(1, top.expand(-1, -1, 4))
        pri = priors.expand(n, -1, -1).gather(1, top.expand(-1, -1, 4))
        boxes = post.decode_boxes(reg, pri, mults=(1, 1), clamp=True)
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

    def __init__(self, c, roi_map_size, clin, conv_depth, mlp_depth, num_classes, bckg_first, ralign_settings):
        super().__init__()
        self.conv = nn.ModuleList([ConvUnit(c, c, 3, 1, 1, 'relu') for _ in range(conv_depth)])
        c1 = c * roi_map_size ** 2
        self.fc = nn.ModuleList([nn.Linear(c1 if i == 0 else clin, clin) for i in range(mlp_depth)])
        self.cls = nn.Linear(clin, 1 + num_classes)
        self.reg = nn.Linear(clin, num_classes * 4)
        self.ralign_set = ralign_settings
        self.bckg_first = bckg_first

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
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes, score_thr, iou_thr, imtop, min_size):
        roi_maps = post.roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides, self.ralign_set)
        reg, log = self.heads(roi_maps)
       
        reg = reg.reshape(reg.shape[0], -1, 4)
        scr = F.softmax(log, dim=-1)
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)
        scr = scr[:, :-1] if not self.bckg_first else scr[:, 1:]
        cls = cls[:, :-1] if not self.bckg_first else cls[:, 1:]

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > score_thr).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = post.convert_to_cwh(proposals)
        boxes = post.decode_boxes(reg, proposals, mults=(0.1, 0.2), clamp=True)
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

    thub = 'https://download.pytorch.org/models/'
    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/'
    links = {
        'tv_resnet50_v1': thub + 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'tv_resnet50_v2': thub + 'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
        'tv_mobilenetv3l_hires': thub + 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
        'tv_mobilenetv3l_lores': thub + 'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
        'mm_resnet50': mmhub + 'faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/'\
                               'faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth',
    }

    def config_base(self):
        cfg = {}
        cfg.update(fpn_batchnorm=None, rpn_convdepth=1, roi_convdepth=0, roi_mlp_depth=2)
        cfg.update(resize_min=800, resize_max=1333)
        cfg.update(lvtop=1000, score_thr1=0.0, imtop1=1000, min_size1=0)
        cfg.update(score_thr2=0.05, imtop2=100, min_size2=0)
        cfg.update(weights_add_nbatches=False)
        return cfg

    def config_torchvision(self, cfg):
        cfg.update(num_classes=90, bckg_class_first=True, ralign_settings=(2, False))
        cfg.update(weights_sub=None, weights_extra=self.tv_conversion)
        cfg.update(prep_resize='torch', priors_patches='fit', round_anchors=True)
        cfg.update(min_size1=1e-3, min_size2=1e-2)
        return cfg
    
    def config_mmdet(self, cfg):
        cfg.update(num_classes=80, bckg_class_first=False, ralign_settings=(0, True))
        cfg.update(weights_sub='state_dict', weights_extra=self.mm_conversion)
        cfg.update(prep_resize='cv2', priors_patches='as_is', round_anchors=False)
        return cfg

    def config_resnet(self, cfg):
        cfg.update(bbone='resnet50', resnet_bn_eps=1e-5)
        cfg.update(cins=[256, 512, 1024, 2048], strides=[4, 8, 16, 32, 64])
        cfg.update(anchors=post.make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5], cfg['round_anchors']))

    def config_mobile(self, cfg):
        cfg.update(bbone='mobilenetv3l')
        cfg.update(cins=[160, 960], strides=[32, 32, 64])
        cfg.update(anchors=post.make_anchors([32, 32, 32], [1, 2, 4, 8, 16], [2, 1, 0.5], cfg['round_anchors']))

    def config(self, src, arch, version=None):
        cfg = self.config_base()
        cfg = self.config_torchvision(cfg) if src == 'tv' else self.config_mmdet(cfg)
        cfg = self.config_resnet(cfg) if arch == 'resnet50' else self.config_mobile(cfg)
        if src == 'tv' and (arch == 'resnet50' and version == 'v1' or arch == 'mobilenetv3l'):
            cfg.update(weights_add_nbatches=True)
        if src == 'tv' and arch == 'resnet50' and version == 'v1':
            cfg.update(resnet_bn_eps=0.0)
        if src == 'tv' and arch == 'resnet50' and version == 'v2':
            cfg.update(fpn_batchnorm=1e-5, rpn_convdepth=2, roi_convdepth=4, roi_mlp_depth=1)
        if src == 'tv' and arch == 'mobilenetv3l' and version == 'lores':
            cfg.update(resize_min=320, resize_max=640, lvtop=150, imtop1=150, score_thr1=0.05)
        return cfg

    def __init__(self, pretrained='tv_resnet50_v1', device='cpu'):
        super().__init__()
        parts = pretrained.split('_')
        src, arch = parts[0:2]
        self.src = src
        version = None if len(parts) <= 2 else parts[2]
        num_classes = 90 if src == 'tv' else 80
        bckg_first = True if src == 'tv' else False
        ralign_set = (2, False) if src == 'tv' else (0, True)
        weights_sub = None if src == 'tv' else 'state_dict'
        weights_extra = self.tv_conversion if src == 'tv' else self.mm_conversion
        fpn_batchnorm, rpn_convdepth, roi_convdepth, roi_mlp_depth = None, 1, 0, 2
        addnbatch = False
        if arch == 'resnet50':
            bn_eps = 0.0 if src == 'tv' and version == 'v1' else 1e-5
            backbone = ResNet50(bn_eps=bn_eps)
            cins = [256, 512, 1024, 2048]
            self.strides = [4, 8, 16, 32, 64]
            anchors = post.make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5], src == 'tv')
            if version == 'v1':
                addnbatch = True
            if version == 'v2':
                fpn_batchnorm, rpn_convdepth, roi_convdepth, roi_mlp_depth = 1e-05, 2, 4, 1
        elif arch == 'mobilenetv3l':
            backbone = MobileNetV3L([13])
            cins = [160, 960]
            self.strides = [32, 32, 64]
            anchors = post.make_anchors([32, 32, 32], [1, 2, 4, 8, 16], [2, 1, 0.5], True)
            addnbatch = True
            if version == 'lores':
                self.resize_min = 320
                self.resize_max = 640
                self.lvtop = 150
                self.imtop1 = 150
                self.score_thr1 = 0.05

        self.bases = list(zip(self.strides, anchors))
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(cins, 256, None, fpn_batchnorm, pool=True, smoothP5=True)
        self.rpn = RegionProposalNetwork(256, len(anchors[0]), rpn_convdepth)
        self.roi = RoIProcessingNetwork(256, 7, 1024, roi_convdepth, roi_mlp_depth, num_classes, bckg_first, ralign_set)
        self.to(device)
        load_weights(self, self.links[pretrained], pretrained, device, sub=weights_sub, add_num_batches=addnbatch, extra_conversion=weights_extra)

    def tv_conversion(self, wd):
        # TorchVision's RoI head predicts boxes for background class too, only to discard
        # them right away, so might as well not calculate it in the first place
        nm = 'roi_heads.box_predictor.bbox_pred.'
        wd[nm + 'weight'] = wd[nm + 'weight'][4:, :] # [364, 1024] -> [360]
        wd[nm + 'bias'] = wd[nm + 'bias'][4:]        # [364] -> [360]
        return wd

    def mm_conversion(self, wd):
        # in MMDet weights for RoI head, representation FC and final reg/log FCs are switched over
        wl = list(wd.items())
        els = [wl.pop(-1) for _ in range(8)][::-1] # last 8 entries
        for el in els[4:] + els[:4]:
            wl.append(el)
        wd = dict(wl)
        return wd

    def forward(self, imgs):
        dv = next(self.parameters()).device
        #if self.src == 'tv':
        #    ts = prep.normalize(imgs, dv, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
        #    ts, sz_orig, sz_used = prep.resize(ts, self.resize_min, self.resize_max)
        #    x = prep.batch(ts, mult=32)
        #else:
        #    imgs, sz_orig, sz_used = prep.resize_cv2(imgs, self.resize_min, self.resize_max)
        #    ts = prep.normalize(imgs, dv, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
        #    x = prep.batch(ts, mult=32)
        x = prep.full(imgs, dv, (self.cfg['resize_min'], self.cfg['resize_max']), self.cfg['prep_resize'])
                
        priors = post.get_priors(x.shape[2:], self.bases, dv, 'corner', 'fit' if self.src =='tv' else 'as_is', concat=False)
        xs = self.body(x)
        xs = self.fpn(xs)
        proposals, imidx = self.rpn(xs, priors, sz_used, lvtop=self.lvtop, imtop=self.imtop1,
                                    score_thr=self.score_thr1, iou_thr=0.7, min_size=1e-3)
        boxes, scores, classes = self.roi(proposals, imidx, xs[:-1], self.strides[:-1], sz_used,
                                          score_thr=self.score_thr2, iou_thr=0.5, imtop=self.imtop2,
                                          min_size=1e-2)
        
        scalebacks = torch.tensor(sz_orig) / torch.tensor(sz_used)
        scalebacks = scalebacks.flip(1).repeat(1, 2)
        boxes = [boxes[i] * scalebacks[i] for i in range(len(imgs))]

        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [boxes, scores, classes]]
        return b, s, c
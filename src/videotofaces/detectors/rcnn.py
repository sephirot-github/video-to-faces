import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from ..backbones.mobilenet import MobileNetV3L
from .components.fpn import FeaturePyramidNetwork
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, convert_to_cwh, decode_boxes, remove_small, scale_boxes
from .operations.loss import get_losses
from .operations.post import get_lvidx, roi_align_multilevel
from .operations.prep import preprocess
from ..utils.weights import load_weights

from .operations.loss import match_with_targets
from .operations.bbox import encode_boxes


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

    def forward(self, fmaps, priors, imsizes, cfg, gtboxes):
        lvtop = cfg['lvtop'][0] if not self.training else cfg['lvtop'][1]
        imtop = cfg['imtop1'][0] if not self.training else cfg['imtop1'][1]

        tuples = [self.head(x) for x in fmaps]
        regs, logs = map(list, zip(*tuples))
        
        dregs = [x.detach() for x in regs]
        dlogs = [x.detach() for x in logs]
        boxes, logits, lvlen = self.filt_dec(dregs, dlogs, priors, lvtop)
        boxes = torch.cat(boxes, axis=1)
        obj = torch.cat(logits, axis=1).sigmoid()

        n, dim = boxes.shape[:2]
        boxes, obj = boxes.reshape(-1, 4), obj.flatten()
        idx = torch.nonzero(obj >= cfg['score_thr1']).squeeze()
        boxes, obj = boxes[idx], obj[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, obj, idx, imidx = remove_small(boxes, cfg['min_size1'], obj, idx, imidx)
        groups = imidx * 10 + get_lvidx(idx % dim, lvlen)
        keep = torchvision.ops.batched_nms(boxes, obj, groups, cfg['iou_thr1'])
        keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
        boxes, imidx = boxes[keep], imidx[keep]

        if not self.training:
            return boxes, imidx, None
        else:
            regs = torch.cat(regs, axis=1)
            logs = torch.cat(logs, axis=1)
            priors = torch.cat(priors)
            loss_obj, loss_reg = get_losses(gtboxes, priors, regs, logs, 0.3, 0.7, True, 256, 0.5)
            return boxes, imidx, (loss_obj, loss_reg)


class RoIProcessingNetwork(nn.Module):

    def __init__(self, c, roi_map_size, clin, cfg):
        super().__init__()
        self.conv = nn.ModuleList([ConvUnit(c, c, 3, 1, 1, 'relu') for _ in range(cfg['roi_convdepth'])])
        c1 = c * roi_map_size ** 2
        self.fc = nn.ModuleList([nn.Linear(c1 if i == 0 else clin, clin) for i in range(cfg['roi_mlp_depth'])])
        self.cls = nn.Linear(clin, 1 + cfg['num_classes'])
        self.reg = nn.Linear(clin, cfg['num_classes'] * 4)
        self.ralign_set = cfg['roialign_settings']
        self.bckg_first = cfg['bckg_class_first']

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
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes, cfg, gtb, gtl):
        if self.training:
            # to list
            proposals = [proposals[imidx == i] for i in range(len(gtb))]
            proposals = [torch.cat([p, b.to(torch.float32)]) for p, b in zip(proposals, gtb)]
            # main
            targets, labels, sidx, _ = match_with_targets(gtb, gtl, proposals, 0.5, 0.5, False, 512, 0.25)
            proposals = [p[sampled] for p, sampled in zip(proposals, sidx)]
            # back to joined
            imidx = [torch.full([len(p)], i) for i, p in enumerate(proposals)]
            proposals, imidx = [torch.cat(x) for x in [proposals, imidx]]
        
        roi_maps = roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides, self.ralign_set)
        reg, log = self.heads(roi_maps)

        if self.training:
            labels = torch.cat(labels)
            targets = torch.cat(targets)
            loss_cls = F.cross_entropy(log, labels)
            reg = reg.reshape(reg.shape[0], -1, 4)
            reg = reg[labels > 0, labels[labels > 0] - 1] # minus 1 because we removed the bckg class
            loss_reg = F.smooth_l1_loss(reg, targets, beta=1/9, reduction='sum') / labels.numel()
            return loss_cls, loss_reg
       
        reg = reg.reshape(reg.shape[0], -1, 4)
        scr = F.softmax(log, dim=-1)
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)
        scr = scr[:, :-1] if not self.bckg_first else scr[:, 1:]
        cls = cls[:, :-1] if not self.bckg_first else cls[:, 1:]

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > cfg['score_thr2']).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = convert_to_cwh(proposals, in_place=True)
        boxes = decode_boxes(reg, proposals, mults=(0.1, 0.2), clamp=True)
        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, scr, cls, imidx = remove_small(boxes, cfg['min_size2'], scr, cls, imidx)
        
        res = []
        for i in range(n):
            bi, si, ci = [x[imidx == i] for x in [boxes, scr, cls]]
            keep = torchvision.ops.batched_nms(bi, si, ci, cfg['iou_thr2'])[:cfg['imtop2']]
            res.append((bi[keep], si[keep], ci[keep]))
        return map(list, zip(*res))
        #groups = imidx * 1000 + cls
        #keep = torchvision.ops.batched_nms(boxes, scr, groups, cfg['iou_thr2'])
        #keep = torch.cat([keep[imidx[keep] == i][:cfg['imtop2']] for i in range(n)])
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
        'mm_resnet50_animefaces': 'https://github.com/hysts/anime-face-detector/'\
                                  'releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth'
    }

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

    def config_base(self):
        cfg = {}
        cfg.update(fpn_batchnorm=None, rpn_convdepth=1, roi_convdepth=0, roi_mlp_depth=2)
        cfg.update(resize_min=800, resize_max=1333)
        cfg.update(score_thr1=0.0, iou_thr1=0.7, imtop1=(1000, 2000), min_size1=0, lvtop=(1000, 2000))
        cfg.update(score_thr2=0.05, iou_thr2=0.5, imtop2=100, min_size2=0)
        cfg.update(weights_add_nbatches=False)
        return cfg

    def config_torchvision(self, cfg):
        cfg.update(num_classes=90, bckg_class_first=True, roialign_settings=(2, False))
        cfg.update(weights_sub=None, weights_extra=self.tv_conversion)
        cfg.update(prep_resize='torch', priors_patches='fit', round_anchors=True)
        cfg.update(min_size1=1e-3, min_size2=1e-2)
        return cfg
    
    def config_mmdet(self, cfg):
        cfg.update(num_classes=80, bckg_class_first=False, roialign_settings=(0, True))
        cfg.update(weights_sub='state_dict', weights_extra=self.mm_conversion)
        cfg.update(prep_resize='cv2', priors_patches='as_is', round_anchors=False)
        return cfg

    def config_resnet(self, cfg):
        cfg.update(bbone='resnet50', resnet_bn_eps=1e-5)
        cfg.update(cins=[256, 512, 1024, 2048], strides=[4, 8, 16, 32, 64])
        cfg.update(anchors=make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5], cfg['round_anchors']))
        return cfg

    def config_mobile(self, cfg):
        cfg.update(bbone='mobilenetv3l')
        cfg.update(cins=[160, 960], strides=[32, 32, 64])
        cfg.update(anchors=make_anchors([32, 32, 32], [1, 2, 4, 8, 16], [2, 1, 0.5], cfg['round_anchors']))
        return cfg

    def get_config(self, src, arch, version=None):
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
        if version == 'animefaces':
            cfg.update(num_classes=1)
        return cfg

    def get_backbone(self, cfg):
        if cfg['bbone'] == 'resnet50':
            return ResNet50(bn_eps=cfg['resnet_bn_eps'])
        if cfg['bbone'] == 'mobilenetv3l':
            return MobileNetV3L([13, 16], bn=(1e-05, 'frozen'), num_freeze=7)

    def __init__(self, pretrained='tv_resnet50_v1', device='cpu'):
        super().__init__()
        cfg = self.get_config(*pretrained.split('_'))
        self.cfg = cfg
        self.bases = list(zip(cfg['strides'], cfg['anchors']))
        self.body = self.get_backbone(cfg)
        self.fpn = FeaturePyramidNetwork(cfg['cins'], 256, None, cfg['fpn_batchnorm'], pool=True, smoothP5=True)
        self.rpn = RegionProposalNetwork(256, len(cfg['anchors'][0]), cfg['rpn_convdepth'])
        self.roi = RoIProcessingNetwork(256, 7, 1024, cfg)
        self.to(device)
        load_weights(self, self.links[pretrained], pretrained, device,
                     cfg['weights_extra'], cfg['weights_sub'], cfg['weights_add_nbatches'])

    def forward(self, imgs, targets=None):
        dv = next(self.parameters()).device
        resize = (self.cfg['resize_min'], self.cfg['resize_max'])
        resize_with = self.cfg['prep_resize']
        priors_patches = self.cfg['priors_patches']
        strides = self.cfg['strides']
        
        x, sz_orig, sz_used = preprocess(imgs, dv, resize, resize_with)
        gtb = None if not self.training else scale_boxes([torch.tensor(t) for t in targets[0]], sz_used, sz_orig)
        gtl = None if not self.training else [torch.tensor(t) for t in targets[1]]
        priors = get_priors(x.shape[2:], self.bases, dv, 'corner', priors_patches, concat=False)
        xs = self.body(x)
        xs = self.fpn(xs)
        p, imidx, p_losses = self.rpn(xs, priors, sz_used, self.cfg, gtb)
        ret = self.roi(p, imidx, xs[:-1], strides[:-1], sz_used, self.cfg, gtb, gtl)
        if self.training:
            d_losses = ret
            return (*p_losses, *d_losses)
        else:
            b, s, c = ret
            b = scale_boxes(b, sz_orig, sz_used)
            b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
            return b, s, c
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from .operations import prep, post
from ..backbones.basic import ConvUnit, BaseMultiReturn
from ..backbones.mobilenet import MobileNetV2
from ..utils.weights import load_weights

# SSD paper: https://arxiv.org/pdf/1512.02325.pdf
# VGG paper: https://arxiv.org/pdf/1409.1556.pdf


def vgg_convunit(cin, cout, k, s, p, d=1):
    return ConvUnit(cin, cout, k, s, p, 'relu', bn=None, d=d)


def mbnet_convunit(cin, cout, k, s, p, grp=1):
    return ConvUnit(cin, cout, k, s, p, 'relu6', 1e-03, grp=grp)


class VGG16(BaseMultiReturn):
    
    def __init__(self):
        super().__init__()
        # configuration D from the paper
        cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        layers = []
        cin = 3
        for c, n in cfg:
            for i in range(n):
                layers.append(vgg_convunit(cin, c, 3, 1, 1))
                cin = c
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)
    
    
class ExtendedVGG16(nn.Module):

    def __init__(self, hires=False):
        super().__init__()
        self.l2_scale = nn.Parameter(torch.ones(512) * 20)
        self.backbone = VGG16()
        self.backbone.layers[9].ceil_mode = True   # adjusting MaxPool3
        # the above is needed in SSD_300 to go from 75 pixels to 38 (and not 37)
        # for other pools (and for all SSD_512), the input size will be even so it doesn't matter
        self.backbone.layers.pop(-1)               # removing MaxPool5
        self.backbone.layers.extend([
            nn.MaxPool2d(3, 1, 1),                 # replacement for MaxPool5
            vgg_convunit(512, 1024, 3, 1, 6, d=6), # FC6
            vgg_convunit(1024, 1024, 1, 1, 0)      # FC7
        ])
        self.backbone.retidx = [12, 19] # right before MaxPool4, last layer with extensions

        self.extra = nn.ModuleList()
        settings = [(1024, 512, 3, 2, 1), (512, 256, 3, 2, 1)]
        if not hires:
            settings.extend([(256, 256, 3, 1, 0), (256, 256, 3, 1, 0)])
        else:
            settings.extend([(256, 256, 3, 2, 1), (256, 256, 3, 2, 1), (256, 256, 4, 1, 1)])
        for cin, cout, k, s, p in settings:
            self.extra.append(nn.Sequential(
                vgg_convunit(cin, cout // 2, 1, 1, 0),
                vgg_convunit(cout // 2, cout, k, s, p)
            ))
        self.couts = [512, 1024] + [s[1] for s in settings]

    def forward(self, x):
        xs = self.backbone(x)
        if hasattr(self, 'l2_scale'):
            xs[0] = F.normalize(xs[0]) * self.l2_scale.view(1, -1, 1, 1)
        x = xs[1]
        for block in self.extra:
            x = block(x)
            xs.append(x)
        return xs


class ExtendedMobileNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV2([5, 8], 'relu6', 1e-03)
        self.extra = nn.ModuleList()
        settings = [(1280, 512), (512, 256), (256, 256), (256, 128)]
        for cin, cout in settings:
            cmid = cout // 2
            self.extra.append(nn.Sequential(
                mbnet_convunit(cin, cmid, 1, 1, 0),
                nn.Sequential(
                    mbnet_convunit(cmid, cmid, 3, 2, 1, grp=cmid),
                    mbnet_convunit(cmid, cout, 1, 1, 0)
                )
            ))
        self.couts = [96, 1280] + [s[1] for s in settings]

    def forward(self, x):
        xs = self.backbone(x)
        x = xs[1]
        for block in self.extra:
            x = block(x)
            xs.append(x)
        return xs


class SSDHead(nn.Module):

    def __init__(self, bbone, c, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        k = 3
        if bbone != 'vgg16':
            self.prep = mbnet_convunit(c, c, 3, 1, 1, grp=c)
            k = 1
        self.conv = nn.Conv2d(c, num_anchors * task_len, k, 1, (k-1)//2)
    
    def forward(self, x):
        if hasattr(self, 'prep'):
            x = self.prep(x)
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class SSD(nn.Module):

    tvhub = 'https://download.pytorch.org/models/'
    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/'
    links = {
        'vgg16_300_tv': tvhub + 'ssd300_vgg16_coco-b556d3b4.pth',
        'vgg16_300_mm': mmhub + 'ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth',
        'vgg16_512_mm': mmhub + 'ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth',
        'mobile2_320_mm': mmhub + 'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',
        'mobile3_320_tv': tvhub + 'ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    }

    def mm_vgg_conversion(self, wd):
        nm = 'neck.l2_norm.weight'
        ret = {nm: wd.pop(nm)}
        ret.update(wd)
        return ret

    def get_config(self, bbone, csize, source):
        cfg = {}
        if source == 'tv':
            cfg.update(wextra=None, wsub=None)
            cfg.update(resize='torch', stdvs=None, means=(122.99925, 116.9991, 103.9992)) #~(0.48235, 0.45882, 0.40784)
            cfg.update(bckg_class_first=False)
            cfg.update(anchors_rounding=False, anchors_clamp=True)
            cfg.update(lvtop=None, cltop=400, score_thr=0.01)
        elif source == 'mm':
            cfg.update(wextra=None, wsub='state_dict')
            cfg.update(resize='cv2', stdvs=None, means='imagenet')
            cfg.update(bckg_class_first=True)
            cfg.update(anchors_rounding=True, anchors_clamp=False)
            cfg.update(lvtop=1000, cltop=None, score_thr=0.02)
            if bbone == 'vgg16':
                cfg.update(wextra=self.mm_vgg_conversion)
            else:
                cfg.update(stdvs='imagenet')
        return cfg

    def __init__(self, backbone, canvas_size, num_classes, pretrained=None, device='cpu'):
        super().__init__()
        self.cfg = self.get_config(*pretrained.split('_'))
        self.canvas_size = canvas_size
        if backbone == 'vgg16':
            self.backbone = ExtendedVGG16(canvas_size == 512)
        else:
            self.backbone = ExtendedMobileNet()
        self.bases, num_anchors_per_level = self.get_bases(canvas_size, self.cfg)
        level_dims = list(zip(self.backbone.couts, num_anchors_per_level))
        self.cls_heads = nn.ModuleList([SSDHead(backbone, c, an, num_classes + 1) for c, an in level_dims])
        self.reg_heads = nn.ModuleList([SSDHead(backbone, c, an, 4) for c, an in level_dims])
        if pretrained:
            extra, sub = self.cfg['wextra'], self.cfg['wsub']
            load_weights(self, self.links[pretrained], pretrained, device, extra, sub)
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        backend, means, stdvs = self.cfg['resize'], self.cfg['means'], self.cfg['stdvs']
        bckg_first = self.cfg['bckg_class_first']

        x, sz_orig, sz_used = prep.full(imgs, dv, self.canvas_size, backend, False, 1, means, stdvs)
        #return x
        xs = self.backbone(x)
        #return xs
        cls = [self.cls_heads[i](xs[i]) for i in range(len(xs))]
        lvlen = [lvl.shape[1] for lvl in cls]
        cls = torch.cat(cls, dim=1)
        reg = torch.cat([self.reg_heads[i](xs[i]) for i in range(len(xs))], dim=1)
        scr = F.softmax(cls, dim=-1)
        scr = scr[:, :, :-1] if bckg_first else scr[:, :, 1:]
        #return reg, cls
        priors = post.get_priors(x.shape[2:], self.bases)
        b, s, c = self.postprocess(reg, scr, priors, sz_used, lvlen, self.cfg)
        b = post.scale_back(b, sz_orig, sz_used)
        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
        return b, s, c

    def get_bases(self, img_size, cfg):
        if img_size == 300:
            strides = [8, 16, 32, 64, 100, 300]
            scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # = [0.07] + np.linspace(0.15, 1.05, 6)
            ar3_idx = [1, 2, 3]
        elif img_size == 512:
            strides = [8, 16, 32, 64, 128, 256, 512]
            scales = [0.04, 0.10, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06] # [0.04] + np.linspace(0.10, 1.06, 7)
            ar3_idx = [1, 2, 3, 4]
        elif img_size == 320:
            strides = [16, 32, 64, 107, 160, 320]
            scales = [0.15, 0.313, 0.47, 0.632, 0.792, 0.95, 1]
            ar3_idx = [0, 1, 2, 3, 4, 5]
            #min_sizes=[48, 100, 150, 202, 253, 304],
            #max_sizes=[100, 150, 202, 253, 304, 320]
            # 0.15, 0.3125, 0.46875, 0.63125, 0.790625, 0.95, 1
            # 0.1625, 0.15625, 0.1625, 0.159375, 0.159375
            #[0.15, 0.3125, 0.47, 0.63, 0.79, 0.95] == np.linspace(0.15, 0.95, 6), 0.31 -> 0.3125
            #[0.15, 0.3125, 0.47, 0.63, 0.79, 0.95, 1.11] == np.linspace(0.15, 1.11, 7), clamp=True
        else:
            raise ValueError(img_size)
        anchors = []
        for i in range(len(strides)):
            r = [1, 2, 0.5]
            if i in ar3_idx:
                r.extend([3, 1/3])
            if not cfg['anchors_rounding']:
                bsize = scales[i] * img_size
                extra = math.sqrt(scales[i] * scales[i + 1]) * img_size
            else:
                # the same thing but with some intermediate rounding to replicate how MMDet does it
                # for SSD_300 it doesn't matter because the multiplications end up ~int anyway
                # but for SSD_512 it does lead to minor difference
                bsize = int(scales[i] * img_size)
                bsizen = int(scales[i + 1] * img_size)
                extra = math.sqrt(bsizen / bsize) * bsize
            a = post.make_anchors([bsize], ratios=r)[0]
            a.insert(1, (extra, extra))
            if cfg['anchors_clamp']:
                a = [(min(x, img_size), min(y, img_size)) for x, y in a]
            anchors.append(a)    
        return list(zip(strides, anchors)), [len(a) for a in anchors]

    def postprocess(self, reg, scr, priors, sz_used, lvlen, cfg):
        n, dim, num_classes = scr.shape
        reg, scr = reg.reshape(-1, 4), scr.flatten()
        fidx = torch.nonzero(scr > cfg['score_thr']).squeeze()
        fidx = post.top_per_level(fidx, scr, cfg['lvtop'], lvlen, n, mult=num_classes)
        scores = scr[fidx]
        classes = fidx % num_classes + (0 if self.cfg['bckg_class_first'] else 1)
        idx = torch.div(fidx, num_classes, rounding_mode='floor')
        imidx = idx.div(dim, rounding_mode='floor')

        if cfg['cltop']:
            sel = post.top_per_class(scores, classes, imidx, cfg['cltop'])
            scores, classes, imidx, idx = [x[sel] for x in [scores, classes, imidx, idx]]
        
        boxes = post.decode_boxes(reg[idx], priors[idx % dim], mults=(0.1, 0.2), clamp=True)
        boxes = post.clamp_to_canvas(boxes, sz_used, imidx)
        boxes, scores, classes, imidx = post.remove_small(boxes, 0, scores, classes, imidx)
        res = []
        for i in range(n):
            bi, si, ci = [x[imidx == i] for x in [boxes, scores, classes]]
            keep = torchvision.ops.batched_nms(bi, si, ci, 0.45)[:200]
            res.append((bi[keep], si[keep], ci[keep]))
        return map(list, zip(*res))
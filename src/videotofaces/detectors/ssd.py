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


def ssd_convunit(cin, cout, k, s, p, d=1):
    return ConvUnit(cin, cout, k, s, p, 'relu', bn=None, d=d)


class VGG16(BaseMultiReturn):
    
    def __init__(self):
        super().__init__()
        # configuration D from the paper
        cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        layers = []
        cin = 3
        for c, n in cfg:
            for i in range(n):
                layers.append(ssd_convunit(cin, c, 3, 1, 1))
                cin = c
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)
    
    
class BackboneExtended(nn.Module):

    def __init__(self, backbone='vgg16', hires=False):
        super().__init__()
        if backbone == 'mobile2':
            self.backbone = MobileNetV2([5, 8])
        else:
            self.l2_scale = nn.Parameter(torch.ones(512) * 20)
            self.backbone = VGG16()
            self.backbone.layers[9].ceil_mode = True   # adjusting MaxPool3
            # the above is needed in SSD_300 to go from 75 pixels to 38 (and not 37)
            # for other pools (and for all SSD_512), the input size will be even so it doesn't matter
            self.backbone.layers.pop(-1)               # removing MaxPool5
            self.backbone.layers.extend([
                nn.MaxPool2d(3, 1, 1),                 # replacement for MaxPool5
                ssd_convunit(512, 1024, 3, 1, 6, d=6), # FC6
                ssd_convunit(1024, 1024, 1, 1, 0)      # FC7
            ])
            self.backbone.retidx = [12, 19] # right before MaxPool4, last layer with extensions
        
        self.extra = nn.ModuleList()
        settings = [(1024, 512, 2, 1), (512, 256, 2, 1), (256, 256, 1, 0), (256, 256, 1, 0)]
        if hires:
            settings.append((256, 256, 4, 0))
        for cin, cout, s, p in settings:
            self.extra.append(nn.Sequential(
                ssd_convunit(cin, cout // 2, 1, 1, 0),
                ssd_convunit(cout // 2, cout, 3, s, p)
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


class Head(nn.Module):

    def __init__(self, cin, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        self.conv = nn.Conv2d(cin, num_anchors * task_len, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class SSD(nn.Module):

    links = {
        'torchvision': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
        'mmdetection': 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth'
        # ssdlite tv: https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
    }

    def mm_conversion(self, wd):
        nm = 'neck.l2_norm.weight'
        ret = {nm: wd.pop(nm)}
        ret.update(wd)
        return ret

    def get_config(self, source):
        cfg = {}
        if source == 'torchvision':
            cfg.update(wextra=None, wsub=None)
            cfg.update(resize='torch', norm=(122.99925, 116.9991, 103.9992)) #~(0.48235, 0.45882, 0.40784)
            cfg.update(bckg_class_first=False)
            cfg.update(lvtop=None, cltop=400, score_thr=0.01)
        elif source == 'mmdetection':
            cfg.update(wextra=self.mm_conversion, wsub='state_dict')
            cfg.update(resize='cv2', norm='imagenet')
            cfg.update(bckg_class_first=True)
            cfg.update(lvtop=1000, cltop=None, score_thr=0.02)
        return cfg

    def __init__(self, canvas_size, num_classes, pretrained=None, device='cpu'):
        super().__init__()
        self.cfg = self.get_config(pretrained)
        self.canvas_size = canvas_size
        self.backbone = BackboneExtended()
        strides = [8, 16, 32, 64, 100, 300]
        anchors = self.get_anchors(canvas_size)
        self.bases = list(zip(strides, anchors))
        level_dims = list(zip(self.backbone.couts, [len(a) for a in anchors]))
        self.cls_heads = nn.ModuleList([Head(c, an, num_classes + 1) for c, an in level_dims])
        self.reg_heads = nn.ModuleList([Head(c, an, 4) for c, an in level_dims])
        if pretrained:
            extra, sub = self.cfg['wextra'], self.cfg['wsub']
            load_weights(self, self.links[pretrained], pretrained, device, extra, sub)
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        backend, means = self.cfg['resize'], self.cfg['norm']
        bckg_first = self.cfg['bckg_class_first']

        x, sz_orig, sz_used = prep.full(imgs, dv, self.canvas_size, backend, False, 1, means, None) 
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

    def get_anchors(self, img_size):
        scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # = [0.07] + np.linspace(0.15, 1.05, 6)
        anchors = []
        for i in range(6):
            r = [1, 2, 0.5]
            if i in (1, 2, 3):
                r.extend([3, 1/3])
            bsize = scales[i] * img_size
            extra = math.sqrt(scales[i] * scales[i + 1]) * img_size
            a = post.make_anchors([bsize], ratios=r)[0]
            a.insert(1, (extra, extra))
            a = [(min(x, img_size), min(y, img_size)) for x, y in a]
            anchors.append(a)
        return anchors

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
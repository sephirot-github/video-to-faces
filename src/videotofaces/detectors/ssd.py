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
    
    def __init__(self, ceil_mode_idx=[2]):
        super().__init__(retidx=None)
        # configuration D from the paper
        cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        layers = []
        cin = 3
        for c, n in cfg:
            for i in range(n):
                layers.append(ssd_convunit(cin, c, 3, 1, 1))
                cin = c
            layers.append(nn.MaxPool2d(2, ceil_mode=i in ceil_mode_idx))
        self.layers = nn.Sequential(*layers)
    
    
class BackboneExtended(nn.Module):

    def __init__(self, backbone='vgg16', hires=False):
        super().__init__()
        if backbone == 'mobile2':
            self.backbone = MobileNetV2([5, 8])
        else:
            self.l2_scale = nn.Parameter(torch.ones(512) * 20)
            self.backbone = VGG16()
            self.backbone.layers.extend([
                nn.MaxPool2d(3, 1, 1),
                ssd_convunit(512, 1024, 3, 1, 6, d=6),
                ssd_convunit(1024, 1024, 1, 1, 0)
            ])
            self.backbone.retidx = [22, 32] # before stage 3 MaxPool, last layer with extensions
        
        self.extra = nn.ModuleList()
        settings = [(1024, 512, 2, 1), (512, 256, 2, 1), (256, 256, 1, 0), (256, 256, 1, 0)]
        if hires:
            settings.append((256, 256, 4, 0))
        for cin, cout, s, p in settings:
            self.extra.append(nn.Sequential(
                ssd_convunit(cin, cout // 2, 1, 1, 0),
                ssd_convunit(cout // 2, cout, 3, s, p)
            ))

    def forward(self, x):
        xs = self.backbone(x)
        if hasattr(self, 'l2norm'):
            xs[0] = F.normalize(xs[0]) * self.l2_scale.view(1, -1, 1, 1)
        x = xs[1]
        for block in self.extra:
            x = block(x)
            xs.append(x)
        return xs


class SSD(nn.Module):

    link = 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth'
    #'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth'
    # ssdlite tv: https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth

    def __init__(self, pretrained=True, device='cpu'):
        super().__init__()
        self.backbone = BackboneExtended()
        if pretrained:
            load_weights(self, self.link, 'tv', device)
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        nrm = ((0.48235, 0.45882, 0.40784), (1 / 255)) # (123, 117, 104)
        # [123.675, 116.28, 103.53]
        x, sz_orig, sz_used = prep.full(imgs, dv, 300, 'torch', keep_ratio=False, norm=nrm, size_divisible=1)
        return x
        xs = self.backbone(x)
        return xs

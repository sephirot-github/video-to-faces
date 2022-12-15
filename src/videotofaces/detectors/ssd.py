import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from .operations import prep, post
from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV2
from ..utils.weights import load_weights

# SSD paper: https://arxiv.org/pdf/1512.02325.pdf
# VGG paper: https://arxiv.org/pdf/1409.1556.pdf


class VGG16(nn.Module):
    
    def __init__(self):
        super().__init__()
        # configuration D from the paper
        cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        layers = []
        cin = 3
        for c, n in cfg:
            for i in range(n):
                layers.append(ConvUnit(cin, c, 3, 1, 1, 'relu', bn=None))
                cin = c
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)
    
    
class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = VGG16()
        


class SSD(nn.Module):

    link = 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth'

    def __init__(self, pretrained=True, device='cpu'):
        super().__init__()
        backbone = VGG16()
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
import torch
import torch.nn as nn

from .basic import ConvUnit

# V1: https://arxiv.org/abs/1704.04861
# V3: https://arxiv.org/pdf/1905.02244


class DepthwiseBlock(nn.Module):

    def __init__(self, cin, cout, stride, relu_type, bn_eps):
        super(DepthwiseBlock, self).__init__()
        self.a = ConvUnit(cin, cin, 3, stride, 1, relu_type, bn_eps, grp=cin)
        self.b = ConvUnit(cin, cout, 1, 1, 0, relu_type, bn_eps)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x
        

class MobileNetV1(nn.Module):

    def __init__(self, width_multiplier, relu_type, bn_eps=1e-05, return_inter=None):
        super(MobileNetV1, self).__init__()
        a = width_multiplier
        self.layers = nn.Sequential(
            ConvUnit(3, 32*a, 3, 2, 1, relu_type, bn_eps),
            DepthwiseBlock(32*a, 64*a, 1, relu_type, bn_eps),
            DepthwiseBlock(64*a, 128*a, 2, relu_type, bn_eps),
            DepthwiseBlock(128*a, 128*a, 1, relu_type, bn_eps),
            DepthwiseBlock(128*a, 256*a, 2, relu_type, bn_eps),
            DepthwiseBlock(256*a, 256*a, 1, relu_type, bn_eps),
            DepthwiseBlock(256*a, 512*a, 2, relu_type, bn_eps),
            *[DepthwiseBlock(512*a, 512*a, 1, relu_type, bn_eps) for _ in range(5)],
            DepthwiseBlock(512*a, 1024*a, 2, relu_type, bn_eps),
            DepthwiseBlock(1024*a, 1024*a, 1, relu_type, bn_eps)
        )
        self.return_inter = return_inter

    def forward(self, x):
        if not self.return_inter:
            return self.layers(x)
        xs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.return_inter:
                xs.append(x)
        xs.append(x)
        return xs


class SqueezeExcitation(nn.Module):

    def __init__(self, cin, csq):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(cin, csq, 1)
        self.fc2 = torch.nn.Conv2d(csq, cin, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x):
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.activation(s)
        s = self.fc2(s)
        s = self.scale_activation(s)
        return x * s


def make_divisable(v, divisor, min_value):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):

    def __init__(self, cin, cmid, cout, k, s, use_se, activ='plain'):
        super().__init__()
        layers = []
        if cin != cmid:
            layers.append(ConvUnit(cin, cmid, 1, 1, 0, activ))
        layers.append(ConvUnit(cmid, cmid, k, s, (k - 1) // 2, activ, grp=cmid))
        if use_se:
            csq = make_divisable(cmid // 4, divisor=8)
            layers.append(SqueezeExcitation(cmid, csq))
        layers.append(ConvUnit(cmid, cout, 1, 1, 0, None))
        
        self.block = nn.Sequential(*layers)
        self.residual = s == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            y += x
        return y


class MobileNetV3L(nn.Module):

    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

    def __init__(self, return_inter=None):
        super().__init__()
        self.layers = nn.Sequential(
            ConvUnit(3, 16, 3, 2, 1, 'hardswish'),
            InvertedResidual(16, 16, 16, 3, 1, False),
            InvertedResidual(16, 64, 24, 3, 2, False), # C1
            InvertedResidual(24, 72, 24, 3, 1, False),
            InvertedResidual(24, 72, 40, 5, 2, True), # C2
            InvertedResidual(40, 120, 40, 5, 1, True),
            InvertedResidual(40, 120, 40, 5, 1, True),
            InvertedResidual(40, 240, 80, 3, 2, False, 'hardswish'), # C3
            InvertedResidual(80, 200, 80, 3, 1, False, 'hardswish'),
            InvertedResidual(80, 184, 80, 3, 1, False, 'hardswish'),
            InvertedResidual(80, 184, 80, 3, 1, False, 'hardswish'),
            InvertedResidual(80, 480, 112, 3, 1, True, 'hardswish'),
            InvertedResidual(112, 672, 112, 3, 1, True, 'hardswish'),
            InvertedResidual(112, 672, 160, 5, 2, True, 'hardswish'), # C4
            InvertedResidual(160, 960, 160, 5, 1, True, 'hardswish'),
            InvertedResidual(160, 960, 160, 5, 1, True, 'hardswish'),
            ConvUnit(160, 6*160, 1, 1, 0, 'hardswish')
        )
        self.return_inter = return_inter
    
    def forward(self, x):
        if not self.return_inter:
            return self.layers(x)
        xs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.return_inter:
                xs.append(x)
        xs.append(x)
        return xs
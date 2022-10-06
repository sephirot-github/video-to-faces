import torch
import torch.nn as nn

from .basic import ConvUnit

# https://arxiv.org/abs/1704.04861

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
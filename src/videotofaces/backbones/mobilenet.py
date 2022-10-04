import torch
import torch.nn as nn

# https://arxiv.org/abs/1704.04861


class ConvUnit(nn.Module):

    def __init__(self, cin, cout, k, s, p, grp=1):
        super(ConvUnit, self).__init__()
        cin, cout, grp = int(cin), int(cout), int(grp)
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=grp, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=0.001)
        self.relu = nn.PReLU(cout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseBlock(nn.Module):

    def __init__(self, cin, cout, stride):
        super(DepthwiseBlock, self).__init__()
        self.aaa = ConvUnit(cin, cin, 3, stride, 1, grp=cin)
        self.bbb = ConvUnit(cin, cout, 1, 1, 0)

    def forward(self, x):
        x = self.aaa(x)
        x = self.bbb(x)
        return x
        

class MobileNetV1(nn.Module):

    def __init__(self, width_multiplier=1):
        super(MobileNetV1, self).__init__()
        a = width_multiplier
        self.layers = nn.Sequential(
            ConvUnit(3, 32*a, 3, 2, 1),
            DepthwiseBlock(32*a, 64*a, 1),
            DepthwiseBlock(64*a, 128*a, 2),
            DepthwiseBlock(128*a, 128*a, 1),
            DepthwiseBlock(128*a, 256*a, 2),
            DepthwiseBlock(256*a, 256*a, 1),
            DepthwiseBlock(256*a, 512*a, 2),
            *[DepthwiseBlock(512*a, 512*a, 1) for _ in range(5)],
            DepthwiseBlock(512*a, 1024*a, 2),
            DepthwiseBlock(1024*a, 1024*a, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
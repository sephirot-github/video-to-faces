import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, cin, cout, k, s, p, relu_type, bn=1e-05, grp=1):
        super().__init__()

        cin, cout, grp = int(cin), int(cout), int(grp)
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=grp, bias=bn is None)
        
        if bn == None:
            self.bn = None
        else:
            self.bn = nn.BatchNorm2d(cout, eps=bn)

        if relu_type == None:
            self.relu = None
        elif relu_type == 'plain':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(cout)
        elif relu_type.startswith('lrelu'):
            leak = float(relu_type.split('_')[1])
            self.relu = nn.LeakyReLU(leak, inplace=True)
        elif relu_type == 'hardswish':
            self.relu = nn.Hardswish()


    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x
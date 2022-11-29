import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, cin, cout, k, s, p, activ, bn=1e-05, grp=1):
        super().__init__()

        cin, cout, grp = int(cin), int(cout), int(grp)
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=grp, bias=bn is None)
        
        if bn == None:
            self.bn = None
        else:
            self.bn = nn.BatchNorm2d(cout, eps=bn)

        if activ == None:
            self.activ = None
        elif activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'prelu':
            self.activ = nn.PReLU(cout)
        elif activ.startswith('lrelu'):
            leak = float(activ.split('_')[1])
            self.activ = nn.LeakyReLU(leak, inplace=True)
        elif activ == 'hardswish':
            self.activ = nn.Hardswish()


    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activ:
            x = self.activ(x)
        return x
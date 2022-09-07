import torch
from torch import nn
import torch.nn.functional as F

# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
# https://arxiv.org/abs/2004.04989

class IRNBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(IRNBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.prelu = nn.PReLU(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = x if not hasattr(self, 'downsample') else self.downsample(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return x + y

class IResNet(nn.Module):
    def __init__(self, layers):
        super(IResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = nn.Sequential(*([IRNBlock(64, 64, 2)] + [IRNBlock(64, 64) for i in range(1, layers[0])]))
        self.layer2 = nn.Sequential(*([IRNBlock(64, 128, 2)] + [IRNBlock(128, 128) for i in range(1, layers[1])]))
        self.layer3 = nn.Sequential(*([IRNBlock(128, 256, 2)] + [IRNBlock(256, 256) for i in range(1, layers[2])]))
        self.layer4 = nn.Sequential(*([IRNBlock(256, 512, 2)] + [IRNBlock(512, 512) for i in range(1, layers[3])]))
        self.bn2 = nn.BatchNorm2d(512)
        #self.dropout = nn.Dropout(p=0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)
                
        #nn.init.constant_(self.features.weight, 1.0)
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.normal_(m.weight, 0, 0.1)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        #x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def iresnet18(): return IResNet([2, 2, 2, 2])
def iresnet34(): return IResNet([3, 4, 6, 3])
def iresnet50(): return IResNet([3, 4, 14, 3])
def iresnet100(): return IResNet([3, 13, 30, 3])
def iresnet200(): return IResNet([6, 26, 60, 6])

def iresnet_irl_encoder(device, architecture, weights_file):
    if architecture == 18: model = iresnet18().to(device)
    if architecture == 34: model = iresnet34().to(device)
    if architecture == 50: model = iresnet50().to(device)
    if architecture == 100: model = iresnet100().to(device)
    weights = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(weights)
    model.eval()
    return model
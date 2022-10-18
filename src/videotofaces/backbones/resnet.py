import torch
import torch.nn as nn

# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


class Bottleneck(nn.Module):

    def __init__(self, cin, width, stride=1):
        super().__init__()
        w = width
        cout = w * 4
        self.conv1 = nn.Conv2d(cin, w, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.conv2 = nn.Conv2d(w, w, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        self.conv3 = nn.Conv2d(w, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        y = x if not hasattr(self, 'downsample') else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)) + y)
        return x


class ResNet50(nn.Module):

    def __init__(self, layers=[3, 4, 6, 3]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*([Bottleneck(64, 64)] + [Bottleneck(256, 64) for i in range(1, layers[0])]))
        self.layer2 = nn.Sequential(*([Bottleneck(256, 128, 2)] + [Bottleneck(512, 128) for i in range(1, layers[1])]))
        self.layer3 = nn.Sequential(*([Bottleneck(512, 256, 2)] + [Bottleneck(1024, 256) for i in range(1, layers[2])]))
        self.layer4 = nn.Sequential(*([Bottleneck(1024, 512, 2)] + [Bottleneck(2048, 512) for i in range(1, layers[3])]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y1 = self.layer2(x)
        y2 = self.layer3(y1)
        y3 = self.layer4(y2)
        return [y1, y2, y3]
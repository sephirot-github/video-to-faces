import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import prep_weights_file

# taken from:  https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
# explanation: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
# paper:       https://arxiv.org/pdf/1602.07261.pdf


class ConvPlus(nn.Module):

    def __init__(self, cin, cout, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = ConvPlus(256, 32, 1)
        self.branch1 = nn.Sequential(ConvPlus(256, 32, 1), ConvPlus(32, 32, 3, p=1))
        self.branch2 = nn.Sequential(ConvPlus(256, 32, 1), ConvPlus(32, 32, 3, p=1), ConvPlus(32, 32, 3, p=1))
        self.conv2d = nn.Conv2d(96, 256, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out
    

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = ConvPlus(896, 128, 1)
        self.branch1 = nn.Sequential(
            ConvPlus(896, 128, 1),
            ConvPlus(128, 128, (1,7), p=(0,3)),
            ConvPlus(128, 128, (7,1), p=(3,0))
        )
        self.conv2d = nn.Conv2d(256, 896, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, relu=True):
        super().__init__()
        self.scale = scale
        self.relu = relu
        self.branch0 = ConvPlus(1792, 192, 1)
        self.branch1 = nn.Sequential(
            ConvPlus(1792, 192, 1),
            ConvPlus(192, 192, (1,3), p=(0,1)),
            ConvPlus(192, 192, (3,1), p=(1,0))
        )
        self.conv2d = nn.Conv2d(384, 1792, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu:
            out = F.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = ConvPlus(256, 384, 3, s=2)
        self.branch1 = nn.Sequential(ConvPlus(256, 192, 1), ConvPlus(192, 192, 3, p=1), ConvPlus(192, 256, 3, s=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(ConvPlus(896, 256, 1), ConvPlus(256, 384, 3, s=2))
        self.branch1 = nn.Sequential(ConvPlus(896, 256, 1), ConvPlus(256, 256, 3, s=2))
        self.branch2 = nn.Sequential(ConvPlus(896, 256, 1), ConvPlus(256, 256, 3, p=1), ConvPlus(256, 256, 3, s=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
        

class InceptionResnetV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d_1a = ConvPlus(3, 32, 3, s=2)
        self.conv2d_2a = ConvPlus(32, 32, 3)
        self.conv2d_2b = ConvPlus(32, 64, 3, p=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = ConvPlus(64, 80, 1)
        self.conv2d_4a = ConvPlus(80, 192, 3)
        self.conv2d_4b = ConvPlus(192, 256, 3, s=2)
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for i in range(5)])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.1) for i in range(10)])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.2) for i in range(5)])
        self.block8 = Block8(relu=False)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, 0.001)

    def forward(self, x):
        # Stem
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
                
        x = self.repeat_1(x) # Inception block A x 5
        x = self.mixed_6a(x) # Reduction block A
        x = self.repeat_2(x) # Inception block B x 10
        x = self.mixed_7a(x) # Reduction block B
        x = self.repeat_3(x) # Inception block C x 5
        x = self.block8(x)

        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x


class IncepResEncoder():

    def __init__(self, device, dataset='vggface2'):
        """TBD"""
        print('Initializing Inception-Resnet V1 model for face feature extraction')
        if dataset == 'casia-webface':
            wf = prep_weights_file('https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt', '20180408-102900-casia-webface.pt')
        else:
            wf = prep_weights_file('https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt', '20180402-114759-vggface2.pt')
        weights = torch.load(wf, map_location=torch.device(device))
        weights.pop('logits.weight')
        weights.pop('logits.bias')
        self.model = InceptionResnetV1().to(device)
        self.model.load_state_dict(weights)
        self.model.eval()
        print()
    
    def __call__(self, images):
        """TBD"""
        inp = cv2.dnn.blobFromImages(images, 1 / 128, (160, 160), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()
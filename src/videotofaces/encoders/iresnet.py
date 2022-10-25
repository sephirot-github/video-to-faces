import cv2
import torch
from torch import nn
import torch.nn.functional as F

#from ..detectors.coord_reg import CoordRegLandmarker
from ..detectors.mtcnn import MTCNNLandmarker
from ..utils import face_align
from ..utils import prep_weights_file

# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
# https://arxiv.org/abs/2004.04989


class IRNBlock(nn.Module):

    def __init__(self, cin, cout, stride=1):
        """TBD"""
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
        """TBD"""
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
        """TBD"""
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
        """TBD"""
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


class IResNetEncoder():

    architecture = {
        'IResNet18': [2, 2, 2, 2],
        'IResNet34': [3, 4, 6, 3],
        'IResNet50': [3, 4, 14, 3],
        'IResNet100': [3, 13, 30, 3],
        'IResNet200': [6, 26, 60, 6]
    }
    options = {
        'glint360k_r18': ('IResNet18', '1-fmCdluzOGEMUFBQeIv_HJiVwAuosrZF', 'glint360k_cosface_r18_fp16_0.1_backbone.pth'),
        'glint360k_r34': ('IResNet34', '16sPIRJGgof6WoERFqMVg9llID6mSMoym', 'glint360k_cosface_r34_fp16_0.1_backbone.pth'),
        'glint360k_r50': ('IResNet50', '1UYIZkHTklpFMGLhFGzzCvUZS-rMLY4Xv', 'glint360k_cosface_r50_fp16_0.1_backbone.pth'),
        'glint360k_r100': ('IResNet100', '1nwZhK33-5KwE8nyKWlBr8zDm0Tx3PYQ9', 'glint360k_cosface_r100_fp16_0.1_backbone.pth'),
        'glint360k_r50_pfc': ('IResNet50', '1eBTfk0Ozsx0hF0l1z06mlzC6mOLxTscK', 'partial_fc_glint360k_r50_insightface.pth'),
        'glint360k_r100_pfc': ('IResNet100', '1XNMRpB0MydK1stiljHoe4vKz4WfNCAIG', 'partial_fc_glint360k_r100_insightface.pth')
    }

    def __init__(self, device, source, align=True, landmarker='mobilenet', tform='similarity'):
        """https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#5-pretrain-models
        https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo"""
        opt = self.options[source]
        print('Initializing %s model for face feature extraction' % opt[0])
        if align:
            self.landmarker = CoordRegLandmarker(device) if landmarker == 'mobilenet' else MTCNNLandmarker(device)
            self.tform = tform
        wf = prep_weights_file('https://drive.google.com/uc?id=%s' % opt[1], opt[2], gdrive=True)
        wd = torch.load(wf, map_location=torch.device(device))
        self.model = IResNet(self.architecture[opt[0]]).to(device)
        self.model.load_state_dict(wd)
        self.model.eval()
        print()

    def __call__(self, images):
        if hasattr(self, 'landmarker'):
            images = [cv2.resize(img, (192, 192)) for img in images]
            lm = self.landmarker(images)
            images = face_align(images, lm, self.tform)
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()


# ==================== EXTRA ====================

class ONNXIResNetEncoder():
    # !pip install onnxruntime
    # !pip install onnxruntime-gpu
    
    def __init__(self, device):
        import onnxruntime
        modelfile = prep_weights_file('https://drive.google.com/uc?id=1AhyD9Zjwy5MZgJIjj2Pb-YfIdeFS3T5E', 'w600k_r50.onnx', gdrive=True)
        self.landmarker = CoordRegLandmarker(device)
        self.session = onnxruntime.InferenceSession(modelfile, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.inpname = self.session.get_inputs()[0].name

    def __call__(self, images):
        import numpy as np
        images = [cv2.resize(img, (192, 192)) for img in images]
        lm = self.landmarker(images)
        images = face_align(images, lm, 'similarity')
        images = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        x = self.session.run(None, {self.inpname: images})[0]
        x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
        return x
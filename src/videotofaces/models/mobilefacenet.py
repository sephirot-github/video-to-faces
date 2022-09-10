import cv2
import torch
import torch.nn as nn

from .mobilenet import MobileNetLandmarker
from .mtcnn import MTCNNLandmarker
from ..utils import face_align
from ..utils import prep_weights_file

# combined from:
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/mobilefacenet.py
# https://github.com/foamliu/MobileFaceNet/blob/master/mobilefacenet.py
# https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/blob/master/face_model.py

# MobileFaceNet paper: https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf
# MobileNetV2 paper:   https://arxiv.org/pdf/1801.04381.pdf

USE_CBIAS = True
USE_BN = True
USE_PRELU = True

class ConvUnit(nn.Module):
    def __init__(self, cin, cout, k, s, p, grp=1, relu=True, cbias_override=None, relu6=True):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=grp, bias=USE_CBIAS if cbias_override is None else cbias_override)
        if USE_BN: self.bn = nn.BatchNorm2d(cout)
        if relu: self.relu = nn.PReLU(cout) if USE_PRELU else (nn.ReLU6(cout) if relu6 else nn.ReLU())
    def forward(self, x):
        x = self.conv(x)
        x = x if not hasattr(self, 'bn') else self.bn(x)
        x = x if not hasattr(self, 'relu') else self.relu(x)
        return x

class MBlock(nn.Module):
    # aka bottleneck aka inverted residual block from MobileNetV2 paper
    def __init__(self, cin, cout, grp, s=1, residual=False):
        super(MBlock, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvUnit(cin, grp, 1, 1, 0),
            ConvUnit(grp, grp, 3, s, 1, grp=grp),
            ConvUnit(grp, cout, 1, 1, 0, relu=False)
        )
    def forward(self, x):
        y = self.layers(x)
        if not self.residual: return y
        return x + y

class MBlockRepeat(nn.Module):
    def __init__(self, c, grp, rep):
        super(MBlockRepeat, self).__init__()
        self.repeats = nn.Sequential(*[MBlock(c, c, grp, residual=True) for _ in range(rep)])
    def forward(self, x):
        return self.repeats(x)

class MobileFaceNet(nn.Module):
    def __init__(self, c=[64, 64, 128], intro_pw=False, outro_gd=True, outro_pw=128, outro_pw_relu=False, outro_lin=None, lin_bias=True, emb_size=128):
        super(MobileFaceNet, self).__init__()
        self.intro = nn.Sequential(
            ConvUnit(3, c[0], 3, 2, 1, relu6=False),
            ConvUnit(c[0], c[0], 3, 1, 1, grp=64, relu6=False),
            ConvUnit(c[0], c[0], 1, 1, 0) if intro_pw else nn.Identity())
        self.main = nn.Sequential(
            # effectively the same as bottlenecks from Table 1 from MobileFaceNet paper
            MBlock(c[0], c[1], grp=128, s=2),
            MBlockRepeat(c[1], grp=128, rep=4),
            MBlock(c[1], c[2], grp=256, s=2),
            MBlockRepeat(c[2], grp=256, rep=6),
            MBlock(c[2], c[2], grp=512, s=2),
            MBlockRepeat(c[2], grp=256, rep=2))
        self.outro = nn.Sequential(
            ConvUnit(c[2], 512, 1, 1, 0),
            ConvUnit(512, 512, 7, 1, 0, grp=512, relu=False) if outro_gd else nn.Identity(),
            ConvUnit(512, outro_pw, 1, 1, 0, cbias_override=True, relu=outro_pw_relu) if outro_pw else nn.Identity(),
            nn.Flatten(),
            nn.Linear(outro_lin, emb_size, bias=lin_bias) if outro_lin else nn.Identity(),
            nn.BatchNorm1d(emb_size) if outro_lin else nn.Identity())
    def forward(self, x):   # [n, 3, 112, 112]
        x = self.intro(x)   # [n, c[0], 56, 56]
        x = self.main(x)    # [n, c[2], 7, 7]
        x = self.outro(x)   # [n, emb_size]
        return x

def get_mbf_pretrained(device, src='insightface'):
    print('Initializing MobileFaceNet model for face feature extraction')
    global USE_CBIAS, USE_BN, USE_PRELU
    if src == 'insightface':
        USE_CBIAS, USE_BN, USE_PRELU = True, False, True
        model = MobileFaceNet([128, 128, 256], outro_gd=False, outro_pw=64, outro_pw_relu=True, outro_lin=3136, emb_size=512).to(device)
        wf = prep_weights_file('https://drive.google.com/uc?id=1fi8jBWJlZYjGNarvcQVPD9q449zA_sLo', 'mbfnet_w600k_insightface.pt', gdrive=True)
        wd = torch.load(wf, map_location=torch.device(device))
    elif src == 'foamliu':
        USE_CBIAS, USE_BN, USE_PRELU = False, True, False
        model = MobileFaceNet(intro_pw=True).to(device)
        wf = prep_weights_file('https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt', 'mbfnet_ms_celeb_1m_by_foamliu.pt')
        wd = torch.load(wf, map_location=torch.device(device))
        wd = _adjust_weights_names(wd, model, src)
    elif src == 'xuexingyu24':
        USE_CBIAS, USE_BN, USE_PRELU = False, True, True
        model = MobileFaceNet(outro_pw=None, outro_lin=512, lin_bias=False, emb_size=512).to(device)
        wf = prep_weights_file('https://raw.githubusercontent.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/master/Weights/MobileFace_Net', 'mbfnet_ms_celeb_1m_by_xuexingyu24')
        wd = torch.load(wf, map_location=torch.device(device))
        wd = _adjust_weights_names(wd, model, src)
    model.load_state_dict(wd)
    model.eval()
    print()
    return model

def _adjust_weights_names(source, model, src):
    names = list(source)
    if src == 'foamliu':
        names.insert(12, names.pop(7))
        for _ in range(19): names.append(names.pop(18))
    result = {}
    for i, w in enumerate(list(model.state_dict())):
        result[w] = source[names[i]]
    return result
    
class MobileFaceNetEncoder():
    def __init__(self, device, src='insightface', align=True, landmarker='mobilenet'):
        if align:
            self.landmarker = MobileNetLandmarker(device) if landmarker == 'mobilenet' else MTCNNLandmarker(device)
        self.encoder = get_mbf_pretrained(device, src)
    
    def __call__(self, paths, tform='similarity', square=True, lminp=192, minsize1=None, minsize2=None):
        images = [cv2.imread(p) for p in paths]
        if hasattr(self, 'landmarker'):
            images = [cv2.resize(img, (lminp, lminp)) for img in images]
            lm = self.landmarker(images)
            images = face_align(images, lm, tform, square)
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp)
        with torch.no_grad():
            return self.encoder(inp).cpu().numpy()

# ==================== EXTRA / NOTES ====================

def convert_weights_mbfnet_insightface():
    '''this is the code I used to convert Insightface's ONNX model's weights to pytorch
       import it from here and call directly if needed
       it should create the same 'mbfnet_w600k_insightface.pt' file that's used above
       need "pip install onnx" to work
       for browsing onnx model, I used https://netron.app mentioned here: https://github.com/onnx/onnx/issues/1425'''
    import shutil, os
    import os.path as osp
    home = os.getcwd()
    dst = prep_weights_file('https://drive.google.com/uc?id=19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG', 'buffalo_sc.zip', gdrive=True)
    os.chdir(osp.dirname(dst))
    print('working at: ' + os.getcwd())
    shutil.unpack_archive('buffalo_sc.zip')
    os.rename('buffalo_sc/w600k_mbf.onnx', 'w600k_mbf.onnx')
    os.remove('buffalo_sc/det_500m.onnx')
    os.remove('buffalo_sc.zip')
    os.rmdir('buffalo_sc')
    print('prepared w600k_mbf.onnx')
        
    import onnx, onnx.numpy_helper
    onnx_model = onnx.load('w600k_mbf.onnx')
    source = dict([(raw.name, onnx.numpy_helper.to_array(raw)) for raw in onnx_model.graph.initializer])
    #for s in SO: print(s, '\t', SO[s].shape)
    match = [
        518, 519, 664, 521, 522, 665,
    
        524, 525, 666, 527, 528, 667, 530, 531,
        533, 534, 668, 536, 537, 669, 539, 540,
        542, 543, 670, 545, 546, 671, 548, 549,
        551, 552, 672, 554, 555, 673, 557, 558,
        560, 561, 674, 563, 564, 675, 566, 567,

        569, 570, 676, 572, 573, 677, 575, 576,
        578, 579, 678, 581, 582, 679, 584, 585,
        587, 588, 680, 590, 591, 681, 593, 594,
        596, 597, 682, 599, 600, 683, 602, 603,
        605, 606, 684, 608, 609, 685, 611, 612,
        614, 615, 686, 617, 618, 687, 620, 621,
        623, 624, 688, 626, 627, 689, 629, 630,

        632, 633, 690, 635, 636, 691, 638, 639,
        641, 642, 692, 644, 645, 693, 647, 648,
        650, 651, 694, 653, 654, 695, 656, 657,

        659, 660, 696, 662, 663, 697,
        'fc.weight', 'fc.bias', 'features.weight', 'features.bias', 'features.running_mean', 'features.running_var'
    ]
    global USE_CBIAS, USE_BN, USE_PRELU
    USE_CBIAS, USE_BN, USE_PRELU = True, False, True
    model = MobileFaceNet([128, 128, 256], outro_gd=False, outro_pw=64, outro_pw_relu=True, outro_lin=3136, emb_size=512)
    dst = model.state_dict()
    nbt_name = list(dst)[-1]
    nbt = dst.pop(nbt_name)
    for i, s in enumerate(dst):
        val = source[str(match[i])]
        if dst[s].dim() == 1:
            val = val.squeeze()
        dst[s] = torch.Tensor(val.copy())
    dst[nbt_name] = torch.Tensor(nbt)
    model.load_state_dict(dst)
    model.eval()
    torch.save(model.state_dict(), 'mobilefacenet_w600k_insightface.pt')
    print('saved mobilefacenet_w600k_insightface.pt')
    os.remove('w600k_mbf.onnx')
    print('removed w600k_mbf.onnx')
    os.chdir(home)
    print('returned to: ' + home)
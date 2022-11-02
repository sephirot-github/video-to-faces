import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..backbones.resnet import ResNet50, ResNet152
from ..utils.download import prep_weights_gdrive, prep_weights_file
from ..utils.bbox import get_priors, decode_boxes, select_boxes 

# Source 1: https://github.com/biubug6/Pytorch_Retinaface
# Source 2: https://github.com/barisbatuhan/FaceDetector
# Paper: https://arxiv.org/pdf/1905.00641.pdf


class FPN(nn.Module):
    """
    FPN paper (section 3 and figure 3) https://arxiv.org/pdf/1612.03144.pdf
    RetinaNet paper (page 4 footnote 2) https://arxiv.org/pdf/1708.02002.pdf
    RetinaFace paper (start of section 4.2) https://arxiv.org/pdf/1905.00641.pdf

    example: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
    tvision: https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py

    ResNet outputs: C2, C3, C4, C5 ({4, 8, 16, 32} stride w.r.t. the input image)
        
    Ti = lateral(Ci) [i=2..5]
    P5 = T5
    P4 = T4 + upsample(P5)
    P3 = T3 + upsample(P4)
    P2 = T2 + upsample(P2)
    Pi = smooth(Pi) [i=2..4] (or 5 too)

    P6 = extra1(C5) [or P5]
    P7 = extra2(relu(P6))

    smoothBeforeMerge is a mistake: https://github.com/biubug6/Pytorch_Retinaface/blob/master/models/net.py
    from the paper: "Finally, we append a 3×3 convolution on each merged map"

    smoothP5 is probably a mistake too: https://github.com/barisbatuhan/FaceDetector/blob/main/BBTNet/components/fpn.py
    but both torchvision and detectron2 implementations seem to be using it
    from the paper, same paragraph: "which is to reduce the aliasing effect of upsampling" (but P5 have no upsampling)
    """

    def __init__(self, cins, cout, relu, bn=1e-05, P6=None, P7=None, smoothP5=False, smoothBeforeMerge=False):
        super().__init__()
        assert P6 in ['fromC5', 'fromP5', None]
        self.P6 = P6
        self.P7 = P7
        self.smoothP5 = smoothP5
        self.smoothBeforeMerge = smoothBeforeMerge
        smooth_n = len(cins) - (0 if smoothP5 else 1)
        self.conv_laterals = nn.ModuleList([ConvUnit(cin, cout, 1, 1, 0, relu, bn) for cin in cins])
        self.conv_smooths = nn.ModuleList([ConvUnit(cout, cout, 3, 1, 1, relu, bn) for _ in range(smooth_n)])
        if P6:
            cin6 = cins[-1] if P6 == 'fromC5' else cout
            self.conv_extra1 = ConvUnit(cin6, cout, 3, 2, 1, relu, bn)
        if P7:
            self.conv_extra2 = ConvUnit(cout, cout, 3, 2, 1, relu, bn)

    def forward(self, C):
        n = len(C)
        P = [self.conv_laterals[i](C[i]) for i in range(n)]
        
        if not self.smoothBeforeMerge:
            for i in range(0, n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
            P = [self.conv_smooths[i](P[i]) for i in range(len(self.conv_smooths))]
        else:
            # no variation with smoothP5 = smoothBeforeMerge = True found so far
            #if self.smoothP5: P[-1] = self.conv_smooths[-1](P[-1])
            for i in range(0, n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
                P[i] = self.conv_smooths[i](P[i])
        
        if self.P6:
            P.append(self.conv_extra1(C[-1] if self.P6 == 'fromC5' else P[-1]))
        if self.P7:
            P.append(self.conv_extra2(F.relu(P[-1])))

        return P


class SSH(nn.Module):

    def __init__(self, cin, cout, relu):
        super().__init__()
        self.conv1 = ConvUnit(cin, cout//2, 3, 1, 1, relu_type=None)
        self.conv2 = ConvUnit(cin, cout//4, 3, 1, 1, relu_type=relu)
        self.conv3 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=None)
        self.conv4 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=relu)
        self.conv5 = ConvUnit(cout//4, cout//4, 3, 1, 1, relu_type=None)

    def forward(self, x):
        y1 = self.conv1(x)
        t = self.conv2(x)
        y2 = self.conv3(t)
        y3 = self.conv5(self.conv4(t))
        out = torch.cat([y1, y2, y3], dim=1)
        out = F.relu(out)
        return out


class Head(nn.Module):

    def __init__(self, cin, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        self.conv = nn.Conv2d(cin, num_anchors * task_len, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


def preprocess(imgs, params, device):
    to0_1, swapRB, mean, std = params
    x = np.stack(imgs)
    x = torch.from_numpy(x).to(device, torch.float32)
    if to0_1: x /= 255
    x = x if not swapRB else x[:, :, :, [2, 1, 0]]
    x -= torch.tensor(list(mean), device=x.device)
    x /= torch.tensor(list(std), device=x.device)
    x = x.permute(0, 3, 1, 2)
    return x


class RetinaFace(nn.Module):

    def __init__(self, backbone, cins, cout, relu, bn, P6, smoothP5, smoothBefore,
                 to0_1, swapRB, mean, std, bases, score_thr, predict_landmarks):
        super().__init__()
        self.prep_params = (to0_1, swapRB, mean, std)
        self.bases = bases
        self.score_thr = score_thr
        num_anchors = len(bases[0][1])
        num_levels = len(cins) + (1 if P6 else 0)

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu, 1e-05, P6, None, smoothP5, smoothBefore)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in range(num_levels)])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(num_levels)])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(num_levels)])
        if predict_landmarks:
            self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in range(num_levels)])

    def forward(self, imgs):
        x = preprocess(imgs, self.prep_params, next(self.parameters()).device)
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        box_reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        classif = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
        if hasattr(self, 'heads_ldmks'):
            ldm_reg = torch.cat([self.heads_ldmks[i](xs[i]) for i in range(len(xs))], dim=1)
     
        scores = F.softmax(classif, dim=-1)[:, :, 1]
        priors = get_priors(x.shape[2:], self.bases).to(x.device)
        boxes = decode_boxes(box_reg, priors)
        l = select_boxes(boxes, scores, score_thr=self.score_thr, iou_thr=0.4, impl='tvis_batched')
        return l


class RetinaNet(nn.Module):

    def __init__(self):
        super(RetinaNet, self).__init__()
        backbone = ResNet50(return_count=3, bn_eps=0.0)
        cins, cout = [512, 1024, 2048], 256
        relu, bn = None, None
        P6, P7, smoothP5 = 'fromP5', True, True

        bases = [
                (8,   [(45, 23), (57, 28), (71, 35),          (32, 32), (40, 40), (50, 50),          (23, 45), (28, 57), (35, 71)]),
                (16,  [(91, 45), (113, 57), (143, 71),        (64, 64), (80, 80), (101, 101),        (45, 91), (57, 113), (71, 143)]),
                (32,  [(181, 91), (228, 114), (287, 144),     (128, 128), (161, 161), (203, 203),    (91, 181), (114, 228), (144, 287)]),
                (64,  [(362, 181), (455, 228), (574, 287),    (256, 256), (322, 322), (406, 406),    (181, 362), (228, 455), (287, 574)]),
                (128, [(724, 362), (912, 456), (1148, 574),   (512, 512), (645, 645), (812, 812),    (362, 724), (456, 912), (574, 1148)])
            ]
        #anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        #aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        #areas = (32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812)
        #ratios = (0.5, 1.0, 2.0)

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu, bn, P6, P7, smoothP5)
        


class RetinaFaceDetector():

    gids = {
        'biubug6_mobilenet': '15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1',
        'biubug6_resnet50': '14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW',
        'bbt_resnet152_mixed': '1xB5RO99bVnXLYesnilzaZL2KWz4BsJfM',
        'bbt_resnet50_mixed': '1uraA7ZdCCmos0QSVR6CJgg0aSLtV4q4m',
        'bbt_resnet50_wider': '1pQLydyUUEwpEf06ElR2fw8_x2-P9RImT',
        'bbt_resnet50_icartoon': '12RsVC1QulqsSlsCleMkIYMHsAEwMyCw8'
    }

    def __init__(self, source, device=None):
        assert source in self.gids
        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        weights_filename = 'retinaface_%s.pth' % source
        print('Initializing RetinaFace model (%s) for face detection' % source)

        bn = 1e-05
        
        if source.startswith('biubug6_'):
            bases = [
                (8, [16, 32]),
                (16, [64, 128]),
                (32, [256, 512])
            ]
            score_thr, predict_landmarks = 0.02, True
            to0_1, swapRB, mean, std = False, False, (104, 117, 123), (1, 1, 1) # mean values from ImageNet in GRB
            P6, smoothP5, smoothBefore = None, False, True
            if source == 'biubug6_mobilenet':
                backbone = MobileNetV1(0.25, relu_type='lrelu_0.1', return_inter=[5, 11])
                cins, cout, relu = [64, 128, 256], 64, 'lrelu_0.1'
            elif source == 'biubug6_resnet50':
                backbone = ResNet50(return_count=3)
                cins, cout, relu = [512, 1024, 2048], 256, 'plain'
            wf = prep_weights_gdrive(self.gids[source], weights_filename)
            wd_src = torch.load(wf, map_location=torch.device(device))

        elif source.startswith('bbt_'):
            bases = [
                # scales: x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)
                (4, [16, 20.16, 25.40]),
                (8, [32, 40.32, 50.80]),
                (16, [64, 80.63, 101.59]),
                (32, [128, 161.26, 203.19]),
                (64, [256, 322.54, 406.37])
            ]
            score_thr, predict_landmarks = 0.5, False
            to0_1, swapRB, mean, std = False, True, (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
            #to0_1, swapRB, mean, std = True, True, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            cins, cout, relu = [256, 512, 1024, 2048], 256, 'plain'
            P6, smoothP5, smoothBefore = 'fromC5', False, True
            backbone = ResNet152() if source == 'bbt_resnet152_mixed' else ResNet50()
            wf = prep_weights_gdrive(self.gids[source], weights_filename)
            wd_src = torch.load(wf, map_location=torch.device(device))
            for s in ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean',
                      'bn.running_var', 'bn.num_batches_tracked']:
                wd_src.pop('fpn.lateral_outs.3.' + s)
            # in this source, FPN extra P6 layer is placed between laterals and smooths, but we need after
            wl = list(wd_src.items())
            idx = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_ins.4')]
            els = [wl.pop(idx[0]) for _ in idx]
            pos = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_outs.')][-1]
            for el in els[::-1]:
                wl.insert(pos + 1, el)
            wd_src = dict(wl)
      
        self.model = RetinaFace(backbone, cins, cout, relu, bn, P6, smoothP5, smoothBefore,
                                to0_1, swapRB, mean, std, bases, score_thr, predict_landmarks).to(device)
        #for w in self.model.state_dict(): print(w, '\t', self.model.state_dict()[w].shape)
        wd_dst = {}
        names = list(wd_src)
        for i, w in enumerate(list(self.model.state_dict())):
            #print(names[i], ' to ', w)
            wd_dst[w] = wd_src[names[i]]
        self.model.load_state_dict(wd_dst)
        self.model.eval()

    def __call__(self, imgs):
        with torch.no_grad():
            res = self.model(imgs)
        return res
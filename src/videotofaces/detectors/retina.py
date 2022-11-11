import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..backbones.resnet import ResNet50, ResNet152
from ..utils.download import prep_weights_gdrive, prep_weights_file
from .operations import prep, post

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

    smoothP5 is probably a mistake, but both torchvision and detectron2 implementations seem to be using it
    from the paper, same paragraph: "which is to reduce the aliasing effect of upsampling" (but P5 have no upsampling)
    """

    def __init__(self, cins, cout, relu, bn=1e-05, P6=None, P7=None,
                 smoothP5=False, smoothBeforeMerge=False, nonCumulative=False):
        super().__init__()
        assert P6 in ['fromC5', 'fromP5', None]
        self.P6 = P6
        self.P7 = P7
        self.smoothBeforeMerge = smoothBeforeMerge
        self.nonCumulative = nonCumulative
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
        
        if self.nonCumulative:
            # mistake from: https://github.com/barisbatuhan/FaceDetector/blob/main/BBTNet/components/fpn.py
            P = [P[i] + F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest') for i in range(len(P) - 1)] + [P[-1]]
            for i in range(len(self.conv_smooths)):
                P[i] = self.conv_smooths[i](P[i])
        elif self.smoothBeforeMerge:
            # mistake from: https://github.com/biubug6/Pytorch_Retinaface/blob/master/models/net.py
            # from the paper: "Finally, we append a 3×3 convolution on each merged map"
            # P5 is never smoothed here (smoothP5 is ignored)
            for i in range(n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
                P[i] = self.conv_smooths[i](P[i])
        else:
            # normal pathway
            for i in range(n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
            for i in range(len(self.conv_smooths)):
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


class HeadShared(nn.Module):

    def __init__(self, c, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        self.conv = nn.Sequential(*[ConvUnit(c, c, 3, 1, 1, relu_type='plain', bn=None) for _ in range(4)])
        self.final = nn.Conv2d(c, num_anchors * task_len, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.final(self.conv(x)).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x
    

class RetinaFace_Biubug6(nn.Module):

    def __init__(self, mobilenet=True):
        super().__init__()
        if mobilenet:
            backbone = MobileNetV1(0.25, relu_type='lrelu_0.1', return_inter=[5, 11])
            cins, cout, relu = [64, 128, 256], 64, 'lrelu_0.1'
        else:
            backbone = ResNet50(return_count=3)
            cins, cout, relu = [512, 1024, 2048], 256, 'plain'

        self.bases = [
            (8, [16, 32]),
            (16, [64, 128]),
            (32, [256, 512])
        ]
        self.bases = list(zip([8, 16, 32], post.make_anchors([16, 64, 256], scales=[1, 2])))
        num_anchors = 2

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu, smoothBeforeMerge=True)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in range(len(cins))])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(len(cins))])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(len(cins))])
        self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in range(len(cins))])

    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = prep.normalize(imgs, dv, [104, 117, 123], stds=None, to0_1=False, toRGB=False)
        x = torch.stack(ts)

        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        box_reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        classif = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
        #ldm_reg = torch.cat([self.heads_ldmks[i](xs[i]) for i in range(len(xs))], dim=1)
        
        scores = F.softmax(classif, dim=-1)[:, :, 1]
        priors = post.get_priors(x.shape[2:], self.bases, dv, loc='center')
        boxes = post.decode_boxes(box_reg, priors, 0.1, 0.2)
        l = post.select_boxes(boxes, scores, score_thr=0.02, iou_thr=0.4, impl='tvis')
        return l

    def get_pretrained_weights(self, dv, source):
        if source.endswith('mobilenet'):
            gid = '15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1'
        else:
            gid = '14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW'
        nm = 'retina_%s.pth' % source
        wf = prep_weights_gdrive(gid, nm)
        wd_src = torch.load(wf, map_location=torch.device(dv))
        wd_dst = {}
        names = list(wd_src)
        for i, w in enumerate(list(self.state_dict())):
            #print(names[i], ' to ', w)
            wd_dst[w] = wd_src[names[i]]
        return wd_dst


class RetinaFace_BBT(nn.Module):

    def __init__(self, resnet50=True):
        super().__init__()
        backbone = ResNet50() if resnet50 else ResNet152()
        cins = [256, 512, 1024, 2048]
        cout = 256
        relu='plain'

        anchors = post.make_anchors([16, 32, 64, 128, 256], scales=[1, 2**(1/3), 2**(2/3)])
        self.bases = list(zip([4, 8, 16, 32, 64], anchors))
        num_anchors = 3

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu, P6='fromC5', nonCumulative=True)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in range(len(cins) + 1)])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(len(cins) + 1)])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(len(cins) + 1)])

    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = prep.normalize(imgs, dv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = torch.stack(ts)
        
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        box_reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        classif = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
             
        scores = F.softmax(classif, dim=-1)[:, :, 1]
        priors = post.get_priors(x.shape[2:], self.bases, dv, loc='center')
        boxes = post.decode_boxes(box_reg, priors, 0.1, 0.2)
        l = post.select_boxes(boxes, scores, score_thr=0.5, iou_thr=0.4, impl='tvis')
        return l

    def get_pretrained_weights(self, dv, source):
        gids = {
            'face_bbt_resnet152_mixed': '1xB5RO99bVnXLYesnilzaZL2KWz4BsJfM',
            'face_bbt_resnet50_mixed': '1uraA7ZdCCmos0QSVR6CJgg0aSLtV4q4m',
            'face_bbt_resnet50_wider': '1pQLydyUUEwpEf06ElR2fw8_x2-P9RImT',
            'face_bbt_resnet50_icartoon': '12RsVC1QulqsSlsCleMkIYMHsAEwMyCw8'
        }
        nm = 'retina_%s.pth' % source
        wf = prep_weights_gdrive(gids[source], nm)
        wd_src = torch.load(wf, map_location=torch.device(dv))
        
        # in this source, smoothP5 is not applied but the layer is still created for it for no reason
        for s in ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']:
            wd_src.pop('fpn.lateral_outs.3.' + s)
        # in this source, FPN extra P6 layer is placed between laterals and smooths, but we need after
        wl = list(wd_src.items())
        idx = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_ins.4')]
        els = [wl.pop(idx[0]) for _ in idx]
        pos = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_outs.')][-1]
        for el in els[::-1]:
            wl.insert(pos + 1, el)
        wd_src = dict(wl)

        wd_dst = {}
        names = list(wd_src)
        for i, w in enumerate(list(self.state_dict())):
            wd_dst[w] = wd_src[names[i]]
        return wd_dst


class RetinaNet_TorchVision(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = ResNet50(return_count=3, bn_eps=0.0)
        cins = [512, 1024, 2048]
        cout = 256
        anchors_per_level = 9
        self.num_classes = 91

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu=None, bn=None, P6='fromP5', P7=True, smoothP5=True)
        self.cls_head = HeadShared(cout, anchors_per_level, self.num_classes)
        self.reg_head = HeadShared(cout, anchors_per_level, 4)
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = prep.normalize(imgs, dv, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
        ts, sz_orig, sz_used = prep.resize(ts, resize_min=800, resize_max=1333)
        x = prep.batch(ts, mult=32)

        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        reg = [self.reg_head(lvl) for lvl in xs]
        log = [self.cls_head(lvl) for lvl in xs]
        
        lvsizes = torch.tensor([lvl.shape[1] for lvl in log])
        lvidx = torch.arange(len(log), device=dv).repeat_interleave(lvsizes)
        reg = torch.cat(reg, axis=1)
        scr = torch.cat(log, axis=1).sigmoid_()

        priors = post.get_priors(x.shape[2:], self.get_bases(), dv, loc='corner', patches='fit')
        b, s, l = post.get_results(reg, scr, priors, 0.05, 0.5, [1, 1], math.log(1000 / 16),
                         lvtop=1000, levels=lvidx, multiclassbox=True, sz_orig=sz_orig, sz_used=sz_used,
                         imtop=300)
        return b, s, l
        #res = []
        #for i in range(x.shape[0]):
        #    idx, si, li = post.select_by_score(scr[i], 0.05, 1000, lsz, multiclassbox=True)
        #    bi = post.decode_boxes(reg[i][idx], priors[idx], 1, 1, math.log(1000 / 16))
        #    bi = post.clamp_to_canvas(bi, sz_used[i])
        #    bi, si, li = post.do_nms(bi, si, li, 0.5, top=300)
        #    bi = post.scale_back(bi, sz_orig[i], sz_used[i])
        #    res.append((bi, si, li))
        #b, s, l = map(list, zip(*res))
        #return b, s, l

    def get_bases(self):
        # equivalent to:
        #strides = [8, 16, 32, 64, 128]
        #anchors = post.make_anchors([32, 64, 128, 256, 512], [1, 2**(1/3), 2**(2/3)], [2, 1, 0.5])
        #return list(zip(strides, anchors))
        # but due to some rounding of intermediate results, torchvision's code ends up with sligthly different numbers
        return [
            (8,   [(46, 22), (56, 28), (70, 36),        (32, 32), (40, 40), (50, 50),       (22, 46), (28, 56), (36, 70)]),
            (16,  [(90, 46), (114, 56), (142, 72),      (64, 64), (80, 80), (100, 100),     (46, 90), (56, 114), (72, 142)]),
            (32,  [(182, 90), (228, 114), (288, 144),   (128, 128), (160, 160), (204, 204), (90, 182), (114, 228), (144, 288)]),
            (64,  [(362, 182), (456, 228), (574, 288),  (256, 256), (322, 322), (406, 406), (182, 362), (228, 456), (288, 574)]),
            (128, [(724, 362), (912, 456), (1148, 574), (512, 512), (644, 644), (812, 812), (362, 724), (456, 912), (574, 1148)])
        ]

    def get_pretrained_weights(self, dv, source):
        link = 'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth'
        nm = 'retina_%s.pth' % source
        wf = prep_weights_file(link, nm)
        wd_src = torch.load(wf, map_location=torch.device(dv))
        
        # source file doesn't have 'num_batches_tracked' entries, but they're used only in
        # train mode and only if BatchNorm2d 'momentum' param = None, so we just fill them with 0
        wd_dst = {}
        names = list(wd_src)
        shift = 0
        for i, w in enumerate(list(self.state_dict())):
            if w.endswith('num_batches_tracked'):
                wd_dst[w] = torch.tensor(0)
                shift += 1
            else:
                wd_dst[w] = wd_src[names[i - shift]]
        return wd_dst


class RetinaDetector():

    variations = [
        'net_torchvision_resnet50_coco',
        'face_biubug6_mobilenet', 'face_biubug6_resnet50',
        'face_bbt_resnet152_mixed', 'face_bbt_resnet50_mixed',
        'face_bbt_resnet50_wider', 'face_bbt_resnet50_icartoon'
    ]

    def __init__(self, source, device=None):
        assert source in self.variations
        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Initializing Retina model for detection (%s)' % source)

        if source == 'net_torchvision_resnet50_coco':
            self.model = RetinaNet_TorchVision()

        elif source.startswith('face_biubug6'):
            self.model = RetinaFace_Biubug6(source.endswith('mobilenet'))
 
        else:
            self.model = RetinaFace_BBT('resnet50' in source)
        
        #for w in self.model.state_dict(): print(w, '\t', self.model.state_dict()[w].shape)
        wd = self.model.get_pretrained_weights(device, source)
        self.model = self.model.to(device)
        self.model.load_state_dict(wd)
        self.model.eval()

    def __call__(self, imgs):
        with torch.no_grad():
            res = self.model(imgs)
        return res
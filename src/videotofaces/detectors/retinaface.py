import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..backbones.resnet import ResNet50, ResNet152
from ..utils.download import prep_weights_gdrive
from ..utils import bbox

# Source 1: https://github.com/biubug6/Pytorch_Retinaface
# Source 2: https://github.com/barisbatuhan/FaceDetector
# Paper: https://arxiv.org/pdf/1905.00641.pdf


class FPN(nn.Module):
    def __init__(self, cins, cout, relu, extra_level):
        super(FPN, self).__init__()
        self.outputs = nn.ModuleList([ConvUnit(cin, cout, 1, 1, 0, relu) for cin in cins])
        if extra_level: # corresponds to "P6" in the paper
            self.extra = ConvUnit(cins[-1], cout, 3, 2, 1, relu)
        self.merges = nn.ModuleList([ConvUnit(cout, cout, 3, 1, 1, relu) for _ in cins[1:]])

    def forward(self, xs):
        n = len(xs)
        ys = [self.outputs[i](xs[i]) for i in range(n)]
        for i in range(0, n - 1)[::-1]:
            ys[i] += F.interpolate(ys[i + 1], size=ys[i].shape[2:], mode='nearest')
            ys[i] = self.merges[i](ys[i])
        if hasattr(self, 'extra'):
            ys.append(self.extra(xs[-1]))
        return ys


class SSH(nn.Module):

    def __init__(self, cin, cout, relu):
        super(SSH, self).__init__()
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
        super(Head, self).__init__()
        self.task_len = task_len
        self.conv = nn.Conv2d(cin, num_anchors * task_len, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class RetinaFace(nn.Module):

    def __init__(self, backbone, arch_params, prep_params, bases, score_thr, predict_landmarks):
        super(RetinaFace, self).__init__()
        self.prep_params = prep_params
        self.bases = bases
        self.score_thr = score_thr

        cins, cout, relu, extra = arch_params
        num_anchors = len(bases[0][1])
        num_levels = len(cins) + (1 if extra else 0)

        self.body = backbone
        self.feature_pyramid = FPN(cins, cout, relu, extra)
        self.context_modules = nn.ModuleList([SSH(cout, cout, relu) for _ in range(num_levels)])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(num_levels)])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(num_levels)])
        if predict_landmarks:
            self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in range(num_levels)])

    def preprocess(self, imgs):
        means, stds, swapRB = self.prep_params
        x = np.stack(imgs)
        x = x if not swapRB else x[:, :, :, [2, 1, 0]]
        x = (x - means) / stds
        x = x.transpose(0, 3, 1, 2)
        x = torch.from_numpy(x).to(next(self.parameters()).device, torch.float32)
        return x

    def forward(self, imgs):
        x = self.preprocess(imgs)
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
        l = select_boxes(boxes, scores, score_thr=self.score_thr, iou_thr=0.4)
        return l


def select_boxes(boxes, scores, score_thr, iou_thr, impl='vect', nms_impl='torch'):
    assert impl in ['vect', 'loop']
    assert nms_impl in ['torch', 'numpy']
    n = boxes.shape[0]
    
    if impl == 'vect':
        k = torch.arange(n).repeat_interleave(boxes.shape[1])
        b, s = boxes.reshape(-1, 4), scores.flatten()
        idx = s > score_thr
        k, b, s = k[idx], b[idx], s[idx]
        if nms_impl == 'torch':
            keep = bbox.batched_nms(b, s, k, iou_thr, 'torch')
            k, b, s = k[keep], b[keep], s[keep]
            r = torch.hstack([b, s.unsqueeze(1)])
            l = [r[k == i] for i in range(n)]
            return [t.detach().cpu().numpy() for t in l]
        if nms_impl == 'numpy':
            b, s, k = [x.detach().cpu().numpy() for x in [b, s, k]]
            keep = bbox.batched_nms(b, s, k, iou_thr, 'numpy')
            k, b, s = k[keep], b[keep], s[keep]
            r = np.hstack([b, np.expand_dims(s, 1)])
            l = [r[k == i] for i in range(n)]
            return l

    if impl == 'loop':
        l = []
        for i in range(n):
            b, s = boxes[i], scores[i]
            idx = s > score_thr
            b, s = b[idx], s[idx]
            r = torch.hstack([b, s.unsqueeze(1)]).detach().cpu().numpy()
            if nms_impl == 'torch':
                keep = bbox.nms(b, s, iou_thr, 'torch')
            else:
                keep = bbox.nms(r[:, :4], r[:, 4], iou_thr, 'numpy')
            l.append(r[keep])
        return l


def get_priors(img_size, bases):
    """For every (stride, anchors) pair in ``bases`` list, walk through every stride-sized
    square patch of ``img_size`` canvas left-right, top-bottom and return anchors-sized boxes
    drawn around each patch's center in a form of (center_x, center_y, width, height).
    
    Example: get_priors((90, 64), [(32, [(8, 4), (25, 15)])])
    Output: shape = (12, 4)
    [[16, 16, 8, 4], [16, 16, 25, 15], [48, 16, 8, 4], [48, 16, 25, 15],
     [16, 48, 8, 4], [16, 48, 25, 15], [48, 48, 8, 4], [48, 48, 25, 15],
     [16, 80, 8, 4], [16, 80, 25, 15], [48, 80, 8, 4], [48, 80, 25, 15]]

    In case of square anchors, only one dimension can be provided, i.e. [(8, [16, 32])]
    will be automatically turned into [(8, [(16, 16), (32, 32)])].
    """
    p = []
    h, w = img_size
    if isinstance(bases[0][1][0], int):
        bases = [(s, [(a, a) for a in l]) for (s, l) in bases]
    for stride, anchors in bases:
        nx = math.ceil(w / stride)
        ny = math.ceil(h / stride)
        xs = torch.arange(nx) * stride + stride // 2
        ys = torch.arange(ny) * stride + stride // 2
        c = torch.dstack(torch.meshgrid(xs, ys, indexing='xy')).reshape(-1, 2)
        # could replace line above by "torch.cartesian_prod(xs, ys)" but that'd be for indexing='ij'
        c = c.repeat_interleave(len(anchors), dim=0)
        s = torch.Tensor(anchors).repeat(nx*ny, 1)
        p.append(torch.hstack([c, s]))
    return torch.cat(p)


def decode_boxes(pred, priors):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    xys = priors[:, 2:] * 0.1 * pred[:, :, :2] + priors[:, :2]
    whs = priors[:, 2:] * torch.exp(0.2 * pred[:, :, 2:])
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


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

        if source.startswith('biubug6_'):
            bases = [
                (8, [16, 32]),
                (16, [64, 128]),
                (32, [256, 512])
            ]
            score_thr, predict_landmarks = 0.02, True
            prep_params = ((104, 117, 123), (1, 1, 1), False)
            if source == 'biubug6_mobilenet':
                backbone = MobileNetV1(0.25, relu_type='lrelu_0.1', return_inter=[5, 11])
                arch_params = ([64, 128, 256], 64, 'lrelu_0.1', False)
            elif source == 'biubug6_resnet50':
                backbone = ResNet50(return_count=3)
                arch_params = ([512, 1024, 2048], 256, 'plain', False)
            wf = prep_weights_gdrive(self.gids[source], weights_filename)
            wd_src = torch.load(wf, map_location=torch.device(device))

        elif source.startswith('bbt_'):
            bases = [
                (4, [16, 20.16, 25.40]),
                (8, [32, 40.32, 50.80]),
                (16, [64, 80.63, 101.59]),
                (32, [128, 161.26, 203.19]),
                (64, [256, 322.54, 406.37])
            ]
            score_thr, predict_landmarks = 0.5, False
            prep_params = ((123.675, 116.28, 103.53), (58.395, 57.12, 57.375), True)
            arch_params = ([256, 512, 1024, 2048], 256, 'plain', True)
            backbone = ResNet152() if source == 'bbt_resnet152_mixed' else ResNet50()
            wf = prep_weights_gdrive(self.gids[source], weights_filename)
            wd_src = torch.load(wf, map_location=torch.device(device))
            for s in ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean',
                      'bn.running_var', 'bn.num_batches_tracked']:
                wd_src.pop('fpn.lateral_outs.3.' + s)
        
        self.model = RetinaFace(backbone, arch_params, prep_params, bases, score_thr, predict_landmarks).to(device)
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
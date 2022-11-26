import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from ..utils import prep_weights_file
from ..utils.weights import load_weights

# adapted from: https://github.com/open-mmlab/mmdetection
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/yolo_head.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py


class ConvModule(nn.Module):

    def __init__(self, cin, cout, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.activate = nn.LeakyReLU(0.1, inplace=True)
  
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvModule(c, c // 2, k=1)
        self.conv2 = ConvModule(c // 2, c, k=3, p=1)
  
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x


class Darknet53(nn.Module):

    def __init__(self, ):
        super().__init__()
        L, C = [1, 2, 8, 8, 4], [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
        self.conv1 = ConvModule(3, 32, k=3, p=1)
        for i in range(len(L)):
            block = nn.Sequential()
            block.add_module('conv', ConvModule(C[i][0], C[i][1], k=3, s=2, p=1))
            for j in range(L[i]):
                block.add_module(f'res{j}', ResBlock(C[i][1]))
            self.add_module(f'conv_res_block{i + 1}', block)
  
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv_res_block1(x)
        x2 = self.conv_res_block2(x1)
        x3 = self.conv_res_block3(x2)
        x4 = self.conv_res_block4(x3)
        x5 = self.conv_res_block5(x4)
        return (x3, x4, x5)
        

class DetectionBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = ConvModule(cin, cout, 1)
        self.conv2 = ConvModule(cout, cout * 2, 3, p=1)
        self.conv3 = ConvModule(cout * 2, cout, 1)
        self.conv4 = ConvModule(cout, cout * 2, 3, p=1)
        self.conv5 = ConvModule(cout * 2, cout, 1)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
        

class YOLOv3Neck(nn.Module):

    def __init__(self):
        super().__init__()
        self.detect1 = DetectionBlock(1024, 512)
        self.conv1 = ConvModule(512, 256, 1)
        self.detect2 = DetectionBlock(768, 256)
        self.conv2 = ConvModule(256, 128, 1)
        self.detect3 = DetectionBlock(384, 128)
  
    def forward(self, x):
        (x1, x2, x3) = x
        y3 = self.detect1(x3)
        t = self.conv1(y3)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x2), 1)
        y2 = self.detect2(t)
        t = self.conv2(y2)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x1), 1)
        y1 = self.detect3(t)
        return (y3, y2, y1)
        

class YOLOv3Head(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.convs_bridge = nn.ModuleList([
            ConvModule(512, 1024, 3, p=1),
            ConvModule(256, 512, 3, p=1),
            ConvModule(128, 256, 3, p=1)
        ])
        cout = (num_classes + 5) * 3
        self.convs_pred = nn.ModuleList([
            nn.Conv2d(1024, cout, 1),
            nn.Conv2d(512, cout, 1),
            nn.Conv2d(256, cout, 1)
        ])
  
    def forward(self, x):
        maps = []
        for i in range(3):
            y = x[i]
            y = self.convs_bridge[i](y)
            y = self.convs_pred[i](y)
            maps.append(y)
        return tuple(maps)


class YOLOv3(nn.Module):

    links = {
        'anime': 'https://github.com/hysts/anime-face-detector/'\
                 'releases/download/v0.0.1/mmdet_anime-face_yolov3.pth',
        'wider': '1pjg1_IeAuzgRzZiY92r71uzd_amfcegu',
        'coco': 'https://download.openmmlab.com/mmdetection/v2.0/yolo/'\
                'yolov3_d53_mstrain-608_273e_coco/'\
                'yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
    }

    def __init__(self, pretrained=None, device='cpu', img_size=608, num_classes=1, extra_thr=None):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.extra_thr = extra_thr
        self.backbone = Darknet53()
        self.neck = YOLOv3Neck()
        self.bbox_head = YOLOv3Head(num_classes)
        if pretrained:
            sub = None if pretrained == 'wider' else 'state_dict'
            load_weights(self, self.links[pretrained], pretrained, device, sub=sub)
  
    def _preprocess(self, imgs, img_size):
        h, w = imgs[0].shape[:2]
        # prepare to scale to fit into (img_size, img_size) while keeping the aspect ratio
        scale = min(img_size / w, img_size / h)
        sw = int(np.ceil(w * scale))
        sh = int(np.ceil(h * scale))
        # prepare to pad so both dimensions are multiples of 32
        ph = 32 * int(np.ceil(sh / 32)) - sh
        pw = 32 * int(np.ceil(sw / 32)) - sw
        # preprocess
        x = np.stack([cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR) for img in imgs])   # [bs, h, w, 3]
        x = x[:, :, :, [2, 1, 0]].astype(np.float32) / 255.0                                        # BGR -> RGB, [0..255] -> [0..1]
        x = np.pad(x, ((0, 0), (0, ph), (0, pw), (0, 0)))                                           # [bs, h+ph, w+pw, 3]
        x = x.transpose(0, 3, 1, 2)                                                                 # [bs, 3, h, w] (channels first)
        x = torch.from_numpy(x).to(next(self.parameters()).device)
        return x, scale

    def forward(self, imgs):
        x, scale = self._preprocess(imgs, self.img_size)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head(x)
        b, s, i, c = self._build_boxes(x)
        b /= scale
        res, cls = self._result_lists(len(imgs), b, s, i, c)
        if self.num_classes == 1:
            return res
        return res, cls

    def _build_boxes(self, pred_maps):
        anchors, e_strides = self._get_anchors(pred_maps)                               # [13167, 4]; [13167, 1]
        bbox_preds, objectness, scores, classes = self._process_pred_maps(pred_maps)    # [bs, 13167, 4]; [bs, 13167]; [bs, 13167]
        bboxes = self._decode_bbox_preds(bbox_preds, anchors, e_strides)                 # [bs, 13167, 4]
        return self._perform_nms(bboxes, objectness, scores, classes)
  
    def _get_anchors(self, pred_maps):
        # https://becominghuman.ai/understanding-anchors-backbone-of-object-detection-using-yolo-54962f00fbbb
        strides = [32, 16, 8]
        # base_sizes = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]
        # centers = [(16., 16.), (8., 8.), (4., 4.)] # half of stride to float tuple
        # base_anchors[i] are drawing boxes around i'th center with sizes from base_sizes[i]
        base_anchors = [
            torch.Tensor([[ -42.,  -29.,   74.,   61.], [ -62.,  -83.,   94.,  115.], [-170.5, -147.,  202.5,  179.]]),
            torch.Tensor([[ -7., -22.5,  23.,  38.5], [-23., -14.5,  39.,  30.5], [-21.5, -51.5,  37.5,  67.5]]),
            torch.Tensor([[ -1.,  -2.5,   9.,  10.5], [ -4., -11.,  12.,  19.], [-12.5,  -7.5,  20.5,  15.5]])
        ]
        al, es = [], []
        for i, map in enumerate(pred_maps):
            mh, mw = map.shape[-2], map.shape[-1]
            shift_x = torch.arange(0, mw).to(torch.float32) * strides[i]        # [mw] e.g. [0, 32, 64, 96, ..., 32*(mw-1)]
            shift_y = torch.arange(0, mh).to(torch.float32) * strides[i]        # [mh] e.g. [0, 32, 64, 96, ..., 32*(mh-1)]
            shift_x = shift_x.repeat(mh)                                        # [mw*mh] e.g. [0, 32, ..., 32*(mw-1)] x mh
            shift_y = shift_y.repeat_interleave(mw)                             # [mh*mw] e.g. [0 x mw, 32 x mw, ..., 32*(mh-1) x mw]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)  # [mw*mh, 4]
            anchors = base_anchors[i].unsqueeze(0) + shifts.unsqueeze(1)        # [mw*mh, 3, 4] = [1, 3, 4] + [mw*mh, 1, 4]
            anchors = anchors.reshape(-1, 4)                                    # [mw*mh*3, 4]
            al.append(anchors)
            es.extend(anchors.shape[0] * [float(strides[i])])
        res1 = torch.cat(al)                                                  # [dim = m1w*m1h*3 + m2w*m2h*3 + m3w*m3h*3, 4]
        res2 = torch.Tensor(es).unsqueeze(-1)                                 # [dim, 1] = [32.] x m1w*m1h*3 + [16.] * m2w*m2h*3 + [8.] * m3w*m3h*3
        dv = pred_maps[0].device
        return res1.to(dv), res2.to(dv)

    def _process_pred_maps(self, pred_maps):
        p = []
        for map in pred_maps:                                                         # [bs, (c+5)*3, miw, mih]
            map = map.permute(0, 2, 3, 1).reshape(map.shape[0], -1, map.shape[1] // 3)  # [bs, miw*mih*3, c+5] (bboxes(4) + objectness(1) + num_classes(c))
            p.append(map)
        f = torch.cat(p, dim=1)                                       # [bs, dim, c+5]
        bbox_preds = f[..., :4]                                       # [bs, dim, 4]
        objectness = f[..., 4].sigmoid()                              # [bs, dim]
        cls_scores = f[..., 5:].sigmoid()                             # [bs, dim, c]
        max_scores, cls_inds = torch.max(cls_scores, dim=-1)          # [bs, dim], [bs, dim]
        return bbox_preds, objectness, max_scores, cls_inds

    def _decode_bbox_preds(self, preds, anchors, stride):
        xy_centers = (anchors[..., :2] + anchors[..., 2:]) * 0.5 + (preds[..., :2].sigmoid() - 0.5) * stride
        whs = (anchors[..., 2:] - anchors[..., :2]) * 0.5 * preds[..., 2:].exp()
        return torch.stack((xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] - whs[..., 1], xy_centers[..., 0] + whs[..., 0], xy_centers[..., 1] + whs[..., 1]), dim=-1)  

    def _perform_nms(self, bboxes, objectness, scores, classes):
        bs, dim = bboxes.shape[:2]                    # bs, dim
        k = torch.arange(bs).repeat_interleave(dim)   # [0] x dim + [1] x dim + ... + [bs-1] x dim
        k = k.to(bboxes.device)
        k = k * 1000 + classes.reshape(-1)            # [bs * dim]
        b = bboxes.reshape(-1, 4)                     # [bs * dim, 4]
        o = objectness.reshape(-1)                    # [bs * dim]
        s = scores.reshape(-1)                        # [bs * dim]
  
        idx = (o >= 0.005) * (s > 0.05)               # filter by thresholds
        b, o, s, k = b[idx], o[idx], s[idx], k[idx]   # (most are eliminated here)
        s *= o                                        # scale scores by objectness
        if self.extra_thr:
            idx = s >= self.extra_thr
            b, s, k = b[idx], s[idx], k[idx]

        idx = batched_nms(b, s, k, 0.45)              # every image + class = separate category as indicated by k
        b, s, k = b[idx], s[idx], k[idx]              # i.e. class 36 in image 5 will be class 5036 for batched_nms
        return b, s, k.div(1000, rounding_mode='floor'), k % 1000

    def _result_lists(self, bs, boxes, scores, imgidx, classes):
        results = torch.cat((boxes, scores.unsqueeze(-1)), dim=1)
        l, c = [], []
        for i in range(bs):
            mask = imgidx == i
            l.append(results[mask].cpu().numpy())
            c.append(classes[mask].cpu().numpy())
        return l, c
        

class YOLOv3DetectorAnime():
    """Returns YOLOv3 net with pre-trained weights, ready to detect faces. Weights by: https://github.com/hysts/anime-face-detector
    Size can be lowered to 416 or 320 (two other common yolo sizes) for a significant speed-up with potentially not that much loss of accuracy
    """
    
    def __init__(self, device, input_size=608):
        """TBD"""
        print('Initializing YOLO v3 model for anime face detection')
        wf = prep_weights_file('https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth', 'mmdet_anime-face_yolov3.pth')
        wd = torch.load(wf, map_location=torch.device(device))['state_dict']
        self.model = YOLOv3(img_size=input_size).to(device)
        self.model.load_state_dict(wd)
        self.model.eval()
        print()
    
    def __call__(self, frames):
        """TBD"""
        with torch.no_grad():
            boxes = self.model(frames)
        return boxes


class YOLOv3Detector():
    """Returns YOLOv3 net with pre-trained weights, ready to detect faces
    Weights are converted from darknet file linked here: https://github.com/juliansprt/OpenCvSharpDNN
    to keras using a technique described here: https://github.com/chinmaykumar06/face-detection-yolov3-keras
    then to pytorch by looping over keras_model.get_weights() and doing torch.from_numpy(np.transpose(w[i], (3, 2, 0, 1)))
    """
    
    def __init__(self, device, input_size=608):
        """TBD"""
        print('Initializing YOLO v3 model for live-action face detection')
        wf = prep_weights_file('https://drive.google.com/uc?id=1pjg1_IeAuzgRzZiY92r71uzd_amfcegu', 'yolov3-wider_16000.pt', gdrive=True)
        wd = torch.load(wf, map_location=torch.device(device))
        self.model = YOLOv3(img_size=input_size, extra_thr=0.1).to(device)
        self.model.load_state_dict(wd)
        self.model.eval()
        print()
        
    def __call__(self, frames):
        """TBD"""
        with torch.no_grad():
            boxes = self.model(frames)
        return boxes


def yolo3_coco_detector(device):
    """Returns YOLOv3 net with pre-trained weights, ready to detect objects
    Weights by: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo
    This isn't used anywhere around here, since it's not applicable to face detection
    (the "person" class will have boxes around whole humans, not just faces/heads),
    but it's the standard option, so I decided to keep it for possible experiments,
    and as an example of multi-class implementation
    """
    print('Initializing YOLO v3 model for object detection (80 classes from COCO dataset)')
    wf = prep_weights_file(
        'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth',
        'yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth')
    model = YOLOv3(num_classes=80).to(device)
    weights = torch.load(wf, map_location=torch.device(device))
    model.load_state_dict(weights['state_dict'])
    model.eval()
    print()
    return model, weights['meta']['CLASSES']
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from ..utils.weights import load_weights


# adapted from https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py

class PNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a[:, 1]


class RNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        # can't do -1 in reshape because it fails for [0, ...]-dim tensors
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        x = self.dense4(x)
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a[:, 1]


class ONet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        x = self.dense5(x)
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a[:, 1]


class MTCNN(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.to(device)
  
    def _preprocess(self, cv2_images, device):
        x = np.stack(cv2_images)                  # [bs, h, w, 3]
        x = x.transpose(0, 3, 1, 2)               # [bs, 3, h, w]
        x = x[:, [2, 1, 0], :, :]                 # BGR -> RGB
        x = (x.astype(np.float32) - 127.5) / 128  # [0..255] -> [-0.5..0.5]
        x = torch.from_numpy(x).to(device)
        return x

    def _scale_pyramid(self, H, W, minsize, factor):
        scales = []
        s = 12.0 / minsize
        while min(H, W) * s >= 12:
            scales.append(s)
            s *= factor
        sizes = [(int(H * s + 1), int(W * s + 1)) for s in scales]
        return scales, sizes

    def _resample(self, x, size):
        return F.adaptive_avg_pool2d(x, size)

    def _get_cropped_candidates(self, x, imgidx, boxes, size):
        H, W = x.shape[2:4]
        l = [torch.zeros([0, x.shape[1], *size], device=x.device)]
        for k in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[k]
            x1, y1, x2, y2 = max(1, int(x1)), max(1, int(y1)), min(W, int(x2)), min(H, int(y2))
            if y2 > y1 - 1 and x2 > x1 - 1:
                crop = x[imgidx[k], :, y1 - 1: y2, x1 - 1: x2]
                crop = self._resample(crop, size)
                l.append(crop.unsqueeze(0))
        return torch.cat(l)
        # could do l=[] above, then l.append(crop); torch.stack(l) here,
        # but that fails for [0, ...] tensors (when no boxes found)

    def forward(self, imgs, minsize=20, return_landmarks=False):
        x = self._preprocess(imgs, next(self.parameters()).device)
        H, W = x.shape[2:4]
        scales, sizes = self._scale_pyramid(H, W, minsize, 0.709)

        # Stage 1: run proposal network for input images in every planned scaling
        # start with boxes as respective 12x12 image patches where scores were above threshold
        # do NMS first within scale + image, then within image
        # and only then refine surviving boxes' coordinates with predictions from pnet
        # (why not refine right away, since we already have the preds from the network? I have no idea)
        boxes, scores, imgidx, preds = [], [], [], []
    
        for i in range(len(scales)):
            xi = self._resample(x, sizes[i])               # [bs, 3, sh, sw]
            pred, prob = self.pnet(xi)                     # [bs, 4, ph, pw], [bs, ph, pw]

            mask = prob >= 0.6                             # [bs, ph, pw] (True/False tensor)
            mask_inds = mask.nonzero()                     # [nz, 3] (3D indices of Trues in mask)
            scores_i = prob[mask]                          # [nz]
            imgidx_i = mask_inds[:, 0]                     # [nz]
            preds_i = pred.permute(1, 0, 2, 3)             # [4, bs, ph, pw] (prep to apply mask)
            preds_i = preds_i[:, mask].permute(1, 0)       # [4, nz] -> [nz, 4]

            stride, cell = 2, 12
            bb = mask_inds[:, 1:].flip(1)                            # [nz, 2] (h, w -> w, h in axis=1)
            q1 = ((stride * bb + 1) / scales[i]).floor()             # [nz, 2] (upper-left corners of areas)
            q2 = ((stride * bb + cell - 1 + 1) / scales[i]).floor()  # [nz, 2] (bottom-right corners of areas)
            boxes_i = torch.cat([q1, q2], dim=1)                     # [nz, 4]

            pick = torchvision.ops.batched_nms(boxes_i, scores_i, imgidx_i, 0.5)
            boxes.append(boxes_i[pick])
            preds.append(preds_i[pick])
            scores.append(scores_i[pick])
            imgidx.append(imgidx_i[pick])
          
        boxes, scores = torch.cat(boxes), torch.cat(scores)
        preds, imgidx = torch.cat(preds), torch.cat(imgidx)
        
        pick = torchvision.ops.batched_nms(boxes, scores, imgidx, 0.7)
        boxes, preds, imgidx = boxes[pick], preds[pick], imgidx[pick]
        boxes = self._refine_bbox(boxes, preds, False)    
        boxes = self._square_bbox(boxes)

        # Stage 2: run refinement network for every candidate using boxes from Stage 1 downscaled to 24 x 24
        # same as before, filter out low scores, do NMS with old boxes and new scores,
        # and only then refine them with new predictions from rnet
        proposals = self._get_cropped_candidates(x, imgidx, boxes, (24, 24))
        preds, scores = self.rnet(proposals)
        ipass = scores > 0.7
        boxes, scores = boxes[ipass, :], scores[ipass] 
        preds, imgidx = preds[ipass, :], imgidx[ipass]
    
        pick = torchvision.ops.batched_nms(boxes, scores, imgidx, 0.7)
        boxes, preds, imgidx = boxes[pick], preds[pick], imgidx[pick]
        boxes = self._refine_bbox(boxes, preds, True)
        boxes = self._square_bbox(boxes)

        # Stage 3: run output network for every candidate using boxes from Stage 2
        # filtering out low scores is exactly the same as Stage 2
        # but there's no usual NMS before refinement, and no squaring of boxes
        # instead, there's a "special" final NMS that checks intersections over minimum, not union
        refinements = self._get_cropped_candidates(x, imgidx, boxes, (48, 48))
        preds, landmarks, scores = self.onet(refinements)
        ipass = scores > 0.7
        boxes, scores = boxes[ipass, :], scores[ipass] 
        preds, imgidx = preds[ipass, :], imgidx[ipass]
        landmarks = landmarks[ipass, :]
    
        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        lm_x = w_i.unsqueeze(1) * landmarks[:, :5] + boxes[:, 0].unsqueeze(1) - 1
        lm_y = h_i.unsqueeze(1) * landmarks[:, 5:] + boxes[:, 1].unsqueeze(1) - 1
        landmarks = torch.stack([lm_x, lm_y], dim=-1)

        boxes = self._refine_bbox(boxes, preds, True)
        pick = self._nms_vectorized(boxes, scores, imgidx, 0.7, 'Min')
        boxes, scores, landmarks, imgidx = boxes[pick], scores[pick], landmarks[pick], imgidx[pick]
        res, ldm = [], []
        for k in range(x.shape[0]):
            idx = imgidx == k
            resk = torch.cat((boxes[idx], scores[idx].unsqueeze(1)), dim=1)
            res.append(resk.cpu().numpy())
            ldm.append(landmarks[idx].cpu().numpy())
        if return_landmarks:
            return res, ldm
        return res

    def _refine_bbox(self, boxes, pred, plus_one=False):
        w = boxes[:, 2] - boxes[:, 0] + (1 if plus_one else 0)
        h = boxes[:, 3] - boxes[:, 1] + (1 if plus_one else 0)
        b1 = boxes[:, 0] + pred[:, 0] * w
        b2 = boxes[:, 1] + pred[:, 1] * h
        b3 = boxes[:, 2] + pred[:, 2] * w
        b4 = boxes[:, 3] + pred[:, 3] * h
        boxes[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)
        return boxes

    def _square_bbox(self, boxes):
        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        l = torch.max(w, h)
        boxes[:, 0] = boxes[:, 0] + w * 0.5 - l * 0.5
        boxes[:, 1] = boxes[:, 1] + h * 0.5 - l * 0.5
        boxes[:, 2:4] = boxes[:, :2] + l.repeat(2, 1).permute(1, 0)
        return boxes

    def _nms_vectorized(self, boxes, scores, classes, thresh, method, chain_suppression=True):
        dv = boxes.device
        if boxes.numel() == 0:
            return torch.Tensor([]).to(dv, torch.int64)
        k = torch.argsort(scores, descending=True)
        classes_sorted = classes[k]
        c = torch.Tensor([]).to(dv, torch.int64)
        for i in classes.unique():
            ci = torch.combinations(k[classes_sorted == i]).to(dv)
            c = torch.cat((c, ci))
        b1 = boxes[c[:, 0]]
        b2 = boxes[c[:, 1]]
        inter_x1 = torch.maximum(b1[:, 0], b2[:, 0])
        inter_y1 = torch.maximum(b1[:, 1], b2[:, 1])
        inter_x2 = torch.minimum(b1[:, 2], b2[:, 2])
        inter_y2 = torch.minimum(b1[:, 3], b2[:, 3])
        inter_w = inter_x2 - inter_x1 + 1
        inter_h = inter_y2 - inter_y1 + 1
        idx = (inter_w > 0) * (inter_h > 0)
        c, b1, b2, inter_w, inter_h = c[idx], b1[idx], b2[idx], inter_w[idx], inter_h[idx]
        inter = inter_w * inter_h
        area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
        area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
        if method == 'Min':
            iom = inter / torch.minimum(area1, area2)
            idx = (iom > thresh)
        else:
            iou = inter / (area1 + area2 - inter)
            idx = (iou > thresh)
        c = c[idx]
        if chain_suppression:
            return torch.Tensor([i for i in k if i not in c[:, 1]]).to(dv, torch.int64)
        keep = k.tolist()
        for pair in c:
            if pair[0] in keep:
                keep.pop(keep.index(pair[1]))
        return torch.Tensor(keep).to(dv, torch.int64)


class RealMTCNN():

    def __init__(self, device=None, min_face_size=5):
        print('Initializing MTCNN model for live-action face detection')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(dv)
        # 3 files from https://github.com/timesler/facenet-pytorch/tree/master/data joined into one
        load_weights(self.model, '1qHW1xoTvuqlUBBhPx1ZLpzUXrWHfW1jN', 'mtcnn_joined')
        self.model.eval()
        self.minsize = min_face_size
        
    def __call__(self, frames):
        with torch.inference_mode():
            boxes = self.model(frames, self.minsize)
        return boxes
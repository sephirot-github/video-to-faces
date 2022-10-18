import numpy as np
import torch

# main source: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
# batch (see coordinate trick): https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html
# on adding +1 to area calculation or not: https://stackoverflow.com/a/51730512/8874388
# (torchvision.ops.nms/batched_nms don't have +1)
# on speed in numpy vs torch: https://discuss.pytorch.org/t/nms-implementation-slower-in-pytorch-compared-to-numpy/36665/7


def nms(boxes, scores, thresh, impl='torch'):
    assert impl in ['torch', 'numpy']
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    if impl == 'numpy':
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ious = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.nonzero(ious <= thresh)[0]
            order = order[inds + 1]

    elif impl == 'torch':
        order = torch.argsort(scores, descending=True)
        while order.numel():
            i = order[0].item()
            keep.append(i)
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            ious = inter / (areas[i] + areas[order[1:]] - inter)
            inds = torch.nonzero(ious <= thresh).squeeze(-1)
            order = order[inds + 1]

    return keep


def batched_nms(boxes, scores, groups, thresh, impl='torch'):
    #if boxes.numel() == 0:
    #    return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    assert impl in ['torch', 'numpy']
    offsets = groups * (boxes.max() + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, thresh, impl)
    return keep
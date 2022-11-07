import math

import numpy as np
import torch
import torchvision.ops

# main source: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
# batch (see coordinate trick): https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html
# on adding +1 to area calculation or not: https://stackoverflow.com/a/51730512/8874388
# (torchvision.ops.nms/batched_nms don't have +1)
# on speed in numpy vs torch: https://discuss.pytorch.org/t/nms-implementation-slower-in-pytorch-compared-to-numpy/36665/7


def nms(boxes, scores, thresh):
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
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
    return keep


def select_decode(reg_maps, cls_maps, priors, sz, score_thr, nms_thr, topk_map=None, topk_img=None):
    """
    """
    bs = cls_maps[0].shape[0]
    num_classes = cls_maps[0].shape[-1]
    reg = torch.cat(reg_maps, dim=1)
    cls = torch.cat(cls_maps, dim=1)
    rb, rs, rl = [], [], []

    for i in range(bs):
        
        s = torch.sigmoid(cls[i]).flatten()
        idx = torch.nonzero(s > score_thr).squeeze()

        if topk_map:
            map_borders = np.cumsum([0] + [m.shape[1] for m in reg_maps]) * num_classes
            sel = []
            for u in range(1, len(map_borders)):
                midx = idx[(idx >= map_borders[u - 1]) * (idx < map_borders[u])]
                _, top = torch.topk(s[midx], min(topk_map, midx.shape[0]))
                sel.append(midx[top])
            idx = torch.cat(sel)

        scores = s[idx]
        labels = idx % num_classes
        sel = torch.div(idx, num_classes, rounding_mode='floor')

        boxes = decode_boxes(reg[i][sel], priors[sel], 1, 1, math.log(1000 / 16))
        boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], max=sz[i][1])
        boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], max=sz[i][0])

        keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_thr)
        if topk_img:
            keep = keep[:topk_img]
        
        rb.append(boxes[keep])
        rs.append(scores[keep])
        rl.append(labels[keep])
    return rb, rs, rl
    

def select_boxes(boxes, scores, score_thr, iou_thr, impl):
    assert impl in ['numpy', 'tvis', 'tvis_batched']
    n = boxes.shape[0]
    if impl == 'tvis_batched':
        k = torch.arange(n).repeat_interleave(boxes.shape[1]).to(boxes.device)
        b, s = boxes.reshape(-1, 4), scores.flatten()
        idx = s > score_thr
        k, b, s = k[idx], b[idx], s[idx]    
        keep = torchvision.ops.batched_nms(b, s, k, iou_thr)
        k, b, s = k[keep], b[keep], s[keep]
        r = torch.hstack([b, s.unsqueeze(1)])
        l = [r[k == i] for i in range(n)]
        return [t.detach().cpu().numpy() for t in l]
    else:
        l = []
        for i in range(n):
            b, s = boxes[i], scores[i]
            idx = s > score_thr
            b, s = b[idx], s[idx]
            r = torch.hstack([b, s.unsqueeze(1)]).detach().cpu().numpy()
            if impl == 'tvis':
                keep = torchvision.ops.nms(b, s, iou_thr)
                keep = keep.detach().cpu().numpy()
            else:
                keep = nms(r[:, :4], r[:, 4], iou_thr)
            l.append(r[keep])
        return l


def get_priors(img_size, bases, dv='cpu', loc='center', patches='as_is'):
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

    If loc='corner', then boxes are drawn around patches' top-left corners instead of centers.

    If patches='fit', then stride is adjusted so that patches fit the canvas without 'going over'.
    E.g. for canvas (h=800, w=1216) and stride 32, ``patches`` param will have no effect (since
    such canvas can be divided into 32x32 areas perfectly), but for stride 128, patches will
    change from 128x128 to 114x121 if the param = 'fit'.
    """
    assert loc in ['center', 'corner']
    assert patches in ['as_is', 'fit']
    p = []
    h, w = img_size
    if isinstance(bases[0][1][0], int):
        bases = [(s, [(a, a) for a in l]) for (s, l) in bases]
    for stride, anchors in bases:
        nx = math.ceil(w / stride)
        ny = math.ceil(h / stride)
        step_x = stride if patches == 'as_is' else w // nx
        step_y = stride if patches == 'as_is' else h // ny
        xs = torch.arange(nx, device=dv) * step_x
        ys = torch.arange(ny, device=dv) * step_y
        if loc == 'center':
            xs += step_x // 2
            ys += step_y // 2
        c = torch.dstack(torch.meshgrid(xs, ys, indexing='xy')).reshape(-1, 2)
        # could replace line above by "torch.cartesian_prod(xs, ys)" but that'd be for indexing='ij'
        c = c.repeat_interleave(len(anchors), dim=0)
        s = torch.tensor(anchors, device=dv).repeat(nx*ny, 1)
        p.append(torch.hstack([c, s]))
    return torch.cat(p)


def decode_boxes(pred, priors, mult_xy=1, mult_wh=1, max_exp_input=None):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    xys = priors[:, 2:] * mult_xy * pred[..., :2] + priors[:, :2]
    whs = priors[:, 2:] * exp_clamped(mult_wh * pred[..., 2:], max_exp_input)
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


def exp_clamped(x, max_=None):
    if not max_:
        return torch.exp(x)
    else:
        return torch.exp(torch.clamp(x, max=max_))
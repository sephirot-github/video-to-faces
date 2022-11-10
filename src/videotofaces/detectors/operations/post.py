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


# todo
def get_results_vect(reg, scr, priors, score_thr, iou_thr, decode_mults):
    n = reg.shape[0]
    k = torch.arange(n).repeat_interleave(reg.shape[1]).to(reg.device)
    reg = reg.reshape(-1, reg.shape[-1])
    scr = scr.reshape(-1, scr.shape[-1])
    idx, scores, labels = select_by_score(scr, score_thr)
    k = k[idx]
    boxes = decode_boxes(reg[idx], priors[idx], *decode_mults)
    if labels:
        groups = k * 1000 + labels
    boxes, scores, groups = do_nms(boxes, scores, groups, iou_thr)
    if labels:
        labels = groups % 1000
        k = groups.div(1000, rounding_mode='floor')
    #l = [r[k == i] for i in range(n)]


# redo
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
        

def select_by_score(scr, score_thr, ltopk=None, lsz=None, multiclassbox=False):
    """"""
    assert (ltopk is None) == (lsz is None), 'if "ltopk" is defined, "lsz" needs to be provided too'
    num_classes = scr.shape[-1]
    if num_classes == 1:
        idx = torch.nonzero(scr > score_thr).squeeze()
        idx = top_per_level(idx, scr, ltopk, lsz)
        scores = scr[idx]
        labels = None
    elif not multiclassbox:
        s, l = torch.max(scr, dim=-1)
        idx = torch.nonzero(s > score_thr).squeeze()
        idx = top_per_level(idx, s, ltopk, lsz)
        scores = s[idx]
        labels = l[idx]
    else:                
        s = scr.flatten()
        idx = torch.nonzero(s > score_thr).squeeze()
        idx = top_per_level(idx, s, ltopk, lsz, num_classes)
        scores = s[idx]
        labels = idx % num_classes
        idx = torch.div(idx, num_classes, rounding_mode='floor')
    return idx, scores, labels


def top_per_level(idx, s, topk, lvl_sizes, mult=1):
    """"""
    if not topk:
        return idx
    borders = np.cumsum([0] + lvl_sizes) * mult
    sel = []
    for u in range(1, len(borders)):
        lidx = idx[(idx >= borders[u - 1]) * (idx < borders[u])]
        _, top = torch.topk(s[lidx], min(topk, lidx.shape[0]))
        sel.append(lidx[top])
    return torch.cat(sel)


def clamp_to_canvas(boxes, img_size):
    boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], max=img_size[1])
    boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], max=img_size[0])
    return boxes


def do_nms(boxes, scores, labels, iou_thr, top=None):
    keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_thr)
    keep = keep if not top else keep[:top]
    b, s, l = [x[keep].detach().cpu().numpy() for x in [boxes, scores, labels]]
    return b, s, l


def scale_back(boxes, size_orig, size_used):
    boxes[:, 0::2] *= size_orig[1] / size_used[1]
    boxes[:, 1::2] *= size_orig[0] / size_used[0]
    return boxes


def make_anchors(dims, scales=[1], ratios=[1]):
    """For every possible combination (D, S, R) of dims, scales and ratios,
    makes a box with area = D*D*S*S and aspect ratio = R,
    returning len(dims) lists, each with len(scales) * len(ratios) tuples.

    Example: make_anchors([16, 32], scales=[1, 0.5, 0.1], ratios=[1, 2])
    Output: [[(16, 16), (8, 8), (1.6, 1.6), (22.63, 11.31), (11.31, 5.66), (2.26, 1.13)],
             [(32, 32), (16, 16), (3.2, 3.2), (45.25, 22.63), (22.63, 11.31), (4.53, 2.26)]]
    """
    mult = [math.sqrt(ar) for ar in ratios]
    anchors = [[(d * s * m, d * s / m) for m in mult for s in scales] for d in dims]
    return anchors


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
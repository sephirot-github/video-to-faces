from ctypes import wstring_at
import math
import sys

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


def do_nms(boxes, scores, groups, iou_thr):
    if groups is None:
        keep = torchvision.ops.nms(boxes, scores, iou_thr)
    else:
        keep = torchvision.ops.batched_nms(boxes, scores, groups, iou_thr)
    return keep


def clamp_to_canvas(boxes, img_size):
    boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=0, max=img_size[1])
    boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=0, max=img_size[0])
    return boxes


def scale_back(boxes, size_orig, size_used):
    boxes[:, 0::2] *= size_orig[1] / size_used[1]
    boxes[:, 1::2] *= size_orig[0] / size_used[0]
    return boxes


def clamp_to_canvas_vect(boxes, sizes_used, imidx):
    mx = torch.tensor(sizes_used).flip(1).repeat(1, 2)[imidx, :]
    boxes.clamp_(min=torch.tensor(0), max=mx)
    return boxes


def scale_back_vect(boxes, sizes_orig, sizes_used, imidx):
    sz_orig = torch.tensor(sizes_orig).to(boxes.device)
    sz_used = torch.tensor(sizes_used).to(boxes.device)
    scl = sz_orig / sz_used
    boxes *= scl.flip(1).repeat(1, 2)[imidx, :]
    return boxes


def remove_small(min_size, boxes, *args):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    mask = (ws >= min_size) & (hs >= min_size)
    if torch.count_nonzero(mask) < boxes.shape[0]:
        boxes = boxes[mask]
        args = [t[mask] if (t is not None) else None for t in args]
    return [boxes, *args]


#def remove_small(boxes, min_size):
#    ws = boxes[:, 2] - boxes[:, 0]
#    hs = boxes[:, 3] - boxes[:, 1]
#    mask = (ws >= min_size) & (hs >= min_size)
#    return mask


def get_results(reg, scr, priors, score_thr, iou_thr, decode, lvtop=None, lvsizes=None,
                scale=False, clamp=False, min_size=None, sizes_used=None, sizes_orig=False,
                nms_per_level=False, multiclassbox=False, imtop=None, implementation='vectorized'):
    """
    """
    assert implementation in ['vectorized', 'loop']
    if scr.ndim == 2:
        scr = scr.unsqueeze(-1)
    if nms_per_level:
        lvidx = torch.arange(len(lvsizes)).repeat_interleave(torch.tensor(lvsizes))
    else:
        lvidx = None
    
    if implementation == 'loop':
        res = []
        for i in range(reg.shape[0]):
            idx, scores, classes = select_by_score(scr[i], score_thr, multiclassbox, (lvtop, lvsizes))
            boxes = decode_boxes(reg[i][idx], priors[idx], settings=decode)
            if clamp:
                boxes = clamp_to_canvas(boxes, sizes_used[i])
            if min_size:
                boxes, scores, classes, lvidx = remove_small(min_size, boxes, scores, classes, lvidx[idx])
            keep = do_nms(boxes, scores, classes, iou_thr)[:imtop]
            b, s, c = boxes[keep], scores[keep], classes[keep] if classes is not None else None
            if scale:
                b = scale_back(b, sizes_orig[i], sizes_used[i])
            res.append((b, s, c))
        bl, sl, cl = map(list, zip(*res))
        return bl, sl, cl
             
    if implementation == 'vectorized':
        n, dim = reg.shape[:2]
        reg = reg.reshape(-1, reg.shape[-1])
        scr = scr.reshape(-1, scr.shape[-1])
        idx, scores, classes = select_by_score(scr, score_thr, multiclassbox, (lvtop, lvsizes, n))
        imidx = idx.div(dim, rounding_mode='floor')
        grpidx = imidx if not nms_per_level else imidx * 10 + lvidx[idx % dim]
        boxes = decode_boxes(reg[idx], priors[idx % dim], settings=decode)
        if clamp:
            boxes = clamp_to_canvas_vect(boxes, sizes_used, imidx)
        if min_size:
            boxes, scores, classes, imidx, grpidx = remove_small(min_size, boxes, scores, classes, imidx, grpidx)
        if classes is None:
            keep = do_nms(boxes, scores, grpidx, iou_thr)
            boxes, scores, imidx = [x[keep] for x in [boxes, scores, imidx]]
            cl = None
        else:
            groups = grpidx * 1000 + classes
            keep = do_nms(boxes, scores, groups, iou_thr)
            boxes, scores, classes, imidx = [x[keep] for x in [boxes, scores, classes, imidx]]
            cl = [classes[imidx == i][:imtop] for i in range(n)]
        if scale:
            boxes = scale_back_vect(boxes, sizes_orig, sizes_used, imidx)
        bl = [boxes[imidx == i][:imtop] for i in range(n)]
        sl = [scores[imidx == i][:imtop] for i in range(n)]
        return bl, sl, cl
     

def select_by_score(scr, score_thr, multiclassbox=False, lvset=None):
    """"""
    num_classes = scr.shape[-1]
    if num_classes == 1:
        s = scr.squeeze()
        idx = torch.nonzero(s > score_thr).squeeze()
        idx = top_per_level(idx, s, lvset)
        scores = s[idx]
        classes = None
    elif not multiclassbox:
        s, c = torch.max(scr, dim=-1)
        idx = torch.nonzero(s > score_thr).squeeze()
        idx = top_per_level(idx, s, lvset)
        scores = s[idx]
        classes = c[idx]
    else:
        s = scr.flatten()
        idx = torch.nonzero(s > score_thr).squeeze()
        idx = top_per_level(idx, s, lvset, num_classes)
        scores = s[idx]
        classes = idx % num_classes
        idx = torch.div(idx, num_classes, rounding_mode='floor')
    return idx, scores, classes


def top_per_level(idx, s, lvset, mult=1):
    """"""
    if lvset is None or lvset[0] is None:
        return idx
    limit, level_sizes = lvset[:2]
    nimg = 1 if len(lvset) < 3 else lvset[2]
    sel = []
    for i in range(nimg):
        borders = np.cumsum([0] + level_sizes)
        borders += i * sum(level_sizes)
        borders *= mult
        for j in range(1, len(borders)):
            lvidx = idx[(idx >= borders[j - 1]) * (idx < borders[j])]
            _, top = torch.topk(s[lvidx], min(limit, lvidx.shape[0]))
            sel.append(lvidx[top])
    return torch.cat(sel)


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


def make_anchors_rounded(dims, scales, ratios):
    """Same as make_anchors but with two intermediate roundings to replicate TorchVision code:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py#L58
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py#L51
    """
    mult = [math.sqrt(ar) for ar in ratios]
    dims_scaled = [[int(d * s) for s in scales] for d in dims]
    anchors = [[(int(d * s) * m, int(d * s) / m) for m in mult for s in scales] for d in dims]
    anchors = [[(round(w / 2) * 2, round(h / 2) * 2) for (w, h) in group] for group in anchors]
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


def decode_boxes(pred, priors, settings):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    mult_xy, mult_wh = settings[:2]
    max_exp_input = None if len(settings) < 3 else settings[2]
    xys = priors[:, 2:] * mult_xy * pred[..., :2] + priors[:, :2]
    whs = priors[:, 2:] * exp_clamped(mult_wh * pred[..., 2:], max_exp_input)
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


def exp_clamped(x, max_=None):
    if not max_:
        return torch.exp(x)
    else:
        return torch.exp(torch.clamp(x, max=max_))
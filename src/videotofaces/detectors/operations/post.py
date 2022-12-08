import math

import numpy as np
import torch
import torchvision.ops


def get_results(reg, scores, priors, score_thr, iou_thr, decode_mults):
    n, dim = reg.shape[:2]
    reg, scores = reg.reshape(-1, 4), scores.flatten()
    idx = torch.nonzero(scores > score_thr).squeeze()
    scores = scores[idx]
    imidx = idx.div(dim, rounding_mode='floor')
    boxes = decode_boxes(reg[idx], priors[idx % dim], mults=decode_mults)
    res = []
    for i in range(n):
        bi, si = boxes[imidx == i], scores[imidx == i]
        keep = torchvision.ops.nms(bi, si, iou_thr)
        res.append((bi[keep], si[keep]))
    return map(list, zip(*res))


def clamp_to_canvas(boxes, imsizes, imidx):
    mx = torch.tensor(imsizes).flip(1).repeat(1, 2)[imidx, :]
    boxes.clamp_(min=torch.tensor(0), max=mx)
    return boxes


def remove_small(boxes, min_size, *args):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    mask = (ws >= min_size) & (hs >= min_size)
    #keep = (boxes[:, 2:] - boxes[:, :2] >= min_size).all(dim=1)
    if torch.count_nonzero(mask) < boxes.shape[0]:
        boxes = boxes[mask]
        args = [t[mask] if (t is not None) else None for t in args]
    return [boxes, *args]


def scale_back(boxes, sz_orig, sz_used):
    scalebacks = torch.tensor(sz_orig) / torch.tensor(sz_used)
    scalebacks = scalebacks.flip(1).repeat(1, 2)
    boxes = [boxes[i] * scalebacks[i] for i in range(len(boxes))]
    return boxes


def get_lvidx(idx, lvsizes):
    """
    dim = 8000
    lsz = [3500, 2500, 1000, 600, 400]
    idx = torch.tensor([0, 5999, 6000, 7999, 8000, 15000, 20000])

    res = tensor([0, 1, 2, 4, 0, 3, 1])
    cumsum = tensor([3500, 6000, 7000, 7600, 8000])
    idx % dim = tensor([   0, 5999, 6000, 7999,    0, 7000, 4000])
    """
    boundaries = torch.tensor(lvsizes).cumsum(0)
    return torch.bucketize(idx, boundaries, right=True)
    #return torch.gt(boundaries, idx[:, None]).to(dtype=int).argmax(dim=1)


def top_per_level(idx, scores, limit, lvlen, nimg, mult=1):
    """"""
    if not limit:
        return idx
    sel = []
    for i in range(nimg):
        borders = np.cumsum([0] + lvlen)
        borders += i * sum(lvlen)
        borders *= mult
        for j in range(1, len(borders)):
            lvidx = idx[(idx >= borders[j - 1]) * (idx < borders[j])]
            if lvidx.shape[0] > limit:
                _, top = torch.topk(scores[lvidx], min(limit, lvidx.shape[0]))
                sel.append(lvidx[top])
            else:
                sel.append(lvidx)
    return torch.cat(sel)


def make_anchors(dims, scales=[1], ratios=[1], rounding=False):
    """For every possible combination (D, S, R) of dims, scales and ratios,
    makes a box with area = D*D*S*S and aspect ratio = R,
    returning len(dims) lists, each with len(scales) * len(ratios) tuples.

    Example: make_anchors([16, 32], scales=[1, 0.5, 0.1], ratios=[1, 2])
    Output: [[(16, 16), (8, 8), (1.6, 1.6), (22.63, 11.31), (11.31, 5.66), (2.26, 1.13)],
             [(32, 32), (16, 16), (3.2, 3.2), (45.25, 22.63), (22.63, 11.31), (4.53, 2.26)]]
    """
    if rounding:
        return _make_anchors_rounded(dims, scales, ratios)
    mult = [math.sqrt(ar) for ar in ratios]
    anchors = [[(d * s * m, d * s / m) for m in mult for s in scales] for d in dims]
    return anchors


def _make_anchors_rounded(dims, scales, ratios):
    """Same as make_anchors but with two intermediate roundings to replicate TorchVision code:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py#L58
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py#L51
    """
    mult = [math.sqrt(ar) for ar in ratios]
    dims_scaled = [[int(d * s) for s in scales] for d in dims]
    anchors = [[(int(d * s) * m, int(d * s) / m) for m in mult for s in scales] for d in dims]
    anchors = [[(round(w / 2) * 2, round(h / 2) * 2) for (w, h) in group] for group in anchors]
    return anchors


def get_priors(img_size, bases, dv='cpu', loc='center', patches='as_is', concat=True):
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
    if not concat:
        return p
    return torch.cat(p)


def convert_to_cwh(boxes):
    """from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    boxes[..., 2:] -= boxes[..., :2]
    boxes[..., :2] += boxes[..., 2:] * 0.5
    return boxes


def decode_boxes(pred, priors, mults=(1, 1), clamp=False, mode='rcnn', strides=None):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    assert mode in ['rcnn', 'yolo']
    if mode == 'yolo':
        assert mults == (1, 1), strides is not None
    mult_xy, mult_wh = mults
    max_exp_input = math.log(1000 / 16) if clamp else None
    if mode == 'rcnn':
        xys = priors[..., 2:] * mult_xy * pred[..., :2] + priors[..., :2]
    elif mode == 'yolo':
        xys = strides * (pred[..., :2].sigmoid() - 0.5) + priors[..., :2]
    whs = priors[..., 2:] * exp_clamped(mult_wh * pred[..., 2:], max_exp_input)
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


def exp_clamped(x, max_=None):
    if not max_:
        return torch.exp(x)
    else:
        return torch.exp(torch.clamp(x, max=max_))


def assign_fpn_levels(boxes, strides):
    """FPN Paper, Eq.1 https://arxiv.org/pdf/1612.03144.pdf"""
    kmin = math.log2(strides[0])
    kmax = math.log2(strides[-1])
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    k = 4 + torch.log2(torch.sqrt(ws * hs) / 224)
    k = torch.clamp(k, min=kmin, max=kmax)
    mapidx = (k - kmin).to(torch.int64)
    return mapidx


def roi_align_multilevel(boxes, imidx, fmaps, strides, settings):
    # https://arxiv.org/pdf/1703.06870.pdf
    # https://github.com/pytorch/vision/issues/4935
    # https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
    # https://chao-ji.github.io/jekyll/update/2018/07/20/ROIAlign.html
    sratio, aligned = settings
    mapidx = assign_fpn_levels(boxes, strides)
    imboxes = torch.hstack([imidx.unsqueeze(-1), boxes])
    roi_maps = torch.zeros((len(mapidx), fmaps[0].shape[1], 7, 7))
    for level in range(len(strides)):
        scale = 1 / strides[level]
        idx = torch.nonzero(mapidx == level).squeeze()
        roi = torchvision.ops.roi_align(fmaps[level], imboxes[idx], (7, 7), scale, sratio, aligned)
        roi_maps[idx] = roi.to(roi_maps.dtype)
    return roi_maps




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
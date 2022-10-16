import numpy as np
import torch

def nms_ref_numpy(dets, scores, thresh):
    """
    https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    but no +1
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 ) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    print(order.size)
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        print(order.size)
    print()
    return keep


def nms_ref_torch(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)
    keep = []
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


def nms_custom_torch(boxes, scores, groups, threshold, method='union'):
    import cProfile, pstats
    pr = cProfile.Profile(); pr.enable()

    dv = boxes.device
    order = torch.argsort(scores, descending=True)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    c = torch.Tensor([]).to(dv, torch.int64)
    for i in groups.unique():
        ci = torch.combinations(order[groups[order] == i]).to(dv)
        c = torch.cat((c, ci))
    pairs = torch.hstack((boxes[c[:, 0]], boxes[c[:, 1]]))
    area1 = areas[c[:, 0]]
    area2 = areas[c[:, 1]]
    ix1 = torch.maximum(pairs[:, 0], pairs[:, 4])
    iy1 = torch.maximum(pairs[:, 1], pairs[:, 5])
    ix2 = torch.minimum(pairs[:, 2], pairs[:, 6])
    iy2 = torch.minimum(pairs[:, 3], pairs[:, 7])
    iw = torch.clamp(ix2 - ix1, min=0)
    ih = torch.clamp(iy2 - iy1, min=0)
    inter = iw * ih
    if method == 'union':
        iou = inter / (area1 + area2 - inter)
        idx = (iou > threshold)
    elif method == 'min':
        iom = inter / torch.minimum(area1, area2)
        idx = (iom > threshold)
    else:
        raise ValueError('unknown method')
    suppressed = set()
    c = c[idx].tolist()
    for i, j in c:
        if i not in suppressed:
            suppressed.add(j)
    keep = torch.isin(order, torch.Tensor(list(suppressed)), assume_unique=True, invert=True)
    res = order[keep]
    
    pr.disable()
    #pstats.Stats(pr).sort_stats('cumtime').print_stats(10)
    #print()
    #print()
    return res

def nms_custom_numpy(boxes, scores, groups, threshold, method='union'):
    import cProfile, pstats
    pr = cProfile.Profile(); pr.enable()

    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    c = []
    for i in np.unique(groups):
        gidx = order[groups[order] == i]
        coord = np.stack(np.meshgrid(gidx, gidx), axis=-1).transpose(1, 0, 2)
        coord = coord[np.triu_indices(len(gidx), k=1)]
        c.append(coord)
    c = np.vstack(c)
    pairs = np.hstack((boxes[c[:, 0]], boxes[c[:, 1]]))
    area1 = areas[c[:, 0]]
    area2 = areas[c[:, 1]]
    ix1 = np.maximum(pairs[:, 0], pairs[:, 4])
    iy1 = np.maximum(pairs[:, 1], pairs[:, 5])
    ix2 = np.minimum(pairs[:, 2], pairs[:, 6])
    iy2 = np.minimum(pairs[:, 3], pairs[:, 7])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    if method == 'union':
        iou = inter / (area1 + area2 - inter)
        idx = (iou > threshold)
    elif method == 'min':
        iom = inter / torch.minimum(area1, area2)
        idx = (iom > threshold)
    else:
        raise ValueError('unknown method')
    suppressed = set()
    c = c[idx]
    for i, j in c:
        if i not in suppressed:
            suppressed.add(j)
    keep = np.isin(order, list(suppressed), assume_unique=True, invert=True)
    res = order[keep]

    pr.disable()
    pstats.Stats(pr).sort_stats('cumtime').print_stats(10)
    print()
    print()

    return res


    """
    Example:
    boxes = [[5, 5, 105, 60], [0, 0, 100, 50], [9, 9, 109, 69], [20, 20, 80, 80],
             [25, 25, 80, 80], [30, 30, 90, 90], [5, 80, 15, 90]]
    scores = [0.88, 0.95, 0.5, 0.1, 0.6, 0.8, 0.9]
    classes = [0, 0, 0, 0, 1, 1, 1]
    boxes' indices sorted by scores in descending order = [1, 6, 0, 5, 4, 2, 3]
    
    i  j  area1 area2 inter union     iou  thr   box j suppressed?
    1  0   5000  5500  4275  6225  0.6867  0.6   yes
    1  2   5000  6000  3731  7269  0.5133  0.6   no
    1  3   5000  3600  1800  6800  0.2647  0.6   no
    0  2   5500  6000  4896  6604  0.7414  0.6   no, since 0 was already eliminated above
    0  3   5500  3600  2400  6700  0.3582  0.6   no
    2  3   6000  3600  2940  6660  0.4414  0.6   no
    6  5   100   3600     0  3700  0.0000  0.6   no
    6  4   100   3025     0  3125  0.0000  0.6   no
    5  4   3600  3025  2500  4125  0.6061  0.6   yes

    returns: [1, 6, 5, 2, 3]
    """
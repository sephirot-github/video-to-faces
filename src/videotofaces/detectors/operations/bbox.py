import torch


def encode(boxes, priors, mults):
    """boxes - (x1, y1, x2, y2), priors - (x1, y1, x2, y2)"""
    bx = convert_to_cwh(boxes)
    pr = convert_to_cwh(priors)
    rel_xys = (bx[..., :2] - pr[..., :2]) / pr[..., 2:] / mults[0]
    rel_whs = torch.log(bx[..., 2:] / pr[..., 2:]) / mults[1]
    return torch.cat([rel_xys, rel_whs], dim=-1)


def convert_to_cwh(boxes):
    """from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    boxes[..., 2:] -= boxes[..., :2]
    boxes[..., :2] += boxes[..., 2:] * 0.5
    return boxes


def convert_to_xyxy(boxes):
    """from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    boxes[..., :2] -= boxes[..., 2:] * 0.5
    boxes[..., 2:] += boxes[..., :2]
    return boxes


def calc_iou_matrix(a, b, plus1=False):
    """Calculates intersection over union for every pair of two groups of boxes.
    I.e. if a.shape = (n, 4) and b.shape = (m, 4), then result.shape = (n, m),
    and result[i][j] = IoU between box a[i] and box b[j].

    Algorithm: https://stackoverflow.com/a/42874377
    Implementation: https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L255
    On doing +1 or not: https://stackoverflow.com/a/51730512/8874388

    In general, doing x[:, None] + y allows you to get pairwise results thanks to
    automatic broadcasting: (N,1,D) + (M,D) <=> (N,1,D) + (1,M,D) = (N,M,D)
    """
    p = 0 if not plus1 else 1
    area1 = (a[:, 2] - a[:, 0] + p) * (a[:, 3] - a[:, 1] + p)
    area2 = (b[:, 2] - b[:, 0] + p) * (b[:, 3] - b[:, 1] + p)
    lt = torch.maximum(a[:, None, :2], b[:, :2])
    rb = torch.minimum(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt + p).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union
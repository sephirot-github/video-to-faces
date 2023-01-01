import torch

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


def assign_gt_to_priors(gtboxes, priors, low_thr, high_thr, match_low_quality):
    """
    """
    m = calc_iou_matrix(gtboxes, priors)
    v, idx = m.max(dim=0)
    idx += 1
    if match_low_quality:
        copied = idx.clone()
    idx[v < low_thr] = 0
    idx[(v >= low_thr) & (v < high_thr)] = -1
    if match_low_quality:
        maxgt = m.max(dim=1)[0]
        extra = torch.where(m == maxgt[:, None])[1]
        idx[extra] = copied[extra]
    return idx
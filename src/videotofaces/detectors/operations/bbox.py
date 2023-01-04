import torch


def encode(reference_boxes, proposals, weights):
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


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
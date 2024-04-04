import math

import torch


def decode_boxes(pred, priors, mult_xy, mult_wh):
    xys = priors[..., 2:] * mult_xy * pred[..., :2] + priors[..., :2]
    whs = priors[..., 2:] * torch.exp(torch.clamp(mult_wh * pred[..., 2:], max=math.log(1000 / 16)))
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


def convert_to_cwh(boxes, in_place=False):
    """from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    ret = boxes if in_place else boxes.clone()
    ret[..., 2:] -= ret[..., :2]
    ret[..., :2] += ret[..., 2:] * 0.5
    return ret


def clamp_to_canvas(boxes, imsizes, imidx):
    dv = imidx.device
    mx = torch.tensor(imsizes).to(dv).flip(1).repeat(1, 2)[imidx, :]
    boxes.clamp_(min=torch.tensor(0).to(dv), max=mx)
    return boxes


def remove_small(boxes, min_size, *args):
    boxes = boxes.view(-1, 4)
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    mask = (ws > min_size) & (hs > min_size)
    if torch.count_nonzero(mask) < boxes.shape[0]:
        boxes = boxes[mask]
        args = [t[mask] if (t is not None) else None for t in args]
    return [boxes.squeeze(), *args]


def scale_boxes(boxes, target_imsizes, current_imsizes):
    scales = torch.tensor(target_imsizes) / torch.tensor(current_imsizes)
    scales = scales.to(boxes[0].device).flip(1).repeat(1, 2)
    boxes = [boxes[i] * scales[i] for i in range(len(boxes))]
    return boxes
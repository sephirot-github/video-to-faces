import math

import numpy as np
import torch
import torch.nn.functional as F

# source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py


def normalize(cv2_images, device, means, stds, to0_1=True, toRGB=True):
    """"""
    ts = []
    for img in cv2_images:
        t = torch.from_numpy(img).to(device, torch.float32)
        if to0_1:
            t /= 255
        if toRGB:
            t = t[:, :, [2, 1, 0]]
        if means:
            t -= torch.tensor(list(means), device=device)
        if stds:
            t /= torch.tensor(list(stds), device=device)
        ts.append(t.permute(2, 0, 1))
    return ts


def resize(ts, resize_min, resize_max):
    """"""
    sz_orig, sz_used = [], []
    for i in range(len(ts)):
        sz = ts[i].shape[1:3]
        scl = min(resize_min / min(sz), resize_max / max(sz))
        ts[i] = F.interpolate(ts[i].unsqueeze(0), None, scl, 'bilinear', recompute_scale_factor=True)[0]
        sz_orig.append(sz)
        sz_used.append(ts[i].shape[1:3])
    scales = torch.tensor(sz_used) / torch.tensor(sz_orig)
    return ts, scales


def batch(ts, mult):
    """"""
    hmax = max([t.shape[1] for t in ts])
    wmax = max([t.shape[2] for t in ts])
    hmax = int(math.ceil(hmax / mult) * mult)
    wmax = int(math.ceil(wmax / mult) * mult)
    x = torch.full((len(ts), 3, hmax, wmax), 0, dtype=torch.float32, device=ts[0].device)
    for i in range(len(ts)):
        x[i, :, :ts[i].shape[1], :ts[i].shape[2]].copy_(ts[i])
    return x
import math

import numpy as np
import torch
import torch.nn.functional as F


def make_batch(ts, sz_orig, resize_min=800, resize_max=1333, mult=32):
    for i in range(len(ts)):
        scl = min(resize_min / min(sz_orig[i]), resize_max / max(sz_orig[i]))
        ts[i] = F.interpolate(ts[i][None], size=None, scale_factor=scl, mode='bilinear',
                              recompute_scale_factor=True, align_corners=False)[0]
    #for t in ts: print(t.shape)
    sz_used = [t.shape[-2:] for t in ts]
    hmax = max([s[0] for s in sz_used])
    wmax = max([s[1] for s in sz_used])
    hmax = int(math.ceil(hmax / mult) * mult)
    wmax = int(math.ceil(wmax / mult) * mult)
    x = torch.full((len(ts), 3, hmax, wmax), 0, dtype=torch.float32, device=ts[0].device)
    for i in range(len(ts)):
        x[i, :, :ts[i].shape[1], :ts[i].shape[2]].copy_(ts[i])
    #print(x.shape)
    scales = torch.tensor(sz_used) / torch.tensor(sz_orig)
    #print(scales)
    return x, scales


def preprocess(imgs, device, norm_params, allow_var_sizes):
    """https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py
    """
    to0_1, swapRB, means, stds = norm_params
    means = torch.tensor(list(means), device=device)
    stds = torch.tensor(list(stds), device=device)
    sz = [img.shape[:2] for img in imgs]
    
    ts = []
    for img in imgs:
        t = torch.from_numpy(img).to(device, torch.float32)
        if to0_1: t /= 255
        t = t if not swapRB else t[:, :, [2, 1, 0]]
        t -= means
        t /= stds
        ts.append(t.permute(2, 0, 1))
    
    if len(np.unique(sz, axis=0)) == 1:
        x = torch.stack(ts)
        scales = None
    elif allow_var_sizes:
        x, scales = make_batch(ts, sz)
    else:
        raise ValueError('Input images are different sizes but allow_var_sizes = False')
    return x, scales
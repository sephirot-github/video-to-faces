import math

import cv2
import torch
import torch.nn.functional as F

# source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py


def full(imgs, dv, resize, resize_with='cv2', keep_ratio=True, norm='imagenet', size_divisible=32):
    assert resize_with in ['cv2', 'torch']
    if not isinstance(resize, tuple):
        resize = (resize, resize)
    if resize_with == 'cv2':
        imgs, sz_orig, sz_used = resize_cv2(imgs, resize, keep_ratio)
        ts = to_tensors(imgs, dv, norm=norm)
    elif resize_with == 'torch':
        ts = to_tensors(imgs, dv, norm=norm)
        ts, sz_orig, sz_used = resize_torch(ts, resize, keep_ratio)
    x = pad_and_batch(ts, mult=size_divisible)
    return x, sz_orig, sz_used


def to_tensors(cv2_images, device, norm, to0_1=True, toRGB=True):
    """"""
    means = [0.485, 0.456, 0.406] if norm == 'imagenet' else (norm[0] if norm else None) # ~ [123.675, 116.28, 103.53]
    stdvs = [0.229, 0.224, 0.225] if norm == 'imagenet' else (norm[1] if norm else None) # ~ [58.395, 57.12, 57.375]
    ts = []
    for img in cv2_images:
        t = torch.from_numpy(img).to(device, torch.float32)
        if to0_1:
            t /= 255
        if toRGB:
            t = t[:, :, [2, 1, 0]]
        if means:
            if isinstance(means, tuple):
                means = list(means)
            t -= torch.tensor(means, device=device)
        if stdvs:
            if isinstance(stdvs, tuple):
                stdvs = list(stdvs)
            t /= torch.tensor(stdvs, device=device)
        ts.append(t.permute(2, 0, 1))
    return ts


def resize_torch(ts, resize, keep_ratio):
    """"""
    sz_orig, sz_used = [], []
    for i in range(len(ts)):
        sz = ts[i].shape[1:3]
        if keep_ratio:
            rmin, rmax = resize
            scl = min(rmin / min(sz), rmax / max(sz))
            ts[i] = F.interpolate(ts[i].unsqueeze(0), None, scl, 'bilinear', recompute_scale_factor=True)[0]
        else:
            r = resize if isinstance(resize, tuple) else (resize, resize)
            ts[i] = F.interpolate(ts[i].unsqueeze(0), size=r, mode='bilinear')[0]
        sz_orig.append(sz)
        sz_used.append(ts[i].shape[1:3])
    return ts, sz_orig, sz_used


def resize_cv2(imgs, resize, keep_ratio):
    res, sz_orig, sz_used = [], [], []
    for i in range(len(imgs)):
        sz = imgs[i].shape[:2]
        if keep_ratio:
            rmin, rmax = resize
            scl = min(rmin / min(sz), rmax / max(sz))
            n = int(sz[0] * scl + 0.5), int(sz[1] * scl + 0.5)
        else:
            n = resize
        im = cv2.resize(imgs[i], n[::-1], interpolation=cv2.INTER_LINEAR)
        res.append(im)
        sz_orig.append(sz)
        sz_used.append(n)
    return res, sz_orig, sz_used


def pad_and_batch(ts, mult):
    """"""
    hmax = max([t.shape[1] for t in ts])
    wmax = max([t.shape[2] for t in ts])
    hmax = int(math.ceil(hmax / mult) * mult)
    wmax = int(math.ceil(wmax / mult) * mult)
    x = torch.full((len(ts), 3, hmax, wmax), 0, dtype=torch.float32, device=ts[0].device)
    for i in range(len(ts)):
        x[i, :, :ts[i].shape[1], :ts[i].shape[2]].copy_(ts[i])
    return x
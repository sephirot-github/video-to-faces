import torch
import torchvision.ops

def final_nms(boxes, scores, classes, imidx, n, thresh, imtop=None):
    res = []
    for i in range(n):
        bi, si, ci = [x[imidx == i] for x in [boxes, scores, classes]]
        keep = torchvision.ops.batched_nms(bi, si, ci, thresh)[:imtop]
        res.append((bi[keep], si[keep], ci[keep]))
    return map(list, zip(*res))


def get_lvidx(idx, lvsizes):
    boundaries = torch.tensor(lvsizes).to(idx.device).cumsum(0)
    return torch.bucketize(idx, boundaries, right=True)
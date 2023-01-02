import torch

from .bbox import calc_iou_matrix


def assign_gt_to_priors(gtboxes, priors, low_thr, high_thr, match_low_quality):
    """"""
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


def random_balanced_sampler(gtidx, num, pos_fraction):
    pos = torch.nonzero(gtidx >= 1).squeeze()
    neg = torch.nonzero(gtidx == 0).squeeze()
    np = min(pos.numel(), int(256 * 0.5))
    nn = min(neg.numel(), 256 - np)
    perm1 = torch.randperm(pos.numel())[:np]
    perm2 = torch.randperm(neg.numel())[:nn]
    perm1 = torch.sort(perm1)[0]
    perm2 = torch.sort(perm2)[0]
    return pos[perm1], neg[perm2]


def get_sidx(gtboxes, priors):
    lp, ln = [], []
    for i, b in enumerate(gtboxes):
        gtidx = assign_gt_to_priors(b, priors, 0.3, 0.7, True)
        pos, neg = random_balanced_sampler(gtidx, 256, 0.5)
        shift = i * len(priors)
        pos += shift
        neg += shift
        lp.append(torch.sort(pos)[0])
        ln.append(torch.sort(neg)[0])
    lp = torch.cat(lp)
    ln = torch.cat(ln)
    lt = torch.cat([lp, ln])
    return lp, lt
import torch
import torch.nn.functional as F

from .bbox import calc_iou_matrix, convert_to_xyxy, convert_to_cwh, encode_boxes


def assign(gtboxes, priors, low_thr, high_thr, match_low_quality):
    """"""
    res = []
    for i in range(len(gtboxes)):
        pr = priors[i] if isinstance(priors, list) else priors
        m = calc_iou_matrix(gtboxes[i], pr)
        v, idx = m.max(dim=0)
        if match_low_quality:
            copied = idx.clone()
        idx[v < low_thr] = -1
        idx[(v >= low_thr) & (v < high_thr)] = -2
        if match_low_quality:
            maxgt = m.max(dim=1)[0]
            extra = torch.where(m == maxgt[:, None])[1]
            idx[extra] = copied[extra]
        res.append(idx)
    return res


def random_balanced_sampler(gtidx, num, pos_fraction):
    pos = torch.nonzero(gtidx >= 0).squeeze()
    neg = torch.nonzero(gtidx == -1).squeeze()
    np = min(pos.numel(), int(num * pos_fraction))
    nn = min(neg.numel(), num - np)
    perm1 = torch.randperm(pos.numel())[:np]
    perm2 = torch.randperm(neg.numel())[:nn]
    return pos[perm1], neg[perm2]


def get_losses(gtboxes, priors, regs, logs, matcher, sampler, types):
    """"""
    batch, pos_ratio = sampler
    lg, lb, inp, trg = [], [], [], []
    
    gtidxs = assign(gtboxes, convert_to_xyxy(priors), *matcher)

    for i in range(len(gtboxes)):
        pos, neg = random_balanced_sampler(gtidxs[i], batch, pos_ratio)
        all_ = torch.cat([pos, neg])
        logits = logs[i][all_].squeeze()
        labels = gtidxs[i][all_].clamp(max=0) + 1
        inputs = regs[i][pos]
        targets = encode_boxes(gtboxes[i][gtidxs[i][pos]], priors[pos], (1, 1))
        lg.append(logits)
        lb.append(labels)
        inp.append(inputs)
        trg.append(targets)
    lg, lb, inp, trg = [torch.cat(x) for x in [lg, lb, inp, trg]]
    loss_obj = F.binary_cross_entropy_with_logits(lg, lb.to(torch.float32))
    loss_reg = F.smooth_l1_loss(inp, trg, beta=1/9, reduction='sum') / (lg.numel())
    return loss_obj, loss_reg


def match_with_targets(gtboxes, gtlabels, priors, bg_iou, fg_iou, match_lq, batch, pos_ratio):
    mt, ml, sa, sp = [], [], [], []
    gtidxs = assign(gtboxes, priors, bg_iou, fg_iou, match_lq)
    for i in range(len(gtboxes)):
        pos, neg = random_balanced_sampler(gtidxs[i], batch, pos_ratio)
        all_ = torch.cat([pos, neg])
        matched_targets = encode_boxes(gtboxes[i][gtidxs[i][pos]], convert_to_cwh(priors[i][pos]), (0.1, 0.2))
        if gtlabels is None:
            matched_labels = gtidxs[i][all_].clamp(max=0) + 1
        else:
            gtidxs[i][pos] = gtlabels[i][gtidxs[i][pos]]
            matched_labels = gtidxs[i][all_]
            matched_labels[matched_labels == -1] = 0
        mt.append(matched_targets)
        ml.append(matched_labels)
        sa.append(all_)
        sp.append(pos)
    return mt, ml, sa, sp








import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

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


def sample_random(gtidxs, num, pos_fraction):
    sidx_all, sidx_pos = [], []
    for gtidx in gtidxs:
        pos = torch.nonzero(gtidx >= 0).squeeze()
        neg = torch.nonzero(gtidx == -1).squeeze()
        np = min(pos.numel(), int(num * pos_fraction))
        nn = min(neg.numel(), num - np)
        perm1 = torch.randperm(pos.numel())[:np]
        perm2 = torch.randperm(neg.numel())[:nn]
        pos = pos[perm1]
        neg = neg[perm2]
        sidx_all.append(torch.cat([pos, neg]))
        sidx_pos.append(pos)
    return sidx_all, sidx_pos


def get_matched_targets(gtboxes, priors, gtidxs, sidx_pos, enc_weights=(1, 1)):
    targets = []
    for i in range(len(gtboxes)):
        pr = priors[i] if isinstance(priors, list) else priors
        pos = sidx_pos[i]
        t = gtboxes[i][gtidxs[i][pos]]
        t = encode_boxes(t, pr[pos], enc_weights)
        targets.append(t)
    return targets
        
        
def get_matched_labels(gtlabels, gtidxs, sidx_all):
    labels = []
    for i in range(len(gtidxs)):
        all_ = sidx_all[i]
        if gtlabels is None:
            lb = gtidxs[i][all_].clamp(max=0) + 1
        else:
            lb = gtidxs[i][all_].clone()
            lb[lb >= 0] = gtlabels[i][lb[lb >= 0]]
            lb[lb == -1] = 0
        labels.append(lb)
    return labels


def get_losses(gtboxes, gtlabels, priors, regs, logs, matcher, sampler, types):
    """"""
    gtidxs = assign(gtboxes, convert_to_xyxy(priors), *matcher)
    if sampler is not None:
        sidx_all, sidx_pos = sample_random(gtidxs, *sampler)
    else:
        sidx_all = [torch.nonzero(gtidx != -2).squeeze() for gtidx in gtidxs]
        sidx_pos = [torch.nonzero(gtidx >= 0).squeeze() for gtidx in gtidxs]    
    
    targets = get_matched_targets(gtboxes, priors, gtidxs, sidx_pos)
    labels = get_matched_labels(gtlabels, gtidxs, sidx_all)

    n = len(gtboxes)
    logs = torch.cat([logs[i][sidx_all[i]].squeeze() for i in range(n)])
    regs = torch.cat([regs[i][sidx_pos[i]] for i in range(n)])
    labels = torch.cat([lb.to(torch.float32) for lb in labels])
    targets = torch.cat(targets)
    loss_obj = F.binary_cross_entropy_with_logits(logs, labels)
    loss_reg = F.smooth_l1_loss(regs, targets, beta=1/9, reduction='sum') / (labels.numel())
    return loss_obj, loss_reg


def match_with_targets(gtboxes, gtlabels, proposals, bg_iou, fg_iou, match_lq, batch, pos_ratio):
    mt, ml, sa, sp = [], [], [], []
    gtidxs = assign(gtboxes, proposals, bg_iou, fg_iou, match_lq)
    sidx_all, sidx_pos = sample_random(gtidxs, batch, pos_ratio)
    proposals = [convert_to_cwh(p) for p in proposals]
    targets = get_matched_targets(gtboxes, proposals, gtidxs, sidx_pos, (0.1, 0.2))
    labels = get_matched_labels(gtlabels, gtidxs, sidx_all)
    return targets, labels, sidx_all
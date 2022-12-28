import numpy as np
import torch

unused_idx = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]

def weights_91_to_80(t):
    idx = np.delete(np.arange(91), unused_idx) + 1
    idx = torch.tensor(idx)
    dims = t.shape[1:]
    z = t.view(-1, 91, *dims)
    t = torch.cat([z[:, idx], z[:, 0:1]], dim=1).reshape(-1, *dims)
    return t

def idx_91_to_80(a91):
    bins91 = np.array(unused_idx)
    a80 = a91 - np.digitize(a91, bins91)
    return a80

def idx_80_to_91(a80):
    bins80 = np.array(unused_idx) - np.arange(len(unused_idx))
    a91 = a80 + np.digitize(a80, bins80)
    return a91


def make_gt_file_from_official_annotations(set_type):
    import json
    import os.path as osp
    with open(osp.join('annotations', 'instances_' + set_type + '2017.json')) as f:
        d = json.load(f)
    gt = {}
    for a in d['annotations']:
        imid = a['image_id']
        if id not in gt:
            gt[imid] = {}
            gt[imid]['boxes'] = []
            gt[imid]['classes'] = []
        gt[imid]['boxes'].append(a['bbox'])
        gt[imid]['classes'].append(a['category_id'])
    with open('gt.json', 'w') as f:
        json.dump(gt, f)
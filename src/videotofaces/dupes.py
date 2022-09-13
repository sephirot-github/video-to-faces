import os
import os.path as osp

import cv2
import numpy as np
import sklearn.metrics


def ahash(img):
    """TBD"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (8, 8))
    diff = tiny > np.mean(tiny)
    return 1 * diff.flatten()


def remove_dupes_nearest(faces, hashes, hash_thr, save_params):
    """TBD"""
    out_dir, _, resize_to, _, _, save_dupes = save_params
    idx, log = [], []
    for k in range(len(faces)):
        img, fn = faces[k]
        h = ahash(img)
        if not hashes:
            hashes.append((h, fn))
        else:
            diffs = [(np.count_nonzero(h != p), pfn) for (p, pfn) in hashes[-5:]]
            md, md_fn = min(diffs, key=lambda a: a[0])
            log.append(','.join([fn, md_fn, str(md), '1' if md <= hash_thr else '0']))
            if md <= hash_thr:
                idx.append(k)
                if save_dupes:
                    img = img if not resize_to else resize_face(img, resize_to)
                    cv2.imwrite(osp.join(out_dir, 'intermediate', 'dupes1', fn), img)
            else:
                hashes.append((h, fn))

    if save_dupes:
        log_fn = osp.join(out_dir, 'intermediate', 'log_dupes1.csv')
        first_time = not osp.exists(log_fn)
        with open(log_fn, 'a') as f:
            if first_time:
                f.write('file_name,nearest_in_prev_5,hash_diff,marked_as_duplicate\n')
            for line in log:
                f.write('%s\n' % line)

    faces = [f for i, f in enumerate(faces) if i not in idx]
    return faces, hashes


def remove_dupes_overall(X, filenames, dup_params):
    """TBD"""
    measure_type, threshold, save_dupes, out_dir = dup_params

    # https://stackoverflow.com/questions/70902177/calculating-euclidean-distance-with-a-lot-of-pairs-of-points-is-too-slow-in-pyth
    if measure_type == 'hash':
        D = sklearn.metrics.pairwise_distances(X, metric=lambda a, b: np.count_nonzero(a != b))
        # D = nsklearn.metrics.pairwise_distances(X, metric='hamming') * 64 # equivalent
        D = D.astype(np.uint16)
    else:
        D = sklearn.metrics.pairwise_distances(X) # by default it's euclidian distance

    D += (1 - np.tri(X.shape[0], k=-1).astype(D.dtype)) * 10000
    mins = D.min(axis=1)
    inds = D.argmin(axis=1)
    idx = (mins <= threshold).nonzero()[0]
    dupes = [fn for i, fn in enumerate(filenames) if i in idx]
    goods = [fn for i, fn in enumerate(filenames) if i not in idx]
    X = np.delete(X, idx, axis=0)

    if not save_dupes:
        for fn in dupes:
            os.remove(osp.join(out_dir, 'faces', fn))
    else:
        if measure_type == 'hash':
            mdigit, mname = '2', 'hash_diff'
        else:
            mdigit, mname = '3', 'distance'
        dir = osp.join(out_dir, 'intermediate', 'dupes' + mdigit)
        os.makedirs(dir, exist_ok=True)
        for fn in dupes:
            os.replace(osp.join(out_dir, 'faces', fn), osp.join(dir, fn))
        with open(osp.join(out_dir, 'intermediate', 'log_dupes' + mdigit + '.csv'), 'w') as f:
            f.write('file_name,nearest_in_prev,' + mname + ',marked_as_duplicate\n')
            for i in range(1, len(filenames)):
                f.write('%s,%s,%s,%s\n' % (filenames[i], filenames[inds[i]], str(mins[i]), '1' if i in idx else '0'))
    
    if measure_type != 'hash' and idx.any():
        print('Removed %u near-duplicates' % idx.shape[0])

    return X, goods
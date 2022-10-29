import math
import os.path as osp

import cv2
import numpy as np

from ...utils.image import resize_keep_ratio, pad_to_square, read_imsize_binary
from ...utils.pbar import tqdm


def get_predictions(load, model, fn, updir, imdir, mult, bs):
    """Runs inference with ``model`` on ``fn`` images from ``imdir`` folder and returns predictions,
    also saving them on disk to ``updir/pred_<set_name>_<model_class_name>.npy`` as one stacked array
    where every row is: "image_index, x1, y1, x2, y2, score" (dtype = np.float32).

    Alternatively, if ``load`` is specified, treats it as a path to a saved array in the same format,
    loads it and converts back into a list with len = image count (while other params are ignored).
    """
    if load is None:
        sn = osp.basename(updir)
        mn = model.__class__.__name__
        print('Getting predictions for the images using %s model' % mn)
        if bs == 1:
            preds = predict(fn, model, imdir)
        else:
            preds = predict_batch(fn, model, imdir, bs)
            #preds = predict_batch_soft(fn, model, imdir, mult, bs)
        fp = osp.join(updir, 'pred_%s_%s.npy' % (sn, mn))
        print('Saving predictions for possible repeated use to: %s' % fp)
        inds = [np.array([i] * len(p))[:, None] for i, p in enumerate(preds)]
        flat = np.hstack([np.concatenate(inds), np.concatenate(preds)])
        np.save(fp, flat)
    else:
        print('Loading saved predictions from: %s' % load)
        flat = np.load(load)
        inds = flat[:, 0].astype(np.int64)
        preds = np.split(flat[:, 1:], np.bincount(inds).cumsum()[:-1])
    return preds


def predict(fn, model, imdir):
    """Runs inference with ``model`` on every image separately."""
    preds = []
    for k in tqdm(range(len(fn))):
        img = cv2.imread(osp.join(imdir, fn[k]))
        pred = model([img])[0]
        pred = pred[pred[:, 4].argsort()[::-1]].astype(np.float32)
        preds.append(pred)
    return preds


def predict_batch(fn, model, imdir, bs):
    preds = []
    with tqdm(total=len(fn)) as pbar:
        for bn in range(math.ceil(len(fn) / bs)):
            imgs = [cv2.imread(osp.join(imdir, p)) for p in fn[bs*bn:bs*(bn+1)]]
            imgs, scls = map(list, zip(*[resize_keep_ratio(im, 608) for im in imgs]))
            imgs = [pad_to_square(im) for im in imgs]
            pred = model(imgs)
            pred = [pred[i] / scls[i] for i in range(len(pred))]
            preds.append(pred)
            pbar.update(len(pred))
    return preds


def predict_batch_soft(fn, model, imdir, mult, bs):
    """Runs inference with ``model`` in batches of max size ``bs`` by trying to group images
    of the same size together. E.g. if a dataset has 140 1024x768 images, 7 1024x683 images and
    1 1020x761 image, then the 1st group will be processed in 5 batches (32, 32, 32, 32, 12),
    the 2nd group in 1 batch (7), and the last image will still be processed separately.

    To reduce the number of groups, also pads every image with 0s on the right and bottom so
    that its dimensions are the multiples of ``mult``. This shouldn't affect the evaluation
    process, since all faces remain at the same coordinates, while the added parts are
    effectively black bars where a detector should find nothing.

    Also prints the mean of actual resulting batch sizes to help gauge whether
    ``mult`` value led to enough group reduction or not.
    """
    whs, pads = [], []
    for f in fn:
        p = osp.join(imdir, f)
        w, h = read_imsize_binary(p)
        #im = cv2.imread(p)
        #assert w == im.shape[1], 'wrong in %s' % p
        #assert h == im.shape[0], 'wrong in %s' % p
        mw = mult * math.ceil(w / mult)
        mh = mult * math.ceil(h / mult)
        whs.append((mw, mh))
        pads.append((mw - w, mh - h))
    whs = np.array(whs)
    pads = np.array(pads)

    sizes, counts = np.unique(whs, axis=0, return_counts=True)
    #print(sizes.shape)
    #print(np.hstack([sizes, counts[:, None]]))
    pred_ind, pred_ret, pred_bsr = [], [], []
    with tqdm(total=len(fn)) as pbar:
        for w, h in sizes:
            mask = (whs[:, 0] == w) * (whs[:, 1] == h)
            idx = np.nonzero(mask)[0]
            for bn in range(math.ceil(len(idx) / bs)):
                bidx = idx[bs*bn:bs*(bn+1)]
                bfn = [fn[i] for i in bidx]
                imgs = [cv2.imread(osp.join(imdir, p)) for p in bfn]
                imgs = [np.pad(imgs[i], ((0, pd[1]), (0, pd[0]), (0, 0))) for i, pd in enumerate(pads[bidx])]
                pred = model(imgs)
                pred = [p[p[:, 4].argsort()[::-1]].astype(np.float32) for p in pred]
                pred_ind.extend(bidx)
                pred_ret.extend(pred)
                pbar.update(len(bidx))
                pred_bsr.append(len(bidx))
    pred_bsr = np.array(pred_bsr)
    unfull = np.count_nonzero(pred_bsr < bs)
    percent = round(unfull / len(pred_bsr) * 100)
    mean = np.mean(pred_bsr[pred_bsr < bs])
    print('Incomplete batches: %u/%u (%u%%) (mean size = %.2f)' % (unfull, len(pred_bsr), percent, mean))
    preds = [pred_ret[i] for i in np.argsort(pred_ind)]
    return preds
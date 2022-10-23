from glob import glob
import os
import os.path as osp
import shutil

import cv2
import numpy as np

from .pbar import tqdm
from .download import url_download

# sources:
# https://github.com/wondervictor/WiderFace-Evaluation
# https://blog.paperspace.com/mean-average-precision/
# https://en.wikipedia.org/wiki/Precision_and_recall


def eval_det_wider(model, load=None, iou_threshold=0.5):
    updir, imdir, gtdir = prepare_dataset('WIDER_FACE')
    fn, gt, st = get_wider_data(gtdir)
    pred = get_predictions(load, model, fn, updir, imdir)
    pred = norm_scores(pred)
    precision, recall, score_thrs, avg_iou = calc_pr_curve_with_settings(pred, gt, st, iou_threshold)
    ap = [calc_ap(precision[i], recall[i]) for i in range(3)]
    f1 = [best_f1(precision[i], recall[i], score_thrs) for i in range(3)]
    sett = ['Easy  ', 'Medium', 'Hard  ']
    for i in range(3):
        print("%s AP: %.16f" % (sett[i], ap[i]))
    for i in range(3):
        print("%s mean IoU: %.3f" % (sett[i], avg_iou[i]))
    for i in range(3):
        print("%s best F1 score: %.3f (for score threshold = %.3f)" % (sett[i], f1[i][0], f1[i][1]))


def eval_det_fddb(model, load=None, iou_threshold=0.5):
    updir, imdir, gtdir = prepare_dataset('FDDB')
    fn, gt = get_fddb_data(imdir, gtdir)
    pred = get_predictions(load, model, fn, updir, imdir)
    precision, recall, score_thrs, avg_iou = calc_pr_curve(pred, gt, iou_threshold)
    ap = calc_ap(precision, recall)
    f1 = best_f1(precision, recall, score_thrs)
    print("AP: %.16f" % ap)
    print("Mean IoU: %.3f" % avg_iou)
    print("Best F1 score: %.3f (for score threshold = %.3f)" % f1)


def prepare_dataset(set_name):
    """Automatically downloads and unpacks ``set_name`` files to ``<project_root>/eval/<set_name>``,
    images to ``images`` subfolder, annotations to ``ground_truth`` subfolder
    (unless those folders already exist; then assumes it's all already been correctly downloaded).
    """
    if set_name not in ['WIDER_FACE', 'FDDB']:
        raise ValueError('Unknown set_name. Possible values are "WIDER_FACE" and "FDDB"')
    cwd = os.getcwd()
    home = osp.dirname(osp.dirname(osp.realpath(__file__))) if '__file__' in globals() else os.getcwd()
    updir = osp.join(home, 'eval', set_name)
    os.makedirs(updir, exist_ok=True)
    os.chdir(updir)
    try:
        imdir = osp.join(updir, 'images')
        if osp.isdir(imdir):
            print('Using %s validation images at: %s' % (set_name, imdir))
        else:
            print('Downloading %s validation images to: %s' % (set_name, imdir))
            if set_name == 'WIDER_FACE':
                download_wider_images()
            elif set_name == 'FDDB':
                download_fddb_images()
        
        gtdir = osp.join(updir, 'ground_truth')
        if osp.isdir(gtdir):
            print('Using %s ground truth data at: %s' % (set_name, gtdir))
        else:
            print('Downloading %s ground truth data to: %s' % (set_name, gtdir))
            if set_name == 'WIDER_FACE':
                download_wider_annotations()
            elif set_name == 'FDDB':
                download_fddb_annotations()
    finally:
        os.chdir(cwd)
    return updir, imdir, gtdir


def download_wider_images():
    # the link is from the dataset main page: http://shuoyang1213.me/WIDERFACE/
    link = 'https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q'
    arcn = 'WIDER_val.zip'
    url_download(link, arcn, gdrive=True)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    shutil.move(osp.join('WIDER_val', 'images'), 'images')
    shutil.rmtree('WIDER_val')


def download_wider_annotations():
    link = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'
    arcn = 'eval_tools.zip'
    url_download(link, arcn)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    shutil.rmtree('__MACOSX')
    shutil.move(osp.join('eval_tools', 'ground_truth'), 'ground_truth')
    shutil.rmtree('eval_tools')


def download_fddb_images():
    # there's a direct link on the dataset page: http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
    # but that's a larger dataset with 28204 images, while only 2845 of them are used for validation
    # so I reuploaded those 2845 images alone to Google Drive
    link = 'https://drive.google.com/uc?id=1GLYnqrKbsHdkptQr1d2pZzDUyZ3NrCGX'
    arcn = 'FDDB_val.zip'
    url_download(link, arcn, gdrive=True)
    shutil.unpack_archive(arcn, 'images')
    os.remove(arcn)


def download_fddb_annotations():
    link = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
    arcn = 'FDBB-folds.tgz'
    url_download(link, arcn)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    os.rename('FDDB-folds', 'ground_truth')


def get_wider_data(gtdir):
    """Parses WIDER Face annotations. They are in Matlab's .mat files, so depends on scipy to read that.

    Source format example:
    event[i]:       13--Interview
    file[i][j]:     13_Interview_Interview_On_Location_13_921
    easy[i]:        []
    medium[i][j]:   [2 3]
    hard[i][j]:     [1 2 3 4]
    facebox[i][j]:  [[200 392  21  26]
                     [660 389  26  32]
                     [753 393  27  34]
                     [935 388  18  29]]

    The split into 3 difficulty settings is implemented by giving subsets of indices for all boxes,
    e.g. for example above for "medium" only boxes 2, 3 need to be detected and are treated as
    "false negatives" if absent, while boxes 1, 4 are ignored (so if a model detects them, they
    don't count as "false positives" (thus not lowering precision), but also don't count as
    "true positives" (thus not improving recall).
    """
    from scipy.io import loadmat
    gt_mat = loadmat(osp.join(gtdir, 'wider_face_val.mat'))
    hard_mat = loadmat(osp.join(gtdir, 'wider_hard_val.mat'))
    medium_mat = loadmat(osp.join(gtdir, 'wider_medium_val.mat'))
    easy_mat = loadmat(osp.join(gtdir, 'wider_easy_val.mat'))
    
    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    fn, gt, st = [], [], []
    for i in range(len(event_list)):
        for j in range(len(file_list[i][0])):
            flname = osp.join(event_list[i][0][0], file_list[i][0][j][0][0]) + '.jpg'
            bboxes = facebox_list[i][0][j][0]
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            idx = [
                easy_mat['gt_list'][i][0][j][0],
                medium_mat['gt_list'][i][0][j][0],
                hard_mat['gt_list'][i][0][j][0]
            ]
            idx = [k if k.shape != (0, 0) else np.empty((0, 1), np.uint8) for k in idx]
            idx = [np.squeeze(k, axis=1) - 1 for k in idx]
            fn.append(flname)
            gt.append(bboxes)
            st.append(idx)
    return fn, gt, st


def get_fddb_data(imdir, gtdir):
    """Parses FDDB annotations. They are given as ellipses instead of boxes, so also converts that.
    
    Source line format: "major_axis_radius minor_axis_radius angle center_x center_y  1"
    (yes, always two spaces and 1 at the end)
    
    Also, here's a code example to draw an ellipse with cv2 if some testing is needed:
      center = (int(e[3]), int(e[4]))
      axesln = (int(e[0]), int(e[1]))
      angle = e[2] * 180 / np.pi
      cv2.ellipse(im, center, axesln, angle, 0, 360, (0, 0, 255), 2)
    """
    listing = sorted(glob(osp.join(gtdir, 'FDDB-fold-[0-9][0-9].txt')))
    details = sorted(glob(osp.join(gtdir, 'FDDB-fold-[0-9][0-9]-ellipseList.txt')))
    fn, gt = [], []
    for c in range(len(listing)):
        with open(listing[c]) as f:
            filenames = f.read().splitlines()
        with open(details[c]) as f:
            txt = f.read().splitlines()
        i = 0
        for f in filenames:
            assert f == txt[i]
            n = int(txt[i + 1])
            i += 2
            v = [list(map(float, s[:-3].split(' '))) for s in txt[i:i+n]]
            v = [get_ellipse_bounding_box(e[3], e[4], e[0], e[1], e[2]) for e in v]
            fn.append(osp.join(imdir, f) + '.jpg')
            gt.append(np.array(v))
            i += n
    return fn, gt


def get_ellipse_bounding_box(center_x, center_y, radius_x, radius_y, angle_radians):
    """Returns the bounding box for an arbitrarily rotated ellipse
    using a solution from here: https://stackoverflow.com/a/14163413
    """
    ux = radius_x * np.cos(angle_radians)
    uy = radius_x * np.sin(angle_radians)
    vx = radius_y * np.sin(angle_radians)
    vy = radius_y * np.cos(angle_radians)
    half_w = np.sqrt(ux * ux + vx * vx)
    half_h = np.sqrt(uy * uy + vy * vy)
    x1, x2 = center_x - half_w, center_x + half_w
    y1, y2 = center_y - half_h, center_y + half_h
    return [x1, y1, x2, y2]


def calc_iou_matrix(a, b):
    """Calculates intersection over union for every pair of two groups of boxes.
    I.e. if a.shape = (n, 4) and b.shape = (m, 4), then result.shape = (n, m),
    and result[i][j] = IoU between box a[i] and box b[j].
    """
    # same as "pairs = np.array(list(itertools.product(a, b))).reshape(-1, 8)" but faster
    idx_a = np.arange(a.shape[0])
    idx_b = np.arange(b.shape[0])
    idx_cartesian = np.array(np.meshgrid(idx_a, idx_b)).T.reshape(-1, 2)
    pairs = np.hstack((a[idx_cartesian[:, 0]], b[idx_cartesian[:, 1]]))

    ix1 = np.maximum(pairs[:, 0], pairs[:, 4])
    iy1 = np.maximum(pairs[:, 1], pairs[:, 5])
    ix2 = np.minimum(pairs[:, 2], pairs[:, 6])
    iy2 = np.minimum(pairs[:, 3], pairs[:, 7])
    iw = np.maximum(0, ix2 - ix1 + 1)
    ih = np.maximum(0, iy2 - iy1 + 1)
    inter = iw * ih
    area1 = (pairs[:, 2] - pairs[:, 0] + 1) * (pairs[:, 3] - pairs[:, 1] + 1)
    area2 = (pairs[:, 6] - pairs[:, 4] + 1) * (pairs[:, 7] - pairs[:, 5] + 1)
    iou = inter / (area1 + area2 - inter)
    iou = iou.reshape(a.shape[0], b.shape[0])
    return iou


def get_predictions(load, model, fn, updir, imdir):
    """Runs inference with ``model`` on ``fn`` images from ``imdir`` folder and returns predictions,
    also saving them on disk to ``updir/pred_<set_name>_<model_class_name>.npy`` as one stacked array
    where very row is: "image_index, x1, y1, x2, y2, score" (dtype = np.float32)

    Alternatively, if ``load`` is specified, treats it as a path to a saved array in the same format,
    loads it and converts back into a list with len = image count (while other params are ignored).
    """
    if load is None:
        sn = osp.basename(updir)
        mn = model.__class__.__name__
        print('Getting predictions for the images using %s model' % mn)
        
        #preds, inds = predict(fn, model, imdir)
        preds, scales = predict_batched(fn, model, imdir, 32, 50)

        #fp = osp.join(updir, 'pred_%s_%s.npy' % (sn, mn))
        #print('Saving predictions for possible repeated use to: %s' % fp)
        #flat = np.hstack([np.concatenate(inds), np.concatenate(preds)])
        #np.save(fp, flat)
    else:
        print('Loading saved predictions from: %s' % load)
        flat = np.load(load)
        inds = flat[:, 0].astype(np.int64)
        preds = []
        for k in range(inds.max() + 1):
            preds.append(flat[inds == k, 1:])
    return preds


def predict(fn, model, imdir):
    preds = []
    inds = []
    for k in tqdm(range(len(fn))):
        img = cv2.imread(osp.join(imdir, fn[k]))
        pred = model([img])[0]
        pred = pred[pred[:, 4].argsort()[::-1]]
        preds.append(pred.astype(np.float32))
        inds.append(np.full([pred.shape[0], 1], k, np.float32))
    return preds, inds


import math
from sklearn.cluster import KMeans
def predict_batched(fn, model, imdir, bs, n_clusters):
    whs = []
    for f in fn:
        p = osp.join(imdir, f)
        w, h = read_jpg_imsize(p)
        whs.append((w, h))
    whs = np.array(whs)

    kmn = KMeans(n_clusters=n_clusters, random_state=0).fit(whs)
    whs_adj = np.rint(kmn.cluster_centers_).astype(int)[kmn.labels_]
    whs_dif = np.abs(whs - whs_adj)
    whs_scl = whs_adj / whs
    #print('max: ', whs_dif.max(), whs_scl.max())
    #print('mean: ', whs_dif.mean(), whs_scl.mean())
    #print(np.hstack([whs, whs_adj, whs_dif, whs_scl])[:20])

    sizes = np.unique(whs_adj, axis=0)
    pred_ind = []
    pred_ret = []
    with tqdm(total=len(fn)) as pbar:
        for w, h in sizes:
            mask = (whs_adj[:, 0] == w) * (whs_adj[:, 1] == h)
            idx = np.nonzero(mask)[0]
            for bn in range(math.ceil(len(idx) / bs)):
                bidx = idx[bs*bn:bs*(bn+1)]
                bfn = [fn[i] for i in bidx]
                imgs = [cv2.imread(osp.join(imdir, p)) for p in bfn]
                imgs = [cv2.resize(im, (w, h)) for im in imgs]
                pred = model(imgs)
                #pred = pred[pred[:, 4].argsort()[::-1]].astype(np.float32)
                pred_ind.extend(bidx)
                pred_ret.extend(pred)
                pbar.update(len(bidx))
    preds = [pred_ret[i] for i in np.argsort(pred_ind)]
    return preds, whs_scl


import struct
# https://jaimonmathew.wordpress.com/2011/01/29/simpleimageinfo/
# http://blog.jaimon.co.uk/simpleimageinfo/SimpleImageInfo.java.html
def read_jpg_imsize(path):
    """Extracts the width and height of a .jpg image located at ``path``
    without reading all data by analyzing jpeg binary markers.
    
    Sources:
    https://github.com/scardine/image_size/blob/master/get_image_size.py
    https://stackoverflow.com/a/63479164
    https://stackoverflow.com/a/35443269
    """
    w, h = None, None
    with open(path, 'rb') as f:
        jpeg_start = f.read(2)
        assert jpeg_start == b'\xFF\xD8'
        b = f.read(1)
        while (b and ord(b) != 0xDA):
            while (ord(b) != 0xFF): b = f.read(1)
            while (ord(b) == 0xFF): b = f.read(1)
            if (ord(b) == 0x01 or ord(b) >= 0xD0 and ord(b) <= 0xD9):
                b = f.read(1)
            elif (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                f.read(3)
                h, w = struct.unpack('>HH', f.read(4))
                break
            else:
                seg_len = int(struct.unpack(">H", f.read(2))[0])
                f.read(seg_len - 2)
                b = f.read(1)
    return w, h


def norm_scores(preds):
    """Normalizes prediction scores, stretching them from [min, max] to [0, max] range,
    where "min" and "max" are calculated across the entire validation dataset.
    """
    min_score = min([np.min(p[:, -1]) for p in preds if p.any()])
    max_score = max([np.max(p[:, -1]) for p in preds if p.any()])
    for p in preds:
        p[:, -1] = (p[:, -1] - min_score) / (max_score - min_score)
    return preds


def calc_pr_curve(pred, gt, iou_thr=0.5, num_score_thr=1000):
    """Calculates a group of precision/recall values for every image by cutting of ``pred``
    at ``num_score_thr`` different score thresholds and then checking how many of the
    remaining predicted boxes has IoU >= ``iou_thr`` with ground truth boxes ``gt``.
    """
    score_thresholds = np.arange(0, 1, 1 / num_score_thr)[::-1]
    total_needed = 0
    det_needed = np.zeros(len(score_thresholds), int)
    det_total = np.zeros(len(score_thresholds), int)
    det_ious = []
    
    print('Comparing predictions with ground truths (IoU threshold = %s)' % str(iou_thr))
    for k in tqdm(range(len(pred))):

        iou = calc_iou_matrix(pred[k][:, :4], gt[k].astype(np.float32))
        mx = iou.max(axis=1)
        mxj = iou.argmax(axis=1)
        pred_n_per_t = [np.count_nonzero(pred[k][:, 4] >= t) for t in score_thresholds]
        
        mxj[mx < iou_thr] = -1
        idup = np.delete(np.arange(len(mx)), np.unique(mxj, return_index=True)[1])
        mxj[idup] = -1
        found_flags = (mxj >= 0).astype(int)
        found_cumsum = np.concatenate(([0], np.cumsum(found_flags)))

        total_needed += len(gt[k])
        det_needed += found_cumsum[pred_n_per_t]
        det_total += pred_n_per_t
        det_ious.extend(list(mx[found_flags == 1]))

    precision = det_needed / det_total
    recall = det_needed / total_needed
    #print(len(det_ious))
    #print(det_needed[100], det_total[100], total_needed, precision[100], recall[100])
    #print(det_needed[-1], det_total[-1], total_needed, precision[-1], recall[-1])
    return precision, recall, score_thresholds, np.mean(det_ious)


def calc_pr_curve_with_settings(pred, gt, st, iou_thr=0.5, num_score_thr=1000):
    """Same as calc_pr_curve but supports different difficulty settings of WIDER
    (or one might also in general call them "subsets" or "filters", and in theory
    use for custom purposes, e.g. adding indices of boxes bigger than certain size)
    """
    num_st = len(st[0])
    score_thresholds = np.arange(0, 1, 1 / num_score_thr)[::-1]
    total_needed = np.zeros((num_st, 1), int)
    det_needed = np.zeros((num_st, len(score_thresholds)), int)
    det_total = np.zeros((num_st, len(score_thresholds)), int)
    det_ious = [[] for _ in range(num_st)]
    
    print('Comparing predictions with ground truths (IoU threshold = %s)' % str(iou_thr))
    for k in tqdm(range(len(pred))):

        iou = calc_iou_matrix(pred[k][:, :4], gt[k].astype(np.float32))
        mx = iou.max(axis=1)
        mxj = iou.argmax(axis=1)
        pred_n_per_t = [np.count_nonzero(pred[k][:, 4] >= t) for t in score_thresholds]
        
        for s, idx in enumerate(st[k]):
            total_needed[s][0] += len(idx)
            found, ignor = 0, 0
            found_cumsum = np.zeros(1 + len(mx), int)
            ignor_cumsum = np.zeros(1 + len(mx), int)
            already = []
            for i in range(len(mx)):
                if mx[i] >= iou_thr and mxj[i] not in already:
                    if mxj[i] in idx:
                        found += 1
                        det_ious[s].append(mx[i])
                    else:
                        ignor += 1
                    already.append(mxj[i])
                found_cumsum[i + 1] = found
                ignor_cumsum[i + 1] = ignor
            det_needed[s] += found_cumsum[pred_n_per_t]
            det_total[s] += pred_n_per_t - ignor_cumsum[pred_n_per_t]

    precision = det_needed / det_total
    recall = det_needed / total_needed
    avg_iou = [np.mean(det_ious[i]) for i in range(num_st)]
    return precision, recall, score_thresholds, avg_iou


def calc_ap(precision, recall):
    """Computes average precision (AP) for a precision-recall curve as described here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    
    Technically, one could instead calculate area under curve with np.trapz or sklearn.metrics.auc,
    but the link above states that for pr-curves the weighted mean is considered superior.

    Also ensures that the curve is monotonically decreasing first by "elevating" the dipping parts,
    so [0.7, 0.3, 0.4, 0.5] becomes [0.7, 0.5, 0.5, 0.5] (but [0.7, 0.6, 0.5, 0.4] stay the same)
    (implemented by going backwards and replacing every element with the current maximum:
    https://stackoverflow.com/a/28563925).

    :param precision: 1D np.array of precisions, representing Y coordinates of dots composing a curve
    :param recall: 1D np.array of recalls, representing X coodrinates of dots composing a curve
    :return: Average precision, a single float value
    """
    R = np.concatenate(([0.], recall)) # [1.]
    P = np.concatenate(([0.], precision)) # [0.]
    P = np.maximum.accumulate(P[::-1])[::-1]
    ap = np.sum((R[1:] - R[:-1]) * P[1:])
    return ap


def best_f1(precision, recall, score_thrs):
    """Returns best F1 score from a range of precisions/recalls,
    along with the score threshold where it occured.
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    return (np.max(F1), score_thrs[np.argmax(F1)])
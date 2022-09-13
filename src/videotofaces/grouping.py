import os
import os.path as osp
import shutil

import cv2
import numpy as np
import sklearn.metrics
import sklearn.cluster

from .utils import tqdm
from .dupes import remove_dupes_overall
from .models import MobileFaceNetEncoderIRL, IncepResEncoderIRL


def get_encoder_model(style, enc_model, device):
    """TBD"""
    if enc_model == 'mbf':
        return MobileFaceNetEncoderIRL(device, landmarker='mtcnn', minsize2=20)
    elif enc_model == 'facenet':
        return IncepResEncoderIRL(device)
    #if style == 'anime':
    #    return vit_anime_encoder(device, enc_model == 'vit_l'), 128        

       
def encode_faces(paths, model, bs):
    """TBD"""
    print('Extracting features from images for grouping')
    x = []
    with tqdm(total=len(paths)) as pbar:
        for bn in range(len(paths) // bs + 1):
            xk = model(paths[bs*bn:bs*(bn+1)])
            x.append(xk)
            pbar.update(xk.shape[0])
    return np.concatenate(x)


def encode_refs(refs, model):
    """TBD"""
    rpaths = [ps[0] for (_, ps) in refs]
    r = model(rpaths)
    return r

    
def classify(X, R, classes, thr, log, paths, out_dir):
    """TBD"""
    dist = sklearn.metrics.pairwise_distances(X, R, metric='euclidean')
    inds = dist.argmin(axis=1)
    if thr:
        mins = dist.min(axis=1)
        inds[mins >= thr] = len(classes)
        classes.append('other')
    if log:
        fnames = [osp.basename(p) for p in paths]
        with open(osp.join(out_dir, 'log_classification.csv'), 'w') as f:
            extra = '(other_threshold=%s)' % str(thr) if thr else ''
            f.write('file_name,' + ','.join(['dist_' + c for c in classes if c != 'other']) + ',assigned_to_class' + extra + '\n')
            for i in range(X.shape[0]):
                f.write('%s,' % fnames[i] + ','.join(['%.4f' % d for d in dist[i]]) + ',%s\n' % classes[inds[i]])
    return inds, classes


def classify_faces(paths, X, model, classif_params):
    """TBD"""
    refs, thr, log, out_dir = classif_params
    classes = [c for (c, _) in refs]
    print('Found %u classes in ref_dir: %s' % (len(classes), ', '.join(classes)))
    print('Extracting features from reference images')
    R = encode_refs(refs, model)
    print('Classifying images')
    inds, classes = classify(X, R, classes, thr, log, paths, out_dir)
    
    img_dir = osp.dirname(osp.abspath(paths[0])) # usually img_dir = out_dir + '/faces', but not always
    for c in classes:
        os.makedirs(osp.join(img_dir, c), exist_ok=True)
    for i in range(len(paths)):
        fn = osp.basename(paths[i])
        cl = classes[inds[i]]
        os.replace(paths[i], osp.join(img_dir, cl, fn))
    
    print('Grouped %u images into %u folders:' % (len(paths), len(classes)))
    for i in range(len(classes)):
        print(classes[i] + ': ' + str(np.count_nonzero(inds == i)))
    print()        


def cluster_faces(paths, X, cluster_params):
    """TBD"""
    clusters, save_all, rstate, log, out_dir = cluster_params
    # leaving only n_clusters <= number of samples (most likely will be all for non-test cases)
    clusters = [c for c in clusters if c <= len(paths)]

    print('Clustering images into %s groups' % ', '.join([str(cl) for cl in clusters]))
    labels = []
    for k in clusters:
        cm = sklearn.cluster.KMeans(n_clusters=k, random_state=rstate).fit(X)
        labels.append(cm.labels_)

    scores = []
    for i in range(len(clusters)):
        s1 = sklearn.metrics.silhouette_score(X, labels[i])
        s2 = sklearn.metrics.calinski_harabasz_score(X, labels[i])
        s3 = sklearn.metrics.davies_bouldin_score(X, labels[i])
        scores.append((clusters[i], s1, s2, s3))
    if log:
        with open(osp.join(out_dir, 'log_clustering.csv'), 'w') as f:
            f.write('n_clusters,silhouette_score,calinski_harabasz_score,davies_bouldin_score\n')
            for score in scores:
                f.write('%u,%s,%s,%s\n' % score)

    if not save_all:
        best_k = max(scores, key=lambda x:x[1])[0]
        i = clusters.index(best_k)
        clusters = [clusters[i]]
        labels = [labels[i]]
        print('The number of groups chosen: %u' % best_k)

    print('Grouped %u images into %s folders:' % (len(paths), '/'.join([str(cl) for cl in clusters])))
    img_dir = osp.dirname(osp.abspath(paths[0]))
    for i in range(len(clusters)):
        k = clusters[i]
        sub = 'G%u' % k if len(clusters) > 1 else ''
        for j in range(k):
            os.makedirs(osp.join(img_dir, sub, str(j)), exist_ok=True)
        for j in range(len(paths)):
            fn = osp.basename(paths[i])
            lb = str(labels[i][j])
            shutil.copyfile(paths[i], osp.join(img_dir, sub, lb, fn))
        values, counts = np.unique(labels[i], return_counts=True)
        print((sub + ': ' if sub else '') + ', '.join(['%u: %u' % (v, c) for v, c in zip(values, counts)]))
    print()
    for p in paths:
        os.remove(paths)


def test_grouping(paths, refs, test_params):
    """TBD"""
    style, mname, device, out_dir, exclude_other, bs, thr, rstate = test_params
    gt, paths, n_clusters = get_ground_truths(paths, out_dir, exclude_other)
    model = get_encoder_model(style, mname, device)
    X = encode_faces(paths, model, bs)
    R = encode_refs(refs, model)
    
    inds, _ = classify(X, R, [c for (c, _) in refs], None if exclude_other else thr, True, paths, out_dir)
    acc = np.count_nonzero(inds + 1 == gt) / gt.size
    
    cm = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=rstate).fit(X)
    rand_scr = sklearn.metrics.rand_score(gt, cm.labels_)
    silh_scr = sklearn.metrics.silhouette_score(X, cm.labels_)

    print('%.4f / %.4f / %.4f' % (acc, rand_scr, silh_scr))
    print('classification accuracy / rand score for clustering / silhouette score for clustering')

    
def get_ground_truths(paths, out_dir, exclude_other):
    """TBD"""
    try:
        with open(osp.join(out_dir, 'labels.txt')) as f:
            gt = np.asarray([int(x) for x in f.read().splitlines()])
    except:
        raise ValueError('Could not load ground truth labels for testing. Expecting file "labels.txt" inside out_dir, filled with line-separated integers')
    if exclude_other:
        other_class = max(gt)
        other_count = np.count_nonzero(gt == other_class)
        paths = [f for i, f in enumerate(paths) if gt[i] != other_class]
        gt = np.asarray([g for g in gt if g != other_class])
        print('Excluded %u images with "other" class' % other_count)
    n_clusters = max(gt)
    return gt, paths, n_clusters
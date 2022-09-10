import os
import os.path as osp
import shutil

import cv2
import torch
import numpy as np
import sklearn.metrics
import sklearn.cluster

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def validate_clusters(c):
    if not c:
        return list(range(2, 9))
    if isinstance(c, int) and c > 0:
        return [c]
    if (isinstance(c, tuple) or isinstance(c, list)) and all(isinstance(el, int) for el in c) and all(el > 0 for el in c):
        return sorted(list(set(c)))
    if isinstance(c, str):
        v = c.split('-')
        if len(v) == 2 and v[0].isdigit() and v[1].isdigit():
            a, b = int(v[0]), int(v[1])
            if a > 0 and a < b:
                return list(range(a, b + 1))
    print('ERROR: incorrent value for clusters. Please specify a natural number, a tuple/list of natural numbers, or a range in a string form "A-B" where 0 < A < B')
    return None
    

def dir_image_count(dir):
    return len([el for el in os.scandir(dir) if el.is_file() and el.name.lower().endswith(IMG_EXTENSIONS)])

def get_img_loader(dir, bs, sz):
  return None

def check_img_dir(out_dir):
    # try a subfolder named "faces" first, according to how we structure output files during detection
    # but if it doesn't exist, fall back to searching images inside out_dir directly
    # (so that grouping can theoretically be used on independent folders)
    dir = osp.join(out_dir, 'faces')
    dir = dir if osp.isdir(dir) else out_dir
    c = dir_image_count(dir)
    if c == 0:
        print('ERROR: no image files for grouping found at: %s' % out_dir)
        return None
    return dir
    
   
def get_encoder_model(style, enc_model, device):
    b = 22
    #if style == 'anime':
    #    return vit_anime_encoder(device, enc_model == 'vit_l'), 128
    #else:
    #    return incepres_irl_encoder(device), 160
        
        
def encode_faces(dl, model):
    print('Extracting features from images at %s' % dl.dataset.dir)
    features = torch.Tensor([])
    with torch.no_grad(), tqdm(total=len(dl.dataset)) as pbar:
        for _, images in enumerate(dl, 0):
            fb = model(images)
            pbar.update(fb.shape[0])
            features = torch.cat((features, fb))
    return features.cpu().numpy()
    
    
def cluster_faces(ds, features, cluster_params):

    clusters, save_all, random_state, group_log = cluster_params

    # leaving only n_clusters <= number of samples (most likely will be all for non-test cases)
    clusters = [c for c in clusters if c <= len(ds)]

    print('Clustering images into %s groups' % ', '.join([str(cl) for cl in clusters]))
    labels = []
    for k in clusters:
        cm = sklearn.cluster.KMeans(n_clusters=k, random_state=random_state).fit(features)
        labels.append(cm.labels_)

    scores = []
    for i in range(len(clusters)):
        s1 = sklearn.metrics.silhouette_score(features, labels[i])
        s2 = sklearn.metrics.calinski_harabasz_score(features, labels[i])
        s3 = sklearn.metrics.davies_bouldin_score(features, labels[i])
        scores.append((clusters[i], s1, s2, s3))
    if group_log:
        with open(osp.join(ds.dir, 'log_clustering.csv'), 'w') as f:
            f.write('n_clusters,silhouette_score,calinski_harabasz_score,davies_bouldin_score\n')
            for score in scores:
                f.write('%u,%s,%s,%s\n' % score)

    if not save_all:
        best_k = max(scores, key=lambda x:x[1])[0]
        i = clusters.index(best_k)
        clusters = [clusters[i]]
        labels = [labels[i]]
        print('The number of groups chosen: %u' % best_k)

    print('Grouped %u images into %s folders:' % (len(ds), '/'.join([str(cl) for cl in clusters])))
    for i in range(len(clusters)):
        k = clusters[i]
        sub = 'G%u' % k if len(clusters) > 1 else ''
        for j in range(k):
            os.makedirs(osp.join(ds.dir, sub, str(j)), exist_ok=True)
        for j in range(len(ds)):
            fn, lb = ds.filenames[j], labels[i][j]
            shutil.copyfile(osp.join(ds.dir, fn), osp.join(ds.dir, sub, str(lb), fn))
        values, counts = np.unique(labels[i], return_counts=True)
        print((sub + ': ' if sub else '') + ', '.join(['%u: %u' % (v, c) for v, c in zip(values, counts)]))
    print()
    
    for fn in ds.filenames:
        os.remove(osp.join(ds.dir, fn))
        
        
def classify_faces(ds, features, model, classif_params):
    
    ref_dir, enc_oth_thr, group_log = classif_params

    rds = None #ImageFolder(ref_dir, ds.transform)
    classes = rds.classes
    print('Found %u classes in ref_dir: %s' % (len(classes), ', '.join(classes)))
    rdl = None #DataLoader(rds, batch_size=len(rds))
    rimages, rlabels = next(iter(rdl))
    print('Extracting features from reference images')
    with torch.no_grad():
        ref = model(rimages)
    
    print('Classifying images')
    dist = sklearn.metrics.pairwise_distances(features, ref)
    inds = dist.argmin(axis=1)
    if enc_oth_thr:
        mins = dist.min(axis=1)
        inds[mins >= enc_oth_thr] = len(classes)
        classes.append('other')
    for c in classes:
        os.makedirs(osp.join(ds.dir, c), exist_ok=True)
    for i in range(len(ds)):
        fn = ds.filenames[i]
        cl = classes[inds[i]]
        os.replace(osp.join(ds.dir, fn), osp.join(ds.dir, cl, fn))
    
    if group_log:
        with open(osp.join(ds.dir, 'log_classification.csv'), 'w') as f:
            extra = '(other_threshold=%s)' % str(enc_oth_thr) if enc_oth_thr else ''
            f.write('file_name,' + ','.join(['dist_' + c for c in classes if c != 'other']) + ',assigned_to_class' + extra + '\n')
            for i in range(len(ds)):
                f.write('%s,' % ds.filenames[i] + ','.join(['%.2f' % d for d in dist[i]]) + ',%s\n' % classes[inds[i]])

    print('Grouped %u images into %u folders:' % (len(ds), len(classes)))
    for i in range(len(classes)):
        print(classes[i] + ': ' + str(np.count_nonzero(inds == i)))
    print()
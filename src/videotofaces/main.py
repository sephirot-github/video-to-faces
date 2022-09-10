import os
import os.path as osp

import cv2
import numpy as np
import sklearn.metrics
import sklearn.cluster
import torch

try:
  import decord
  HAS_DECORD = True
except ImportError:
  HAS_DECORD = False

# it's the same tqdm there but with a placeholder in case it's not installed
from .utils import tqdm

from .detection import detect_faces, get_video_list, get_detector_model
from .grouping import check_img_dir, get_encoder_model, validate_clusters, validate_ref_dir, get_img_loader, encode_faces, cluster_faces, classify_faces

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def video_to_faces(input=None, input_ext=None,
                   mode='full', style='anime', device=None,
                   out_dir=None, out_prefix='', resize_to=None, save_frames=False, save_rejects=False, save_dupes=False,
                   video_step=1, video_fragment=None, video_area=None, video_reader='opencv',
                   det_model='yolo', det_batch_size=32, det_min_score=0.6, det_min_size=100, det_min_border=5, det_scale=(1.5, 1.5, 2.2, 1.2), det_square=True,
                   hash_thr=8,
                   enc_model='vit_l', enc_batch_size=128,
                   group_mode='clustering', clusters=None, clusters_save_all=False, ref_dir=None, random_state=0, group_log=True,
                   enc_dup_thr=30, enc_oth_thr=70
                   ):
    if style not in ['live', 'anime']:
        print('ERROR: unknown style. Available options are "live", "anime"')
        return
    if mode not in ['full', 'detection', 'grouping']:
        print('ERROR: unknown mode. Available options are "full", "detection", "grouping"')
        return
    if group_mode not in ['clustering', 'classification']:
        print('ERROR: unknown group_mode. Avalialable options are "clustering", "classification"')
        return
    if video_reader not in ['opencv', 'decord']:
        print('ERROR: unknown video_reader. Avalialable options are "opencv", "decord"')
        return
    if input and not osp.exists(input):
        print('ERROR: specified input doesn\'t exist. Please provide a valid path to a file, a directory with files, or a .txt file with full paths inside')
        return
    if out_dir and not osp.isdir(out_dir):
        print('ERROR: specified output path doesn\'t exist or isn\'t a directory. Please provide a valid path to a directory')
        return
    if not input and mode != 'grouping':
        print('ERROR: please specify input')
        return
    if not input and mode == 'grouping' and not out_dir:
        print('ERROR: for grouping, please specify either out_dir or the same input used during detection')
        return
    if mode != 'detection' and group_mode == 'clustering':
        clusters = validate_clusters(clusters)
        if not clusters:
            return
    if mode != 'detection' and group_mode == 'classification' and not validate_ref_dir(ref_dir):
        return
    
    # if unspecified, output dir = input dir
    if not out_dir:
        out_dir = input if osp.isdir(input) else osp.dirname(osp.abspath(input))

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if mode == 'full' or mode == 'detection':
        files = get_video_list(input, input_ext)
        if not files:
            return
        vid_params = (video_step, video_fragment, video_area, video_reader)
        det_params = (det_batch_size, det_min_score, det_min_size, det_min_border, det_scale, det_square)
        save_params = (out_dir, out_prefix, resize_to, save_frames, save_rejects, save_dupes)
        
        detector = get_detector_model(style, det_model, device)
        detect_faces(files, detector, vid_params, det_params, save_params, hash_thr)

    if mode == 'full' or mode == 'grouping':
        img_dir = check_img_dir(out_dir)
        if not img_dir:
            return
        cluster_params = (clusters, clusters_save_all, random_state, group_log)
        classif_params = (ref_dir, enc_oth_thr, group_log)
        dup_params = ('enc', enc_dup_thr, save_dupes, out_dir)

        encoder, inp_size = get_encoder_model(style, enc_model, device)
        dl = get_img_loader(img_dir, enc_batch_size, inp_size)
        features = encode_faces(dl, encoder)
        if enc_dup_thr:
            features, dl.dataset.filenames = remove_dupes_overall(features, dl.dataset.filenames, dup_params)
        if group_mode == 'clustering':
            cluster_faces(dl.dataset, features, cluster_params)
        elif group_mode == 'classification':
            classify_faces(dl.dataset, features, encoder, classif_params)

    print('Done')
    
    
def remove_dupes_overall(X, filenames, dup_params):
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
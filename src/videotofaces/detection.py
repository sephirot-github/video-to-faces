import os
import os.path as osp

import cv2
import numpy as np
import torch
try:
  import decord
  HAS_DECORD = True
except ImportError:
  HAS_DECORD = False
  
from .utils import tqdm, resize_keep_ratio
from .dupes import ahash, remove_dupes_nearest, remove_dupes_overall
#from .detectors.yolo import YOLOv3Detector, YOLOv3DetectorAnime
from .detectors.mtcnn import MTCNNDetector


def get_detector_model(style, det_model, device):
    """TBD"""
    #if style == 'anime':
    #    return YOLOv3DetectorAnime(device)
    #elif det_model == 'mtcnn':
    #    return MTCNNDetector(device)
    #return YOLOv3Detector(device)
    return 0
    
    
def detect_faces(files, model, vid_params, det_params, save_params, hash_thr):
    """TBD"""
    out_dir, out_prefix, _, save_frames, save_rejects, save_dupes = save_params
    
    os.makedirs(osp.join(out_dir, 'faces'), exist_ok=True)
    if save_frames:
        os.makedirs(osp.join(out_dir, 'intermediate', 'frames'), exist_ok=True)
    if save_rejects:
        os.makedirs(osp.join(out_dir, 'intermediate', 'rejects'), exist_ok=True)
    if save_dupes:
        os.makedirs(osp.join(out_dir, 'intermediate', 'dupes1'), exist_ok=True)

    if len(files) > 1:
        print('File count: ' + str(len(files)))

    hashes = []
    fnames = []
    for k in range(len(files)):
        print('Processing ' + files[k])
        out_prefix_file = out_prefix + ('' if len(files) == 1 else '%02d_' % (k + 1))
        save_params = (out_dir, out_prefix_file, *save_params[2:])
        fnames_k, hashes_k = process_video(files[k], model, vid_params, det_params, save_params, hash_thr)
        fnames.extend(fnames_k)
        hashes.extend(hashes_k)

    if hash_thr:
        dup_params = ('hash', hash_thr, save_dupes, out_dir)
        _, fnames = remove_dupes_overall(np.stack(hashes), fnames, dup_params)
    
    paths = [osp.join(out_dir, 'faces', fn) for fn in fnames]
    print()
    print('Saved a total of %u faces to: %s' % (len(paths), osp.join(out_dir, 'faces')))
    print()
    return paths
    
    
def process_video(path, model, vid_params, det_params, save_params, hash_thr):
    """TBD"""
    video_step, video_fragment, video_area, video_reader = vid_params
    bs, _, _, _, _, _ = det_params
    use_decord = HAS_DECORD and video_reader == 'decord'

    if use_decord:
        try:
            vr = decord.VideoReader(path, decord.gpu())
        except decord.DECORDError:
            vr = decord.VideoReader(path)
        lng = len(vr)
        fps = round(vr.get_avg_fps())
    else:
        cap = cv2.VideoCapture(path)
        lng = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        c = 0
    step = round(fps * video_step)
    bgn = step if not video_fragment or video_fragment[0] < 0 else max(step, round(60 * video_fragment[0] * fps))
    end = lng if not video_fragment or video_fragment[1] < 0 else min(lng, round(60 * video_fragment[1] * fps + 1))

    fnames = []
    hashes = []
    fi = list(range(bgn, end, step))
    pbar = tqdm(total=len(fi))
    for k in range(-(len(fi) // -bs)): # ceil
        bi = fi[bs * k : bs * (k + 1)]
        if use_decord:
            frames = vr.get_batch(bi).asnumpy()[..., [2, 1, 0]] # to BGR
            vr.seek(0) # https://github.com/dmlc/decord/issues/208
        else:
            frames = []
            for i in bi:
                if step > 50:
                    # step big enough - should be faster to seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
                    _, frame = cap.read()
                else:
                    # step small enough - should be faster to step through sequentially
                    for j in range(c, i + 1):
                        cap.grab()
                    c = i + 1
                    _, frame = cap.retrieve()
                frames.append(frame)
            frames = np.stack(frames)
        if video_area:
            cx1, cy1, cx2, cy2 = video_area
            frames = frames[:, cy1: cy2, cx1: cx2, :]
        fnames_b, hashes = process_frames_batch(frames, bi, model, det_params, save_params, hash_thr, hashes)
        fnames.extend(fnames_b)
        pbar.update(len(bi))
    pbar.close()
    if not use_decord:
        cap.release()
    return fnames, [h for (h, fn) in hashes]
    
    
def process_frames_batch(frames, indices, model, det_params, save_params, hash_thr, hashes):
    """TBD"""
    _, mscore, msize, mborder, scale, square = det_params
    out_dir, out_prefix, resize_to, _, _, _ = save_params
    imsize = frames[0].shape[:2]
    # 1. Do a forward pass through detection network for a batch of frames, receive list[np.array(ndet, 5)] (len = batch_size)
    boxes = model(frames)
    # 2. Remove boxes that don't satisfy specified basic conditions
    # Boxes' coordinates are rounded to int, each array becomes a list of tuples
    boxes = [filter_boxes(b, imsize, mscore, msize, mborder, save_params, f, i) for (b, f, i) in zip(boxes, frames, indices)] # list[list[tuple(int, int, int, int, float)]]
    # 3. Scale and/or square each box according to settings
    boxes = [adjust_boxes(b, imsize, scale, square) for b in boxes]                                                 # list[list[tuple(int, int, int, int, float)]]
    # 4. Get cropped images, and also save frame number corresponding to each
    faces = [(get_crops(f, b), i) for (f, i, b) in zip(frames, indices, boxes)]                     # list[(list[imgs], frame_index)]
    # 5. Flatten the list of lists, while also saving the order of faces
    faces = [(img, i, j) for (imgs, i) in faces for j, img in enumerate(imgs)]                            # list[(img, frame_index, face_index_within_frame)]
    # 6. Create filenames from indices saved during 2 prev steps
    faces = [(img, out_prefix + '%06d_%u.jpg' % (i, j)) for (img, i, j) in faces]                     # list[(img, filename)]
    # 7. Resize all faces if needed
    if resize_to:
        faces = [(resize_keep_ratio(img, resize_to), fn) for (img, fn) in faces]
    # 8. Remove all faces that are near-identical to one of the N preceeding faces (N = 5)
    if hash_thr:
        faces, hashes = remove_dupes_nearest(faces, hashes, hash_thr, save_params)
    # 9. Save results on disk
    for (img, fn) in faces:
        cv2.imwrite(osp.join(out_dir, 'faces', fn), img)
    # 10. Return resulting filenames and hashes
    return [fn for (_, fn) in faces], hashes
    

def get_crops(img, boxes):
    """TBD"""
    return [img[y1: y2, x1: x2] for (x1, y1, x2, y2, _) in boxes]

    
def check_box(box, img_size, mscore, msize, mborder):
    """TBD"""
    x1, y1, x2, y2, score = box
    H, W = img_size
    c1 = score < mscore
    c2 = x2 - x1 < msize or y2 - y1 < msize
    c3 = mborder and (x1 < mborder or y1 < mborder or x2 > W - mborder or y2 > H - mborder)
    return (c1, c2, c3)


def filter_boxes(boxes, img_size, mscore, msize, mborder, save_params, frame, frame_index):
    """TBD"""
    boxes = [(int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2)), score) for (x1, y1, x2, y2, score) in boxes]
    boxes = [(b, check_box(b, img_size, mscore, msize, mborder)) for b in boxes]
    passed = [b for (b, c) in boxes if not any(c)]

    out_dir, out_prefix, _, save_frames, save_rejects, _ = save_params

    if save_frames:
        scale = 1024 / max(img_size)
        fm = cv2.resize(frame, (int(img_size[1] * scale), int(img_size[0] * scale)))
        for (b, c) in boxes:
            x1, y1, x2, y2 = (np.array(b[:4]) * scale).astype(int)
            color = (0, 0, 255) if any(c) else (0, 255, 0)
            cv2.rectangle(fm, (x1, y1), (x2, y2), color, 2)
            cv2.putText(fm, str(round(b[4], 2)), (x1, y1 - 2 if y1 > 10 else y2 - 2), 0, 0.6, color, 1, lineType=cv2.LINE_AA)
        cv2.imwrite(osp.join(out_dir, 'intermediate', 'frames', out_prefix + '%06d.jpg' % frame_index), fm, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    if not save_rejects:
        return passed

    H, W = img_size
    i, j, log = 0, 0, []
    for ((x1, y1, x2, y2, score), (c1, c2, c3)) in boxes:
        r = c1 or c2 or c3
        fn = out_prefix + '%06d_' % frame[1] + ('r%u' % j if r else '%u' % i) + '.jpg'
        data = [fn, '%.2f' % score, x2 - x1, y2 - y1, x1, y1, x2, y2, int(c1), int(c2), int(c3), int(r)]
        log.append(','.join([str(el) for el in data]))
        if r:
            cv2.imwrite(osp.join(out_dir, 'intermediate', 'rejects', fn), frame[0][max(0, y1): min(H, y2), max(0, x1): min(W, x2)])
            j += 1
        else:
            i += 1

    log_fn = osp.join(out_dir, 'intermediate', 'log_rejects.csv')
    first_time = not osp.exists(log_fn)
    with open(log_fn, 'a') as f:
        if first_time:
            f.write('file_name,score,width,height,x1,y1,x2,y2')
            f.write(',too_low(mscore=%s),too_small(msize=%u),too_close(mborder=%s),rejected' % (str(mscore), msize, str(mborder)))
            f.write('\n')
        for line in log:
            f.write('%s\n' % line)
    return passed

 
def adjust_boxes(boxes, img_size, scale, square):
    """TBD"""
    if isinstance(scale, int):
        scale = (scale, scale, scale, scale)
    (sx1, sx2, sy1, sy2) = scale
    H, W = img_size
    adjusted = []
    for (x1, y1, x2, y2, score) in boxes:
        # 1. scale a box (if scale = 1, then it will just round them up to int)
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
        x1 = int(np.floor(max(0, xc - sx1 * w / 2)))
        x2 = int(np.ceil(min(W, xc + sx2 * w / 2)))
        y1 = int(np.floor(max(0, yc - sy1 * h / 2)))
        y2 = int(np.ceil(min(H, yc + sy2 * h / 2)))
        w, h = x2 - x1, y2 - y1
        # 2. square a box
        if square:
            if h > w:
                d = h - w
                x1 -= d // 2
                x2 += d - d // 2
                # in case we went out of bounds
                if x1 < 0: x2 += abs(x1); x1 = 0; x2 = min(W, x2)
                if x2 > W: x1 -= x2 - W; x2 = W; x1 = max(0, x1)
            elif w > h:
                d = w - h
                y1 -= d // 2
                y2 += d - d // 2
                # in case we went out of bounds
                if y1 < 0: y2 += abs(y1); y1 = 0; y2 = min(H, y2)
                if y2 > H: y1 -= y2 - H; y2 = H; y1 = max(0, y1)
            # final shrinking if box's w/h got larger than frame's h/w
            w, h = x2 - x1, y2 - y1
            if w > H:
                d = w - H
                x1 += d // 2
                x2 -= d - d // 2
            elif h > W:
                d = h - W
                y1 += d // 2
                y2 -= d - d // 2
        adjusted.append((x1, y1, x2, y2, score))
    return adjusted
import os
import os.path as osp

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_img_paths(target_dir):
    return sorted([e.path for e in os.scandir(target_dir) if e.is_file() and e.name.lower().endswith(IMG_EXTENSIONS)])


def check_limited_option(val, arg_name, possible_vals):
    if val not in possible_vals:
        print('ERROR: unknown %s. Available options are %s' % (arg_name, ', '.join(['"' + v + '"' for v in possible_vals])))
        return False
    return True


def validate_args(mode, input_path, out_dir, style, group_mode, video_reader, det_model, enc_model):
    if not check_limited_option(mode, 'mode', ['full', 'detection', 'grouping']):
        return False
    
    res = True
    if input_path and not osp.exists(input_path):
        print('ERROR: specified input_path doesn\'t exist. Please provide a valid path to a file, a directory with files, or a .txt file with full paths inside')
        res = False
    if out_dir and not osp.isdir(out_dir):
        print('ERROR: specified out_dir doesn\'t exist or isn\'t a directory. Please provide a valid path to a directory')
        res = False
    if not input_path and mode != 'grouping':
        print('ERROR: please specify input_path')
        res = False
    if not input_path and mode == 'grouping' and not out_dir:
        print('ERROR: for grouping, please specify either out_dir or the same input_path used during detection')
        res = False
            
    res = res and check_limited_option(style, 'style', ['live', 'anime'])
    res = res and check_limited_option(group_mode, 'group_mode', ['clustering', 'classification'])
    res = res and check_limited_option(video_reader, 'video_reader', ['opencv', 'decord'])
    if style == 'live':
        res = res and check_limited_option(det_model, 'det_model', ['default', 'yolo', 'mtcnn'])
        res = res and check_limited_option(enc_model, 'enc_model', ['default', 'facenet_vgg', 'facenet_casia'])
    if style == 'anime':
        res = res and check_limited_option(det_model, 'det_model', ['default', 'rcnn'])
        res = res and check_limited_option(enc_model, 'enc_model', ['default', 'vit_b', 'vit_l'])
    return res
    
    
def get_clusters(c):
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


def get_class_ref(ref_dir, out_dir):
    explanation = 'Please prepare a directory with 1 or more subfolders representing groups, each with 1 or more reference images inside'
    if not ref_dir:
        tdir = osp.join(out_dir, 'ref')
        if osp.isdir(tdir):
            print('NOTE: ref_dir is unspecified, but found "ref" folder inside out_dir. Will search for reference images there')
            ref_dir = tdir
        else:
            print('ERROR: for group_mode="classification", ref_dir needs to be specified')
            print(explanation)
            return None
    if not osp.isdir(ref_dir):
        print('ERROR: specified ref_dir doesn\'t exist or isn\'t a directory. Please provide a valid path to a directory')
        return None
    
    classes = sorted([e.name for e in os.scandir(ref_dir) if e.is_dir()])
    if not classes:
        print('ERROR: specified ref_dir doesn\'t contain any subfolders')
        print(explanation)
        return None
    
    refs = []
    warn = []
    for c in classes:
        cref = sorted([e.path for e in os.scandir(osp.join(ref_dir, c)) if e.is_file() and e.name.lower().endswith(IMG_EXTENSIONS)])
        if len(cref) == 0:
            warn.append('WARNING: ref_dir\'s subfolder "%s" doesn\'t contain any images. During classification, this class will be ignored' % c)
        else:
            refs.append((c, cref))
    if not refs:
        print('ERROR: none of the ref_dir\'s subfolders contain any images')
        print('Supported extensions are: %s' % ', '.join(IMG_EXTENSIONS))
        return None
    if warn:
        for w in warn:
            print(w)
    return refs
    
    
def get_paths_for_grouping(out_dir):
    # try a subfolder named "faces" first, according to how we structure output files during detection
    # but if it doesn't exist, fall back to searching images inside out_dir directly
    tdir = osp.join(out_dir, 'faces')
    paths = get_img_paths(tdir)
    if not paths:
        tdir = out_dir
        paths = get_img_paths(tdir)
        if not paths:
            print('ERROR: no image files for grouping found at: %s' % out_dir)
            return None
    print('Found %u images at: %s' % (len(paths), tdir))
    return paths
    
    
def get_video_list(input, ext):
    # if input is a .txt file, read it and return lines that are valid file paths
    if osp.isfile(input) and input.lower().endswith('.txt'):
        with open(input) as f:
            files = [l.strip() for l in f.read().splitlines() if osp.isfile(l.strip())]
            if not files:
                print('ERROR: specified .txt file doesn\'t contain any valid paths. Please provide a file with paths to videos, each on a separate line')
            return files
    
    # if input is any other file, then consider it to be a single video file (no extra check for extensions,
    # cv2.VideoCapture.isOpened() will tell later if it's not a valid video or have some unknown codec)
    if osp.isfile(input):
        return [input]
    
    # if input is directory, list all contents non-recursively, sort alphabetically and return only those that are files
    files = [osp.join(input, p) for p in sorted(os.listdir(input)) if osp.isfile(osp.join(input, p))]
    if not files:
        print('ERROR: no files are found in the specified input directory')
    if ext:
        print(ext)
        # filtering by extensions specified inside semicolon-separated string
        files = [s for s in files if s.lower().split('.')[-1] in ext.split(';')]
        if not files:
            print('ERROR: no files with specified extensions (%s) are found in the input directory' % ext)
    return files
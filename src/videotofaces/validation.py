import os
import os.path as osp

# doing some LBYL instead of EAFP here to avoid reporting problems after half of the processing is already done

def validate_ref_dir(ref_dir):
    explanation = 'Please prepare a directory with 1 or more subfolders representing groups, each with 1 or more reference images inside'
    if not ref_dir:
        print('ERROR: for group_mode="classification", ref_dir needs to be specified')
        print(explanation)
        return False
    if not osp.isdir(ref_dir):
        print('ERROR: specified ref_dir doesn\'t exist or isn\'t a directory. Please provide a valid path to a directory')
        return False
    classes = [e.path for e in os.scandir(ref_dir) if e.is_dir()]
    if not classes:
        print('ERROR: specified ref_dir doesn\'t contain any subfolders')
        print(explanation)
        return False
    for c in classes:
        if dir_image_count(c) == 0:
            print('ERROR: ref_dir\'s subfolder "%s" doesn\'t contain any images' % c)
            print(explanation)
            print('Supported extensions are: %s' % ', '.join(IMG_EXTENSIONS))
            return False
    return True
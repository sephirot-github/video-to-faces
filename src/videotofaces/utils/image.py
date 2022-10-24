import struct

import cv2


def resize_keep_ratio(img, resize_to):
    """TBD"""
    h, w = img.shape[:2]
    scale = resize_to / max(h, w)
    if scale < 1: # smaller images stay that way, no upscaling
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def crop_to_area(img, area):
    h, w = img.shape[:2]
    px1, py1, px2, py2 = area
    x1, x2 = int(px1 * w), int(px2 * w + 1)
    y1, y2 = int(py1 * h), int(py2 * h + 1)
    return img[y1:y2, x1:x2, :]


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
import struct

import cv2
import numpy as np


def resize_keep_ratio(img, resize_to, upscale=True):
    """Resizes an image to fit into (resize_to, resize_to) square while keeping the aspect ratio.
    Bigger images will always be resized; smaller images will be resized only if ``upscale``=True.
    """
    h, w = img.shape[:2]
    scale = resize_to / max(h, w)
    if scale != 1 and (upscale or scale < 1):
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def pad_to_square(img, align='left'):
    """"""
    assert align in ['left', 'right', 'center']
    h, w = img.shape[:2]
    p = abs(h - w)
    if p == 0:
        return img

    if align == 'left':
        t = (0, p)
    elif align == 'right':
        t = (p, 0)
    else:
        t = (p // 2, p - p // 2)
    
    if w > h:
        img = np.pad(img, (t, (0, 0), (0, 0)))
    else:
        img = np.pad(img, ((0, 0), t, (0, 0)))
    return img


def crop_to_area(img, area):
    h, w = img.shape[:2]
    px1, py1, px2, py2 = area
    x1, x2 = int(px1 * w), int(px2 * w + 1)
    y1, y2 = int(py1 * h), int(py2 * h + 1)
    return img[y1:y2, x1:x2, :]


def read_imsize_binary(path):
    """Extracts the width and height of an image located at ``path``
    without reading all data by analyzing the beginning as raw bytes.
    
    Sources:
    https://github.com/scardine/image_size/blob/master/get_image_size.py
    JPG: https://stackoverflow.com/a/63479164
    JPG: https://stackoverflow.com/a/35443269
    PNG: https://stackoverflow.com/a/5354562
    """
    w, h = None, None
    with open(path, 'rb') as f:
        start = f.read(2)
        if start == b'\xFF\xD8': # JPEG
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
        elif start == b'\x89\x50': # PNG
            f.read(14)
            w, h = struct.unpack(">LL", f.read(8))
    return w, h
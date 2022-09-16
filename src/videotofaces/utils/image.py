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
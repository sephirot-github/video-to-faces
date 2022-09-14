import cv2


def resize_keep_ratio(img, resize_to):
    """TBD"""
    h, w = img.shape[:2]
    scale = resize_to / max(h, w)
    if scale < 1: # smaller images stay that way, no upscaling
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img
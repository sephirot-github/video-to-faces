import cv2


def resize_keep_ratio(img, to_area, upscale=True):
    """Resizes an image to fit into the specified area while keeping the aspect ratio.
    If ``to_area`` is not tuple, the area is inferred to be square (to_area, to_area).
    If ``upscale`` is False, only bigger images are resized, while smaller ones return unaltered.
    """
    h, w = img.shape[:2]
    aw, ah = to_area if isinstance(to_area, tuple) else (to_area, to_area)
    scale = min(aw / w, ah / h)
    if scale != 1 and (upscale or scale < 1):
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def crop_to_area(img, area):
    h, w = img.shape[:2]
    px1, py1, px2, py2 = area
    x1, x2 = int(px1 * w), int(px2 * w + 1)
    y1, y2 = int(py1 * h), int(py2 * h + 1)
    return img[y1:y2, x1:x2, :]
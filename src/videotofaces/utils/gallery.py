from base64 import b64encode
from glob import glob
import os
import os.path as osp

import cv2
from IPython.display import display, HTML
import pandas as pd

def get_base64(path, h):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    img = cv2.resize(img, (int(W/H*h+0.5), h))
    enc = cv2.imencode('.jpg', img)[1]
    return "data:image.jpg;base64," + b64encode(enc).decode()

def image_gallery(dir, page_size=None, page_number=0, height=150, extensions='.jpg'):
    paths = [osp.join(dir, f) for f in sorted(os.listdir(dir)) if osp.isfile(osp.join(dir, f)) and f.lower().endswith(extensions)]
    captions = [osp.basename(p) for p in paths]
    bs, p, l = page_size if page_size else len(paths), page_number, len(paths)
    if bs*p+1 > l:
        print('starting image index (%d) exceeds the number of files in folder (%d)' % (bs*p+1, l))
        return
    # https://mindtrove.info/jupyter-tidbit-image-gallery/
    s = '<div style="display: flex; flex-flow: row wrap; text-align: center;">'
    for i in range(bs*p, min(bs*(p+1), l)):
        s += '<figure style="margin: 5px !important;">'
        s += '<img src="' + get_base64(paths[i], height) + '" style="height: ' + str(height) + 'px">'
        s += '<figcaption style="font-size: 0.9em">' + captions[i] + '</figcaption>'
        s += '</figure>'
    s += '</div>'
    print('%d-%d out of %d' % (bs*p+1, min(bs*(p+1), l), l))
    display(HTML(s))

# https://stackoverflow.com/questions/47113934/how-to-display-table-with-text-and-images-in-jupyter-notebook
def dataframe_with_images(csv_path, img_root_dir, height=120, sort_by=None, filter=None, extensions='.jpg'):
    df = pd.read_csv(csv_path)
    fn_cols = [k for k, v in df.loc[0].to_dict().items() if isinstance(v, str) and v.endswith('.jpg')]
    if sort_by:
        df = df.sort_values(sort_by)
    if filter:
        fcol, fmin, fmax = filter
        df = df.loc[(df[fcol] >= fmin) & (df[fcol] <= fmax)]
    for col in fn_cols:
        nn = '[img]' + col
        df[nn] = df[col].apply(lambda x: glob(osp.join(img_root_dir, '**', x), recursive=True)[0])
        df[nn] = df[nn].apply(lambda x: '<img src="' + get_base64(x, height) + '" style="height: ' + str(height) + 'px">')
    s = df.to_html(escape=False)
    print('Rows selected: %u' % df.shape[0])
    display(HTML(s))
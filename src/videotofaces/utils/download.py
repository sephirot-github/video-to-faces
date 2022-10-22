import os
import os.path as osp
import re
import requests

from .pbar import tqdm


def url_download(url, dst, gdrive=False):
    # adapted from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # and https://github.com/wkentaro/gdown/blob/main/gdown/download.py
    CHUNK_SIZE = 1024 * 1024 # 1MB
    session = requests.session()
    headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36' } # NOQA
    params = { 'confirm': 1 }
    response = session.get(url, headers=headers, params=params, stream=True, verify=True)
  
    # for Google Drive in case it returns a "file too big for virus scan, confirm to download anyway" page
    # (adding 'confirm 1' to params like above seems to be enough but let's keep this too for failproofing)
    if gdrive and 'Content-Disposition' not in response.headers:
        m = re.search('id="downloadForm" action="(.+?)"', response.text)
        if m:
            url = m.groups()[0].replace('&amp;', '&')
            response = session.get(url, headers=headers, stream=True, verify=True)
        else:
            print('Unable to download from Google Drive')
            return
  
    try:
        total = response.headers.get('Content-Length')
        total = int(total) if total else None
        with open(dst, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
    finally:
        session.close()


def prep_weights_file(url, fn, gdrive=False):
    home = osp.dirname(osp.dirname(osp.realpath(__file__))) if '__file__' in globals() else os.getcwd()
    tdir = osp.join(home, 'weights')
    os.makedirs(tdir, exist_ok=True)
    dst = osp.join(tdir, fn)
    if osp.isfile(dst):
        print('Using weights from: ' + dst)
    else:
        print('Downloading weights from: ' + url)
        print('To: ' + dst)
        url_download(url, dst, gdrive)
    return dst


def prep_weights_gdrive(gid, fn):
    url = 'https://drive.google.com/uc?id=%s' % gid
    return prep_weights_file(url, fn, gdrive=True)
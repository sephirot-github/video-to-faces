import os
import os.path as osp

import torch

from .download import url_download


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


def load_weights_from_list(model, wl, print_log=False):
    wd = {}
    for i, w in enumerate(list(model.state_dict())):
        if print_log:
            print(wl[i][0], ' to ', w)
        wd[w] = wl[i][1]
    model.load_state_dict(wd)
    return model


def load_weights(model, link, suffix, device, extra_conversion=None, sub=None, add_num_batches=False):
    """"""
    link = link if '://' in link else 'https://drive.google.com/uc?id=%s' % link
    fn = '%s_%s.pt' % (model.__class__.__name__.lower(), suffix)
    wf = prep_weights_file(link, fn)
    wd_src = torch.load(wf, map_location=torch.device(device))
    if sub:
        wd_src = wd_src[sub]
    if extra_conversion:
        wd_src = extra_conversion(wd_src)
    wd_dst = {}
    names = list(wd_src)
    shift = 0
    for i, w in enumerate(list(model.state_dict())):
        if add_num_batches and w.endswith('num_batches_tracked'):
            #print('0 to ', w)
            wd_dst[w] = torch.tensor(0)
            shift += 1
        else:
            #print(names[i - shift], ' to ', w)
            wd_dst[w] = wd_src[names[i - shift]]
    model.load_state_dict(wd_dst)
    #for w in model.state_dict(): print(w, '\t', model.state_dict()[w].shape)
    # source file doesn't have 'num_batches_tracked' entries, but they're used only in
    # train mode and only if BatchNorm2d 'momentum' param = None, so we just fill them with 0
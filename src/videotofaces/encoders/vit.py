import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.weights import prep_weights_file, load_weights_from_list
from ..utils.wconvert.w_vit import wconv_openai, wconv_hface, wconv_facetr, wconv_animesion, wconv_tv

# adapted from
# https://github.com/arkel23/animesion/tree/main/classification_tagging/models/vit_animesion
# https://github.com/zhongyy/Face-Transformer/tree/main/copy-to-vit_pytorch-path
# all dropouts are removed since using only for inference
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L235
# https://github.com/openai/CLIP/blob/main/clip/model.py

class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, dim, heads, att_scale):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = heads
        self.scale = dim if att_scale != 'per_head' else dim // heads
    
    def split(self, x):
        return x.view(*x.shape[:2], self.n_heads, -1).transpose(1, 2)

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x) # [bs, n*n+1, dim]
        q, k, v = [self.split(el) for el in [q, k, v]]           # [bs, heads, n*n+1, dim/heads]
        scores = q @ k.transpose(2, 3) / np.sqrt(self.scale)     # [bs, heads, n*n+1, n*n+1]
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2)   # [bs, n*n+1, heads, dim/heads]
        h = h.reshape(*x.shape[:2], -1)    # [bs, n*n+1, dim]
        return h


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, dim, ff_dim, gelu_type):
        super().__init__()
        assert gelu_type in ['exact', 'quick']
        self.act = F.gelu if gelu_type == 'exact' else self.quick_gelu
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        
    def quick_gelu(self, x):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    
    def __init__(self, dim, heads, ff_dim, eps, att_scale, gelu_type):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = MultiHeadedSelfAttention(dim, heads, att_scale)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.pwff = PositionWiseFeedForward(dim, ff_dim, gelu_type)
    
    def forward(self, x):
        h = self.attn(self.norm1(x))
        h = self.proj(h)
        x = x + h
        h = self.pwff(self.norm2(x))
        x = x + h
        return x


class Transformer(nn.Module):
    
    def __init__(self, dim, depth, heads, ff_dim, eps, att_scale, gelu_type):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, ff_dim, eps, att_scale, gelu_type) for _ in range(depth)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    
    def __init__(self, img_size, patch_size, dim, depth, eps=1e-05, stem='conv',
                 pre_norm=False, att_scale='total', gelu_type='exact', projection=None,
                 classes=None):
        super().__init__()
        self.patch_size = patch_size
        self.classes = classes
        self.stem = stem
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, dim))
        if stem == 'conv': self.patch_embedding = nn.Conv2d(3, dim, (patch_size, patch_size), (patch_size, patch_size), bias = not pre_norm)
        if stem == 'linr': self.patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
        if stem == 'linr_ov': self.patch_embedding = nn.Linear(3 * 12 ** 2, dim)
        if pre_norm:
            self.norm_pre = nn.LayerNorm(dim, eps=eps)
        self.transformer = Transformer(dim, depth, dim//64, dim*4, eps, att_scale, gelu_type)
        self.norm = nn.LayerNorm(dim, eps=eps)
        if projection is not None:
            self.projection = nn.Linear(dim, projection, bias=False) # nn.Parameter(torch.zeros(dim, projection))
        if classes:
            self.fc = nn.Linear(dim, classes)
    
    def forward(self, x):
        if self.stem == 'conv':
            x = self.patch_embedding(x)             # [bs, dim, n, n] (n = img_size // patch_size = number of patches)
            x = x.flatten(2).transpose(1, 2)        # [bs, n*n, dim]
        else:
            x = self._unfold_input(x)               # [bs, n*n, 3*p*p]
            x = self.patch_embedding(x)             # [bs, n*n, dim]
        t = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((t, x), dim=1)                # [bs, n*n+1, dim]
        x = x + self.pos_embedding                  # [bs, n*n+1, dim]
        if hasattr(self, 'norm_pre'):
            x = self.norm_pre(x)
        x = self.transformer(x)                     # [bs, n*n+1, dim]
        x = x[:, 0]                                 # [bs, dim]
        x = self.norm(x)                            # [bs, dim]
        if hasattr(self, 'projection'):
            x = self.projection(x)
        if not self.classes:
            return x
        y = self.fc(x)
        y = torch.softmax(y, dim=-1)
        return y, x

    def _unfold_input(self, x): # [bs, 3, 112, 112]
        """If stem (1st layer) isn't conv but linear, need to unwrap every [3, p, p] patch from input images into a flattened column
        Pretrained P12S8 corresponds to 12x12 patches with stride 8, i.e. having overlapping ('linr_ov')
        Pretrained P8S8 is 8x8 patches with stride 8, i.e. the usual division of images into 8x8 regions ('linr')
        There is also an implementation difference in how the flattened columns assembled: a 3x2x2 patch from RGB image can go like
        [0 1] [16 17] [32 33] => a) [0 16 32 1 17 33 4 20 36 5 21 37] (1st pixel from all channels, then 2nd pixel from all channels, etc)
        [4 5] [20 21] [36 37] => b) [0 1 4 5 16 17 20 21 32 33 36 37] (all pixels from R channel, then all pixel from G channel, etc)
        tensorflow.space_to_depth or einops.rearrange does a), but torch.nn.functional's unfold or pixel_unshuffle does b)
        I'm doing a) for P8S8 and b) for P12S8, since github.com/zhongyy/Face-Transformer's models were trained like this
        (though it's probably doesn't matter if you train yourself)
        a) is from https://stackoverflow.com/a/44359328
        b) have hardcoded values (there's enough semi-generalized parameters passed around as it is)"""
        if self.stem == 'linr_ov':
            return F.unfold(x, 12, stride=8, padding=4).transpose(1, 2) # [bs, 196, 432]
        n, c, h, w = x.shape; p = self.patch_size
        x = x.permute(0, 2, 3, 1).reshape(n, h // p, p, w // p, p, c).transpose(2, 3).reshape(n, h // p * w // p, -1) # [bs, 196, 192]
        #x = F.pixel_unshuffle(x, p).reshape(n, c * p ** 2, -1).transpose(1, 2)
        #x = F.unfold(x, p, stride=p).transpose(1, 2)
        return x


class VitTorchVision():

    base = 'https://download.pytorch.org/models/'
    links = {
        'B-16': base + 'vit_b_16_swag-9ac1b537.pth',
        'L-16': base + 'vit_l_16_swag-4f3808c9.pth',
    }

    def __init__(self, device, typ, classify=False):
        assert typ in self.links
        print('Initializing ViT-%s model from TorchVision' % typ)
        wf = prep_weights_file(self.links[typ], 'ViT-%s-TorchVision.pth' % typ)
        wd = torch.load(wf, map_location=torch.device(device))
        wl = wconv_tv(wd, classify)
        ps = int(typ.split('-')[1])
        imsz = 384 if typ[0] != 'L' else 512 #518
        dim =  768 if typ[0] != 'L' else 1024 #1280
        depth = 12 if typ[0] != 'L' else 24 #32
        ncls = None if not classify else 1000
        self.model = ViT(img_size=imsz, patch_size=ps, dim=dim, depth=depth, eps=1e-6,
                         att_scale='per_head', classes=ncls).to(device)
        self.model = load_weights_from_list(self.model, wl)
        self.model.eval()
        print()
        self.imsz = imsz
        
    def __call__(self, imagesPIL):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        prep = Compose([Resize(self.imsz, interpolation=InterpolationMode.BICUBIC), CenterCrop(self.imsz),
                               ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        with torch.no_grad():
            dv = next(self.model.parameters()).device
            inp = torch.stack([prep(im) for im in imagesPIL]).to(dv)
            out = self.model(inp)
        if isinstance(out, tuple):
            return (out[0].cpu().numpy(), out[1].cpu().numpy())
        return out.cpu().numpy()

    def get_predictions(self, probs, topk=5):
        home = osp.dirname(osp.dirname(osp.realpath(__file__))) if '__file__' in globals() else os.getcwd()
        # https://github.com/pytorch/vision/blob/main/torchvision/models/_meta.py#L7
        clsf = osp.join(home, 'classes', 'imagenet1k.txt')
        with open(clsf, encoding='utf-8') as f:
            classes = [l for l in f.read().splitlines()]
        res = []
        for pb in probs:
            # https://stackoverflow.com/a/38772601
            idx = np.argpartition(pb, -topk)[-topk:]    # returns largest k unordered
            idx = idx[np.argsort(pb[idx])[::-1]]        # makes them ordered
            res.append([(classes[ind], pb[ind] * 100) for ind in idx])
        return res


class VitClip():
    
    # supporting weights from 2 sources, even though the numbers in them (and thus the produced results) are the same
    # openai files are smaller because they are FP16, but huggingface should be more consistent with download speeds
    openai_base = 'https://openaipublic.azureedge.net/clip/models/'
    links = {
        'openai': {
            'B-32': openai_base + '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
            'B-16': openai_base + '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
            'L-14': openai_base + 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
        },
        'huggingface': {
            'B-32': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin',
            'B-16': 'https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin',
            'L-14': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin',
        }
    }

    def __init__(self, device, typ, src='openai'):
        assert src in self.links
        assert typ in self.links[src]
        print('Initializing ViT-%s model from CLIP' % typ)
        wf = prep_weights_file(self.links[src][typ], 'ViT-%s-%s.pt' % (typ, src))
        if src == 'openai':
            wd_clip = torch.jit.load(wf, map_location=torch.device(device)).eval().state_dict()
            wl = wconv_openai(wd_clip)
        else:
            wd_clip = torch.load(wf, map_location=torch.device(device))
            wl = wconv_hface(wd_clip)
        ps = int(typ.split('-')[1]) # patch size is contained in the name (e.g. 'L-14' => size=14)
        dim =  768 if typ != 'L-14' else 1024
        depth = 12 if typ != 'L-14' else 24
        proj = 512 if typ != 'L-14' else 768
        self.model = ViT(img_size=224, patch_size=ps, dim=dim, depth=depth,
                         pre_norm=True, att_scale='per_head', gelu_type='quick', projection=proj).to(device)
        self.model = load_weights_from_list(self.model, wl)
        self.model.eval()
        print()
        
    def __call__(self, imagesPIL):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        prep = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        with torch.no_grad():
            dv = next(self.model.parameters()).device
            inp = torch.stack([prep(im) for im in imagesPIL]).to(dv)
            out = self.model(inp)
        return out.cpu().numpy()

    
class VitEncoderAnime():

    # https://drive.google.com/drive/folders/1fzLLGvu7IzmjFy8LZolwkT-NTzZPYonb
    # 1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC - verify_danbooruFaces_b16_ptTrue_bat....ckpt  336.6MB Aug 20, 2021 : ViT-B-16
    # 1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah - verify_danbooruFaces_l16_ptTrue_bat....ckpt  1.14GB  Aug 20, 2021 : ViT-L-16
    # 1kJx8eLmY0kv4m8QV-N8MwoLLbxLfN87g - danbooruFaces_B_16_image128_batch16....ckpt  336.6MB Jan 15, 2022 : ViT-B-16-IFA
    # 1V0kF67t9bEsO3sHtcHtPAePGmjfYdvHc - danbooruFaces_L_16_image128_batch16....ckpt  1.14GB  Jan 17, 2022 : ViT-L-16-IFA
    # 1pFADAEGz8woim_MRhDhtBN4hW6BrQByH - danbooruFull_B_16_image128_batch16_....ckpt  378.5MB Aug 20, 2021 : ViLT-B-16
    gids = {
        'B-16-Danbooru-Faces': '1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC',
        'L-16-Danbooru-Faces': '1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah',
        'B-16-Danbooru-Full': '1pFADAEGz8woim_MRhDhtBN4hW6BrQByH',
    }

    def __init__(self, device, typ='B-16-Danbooru-Faces', classify=False):
        assert typ in self.gids
        num_cls = None if not classify else 3263
        print('Initializing ViT-%s model' % typ)
        wf = prep_weights_file('https://drive.google.com/uc?id=%s' % self.gids[typ], 'ViT-%s.ckpt' % typ)
        dim =  768 if typ[0] != 'L' else 1024
        depth = 12 if typ[0] != 'L' else 24
        self.model = ViT(128, 16, dim, depth, 1e-12, att_scale='per_head', classes=num_cls).to(device)
        weights = torch.load(wf, map_location=torch.device(device))
        wl = wconv_animesion(weights, classify)
        self.model = load_weights_from_list(self.model, wl)
        self.model.eval()
        print()
        
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (128, 128), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model(inp)
        if isinstance(out, tuple):
            return (out[0].cpu().numpy(), out[1].cpu().numpy())
        return out.cpu().numpy()

    def get_predictions(self, probs, topk=5):
        home = osp.dirname(osp.dirname(osp.realpath(__file__))) if '__file__' in globals() else os.getcwd()
        # https://raw.githubusercontent.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/master/labels/classid_classname.csv
        clsf = osp.join(home, 'classes', 'dafre.csv')
        with open(clsf, encoding='utf-8') as f:
            classes = [l.split(',') for l in f.read().splitlines()]
            classes = dict([(int(ind), nm) for ind, nm in classes])
        res = []
        for pb in probs:
            # https://stackoverflow.com/a/38772601
            idx = np.argpartition(pb, -topk)[-topk:]    # returns largest k unordered
            idx = idx[np.argsort(pb[idx])[::-1]]        # makes them ordered
            res.append([(classes[ind], pb[ind] * 100) for ind in idx])
        return res


class VitEncoder():

    def __init__(self, device, isP12S8, classify=False):
        num_cls = None if not classify else 93431
        if isP12S8:
            print('Initializing ViT-P12S8 model for face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1U7c_ojiuRPBfolvziB_VthksABHaFKud', 'Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth', gdrive=True)
        else:
            print('Initializing ViT-P8S8 model for face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu', 'Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth', gdrive=True)
    
        self.model = ViT(img_size=112, patch_size=8, dim=512, depth=20, stem='linr_ov' if isP12S8 else 'linr', classes=num_cls).to(device)
        weights = torch.load(wf, map_location=torch.device(device))
        wl = wconv_facetr(weights, classify)
        self.model = load_weights_from_list(self.model, wl)
        self.model.eval()
        print()
    
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1, (112, 112), (0, 0, 0), swapRB=True)
        inp = torch.from_numpy(inp)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()
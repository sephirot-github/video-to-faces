import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import prep_weights_file

# adapted from
# https://github.com/arkel23/animesion/tree/main/classification_tagging/models/vit_animesion
# https://github.com/zhongyy/Face-Transformer/tree/main/copy-to-vit_pytorch-path
# all dropouts are removed since using only for inference
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L235
# https://github.com/openai/CLIP/blob/main/clip/model.py

class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, dim, num_heads, att_scale):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = num_heads
        self.scale = dim if att_scale != 'per_head' else dim // num_heads
    
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
    
    def __init__(self, img_size, patch_size, dim, depth, heads, ff_dim, eps=1e-05, stem='conv',
                 pre_norm=False, att_scale='total', gelu_type='exact', projection=None,
                 classes=None, loss=None):
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
        self.transformer = Transformer(dim, depth, heads, ff_dim, eps, att_scale, gelu_type)
        self.norm = nn.LayerNorm(dim, eps=eps)
        if projection is not None:
            self.projection = nn.Linear(dim, projection, bias=False) # nn.Parameter(torch.zeros(dim, projection))
        if classes and loss == 'CE':
            self.fc = nn.Linear(dim, classes)
        elif classes and loss == 'CosFace':
            self.loss = None
    
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
            x = self.projection(x) #x = x @ self.projection
        if not self.classes: return x
        return self.fc(x)

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


class VitClip():
    
    links = {
        'B-32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
        'B-16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
        'L-14': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
    }

    def convert_weights(self, wd_clip):
        wl = []
        proj = None
        for nm in wd_clip:
            if not nm.startswith('visual.'):
                # ignoring the text part, we only ViT
                continue
            if 'attn.in_proj_weight' in nm:
                # proj_k/q/v are joined into one in OG CLIP dict, we need separate
                cb = nm.replace('_weight', '_bias')
                ws = wd_clip[nm].chunk(3)
                bs = wd_clip[cb].chunk(3)
                for i in range(3):
                    wl.append((nm, ws[i]))
                    wl.append((cb, bs[i]))
            elif 'attn.in_proj_bias' in nm:
                # skipping bias since we did it above alongside main weights
                continue
            elif 'ln_1' in nm:
                # norm 1 is after attention in OG CLIP dict, we need before
                wl.insert(len(wl) - 8, (nm, wd_clip[nm]))
            elif 'ln_2' in nm:
                # norm 2 is after feedforward in OG CLIP dict, we need before
                wl.insert(len(wl) - 4, (nm, wd_clip[nm]))
            elif 'class_embedding' in nm:
                wl.append((nm, wd_clip[nm].unsqueeze(0).unsqueeze(0)))
            elif 'positional_embedding' in nm:
                wl.append((nm, wd_clip[nm].unsqueeze(0)))
            elif nm == 'visual.proj':
                # projection weights are near the beginning, we need them last
                proj = (nm, wd_clip[nm].transpose(1, 0))
            else:
                wl.append((nm, wd_clip[nm]))
        wl.append(proj)
        return wl

    def __init__(self, device, typ):
        import os.path as osp
        assert typ in ['B-32', 'B-16', 'L-14']
        print('Initializing ViT-%s model from CLIP' % typ)
        wf = prep_weights_file(self.links[typ], osp.basename(self.links[typ]))
        wd_clip = torch.jit.load(wf, map_location='cpu').eval().state_dict()
        wl = self.convert_weights(wd_clip)
        ps = int(typ.split('-')[1]) # patch size is contained in the name (e.g. 'L-14' => size=14)
        dim =  768 if typ != 'L-14' else 1024
        depth = 12 if typ != 'L-14' else 24
        proj = 512 if typ != 'L-14' else 768
        self.model = ViT(img_size=224, patch_size=ps, dim=dim, depth=depth, heads=dim//64, ff_dim=dim*4,
                         pre_norm=True, att_scale='per_head', gelu_type='quick', projection=proj).to(device)
        wd = {}
        for i, w in enumerate(list(self.model.state_dict())):
            #print(wl[i][0], ' to ', w)
            wd[w] = wl[i][1]
        self.model.load_state_dict(wd)
        self.model.eval()
        print()
        
    def __call__(self, imagesPIL):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        prep = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        with torch.no_grad():
            inp = torch.stack([prep(im) for im in imagesPIL])
            out = self.model(inp)
        return out.cpu().numpy()


class VitEncoder():

    def __init__(self, device, isP12S8, classify=False):
        num_cls = None if not classify else 93431
        if isP12S8:
            print('Initializing ViT-P12S8 model for face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1U7c_ojiuRPBfolvziB_VthksABHaFKud', 'Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth', gdrive=True)
        else:
            print('Initializing ViT-P8S8 model for face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu', 'Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth', gdrive=True)
    
        self.model = ViT(img_size=112, patch_size=8, dim=512, depth=20, heads=8, ff_dim=2048, stem='linr_ov' if isP12S8 else 'linr', classes=num_cls).to(device)
        weights = torch.load(wf, map_location=torch.device(device))

        weights['class_token'] = weights.pop('cls_token')
        weights['norm.weight'] = weights.pop('mlp_head.0.weight')
        weights['norm.bias'] = weights.pop('mlp_head.0.bias')
        weights['patch_embedding.weight'] = weights.pop('patch_to_embedding.weight')
        weights['patch_embedding.bias'] = weights.pop('patch_to_embedding.bias')
        for i in range(20):
            weights['transformer.blocks.%u.norm1.weight' % i] = weights.pop('transformer.layers.%u.0.fn.norm.weight' % i)
            weights['transformer.blocks.%u.norm1.bias' % i] = weights.pop('transformer.layers.%u.0.fn.norm.bias' % i)
            weights['transformer.blocks.%u.norm2.weight' % i] = weights.pop('transformer.layers.%u.1.fn.norm.weight' % i)
            weights['transformer.blocks.%u.norm2.bias' % i] = weights.pop('transformer.layers.%u.1.fn.norm.bias' % i)
            weights['transformer.blocks.%u.proj.weight' % i] = weights.pop('transformer.layers.%u.0.fn.fn.to_out.0.weight' % i)
            weights['transformer.blocks.%u.proj.bias' % i] = weights.pop('transformer.layers.%u.0.fn.fn.to_out.0.bias' % i)
            weights['transformer.blocks.%u.pwff.fc1.weight' % i] = weights.pop('transformer.layers.%u.1.fn.fn.net.0.weight' % i)
            weights['transformer.blocks.%u.pwff.fc1.bias' % i] = weights.pop('transformer.layers.%u.1.fn.fn.net.0.bias' % i)
            weights['transformer.blocks.%u.pwff.fc2.weight' % i] = weights.pop('transformer.layers.%u.1.fn.fn.net.3.weight' % i)
            weights['transformer.blocks.%u.pwff.fc2.bias' % i] = weights.pop('transformer.layers.%u.1.fn.fn.net.3.bias' % i)
            qkv = weights.pop('transformer.layers.%u.0.fn.fn.to_qkv.weight' % i).chunk(3)
            weights['transformer.blocks.%u.attn.proj_q.weight' % i] = qkv[0]
            weights['transformer.blocks.%u.attn.proj_k.weight' % i] = qkv[1]
            weights['transformer.blocks.%u.attn.proj_v.weight' % i] = qkv[2]
            weights['transformer.blocks.%u.attn.proj_q.bias' % i] = torch.zeros(512)
            weights['transformer.blocks.%u.attn.proj_k.bias' % i] = torch.zeros(512)
            weights['transformer.blocks.%u.attn.proj_v.bias' % i] = torch.zeros(512)
        if not classify:
            weights.pop('loss.weight')
        self.model.load_state_dict(weights)
        self.model.eval()
        print()
    
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1, (112, 112), (0, 0, 0), swapRB=True)
        inp = torch.from_numpy(inp)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()
        
    
class VitEncoderAnime():

    def __init__(self, device, isL, classify=False):
        num_cls = None if not classify else 3263
        if isL:
            print('Initializing ViT-L/16 model for anime face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah',
                                   'verify_danbooruFaces_l16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt', gdrive=True)
            self.model = ViT(img_size=128, patch_size=16, dim=1024, depth=24, heads=16, ff_dim=4096, eps=1e-12, att_scale='per_head', classes=num_cls, loss='CE').to(device)
        else:
            print('Initializing ViT-B/16 model for anime face feature extraction')
            wf = prep_weights_file('https://drive.google.com/uc?id=1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC',
                                   'verify_danbooruFaces_b16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt', gdrive=True)
            self.model = ViT(img_size=128, patch_size=16, dim=768, depth=12, heads=12, ff_dim=3072, eps=1e-12, att_scale='per_head', classes=num_cls, loss='CE').to(device)
        weights = torch.load(wf, map_location=torch.device(device))
        weights = dict((key.replace('model.', ''), value) for (key, value) in weights.items())
        weights['pos_embedding'] = weights.pop('positional_embedding.pos_embedding')
        if not classify:
            weights.pop('fc.weight')
            weights.pop('fc.bias')
        self.model.load_state_dict(weights)
        self.model.eval()
        print()
        
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (128, 128), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()
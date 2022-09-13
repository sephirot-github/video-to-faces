import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from
# https://github.com/arkel23/animesion/tree/main/classification_tagging/models/vit_animesion
# https://github.com/zhongyy/Face-Transformer/tree/main/copy-to-vit_pytorch-path
# all dropouts are removed since using only for inference


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, dim, num_heads, att_scale):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = num_heads
        self.scale = dim if att_scale != 'per_head' else dim // num_heads
    
    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(self.scale)
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        return h


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    
    def __init__(self, dim, heads, ff_dim, eps, att_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = MultiHeadedSelfAttention(dim, heads, att_scale)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
    
    def forward(self, x):
        h = self.attn(self.norm1(x))
        h = self.proj(h)
        x = x + h
        h = self.pwff(self.norm2(x))
        x = x + h
        return x


class Transformer(nn.Module):
    
    def __init__(self, dim, depth, heads, ff_dim, eps, att_scale):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, ff_dim, eps, att_scale) for _ in range(depth)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    
    def __init__(self, img_size, patch_size, dim, depth, heads, ff_dim, eps=1e-05, stem='conv', att_scale='total', classes=None, loss=None):
        super().__init__()
        self.patch_size = patch_size
        self.classes = classes
        self.stem = stem
        if stem == 'conv': self.patch_embedding = nn.Conv2d(3, dim, (patch_size, patch_size), (patch_size, patch_size))
        if stem == 'linr': self.patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
        if stem == 'linr_ov': self.patch_embedding = nn.Linear(3 * 12 ** 2, dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, dim))
        self.transformer = Transformer(dim, depth, heads, ff_dim, eps, att_scale)
        self.norm = nn.LayerNorm(dim, eps=eps)
        if classes and loss == 'CE':
            self.fc = nn.Linear(dim, classes)
        elif classes and loss == 'CosFace':
            self.loss = None
    
    def forward(self, x):
        if self.stem == 'conv':
            x = self.patch_embedding(x)             # [bs, dim, n, n] (n = img_size // patch_size = number of patches)
            x = x.flatten(2).transpose(1, 2)    # [bs, n*n, dim]
        else:
            x = self._unfold_input(x)                 # [bs, n*n, 3*p*p]
            x = self.patch_embedding(x)             # [bs, n*n, dim]
        t = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((t, x), dim=1)                # [bs, n*n+1, dim]
        x = x + self.pos_embedding                    # [bs, n*n+1, dim]
        x = self.transformer(x)                         # [bs, n*n+1, dim]
        x = x[:, 0]                                                 # [bs, dim]
        x = self.norm(x)                                        # [bs, dim]
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


def vit_irl_encoder(device, isP12S8, classify=False):
    num_cls = None if not classify else 93431
    if isP12S8:
        print('Initializing ViT-P12S8 model for face feature extraction')
        wf = prep_weights_file('https://drive.google.com/uc?id=1U7c_ojiuRPBfolvziB_VthksABHaFKud', 'Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth', gdrive=True)
    else:
        print('Initializing ViT-P8S8 model for face feature extraction')
        wf = prep_weights_file('https://drive.google.com/uc?id=1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu', 'Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth', gdrive=True)
    
    model = ViT(img_size=112, patch_size=8, dim=512, depth=20, heads=8, ff_dim=2048, stem='linr_ov' if isP12S8 else 'linr', classes=num_cls).to(device)
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
    model.load_state_dict(weights)
    model.eval()
    print()
    return model
    
    
def vit_anime_encoder(device, isL, classify=False):
    num_cls = None if not classify else 3263
    if isL:
        print('Initializing ViT-L/16 model for anime face feature extraction')
        wf = prep_weights_file('https://drive.google.com/uc?id=1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah',
                                                     'verify_danbooruFaces_l16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt', gdrive=True)
        model = ViT(img_size=128, patch_size=16, dim=1024, depth=24, heads=16, ff_dim=4096, eps=1e-12, att_scale='per_head', classes=num_cls, loss='CE').to(device)
    else:
        print('Initializing ViT-B/16 model for anime face feature extraction')
        wf = prep_weights_file('https://drive.google.com/uc?id=1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC',
                                                     'verify_danbooruFaces_b16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt', gdrive=True)
        model = ViT(img_size=128, patch_size=16, dim=768, depth=12, heads=12, ff_dim=3072, eps=1e-12, att_scale='per_head', classes=num_cls, loss='CE').to(device)
    weights = torch.load(wf, map_location=torch.device(device))
    weights = dict((key.replace('model.', ''), value) for (key, value) in weights.items())
    weights['pos_embedding'] = weights.pop('positional_embedding.pos_embedding')
    if not classify:
        weights.pop('fc.weight')
        weights.pop('fc.bias')
    model.load_state_dict(weights)
    model.eval()
    print()
    return model
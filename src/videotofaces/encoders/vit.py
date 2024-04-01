import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.weights import load_weights


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
        scores = q @ k.transpose(2, 3) / (self.scale ** .5)      # [bs, heads, n*n+1, n*n+1]
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
    
    def __init__(self, device, img_size, patch_size, dim, depth, eps=1e-12,
                 att_scale='per_head', gelu_type='exact'):
        super().__init__()
        p = patch_size
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // p) ** 2 + 1, dim))
        self.patch_embedding = nn.Conv2d(3, dim, (p, p), (p, p), bias=True)
        self.transformer = Transformer(dim, depth, dim // 64, dim * 4, eps, att_scale, gelu_type)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.to(device)
    
    def forward(self, x):                           # (n = img_size // patch_size = number of patches)
        x = self.patch_embedding(x)                 # [bs, dim, n, n]
        x = x.flatten(2).transpose(1, 2)            # [bs, n*n, dim]
        t = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((t, x), dim=1)                # [bs, n*n+1, dim]
        x = x + self.pos_embedding                  # [bs, n*n+1, dim]
        x = self.transformer(x)                     # [bs, n*n+1, dim]
        x = x[:, 0]                                 # [bs, dim]
        x = self.norm(x)                            # [bs, dim]
        return x


class AnimeVIT():

    links = {
        'B16': '1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC',
        'L16': '1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah',
    }

    def wconv(self, src):
        wl = []
        for nm in src:
            if any(skip in nm for skip in ['text_embeddings', 'decoder', 'mlm_head']):
                continue
            if any(skip in nm for skip in ['model.fc', 'class_head.1']):
                continue
            elif 'positional_embedding' in nm:
                wl.insert(len(wl) - 2, (nm, src[nm]))
            elif 'norm1' in nm:
                wl.insert(len(wl) - 8, (nm, src[nm]))
            elif 'norm2' in nm:
                wl.insert(len(wl) - 4, (nm, src[nm]))
            else:
                wl.append((nm, src[nm]))
        return dict(wl)
        
    def __init__(self, device=None, isL=False):
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        src = 'B16' if not isL else 'L16'
        dim =   768 if not isL else 1024
        depth =  12 if not isL else 24
        print('Initializing ViT_%s model for anime face encoding' % src)
        self.model = ViT(dv, 128, 16, dim, depth)
        load_weights(self.model, self.links[src], 'vit_anime_' + src.lower(), self.wconv)
        self.model.eval()
        print()
    
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (128, 128), (127.5, 127.5, 127.5), swapRB=True)
        dv = next(self.model.parameters()).device
        inp = torch.from_numpy(inp).to(dv)
        with torch.inference_mode():
            out = self.model(inp)
        return out.cpu().numpy()
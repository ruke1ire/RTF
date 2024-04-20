import math
from einops import rearrange, pack
import torch
import torch.nn as nn
from torch.fft import rfft, irfft
from torch.nn.functional import pad
from typing import Optional
import src.utils as utils
from src.utils import registry
from src.models.nn import LinearActivation, Activation, DropoutNd

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)

class MultiHyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            filter_cfg,
            dropout=0.0,  
            dtype=torch.float32,
            skip=1.0,
            **kwargs
            ):
        super().__init__()
        self.dtype = dtype
        self.channels = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_channels = d_model//num_heads
        inner_width = d_model * 3
        self.inner_width = inner_width
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.short_filter = nn.Conv1d(
                inner_width, 
                inner_width, 
                3,
                groups=inner_width,
                dtype = dtype
                )

        self.filter = utils.instantiate(registry.filter, config=filter_cfg)
        self.exp_feat = False
        self.norm_t = False
        self.d_output = self.channels
        self.norm = RMSNorm(dim = self.channels)
        self.skip = skip

    def forward(self, inputs, **kwargs):
        u = inputs
        l = u.size(-2)

        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        u = pad(u, (2,0))
        uc = self.short_filter(u)[...,:l] 
        uc = rearrange(uc, "B (N C) L -> B N C L", N = self.num_heads) # B N 3C L
        *x, v = uc.split(self.head_channels, dim=2) # B N C L

        if self.exp_feat:
            x[0], x[1] = torch.softmax(x[0], dim=-2), torch.softmax(x[1], dim=-2)
        elif not self.exp_feat and self.norm_t: # if temporal normalization is applied, make Q and K positive so that the normalizing constant doesn't become 0.
            x[0], x[1] = x[0].abs(), x[1].abs()

        h, _ = self.filter.kernel(l, rate=1.0, state=None)
        h = h.to(torch.float32)
        if(h.dim() == 3):
            h = h[0]

        kv = self.dropout(torch.einsum("bnil,bnjl->bijnl", x[1], v)) # KV projections
        hkv = fftconv(kv, h) # apply n conv filters to kv projections

        qhkv = torch.einsum("bijnl,bnil->blnj", hkv, x[0])

        if self.norm_t:
            hk = fftconv(h, rearrange(x[1],"b n i l -> b i n l").to(torch.float32))
            qhk = torch.einsum("bnil,binl->bln", x[0], hk.to(x[0].dtype))+1e-5
            qhkv = qhkv/qhk[:,:,:,None]

        y = self.norm(rearrange(qhkv, "B L N C -> B L (N C)"))

        y = self.out_proj(y)
        y = y+inputs*self.skip
        return y # TODO: recurrent state

def fftconv(u, k):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    return y.to(dtype=u.dtype)

class HyenaBlock(nn.Module):
    def __init__(self, d_model, dropout, expand=1, **kwargs):
        super().__init__()
        self.model = MultiHyenaOperator(d_model, dropout = dropout, **kwargs)
        self.model2 = MultiHyenaOperator(d_model, dropout = dropout, **kwargs)
        self.d_output = self.model.channels

        self.l1 = nn.Linear(self.d_output, self.d_output*expand*2) # *2 is for GLU which halves the dims
        self.l2 = nn.Linear(self.d_output*expand, self.d_output)
        self.norm1 = RMSNorm(self.d_output)
        self.norm2 = RMSNorm(self.d_output)
        self.norm3 = RMSNorm(self.d_output)
        self.glu = nn.GLU()
        self.drop = nn.Dropout(dropout)
   
    def forward(self, inputs, **kwargs):
        skip = inputs
        x = inputs
        x = self.norm1(x)
        x = self.model(x)
        x = x + skip
        skip = x
        x = self.norm2(x)
        x = self.model2(x)
        x = x + skip
        skip = x
        x = self.norm3(x)
        x = self.drop(self.glu(self.l1(x)))
        x = self.l2(x) + skip
        return x
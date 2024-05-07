import sys
sys.path.append('../../../../../') # root of repo
from rtf import RTF as RTF_
from src.utils.train import OptimModule
from src.ops.fftconv import fftconv_ref, fftconv_func
from einops import rearrange
import torch

class RTF(RTF_, OptimModule): # RTF integration into Hyena operator
    def __init__(self,
        d_model: int, 
        state_size: int,
        trunc_len: int,
        num_a: int=None,
        dropout: float=0.0,
        bidirectional: bool=False,
        fused_fft_conv: bool=False,
        init: str='zeros',
        constraint='no',
        transposed: bool=False,
        lr: float=0.001,
        wd: float=0.0,
        **kwargs):
        super().__init__(d_model=d_model, state_size=state_size, trunc_len=trunc_len, num_a=num_a, dropout=dropout, bidirectional=bidirectional, flash_fft_conv=False, init=init, constraint=constraint)
        if transposed:
            raise NotImplementedError()
        
        self.register("ab", self.ab.data, lr, wd)
        self.register("h_0", self.h_0.data, lr, wd)
        self.d_output = d_model

        self.use_bias = True
        self.fused_fft_conv = fused_fft_conv

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 
        if bias is None: bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        if self.fused_fft_conv: 
            bias = bias.squeeze(0,-1)
            x = rearrange(x, "b () d () l -> b d l")
            bias = bias.to(dtype=torch.float32)
            y = fftconv_func(
                x, k, bias, dropout_mask=None, gelu=False, 
                force_fp16_output=torch.is_autocast_enabled()
            )
            y = rearrange(y, "b d l -> b () d () l")
        else:
            y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y

    def filter(self, L, **kwargs):
        return rearrange(self.get_k(L).unsqueeze(0), "B C L -> B L C")

    @property
    def bias(self):
        return self.h_0
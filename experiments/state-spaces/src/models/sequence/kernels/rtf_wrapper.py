import sys
sys.path.append('../../../../../') # root of repo
from rtf import RTF as RTF_
from src.models.sequence import SequenceModule
from src.models.nn import Activation

class RTF(RTF_, SequenceModule):
    def __init__(
        self, 
        d_model: int,
        state_size: int,
        trunc_len: int,
        num_a: int  = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
        flash_fft_conv: bool = False,
        activation: str=None,
        lr: float=0.001,
        wd: float=0.0,
        **kwargs
    ):
        print("RTF: unused kwargs:", kwargs)
        super().__init__(d_model, state_size, trunc_len, num_a, dropout, bidirectional, flash_fft_conv)
        self.activation = Activation(activation, dim=-1)
        self.register("ab", self.ab.data, lr, wd)
        self.register("h_0", self.h_0.data, lr, wd)
        self.d_output = d_model
    
    def forward(self, x, state=None, **kwargs):
        x = super().forward(x)
        y = self.activation(x)
        return y, None
    
    def step(self, x, state=None, **kwargs):
        if state == None:
            state = super().x_0(x.shape[:-2])
        x, state = super().step(x, state)
        y = self.activation(x)
        return y
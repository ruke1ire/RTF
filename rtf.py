import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from torch.nn.functional import pad
from torch.fft import rfft, irfft
from functools import partial
try:
    from flashfftconv import FlashFFTConv
    imported_flash_fft_conv = True
except:
    imported_flash_fft_conv = False

print("Flash FFT Conv imported?", imported_flash_fft_conv)

class RTF(nn.Module):
    def  __init__(
        self, 
        d_model: int,
        state_size: int,
        trunc_len: int,
        num_a: int  = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
        flash_fft_conv: bool = False,
        init: str = 'zeros',
        constraint: str = 'no',
    ):
        """
        Args:
            - d_model: Number of SISO channels.
            - state_size: State size of SISO SSM.
            - trunc_len: Truncation length (maximum length) for parallel inference.
            - num_a: Number of unique set of denominator parameters (a). Must divide d_model, and if set to None, num_a => d_model.
            - dropout: Dropout applied to the kernel.
            - bidirectional: If set True, will process input signals with both a causal and an anti-causal SSM.
            - flash_fft_conv: If set True, will use FlashFFTConv.
            - init: Initialization function's name. (zeros, xavier, montel)
            - constraint: Denominator constraint (to keep poles within unit circle). (no, l1_montel)
        """
        super().__init__()

        assert trunc_len > state_size, f"Truncation length {trunc_len} must be larger than the state size {state_size}."

        self.D = d_model
        self.N = state_size 
        if(num_a == None):
            self.num_a = d_model
        else:
            assert d_model%num_a == 0, "num_a must divide d_model"
            self.num_a = num_a
        self.L = trunc_len
        self.bdir = bidirectional

        init_fn = globals()[init+"_init"]
        self.ab = nn.Parameter(init_fn((1+bidirectional)*(self.D + self.num_a), self.N)) # a, b parameters
        self.h_0 = nn.Parameter(torch.randn((1+bidirectional)*self.D)) # h_0 parameter
        self.a_channels = (1+bidirectional)*self.num_a

        self.constraint_flag = False
        if constraint is None:
            constraint = "no"
        a_constraint = globals()[constraint+"_constraint"]
        if constraint in ["l1_montel"]:
            self.scalar = torch.nn.Parameter(torch.rand(self.num_a))
            self.a_constraint = partial(a_constraint, scalar=self.scalar)
            self.constraint_flag = True
        else:
            self.a_constraint = a_constraint

        self.dropout = torch.nn.Dropout(dropout)

        if flash_fft_conv and imported_flash_fft_conv:
            self.flash_fft_conv = FlashFFTConv(2*self.L, dtype=torch.bfloat16)
        else:
            self.flash_fft_conv = None
        
    def get_k(self, L=None):
        """
        RTF kernel generation algorithm.
        """
        if L is None:
            L = self.L
        assert L <= self.L
        if self.constraint_flag:
            ab = torch.cat((self.a, self.ab[self.a_channels:]), dim = 0)
        else:
            ab = self.ab
        ab = pad(ab, (1, self.L-self.N-1+self.L%2))# zero padding params. +self.L%2 is rFFT specific
        ab[:self.a_channels,0] = 1.0 # setting the monic term
        AB = rfft(ab,dim=-1) # polynomial evaluation on points of unity
        K = AB[self.a_channels:]/repeat(AB[:self.a_channels], "D N -> (D R) N", R=self.D//self.num_a) + self.h_0[:,None] # get kernel spectrum
        k = irfft(K,dim=-1)[:,:L] # return time domain kernel
        if self.bdir:
            k = torch.cat((k[:self.D], k[self.D:].flip(-1)), dim=-1) # flip half of the kernels
        return k
        
    def forward(self, u, **kwargs):
        """
        u: (batch, length, channels)
        """
        l = u.size(-2)
        k = self.dropout(self.get_k(l))
        self.k = k
        # below this is functionally identical to s4/s4d
        if self.flash_fft_conv is not None: 
            if self.bdir:
                raise NotImplementedError("Strange behavior with FlashFFTConv, not allowing non-causal convolutions.")
            u = u.permute(0,2,1).to(torch.bfloat16).contiguous()
            y = self.flash_fft_conv(u, k.to(torch.float32))
            y = rearrange(y, "B D L -> B L D").to(u.dtype)
        else:
            if self.bdir:
                u = rearrange(u, "B L D -> (B D) L")
                u = pad(u, (0, l))
                KU = rfft(torch.cat((k, u), dim=0), dim=-1)
                Y = KU[:self.D].T*rearrange(KU[self.D:], "(B D) L -> B L D", D=self.D)
                y = irfft(Y, dim=-2, n=2*l-l%2)[...,:l,:]
            else:
                u = rearrange(u, "B L D -> L (B D)")
                KU = rfft(torch.cat((k.T,u),dim=1),n=2*l-l%2, dim=0)
                U = rearrange(KU[:,self.D:], "L (B D) -> B L D", D=self.D)
                Y = KU[:,:self.D]*U
                y = irfft(Y, dim=-2, n=2*l-l%2)[:,:l]
        return y
        
    def step(self, u, x_i):
        assert self.bdir == False
        c = self.get_C() # c can be cached 
        a = repeat(self.a, "D N -> (D R) N", R=self.D//self.num_a) # repeated a can be cached
        y = torch.einsum("BNC,CN->BC", x_i, c) + self.h_0*u
        x_f = torch.roll(x_i, 1, 1)
        x_f[:,0] = torch.einsum("CN,BNC->BC",-a,x_i) + u
        return y, x_f

    @torch.no_grad()
    def get_C(self):
        """
        returns the corrected C matrix (AKA the numerator "b", for RTF)
        """
        assert self.bdir == False
        device = self.ab.device
        N = self.N
        A = torch.roll(torch.eye(self.N, device=device),1,0)
        A = torch.clone(repeat(A, "N M -> C N M",C=self.num_a))
        A[:,0] = -self.a # construct A matrix
        I_AL = repeat(torch.eye(N, device=device) - torch.matrix_power(A, self.L), "C N M -> (C R) N M", R = self.D//self.num_a) # (I-A^L)
        return torch.linalg.solve(I_AL, self.ab[self.a_channels:], left=True) # solves for C in, C_prime = C(I-A^L)
        
    def x_0(self, batch_shape, device=None):
        return torch.zeros(batch_shape, self.N, self.D, device=device)
    
    def get_k_step(self):
        """
        Get the conv kernel recurrently. Used mainly for testing whether get_k() corresponds with get_k_step() or not.
        """
        u = torch.zeros(1, self.L, self.D, device=self.ab.device)
        u[0,0] = 1.0
        x = self.x_0(1, device=self.ab.device)
        k = []
        for i in range(self.L):
            k_, x = self.step(u[0:1, i], x)
            k.append(k_)
        return torch.cat(k, dim = -2).permute(1,0)
    
    @property
    def a(self):
        return self.a_constraint(self.ab[:self.a_channels])

def zeros_init(channels, order):
    return torch.zeros(channels, order)

def xavier_init(channels, order): # xavier init can sometimes initialize an unstable system
    stdv = 1. / math.sqrt(order)
    return torch.FloatTensor(channels, order).uniform_(-stdv, stdv)

def montel_init(channels, order):
    stdv = 1. / order
    return torch.FloatTensor(channels, order).uniform_(-stdv, stdv)

def no_constraint(coefs, **kwargs):
    return coefs

def l1_montel_constraint(coefs, scalar, **kwargs):
    return coefs/(torch.sum(coefs.abs(), dim = -1) + scalar.abs() + 1e-6)[:,None]

class RTF2(nn.Module):
    def  __init__(
        self, 
        d_model: int,
        state_size: int,
        trunc_len: int,
        num_a: int  = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
        flash_fft_conv: bool = False,
        init: str = 'zeros',
        constraint: str = 'no',
    ):
        """RTF with numerator correction done on the convolutional mode. 
        Args:
            - d_model: Number of SISO channels.
            - state_size: State size of SISO SSM.
            - trunc_len: Truncation length (maximum length) for parallel inference.
            - num_a: Number of unique set of denominator parameters (a). Must divide d_model, and if set to None, num_a => d_model.
            - dropout: Dropout applied to the kernel.
            - bidirectional: If set True, will process input signals with both a causal and an anti-causal SSM.
            - flash_fft_conv: If set True, will use FlashFFTConv.
            - init: Initialization function's name. (zeros, xavier, montel)
            - constraint: Denominator constraint (to keep poles within unit circle). (no, l1_montel)
        """
        super().__init__()

        assert trunc_len > state_size, f"Truncation length {trunc_len} must be larger than the state size {state_size}."

        self.D = d_model
        self.N = state_size 
        if(num_a == None):
            self.num_a = d_model
        else:
            assert d_model%num_a == 0, "num_a must divide d_model"
            self.num_a = num_a
        self.L = trunc_len
        self.bdir = bidirectional

        init_fn = globals()[init+"_init"]
        self.ab = nn.Parameter(init_fn((1+bidirectional)*(self.D + self.num_a), self.N)) # a, b parameters
        self.h_0 = nn.Parameter(torch.randn((1+bidirectional)*self.D)) # h_0 parameter
        self.a_channels = (1+bidirectional)*self.num_a

        self.constraint_flag = False
        if constraint is None:
            constraint = "no"
        a_constraint = globals()[constraint+"_constraint"]
        if constraint in ["l1_montel"]:
            self.scalar = torch.nn.Parameter(torch.rand(self.num_a))
            self.a_constraint = partial(a_constraint, scalar=self.scalar)
            self.constraint_flag = True
        else:
            self.a_constraint = a_constraint

        self.dropout = torch.nn.Dropout(dropout)

        if flash_fft_conv and imported_flash_fft_conv:
            self.flash_fft_conv = FlashFFTConv(2*self.L, dtype=torch.bfloat16)
        else:
            self.flash_fft_conv = None
        
    def get_k(self, L=None):
        """
        RTF kernel generation algorithm.
        """
        if L is None:
            L = self.L
        assert L <= self.L
        a = pad(self.a, (1, self.L-self.N-1+self.L%2))
        b = pad(self.get_C_prime(), (0, self.L-self.N+self.L%2))
        ab = torch.cat((a, b), dim = 0)
        ab[:self.a_channels,0] = 1.0 # setting the monic term
        AB = rfft(ab,dim=-1) # polynomial evaluation on points of unity
        K = AB[self.a_channels:]/repeat(AB[:self.a_channels], "D N -> (D R) N", R=self.D//self.num_a) + self.h_0[:,None] # get kernel spectrum
        k = irfft(K,dim=-1)[:,:L] # return time domain kernel
        if self.bdir:
            k = torch.cat((k[:self.D], k[self.D:].flip(-1)), dim=-1) # flip half of the kernels
        return k
        
    def forward(self, u, **kwargs):
        """
        u: (batch, length, channels)
        """
        l = u.size(-2)
        k = self.dropout(self.get_k(l))
        self.k = k
        # below this is functionally identical to s4/s4d
        if self.flash_fft_conv is not None: 
            if self.bdir:
                raise NotImplementedError("Strange behavior with FlashFFTConv, not allowing non-causal convolutions.")
            u = u.permute(0,2,1).to(torch.bfloat16).contiguous()
            y = self.flash_fft_conv(u, k.to(torch.float32))
            y = rearrange(y, "B D L -> B L D").to(u.dtype)
        else:
            if self.bdir:
                u = rearrange(u, "B L D -> (B D) L")
                u = pad(u, (0, l))
                KU = rfft(torch.cat((k, u), dim=0), dim=-1)
                Y = KU[:self.D].T*rearrange(KU[self.D:], "(B D) L -> B L D", D=self.D)
                y = irfft(Y, dim=-2, n=2*l-l%2)[...,:l,:]
            else:
                u = rearrange(u, "B L D -> L (B D)")
                KU = rfft(torch.cat((k.T,u),dim=1),n=2*l-l%2, dim=0)
                U = rearrange(KU[:,self.D:], "L (B D) -> B L D", D=self.D)
                Y = KU[:,:self.D]*U
                y = irfft(Y, dim=-2, n=2*l-l%2)[:,:l]
        return y
        
    def step(self, u, x_i):
        assert self.bdir == False
        c = self.ab[self.a_channels:] # c can be cached 
        a = repeat(self.a, "D N -> (D R) N", R=self.D//self.num_a) # repeated a can be cached
        x_f = torch.roll(x_i, 1, 1)
        x_f[:,0] = torch.einsum("CN,BNC->BC",-a,x_i) + u
        y = torch.einsum("BNC,CN->BC", x_f, c) + self.h_0*u
        return y, x_f

    def get_C_prime(self):
        """
        returns the corrected C matrix (AKA the numerator "b", for RTF)
        """
        device = self.ab.device
        N = self.N
        A = torch.roll(torch.eye(self.N, device=device),1,0)
        A = torch.clone(repeat(A, "N M -> C N M",C=self.num_a*(1+self.bdir)))
        A[:,0] = -self.a # construct A matrix
        I_AL = repeat(torch.eye(N, device=device) - torch.matrix_power(A, self.L), "C N M -> (C R) N M", R = self.D//self.num_a) # (I-A^L)
        return torch.einsum("CN,CNM->CM", self.ab[self.a_channels:],I_AL)
        
    def x_0(self, batch_shape, device=None):
        return torch.zeros(batch_shape, self.N, self.D, device=device)
    
    def get_k_step(self):
        """
        Get the conv kernel recurrently. Used mainly for testing whether get_k() corresponds with get_k_step() or not.
        """
        u = torch.zeros(1, self.L, self.D, device=self.ab.device)
        u[0,0] = 1.0
        x = self.x_0(1, device=self.ab.device)
        k = []
        for i in range(self.L):
            k_, x = self.step(u[0:1, i], x)
            k.append(k_)
        return torch.cat(k, dim = -2).permute(1,0)
    
    @property
    def a(self):
        return self.a_constraint(self.ab[:self.a_channels])

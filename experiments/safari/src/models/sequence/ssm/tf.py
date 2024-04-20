from src.ops.fftconv import fftconv_ref, fftconv_func
from src.utils.train import OptimModule
import torch
from einops import rearrange, repeat
from opt_einsum import contract
from torch.nn.functional import pad, softmax
from src.models.sequence import SequenceModule
import warnings
import numpy as np
from numpy.polynomial.polynomial import polyfromroots

contract=torch.einsum

def softmax_norm(coefs, scalar=None):
    if(scalar is not None):
        return contract("m h o, m h -> m h o", torch.exp(coefs), 1/(torch.sum(torch.exp(coefs), dim = -1) + torch.exp(scalar)))
    else:
        return contract("m h o, m h -> m h o", torch.exp(coefs), 1/(torch.sum(torch.exp(coefs), dim = -1)))

def scaled_l1_norm(coefs, scalar=None):
    if(scalar is not None):
        return contract("m h o, m h -> m h o", coefs, 1/(torch.sum(coefs, dim = -1) + scalar))
    else:
        return contract("m h o, m h -> m h o", coefs, 1/(torch.sum(coefs, dim = -1)))

def Identity(x, *args, **kwargs):
    return x

def rand_init(order, shape, spread=1.0):
    return torch.rand(*shape, order)*spread

def zeros_init(order, shape):
    return torch.zeros(*shape, order)

def uniform_ring_init(order, shape, r_min = 0.7, r_max = 0.8):
    '''
    roots are samples uniformly around a ring with band [r_min, r_max] within the unit circle, then converted into polynomial
    coefficients. taken from https://arxiv.org/pdf/2303.06349.pdf
    '''
    assert order % 2 == 0, "order must be even."
    v = -0.5*np.log(np.random.rand(*shape, order//2)*(r_max**2-r_min**2)+r_min**2)
    theta = 2*np.pi*np.random.rand(*shape, order//2)
    roots = np.exp(-v+1j*theta)
    roots = np.concatenate((roots, np.conjugate(roots)), axis=-1)
    roots = roots.reshape(-1, order)
    coefs = []
    for root in roots:
        coefs.append((polyfromroots(root)[:-1].real))
    coefs = np.stack(coefs)
    coefs = coefs.reshape(*shape, order)
    return torch.from_numpy(coefs).to(torch.float32)

class TransferFunction(OptimModule):
    def __init__(
            self, 
            d_model: int, 
            order: int,
            truncated_len: int,
            order_multiplier: int=1,
            num_sys: int=None,
            normalization: str=None,
            init: str='rand',
            init_kwargs: dict=dict(),
            activation: str=None,
            dropout: float=0.0,
            bidirectional: bool=False,
            transposed: bool=False,
            lr: float=0.001,
            wd: float=0.0,
            **kwargs
            ):
        """ Transfer function filter.

        Args:
            - d_model: Input dimension. It is also the number of SISO heads/channels.
            - order: Order of transfer function.
            - order_multiplier: Order of the system becomes `order*order_multiplier` by summing multiple lower order rational functions.
            - num_sys: Number of unique A matrix across the heads. If None, it will be set to d_model.
            - truncated_len: Length of convolutional kernel. It is also the max input signal length.
            - normalization: Polynomial coefficient normalization. "l1" and "softmax" is available.
            - activation: Activation function to apply after the transfer function filter.
            - bidirectional: If true, the model will have a pair of kernels for each direction. This will also make the recurrent inference not work.
            - transposed: If False, input shape is [batch x sequence length x d_model] and output shape is [batch x sequence length x d_output]. If True, input shape is [batch x d_model x sequence length] and output shape is [batch x d_output x sequence length].
        """

        super().__init__()
        assert truncated_len > order, f"Truncated length {truncated_len} must be larger than the order {order}."

        self.d_model = d_model
        self.d_output = d_model
        self.order = order
        self.order_multiplier=order_multiplier
        if(num_sys == None):
            self.num_sys = d_model
        else:
            assert d_model%num_sys == 0, "num_sys must divide d_model"
            self.num_sys = num_sys
        self.truncated_len = truncated_len
        self.transposed = transposed
        self.normalization = normalization
        self.bidirectional = bidirectional

        self.head_size = self.d_model*(1+self.bidirectional)
        self.a_shape = (self.order_multiplier, self.num_sys*(1+self.bidirectional), self.order)
        self.c_shape = (self.order_multiplier, self.head_size, self.order)

        if(init == 'rand'):
            self.init = rand_init
            self.init_kwargs = init_kwargs
        elif(init == 'uniform_ring'):
            self.init = uniform_ring_init
            self.init_kwargs = init_kwargs
        elif(init == 'zeros'):
            self.init = zeros_init
            self.init_kwargs = dict()
        else:
            raise NotImplementedError()

        if(normalization == 'l1'):
            self.forward_norm = scaled_l1_norm
            self._a_scalar = torch.nn.Parameter(torch.rand(*self.a_shape[:-1]))
        elif(normalization == 'softmax'):
            self.forward_norm = softmax_norm
            self._a_scalar = torch.nn.Parameter(torch.rand(*self.a_shape[:-1]))
        else:
            self.forward_norm = Identity
            self._a_scalar = None
        a_vec, c_prime, skip = self._init_coefs()

        #self._a_vec = torch.nn.Parameter(a_vec)
        #self.c_prime = torch.nn.Parameter(c_prime)
        #self.skip = torch.nn.Parameter(skip)
        self.register("_a_vec", a_vec, lr, wd)
        self.register("c_prime", c_prime, lr, wd)
        self.register("skip", skip, lr, wd)

        if(activation is not None):
            if(activation == 'relu'):
                self.activation = torch.nn.ReLU()
            elif(activation == 'gelu'):
                self.activation = torch.nn.GELU()
            else:
                raise NotImplementedError()
        else:
            self.activation = torch.nn.Identity()

        self.dropout = torch.nn.Dropout(dropout)

        self.updated = True
        self._a_vec.register_hook(self._updated_weights)
        self.c_cache = None

        self.use_bias = True
        self.fused_fft_conv = True

    def a_vec(self, unique_sys=False):
        '''
        Args:
            - unique_sys: If True, will only return the `num_sys` unique `a_vec`s. In other words, it will not make multiple copies of the system's dynamics.
        '''
        if(unique_sys):
            return self.forward_norm(self._a_vec, self._a_scalar)
        else:
            return repeat(self.forward_norm(self._a_vec, self._a_scalar), "m s o -> m (s r) o", r=self.d_model//self.num_sys)

    def _updated_weights(self, grad):
        self.updated = True
        return grad

    def _init_coefs(self):
        c_prime = self.init(order=self.c_shape[-1], shape=self.c_shape[:-1], **self.init_kwargs)
        #c_prime = rand_init(order=self.c_shape[-1], shape=self.c_shape[:-1], **self.init_kwargs)
        a_vec = self.init(order=self.c_shape[-1], shape=self.a_shape[:-1], **self.init_kwargs)
        skip = torch.randn(self.head_size)
        return a_vec, c_prime, skip

    def _pad_coefs(self, coefs, monic):
        pad_len = self.truncated_len-coefs.shape[-1]
        if(monic):
            padded_coefs = pad(coefs, (0,pad_len))
            padded_coefs[...,coefs.shape[-1]] = 1.0
        else:
            padded_coefs = pad(coefs, (1,pad_len-1))
        return padded_coefs

    def _pad_causal(self, signal):
        pad_len = signal.shape[-1]
        padded_signal = pad(signal, (0,pad_len))
        return padded_signal

    def _poly_val(self, padded_coefs):
        L = padded_coefs.size(-1)
        return torch.fft.rfft(padded_coefs, dim=-1, n = L+L%2).conj()

    @torch.no_grad()
    def get_c(self):
        a_vec = self.a_vec()
        device = a_vec.device
        N = self.order
        A = torch.cat(
            (torch.cat((torch.zeros(self.order_multiplier, self.d_model, N-1, 1).to(device), 
                repeat(torch.eye(N-1, N-1).to(device), "a b -> m h a b", h = self.d_model, m = self.order_multiplier)), dim = 3), 
             -rearrange(a_vec, "m h n -> m h () n")), dim = 2)
        trunc_correction = torch.eye(N).to(device) - torch.matrix_power(A, self.truncated_len)
        return torch.linalg.solve(trunc_correction, self.c_prime, left=True)

    def truncated_frequency_response(self):
        num_den = self._poly_val(torch.cat((
            self._pad_coefs(self.c_prime, monic=False), 
            self._pad_coefs(self.a_vec(unique_sys=True), monic=True)), dim=1))
        tf = num_den[:,:self.head_size]/repeat(num_den[:,self.head_size:], "m s t -> m (s r) t", r=self.d_model//self.num_sys)
        tf = torch.sum(tf, dim=0)
        return (tf.T + self.skip).T

    def truncated_impulse_response(self):
        fr = self.truncated_frequency_response()
        L = fr.size(-1)
        return torch.fft.irfft(fr)[...,:self.truncated_len]

    def fft_conv(self, u, k, bidirectional=False):
        if(bidirectional == True):
            heads = k.size(0)
            k = pad(k[:heads//2], (0,self.truncated_len)) + pad(k[heads//2:].flip(-1), (self.truncated_len, 0))
            u = pad(u, (0,self.truncated_len))
        else:
            k = pad(k, (0,self.truncated_len))
            u = pad(u, (0,self.truncated_len))
        if(k.dim() == 2):
            k = k.unsqueeze(0)
        length = k.size(-1)
        u_k_omega = torch.fft.rfft(torch.cat((u, k), dim=0),n=length-length%2)
        #print(u_k_omega.shape)
        y_omega = u_k_omega[:-1]*u_k_omega[-1]
        y = torch.fft.irfft(y_omega, n=length-length%2)[...,:self.truncated_len]
        return y

#    def forward(self, x, state=None, **kwargs):
#        if(state is not None):
#            raise NotImplementedError()
#
#        if(self.transposed == False):
#            x = rearrange(x, "b l h -> b h l")
#        if(x.shape[-1] > self.truncated_len):
#            warnings.warn(f"Input length {x.shape[-1]} should be less than or equal to self.truncated_len {self.truncated_len}, truncating input to self.truncated_len")
#            x = x[...,:self.truncated_len]
#            T = self.truncated_len
#        else:
#            T = x.shape[-1]
#            x = pad(x, (0, self.truncated_len - x.shape[-1]))
#
#        y = self.fft_conv(x, self.dropout(self.truncated_impulse_response()), self.bidirectional)[...,:T]
#        y = self.activation(y)
#
#        if(self.transposed == False):
#            y = rearrange(y, "b h l -> b l h")
#        return y, None

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(
                self.order_multiplier,
                *batch_shape, 
                self.d_model,
                self.order, 
                device=device, 
                requires_grad=False)

    def step_ssm(self, u, state, a_vec, c, skip):
        *B, H = u.shape
        if(state == None):
            state = self.default_state(*B)
        state = torch.cat((state[...,1:], (contract("m h o, m b h o -> m b h", -a_vec, state) + u).unsqueeze(-1)), dim = -1)
        y = contract("m h o, m b h o -> b h", c, state) + skip*u
        return y, state

    def step(self, x, state=None, **kwargs):
        if(self.bidirectional == True):
            raise NotImplementedError()

        # Caching C so that it gets computed just once.
        if(self.updated == True):
            c = self.get_c()
            self.c_cache = c
            self.updated = False
        else:
            c = self.c_cache

        x_dim = x.dim()
        if(x_dim == 3):
            x = x.squeeze(1)

        y, state = self.step_ssm(u = x, state = state, a_vec = self.a_vec(), c = c, skip = self.skip)

        y = self.activation(y)

        if(x_dim ==3):
            y = y.unsqueeze(1)
        return y, state
    
    def kernel(self, L, **kwargs):
        return self.truncated_impulse_response()[...,:L], None

    def filter(self, L, **kwargs):
        return rearrange(self.truncated_impulse_response()[...,:L].unsqueeze(0), "B C L -> B L C")
    
    @property
    def bias(self):
        return self.skip

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
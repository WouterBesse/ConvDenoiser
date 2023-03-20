import torch
from torch import nn
import numpy as np
import math
from typing import Callable, Optional, Sequence, Tuple, Union
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchaudio import functional as F

"""
Util functions
"""    
# def sampleold(zmean, zlsig, eps=None):
#     b, l = zmean.size()
#     device = zmean.get_device()

#     if eps is None:
#         eps = torch.randn(b, l)
#         if zmean.is_cuda:
#             eps = eps.to(device)
#         eps = eps

#     return zmean + eps * (zlsig * 0.5).exp()

def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

def dimensionSize(in_size, division):
        return in_size // division

# def gated_cnn(net, in_channels, out_channels, kernel_size, dilations, 
#     local_condition):
#     x = CausalConv1d(in_channels, 2 * out_channels, kernel_size, dilations)
 
#     net = add_condition(net, local_condition)

#     net_filter, net_gate = net[:, :, : dilation_filters], net[:, :, dilation_filters: ]
#     net = tf.nn.tanh(net_filter) * tf.nn.sigmoid(net_gate)
#     return net

# def add_condition(net, condition):
#     if condition is not None:
#         _, net_len, out_channels = net.shape.as_list()
#         T = condition.size()[-1]
#         encoding = conv1d_v2(condition, out_channels, kernel_size=1, log=True, use_bias=False)
        
#         net = tf.reshape(net, [-1, T, net_len // T, out_channels])
#         net += tf.expand_dims(encoding, 2)
#         net = tf.reshape(net, [-1, net_len, out_channels])
#     return net

def receptive_field_size(total_layers, stacks, kernel_size,
                         dilation=lambda x: 2**x, target_field_length = 1):
    """Compute receptive field size
    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.
    Returns:
        int: receptive field size in sample
    """
    assert total_layers % stacks == 0
    # print(total_layers, num_cycles, kernel_size)
    layers_per_cycle = total_layers // stacks
    # print(layers_per_cycle)
    dilations = [dilation(i) for i in range(total_layers)]
    print(dilations)
    
    half_filter_length = (kernel_size-1)/2
    length = 0
    for d in dilations:
        length += d*half_filter_length
    length = 2*length
    length = stacks * length
    length += target_field_length
    
    return length


"""
Toch Modules
"""

class AddCond(nn.Module):
    
    def __init__(self):
        super(AddCond, self).__init__()
        self.convy = normalisedConv1d(64, res_channels, kernel_size=1, padding=0, dilation=1, bias=False, std_mul=1.0)
        
    def forward(self, input, condition):
        b, out_channels, input_len = input.size()
        T = condition.size()[-1]
        encoding = self.convy(condition)
        
        net = torch.reshape(net, (b, out_channels, T, input_len // T))
        net += encoding.unsqueeze(1)
        net = torch.reshape(net, (b, out_channels, input_len))
        return net


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self.probability = probability

    def forward(self, quantized):
        original_quantized = quantized.clone()
        length = original_quantized.size(2)

        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self.probability, 1 - self.probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized

def normalisedConv1d(in_channels, out_channels, kernel_size, dropout=0.05, std_mul=4.0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def normalisedConvTranspose2d(in_channels, out_channels, kernel_size,
                    weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    if weight_normalization:
        return nn.utils.weight_norm(m)
    else:
        return m

def CustomConv1D(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)

        return conv

class ResidualConv1dGLU(nn.Module):

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels = None, cin_channels = -1, dropout= 1 - 0.95, dilation = 1, causal = True, bias = True, weight_normalisation = True):
        super(ResidualConv1dGLU, self).__init__()

        self.dropout = nn.Dropout(p = dropout)
#       dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # padding = (kernel_size - 1) * dilation - 1
        padding = dilation
        # padding = (kernel_size - 1) * dilation # 2 Kernel padding

        if weight_normalisation:
            assert bias
            self.dil_conv = normalisedConv1d(residual_channels, gate_channels, kernel_size,
                               padding=padding, dilation=dilation,
                               bias=bias, std_mul=1.0)
        else:
            self.dil_conv = nn.Conv1d(residual_channels, gate_channels, kernel_size,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        self.conv1cond = normalisedConv1d(cin_channels, gate_channels, kernel_size=1, padding=0, dilation=1, bias=bias, std_mul=1.0)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1_out = normalisedConv1d(gate_out_channels, residual_channels, kernel_size = kernel_size, bias=bias, padding = 'same')
        # self.conv1_skip = normalisedConv1d(gate_out_channels, skip_out_channels, kernel_size = kernel_size, bias=bias, padding = 1)
        self.conv1_skip = normalisedConv1d(gate_out_channels, skip_out_channels, kernel_size = kernel_size, bias=bias, padding = 'same')
        self.splitdim = 1

    def forward(self, x, c, skip):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
        Returns:
            Tensor: output
        """
        residual = x
        x = self.dropout(x)
        # print('c0', c.size())
        condition = self.conv1cond(c)
        # print('x1', x.size())
        # print('c1', condition.size())

        
        
        # dilation with remove future time steps
        x = self.dil_conv(x)
        # print('x2', x.size())
        # x = x[:, :, :residual.size(-1)]
        # print('x3', x.size())
        
        # combined_x = x + condition
        # filt, gate = combined_x.split(x.size(self.splitdim) // 2, dim=self.splitdim)
        a, b = x.split(x.size(self.splitdim) // 2, dim=self.splitdim)

        # local conditioning
        
        ca, cb = c.split(condition.size(self.splitdim) // 2, dim=self.splitdim)
        filt, gate = a + ca, b + cb

        x = torch.tanh(filt) * torch.sigmoid(gate)

        # For skip connection
        s = self.conv1_skip(x)
        # print('s', s.size())
        # print('skip', skip.size())
        skip = skip + s

        # For residual connection
        x = self.conv1_out(x)
        # print(x.size(), s.size())

        x = (x + residual) * math.sqrt(0.5)
        
        return x, skip
    
def add_noise(
    waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise # (*, L)
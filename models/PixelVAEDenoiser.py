import math
import torch
from torch import nn, from_numpy
import torch.nn.functional as F
from torch.utils.data import Dataset
import CustomModules
from tqdm import tqdm
import numpy as np
import os

"""
Util functions
"""    
def dimensionSize(in_size, upsamples):
        division = 1
        for upsample in upsamples:
            multiplier *= upsample
        return in_size // division

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)


class convBlock(nn.Module):
    """
    Convolution block for extracting features
    """    
    def __init__(self, in_channels, out_channels, num_convs = 3, kernel_size = 3, batch_norm = False, use_weight = True, use_res = True, deconv = False):
        super().__init__()

        convLayers = []
        self.use_weight = use_weight
        self.use_res = use_res

        padding = int(math.floor(kernel_size / 2))

        self.upchannels = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        for i in range(num_convs):
            if deconv:
                convLayers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding, bias = not batch_norm))
            else:
                convLayers.append(nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding, bias = not batch_norm))

            if batch_norm:
                convLayers.append(nn.BatchNorm2d(out_channels))

            convLayers.append(nn.ReLU())

        self.block = nn.Sequential(*convLayers)

        if use_weight:
            self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.upchannels(x)

        out = self.seq(x)

        return out + self.weight * x


class LMaskedConv2d(nn.Module):
    """
    Masked convolution, with location dependent conditional.
    The conditional must be an 'image' tensor (BCHW) with the same resolution as the instance (no of channels can be different)
    """
    def __init__(self, input_size, conditional_channels, channels, colors=1, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        # Kernels are shaped as such [num_filters, input_channels, height, width]
        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

      
        f, t = 0 * pc, (0+1) * pc

        # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
        if self_connection:
            self.hmask[f:t, :f+pc, 0, m] = 1 # Sets the left horizontal middle position to 1. Place for us = [0:20, :20, 0, m]
            self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1 # [20:40, :20, 0, m]

        print(self.hmask[:, :, 0, m])

        # The conditional weights
        self.vhf = nn.Conv2d(conditional_channels, channels, 1)
        self.vhg = nn.Conv2d(conditional_channels, channels, 1)
        self.vvf = nn.Conv2d(conditional_channels, channels, 1)
        self.vvg = nn.Conv2d(conditional_channels, channels, 1)

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.
        Conditional and weights are used to compute a bias based on the conditional element
        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond)
        sig_bias = vg(cond)

        # compute convolution term
        b = x.size(0)
        c = x.size(1)

        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return F.tanh(top + tan_bias) * F.sigmoid(bottom + sig_bias)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h,  (self.vvf, self.vvg))
            hx = self.gate(hx, h,  (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, 


class LGated(nn.Module):
    """
    Gated model with location specific conditional
    """

    def __init__(self, input_size, conditional_channels, hidden_channels, num_layers, k=7, padding=3):
        super().__init__()

        c, features, timesteps = input_size

        self.conv1 = nn.Conv2d(c, hidden_channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                LMaskedConv2d(
                    (hidden_channels, features, timesteps),
                    conditional_channels,
                    hidden_channels, 
                    colors = c, 
                    self_connection = i > 0,
                    res_connection = i > 0,
                    gates = True,
                    hv_connection = True,
                    k = k, 
                    padding = padding)
            )

        self.conv2 = nn.Conv2d(hidden_channels, 256*c, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x.view(b, c, 256, h, w).transpose(1, 2)


class Encoder(nn.Module):
    """
    VAE Encoder
    """    
    def __init__(self, input_size, channel_sizes, upsamples, zsize=32, depth = 0, use_bn=False):
        super().__init__()

        self.zsize = zsize
        features, timesteps = input_size

        modules = [
            convBlock(1, channel_sizes[0], batch_norm = use_bn, use_res = True),
            nn.MaxPool2d(upsamples[0]),
            convBlock(channel_sizes[0], channel_sizes[1], batch_norm = use_bn, use_res = True),
            nn.MaxPool2d(upsamples[0]),
            convBlock(channel_sizes[1], channel_sizes[2], batch_norm = use_bn, use_res = True),
            nn.MaxPool2d(upsamples[0]),
        ]

        for _ in range(depth):
            modules.append(convBlock(channel_sizes[2], channel_sizes[2], batch_norm = use_bn, use_res = True))

        modules.extend([
            Flatten(),
            nn.Linear(dimensionSize(features, upsamples) * dimensionSize(timesteps, upsamples) * channel_sizes[-1], zsize * 2)
        ])

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):

        zcomb = self.encoder(x)
        return zcomb[:, :self.zsize], zcomb[:, self.zsize:]


class Decoder(nn.Module):
    """
    VAE Decoder
    """
    def __init__(self, input_size, channel_sizes, upsamples, zsize=32, depth = 0, use_bn=False, out_channels = 20):
        super().__init__()

        self.zsize = zsize
        features, timesteps = input_size

        upmode = 'bilinear'
        bigfeaturedim = dimensionSize(features, upsamples)
        bigtimedim = dimensionSize(timesteps, upsamples)
        modules = [
            nn.Linear(zsize, bigfeaturedim * bigtimedim * channel_sizes[-1]), 
            nn.ReLU(),
            Reshape((channel_sizes[-1], bigfeaturedim, bigtimedim))
        ]

        for _ in range(depth):
            modules.append(convBlock(channel_sizes[-1], channel_sizes[-1], deconv = True, batch_norm = use_bn, use_res = True))


        modules.extend([
            nn.Upsample(scale_factor = upsamples[2], mode = upmode),
            convBlock(channel_sizes[2], channel_sizes[2], deconv = True, batch_norm = use_bn, use_res = True),
            nn.Upsample(scale_factor = upsamples[1], mode = upmode),
            convBlock(channel_sizes[2], channel_sizes[1], deconv = True, batch_norm = use_bn, use_res = True),
            nn.Upsample(scale_factor = upsamples[0], mode = upmode),
            convBlock(channel_sizes[1], channel_sizes[0], deconv = True, batch_norm = use_bn, use_res = True),
            nn.ConvTranspose2d(channel_sizes[0], out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*modules)

    def forward(self, zsample):

        return self.decoder(zsample)

class SPDataset(Dataset):

    def __init__(self, trainN_dir, trainC_dir, transform=None, target_transform=None, segments=8):
        print("Directories: ", trainN_dir, "|", trainC_dir)
        
        nTrain_dirlist = os.listdir(trainN_dir)
        cTrain_dirlist = os.listdir(trainC_dir)

        self.nTrain_samples = []
        self.cTrain_samples = []
        
        self.sp_min = 99999999999
        self.sp_max = -99999999999
        
        

        print("Noisy file amount: ", len(nTrain_dirlist))
        print("Clean file amount: ", len(cTrain_dirlist))

        with tqdm(total=len(nTrain_dirlist), desc='Loading files to dataset') as pbar:
            for noisy, clean in zip(nTrain_dirlist, cTrain_dirlist):
                noisy_path = os.path.join(trainN_dir, noisy)
                clean_path = os.path.join(trainC_dir, clean)

                noisy_file = np.load(noisy_path).astype(np.float32)
                clean_file = np.load(clean_path).astype(np.float32)

                sp_min = min(np.min(clean_file), np.min(noisy_file))
                sp_max = max(np.max(clean_file), np.max(noisy_file))
                
                
                self.sp_min, self.sp_max = CustomModules.getMinMax(sp_min, sp_max, self.sp_min, self.sp_max)

                

                # meannumb = torch.mean(torch.cat((noisy_file, clean_file), dim=0))

                # noisy_file = nn.functional.normalize(from_numpy(noisy_file))
                

                # clean_file = nn.functional.normalize(from_numpy(clean_file))

                # clean_file = (clean_file - sp_min) / (sp_max - sp_min) - 0.5
                # noisy_file = (noisy_file - sp_min) / (sp_max - sp_min) - 0.5
                
                
                

                assert len(clean_file) == len(noisy_file)

                i = 0
                while i < len(clean_file) - segments:
                    self.nTrain_samples.append(noisy_file[i:i+segments])
                    self.cTrain_samples.append(clean_file[i:i+segments])
                    i += 1

                pbar.update(1)

    def __len__(self):
        return len(self.cTrain_samples)

    def __getitem__(self, idx):
        noisy = self.nTrain_samples[idx]
        # cleany = self.cTrain_samples[idx]
        clean = self.cTrain_samples[idx]
        
        clean = (clean - self.sp_min) / (self.sp_max - self.sp_min)
        # cleany = (cleany - self.sp_min) / (self.sp_max - self.sp_min) - 0.5

        noisy = from_numpy(noisy).transpose(0,1)
        # cleanog = from_numpy(clean).transpose(0,1).unsqueeze(0)
        clean = from_numpy(clean).transpose(0,1).unsqueeze(0)
        clean2 = torch.clone(clean)

        # noisy = nn.functional.normalize(noisy)
        # clean = nn.functional.normalize(clean)
        
        # clean = (clean_file - sp_min) / (sp_max - sp_min) - 0.5
        
        return clean
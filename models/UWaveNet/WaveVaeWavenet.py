import math
import numpy as np

import torch
from torch import nn
import models.UWaveNet.WaveVaeOperations as WOP

class Wavenet(nn.Module):

    def __init__(self, out_channels, layers = 1, stacks = 2, res_channels = 512, skip_channels = 512, gate_channels = 512, condition_channels = -1, kernel_size = 3, freq_axis_kernel_size=3, dropout = 1 - 0.95, timesteps = 512, upsample_conditional_features = False, upsample_scales = None):
        super(Wavenet, self).__init__()

        #assert layers % stacks == 0
        layers_per_stack = layers // stacks
        self.upsample = upsample_conditional_features

        self.first_conv = WOP.normalisedConv1d(1, res_channels, kernel_size=1, padding=0, dilation=1, bias=True, std_mul=1.0)
        self.skip_conv = WOP.normalisedConv1d(res_channels, res_channels, kernel_size=1, padding=0, dilation=1, bias=True, std_mul=1.0)
        self.cond_conv = nn.Conv1d(64, res_channels, kernel_size = 1, padding=0)

        # Wavenet layers
        dilation = 1
        receptive_field = 1
        self.dropout = nn.Dropout(p = dropout)
        
        self.conv_layers = nn.ModuleList()
        for stack in range(stacks):
            for layer in range(layers):
                dilation = 2**(layer % layers_per_stack)

                resdilconv = WOP.ResidualConv1dGLU(
                    residual_channels = res_channels,
                    gate_channels = gate_channels,
                    kernel_size = kernel_size,
                    skip_out_channels = skip_channels,
                    cin_channels = condition_channels,
                    dropout = dropout,
                    weight_normalisation = True,
                )
                self.conv_layers.append(resdilconv)
        
        self.final_convs_1 = nn.Sequential(
            WOP.normalisedConv1d(skip_channels, 1024, kernel_size = 3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p = dropout)
            # nn.Linear(int(timesteps * skip_channels), int(timesteps * skip_channels))
        )
        
        self.final_convs_2 = nn.Sequential(
            WOP.normalisedConv1d(1024, 512, kernel_size = 3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(512),
            #nn.Dropout(p = dropout),
            # nn.Conv1d(512, 256, kernel_size = 3, padding = 'same'),
            # nn.Tanh()
            # nn.Linear(int(timesteps * out_channels), int(timesteps * out_channels))
        )
        
        self.final_convs_3 = nn.Sequential(
            WOP.normalisedConv1d(512, 256, kernel_size = 3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(256),
            #nn.Dropout(p = dropout),
            nn.Conv1d(256, out_channels, kernel_size = 3, padding = 'same'),
            # nn.Tanh()
            # nn.Linear(int(timesteps * out_channels), int(timesteps * out_channels))
        )


        # Upsample conv net
        # self.upsample_conv_seq = nn.Sequential()
        if upsample_conditional_features:
            self.upsample_conv = []
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = WOP.normalisedConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization= True)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace = True))
                
            self.upsample_conv_seq = nn.Sequential(*self.upsample_conv) 
        else:
            self.upsample_conv = None

        self.receptive_field = WOP.receptive_field_size(layers, stacks, kernel_size)
        print("Receptive field = ", self.receptive_field)

    def forward(self, x, c = None, softmax = True):
        """Forward step
        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
              Also type of input tensor must be FloatTensor, not LongTensor
            softmax (bool): Whether applies softmax or not.
        Returns:
            Tensor: output, shape B x out_channels x T
        """

        B, _, T = x.size()
        # print(c.size())

        # B x 1 x C x T
        # c = c.unsqueeze(1)

        c = self.upsample_conv_seq(c)
        # B x C x T
        c = c.squeeze(1)
        # print(c.size(-1), x.size(-1))
        # print(c.size())
        assert c.size(-1) == x.size(-1)
        
        c = self.cond_conv(c)
        
        # Feed data to network
        x = self.first_conv(x)
        skip = self.skip_conv(x)
        for layer in self.conv_layers:
            x, skip = layer(x, c, skip)

        x = skip
        x = self.final_convs_1(x)
        x = self.dropout(x)
        x = self.final_convs_2(x)
        x = self.final_convs_3(x)
        # x = self.dropout(x)

        return x
        
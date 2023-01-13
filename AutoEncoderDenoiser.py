from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import math

class ConvLayer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()

        padding = 'same'
        if stride > 1:
            padding = math.floor((kernel_size-1)/stride)

        print(padding, stride)

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(output_channels)

    def forward(self, input):
        x = input

        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)

        return x

class Encoder(nn.Module):

    def __init__(self, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, features, rec_field):
        super().__init__()

        layeramt = len(layer_channels)
        self.convlayers = nn.ModuleList()

        for i in range(layeramt):
            input_channels = 1 if i == 0 else layer_channels[i-1]
            convlayer = ConvLayer(input_channels, layer_channels[i], layer_kernel_sizes[i], layer_strides[i])
            self.convlayers.append(convlayer)

        # print(self.convlayers)

        self.flatten = nn.Flatten()

        dense_input = int(math.ceil(features / 4) * math.ceil(rec_field / 4) * layer_channels[-1])

        self.dense = nn.Linear(dense_input, 2)

        
    def forward(self, input):
        x = input

        for layer in self.convlayers:
            x = layer(x)

        x = self.flatten(x)
        x = self.dense(x)
        
        return x


class Decoder(nn.Module):

    def __init__(self, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, features, rec_field):
        super().__init__()

        

    def forward(self, x):
        

        return x


class SPDenoiser(nn.Module):
    def __init__(self, features, rec_field, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, device):
        super().__init__()
        # Input size = [c, features (60), receptive field (256)]

        self.encoder = Encoder(layer_channels, layer_kernel_sizes, layer_strides, latent_dim, features, rec_field)
        self.decoder = Decoder(layer_channels, layer_kernel_sizes, layer_strides, latent_dim, features, rec_field)


        self.device = device

        
    def forward(self, input):
        latent_rep = self.encoder(input)
        # clean_rep = self.decoder(latent_rep)


        return latent_rep
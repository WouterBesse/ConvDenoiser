from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        

    def forward(self, x):
        

        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        

    def forward(self, x):
        

        return x


class SPDenoiser(nn.Module):
    def __init__(self, features, rec_field, steps, latent_dim, device):
        super().__init__()
        # Input size = [c, features (60), receptive field (256)]

        self.encoder = Encoder()
        self.decoder = Decoder()


        self.device = device

        
    def forward(self, input):
        latent_rep = self.encoder(input)
        clean_rep = self.decoder(latent_rep)


        return clean_rep
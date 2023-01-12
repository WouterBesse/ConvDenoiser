from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np

class SPDenoiser(nn.Module):
    def __init__(self, features, rec_field, steps, device):
        super().__init__()
        # Input size = [c, features (60), receptive field (256)]

        


        self.device = device

        
    def forward(self, input):

        return x
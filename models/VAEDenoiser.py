from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from torch.distributions.normal import Normal
import numpy as np
import math
import sys
import models.CustomModules as CustomModules


class Encoder(nn.Module):

    def __init__(self, input_channels, hidden_channels, embedding_dim, num_residual_layers, verbose = False):
        super().__init__()
        
        self.verbose = verbose
        self.duo_conv_1 = CustomModules.DoubleConvStack(input_channels, hidden_channels, 3, 1)
        self.duo_conv_2 = CustomModules.DoubleConvStack(hidden_channels, hidden_channels, 3, 1)

        self.strided_convolution = nn.Conv1d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = 4,
            stride = (1, 2),
            padding = 1
        )


        # 4 feedforward ReLu layers with residual connections.
        self.residual_stack = CustomModules.ResidualStack(
            in_channels = hidden_channels,
            num_hiddens = hidden_channels,
            num_residual_layers = num_residual_layers,
            num_residual_hiddens = hidden_channels
        )

        self.end_conv = nn.Conv1d(
            in_channels = hidden_channels,
            out_channels = embedding_dim,
            kernel_size = 3,
            padding = 1
        )
        
        
    def forward(self, x):
        if self.verbose:
            print("##################")
            print("Encoder preconv size: ", x.size())
        x = self.duo_conv_1(x)
        if self.verbose:
            print("Encoder duo_conv_1 size: ", x.size())
        x = self.strided_convolution(x)
        if self.verbose:
            print("Encoder strided_conv size: ", x.size())
        x = self.duo_conv_2(x)
        if self.verbose:
            print("Encoder duo_conv_2 size: ", x.size())
        x = self.residual_stack(x)
        if self.verbose:
            print("Encoder residual_stack size: ", x.size())
        x = self.end_conv(x)
        if self.verbose:
            print("Encoder end_conv size: ", x.size())
        
        return x


class Decoder(nn.Module):

    def __init__(self, out_channels, hidden_channels, embedding_dim, num_residual_layers, verbose = False):
        super().__init__()
        
        self.verbose = verbose

        self.preconv = nn.Conv1d(
            in_channels = embedding_dim,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )

        self.upsample = nn.Upsample(scale_factor=2)

        # 4 feedforward ReLu layers with residual connections.
        self.residual_stack = CustomModules.ResidualStack(
            in_channels = hidden_channels,
            num_hiddens = hidden_channels,
            num_residual_layers = num_residual_layers,
            num_residual_hiddens = hidden_channels
        )

        self.conv_trans_1 = nn.ConvTranspose1d(
            in_channels = hidden_channels, 
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )

        self.conv_trans_2 = nn.ConvTranspose1d(
            in_channels = hidden_channels, 
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )
        
        self.conv_trans_3 = nn.ConvTranspose1d(
            in_channels = hidden_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1
        )
        

    def forward(self, x):
        if self.verbose:
            print("##################")
            print("Decoder start size: ", x.size())
        x = self.preconv(x)
        if self.verbose:
            print("Decoder preconv size: ", x.size())
        x = self.upsample(x)
        if self.verbose:
            print("Decoder upsample size: ", x.size())
        x = self.residual_stack(x)
        if self.verbose:
            print("Decoder residual stack size: ", x.size())
        x = nn.functional.relu(self.conv_trans_1(x))
        if self.verbose:
            print("Decoder conv_trans_1 size: ", x.size())
        x = nn.functional.relu(self.conv_trans_2(x))
        if self.verbose:
            print("Decoder conv_trans_2 size: ", x.size())
        x = self.conv_trans_3(x)
        if self.verbose:
            print("Decoder conv_trans_3 size: ", x.size())
        return x


class SPDenoiser(nn.Module):
    def __init__(self, features, rec_field, hidden_channels, embedding_dim, latent_dim, device, verbose = False):
        super().__init__()
        # Input size = [c, features (60), receptive field (256)]

        self.device = device        
        self.verbose = verbose
        
        self.encoder = Encoder(features, hidden_channels, embedding_dim, 2, verbose)
        self.latentspace = CustomModules.VariationalLatentConverter(rec_field, hidden_channels, embedding_dim, latent_dim, device, verbose)
        self.decoder = Decoder(features, hidden_channels, embedding_dim, 2, verbose)
        
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.device = device

        
    def forward(self, input):
        if self.verbose:
            print("Input size: ", input.size())
        # print("Input:",torch.min(input), torch.max(input), torch.isnan(input).sum().item())
        x = self.encoder(input)
        # print("Encoder out:",torch.min(x), torch.max(x), torch.isnan(x).sum().item())
        z, mu, sigma = self.latentspace(x)
        # print("Z out:",torch.min(z), torch.max(z), torch.isnan(z).sum().item())
        x = self.decoder(z)
        
        # filter, gate = torch.chunk(x, 2, dim=1)
        
#         xf = self.tanh(x)
#         xg = self.sig(x)

#         x = xf * xg
        
        # print("Decoder out:",torch.min(x), torch.max(x), torch.isnan(x).sum().item())

        return x, mu, sigma


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
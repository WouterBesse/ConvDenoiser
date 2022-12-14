from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import random

def decomposition(i):
    while i > 0:
        n = random.randint(16, 256)
        yield n
        i -= n

def normalize2(clean_file, noisy_file):
    noisy_file = from_numpy(noisy_file)
    clean_file = from_numpy(clean_file)

    noisy_min = torch.min(noisy_file)
    clean_min = torch.min(clean_file)

    noisy_file = noisy_file - min(noisy_min, clean_min)
    clean_file = clean_file - min(noisy_min, clean_min)

    noisy_max = torch.max(noisy_file)
    clean_max = torch.max(clean_file)

    noisy_file = noisy_file / max(noisy_max, clean_max)
    clean_file = clean_file / max(noisy_max, clean_max)

    # print(torch.max(noisy_file), torch.min(noisy_file))

    noisy_file = noisy_file.cpu().detach().numpy()
    clean_file = clean_file.cpu().detach().numpy()

    return clean_file, noisy_file

# Torch implementation of https://github.com/sthalles/cnn_denoiser

class DilBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, padding = 'same'):
        super().__init__()
        # Maybe check the input and output channels
        pad = (kernel_size - 1) * dilation
        self.preconv = nn.Conv1d(in_channels, out_channels, padding = 'same', kernel_size = 1)
        self.postconv = nn.Conv1d(out_channels, out_channels, padding = 'same', kernel_size = 1)
        self.dilconv = nn.Conv1d(in_channels, out_channels, dilation = dilation, kernel_size = kernel_size, padding = pad)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.preconv(x)
        xc = self.dilconv(x)
        xt = self.tanh(xc)
        xs = self.sig(xc)

        z = xt * xs
        z = self.postconv(x)

        res = z + x
        skip = z

        return res, skip

class EndBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.preconv = nn.Conv1d(in_channels, in_channels * 128, kernel_size = 1, padding = 'same')
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout1d(0.2)
        self.lastconv = nn.Conv1d(in_channels * 128, out_channels, kernel_size = 1, padding='same')

    def forward(self, x):
        out = self.relu(x)
        out = self.preconv(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.lastconv(out)

        return out



    

# Denoiser will get tensors of 129 features (frequency) by 8 segments (timesteps)
class SPWavenetDenoiser(nn.Module):
    def __init__(self, features, device):
        super().__init__()

        self.DilBlock1 = DilBlock(features, features, dilation = 1, kernel_size = 2, padding = 'causal') # rec: 256
        self.DilBlock2 = DilBlock(features, features, dilation = 2, kernel_size = 2, padding = 'causal') # rec: 128
        self.DilBlock3 = DilBlock(features, features, dilation = 4, kernel_size = 2, padding = 'causal') # rec: 64
        self.DilBlock4 = DilBlock(features, features, dilation = 8, kernel_size = 2, padding = 'causal') # rec: 32
        self.DilBlock5 = DilBlock(features, features, dilation = 16, kernel_size = 2, padding = 'causal') # rec: 16
        self.DilBlock6 = DilBlock(features, features, dilation = 32, kernel_size = 2, padding = 'causal') # rec: 8
        self.DilBlock7 = DilBlock(features, features, dilation = 64, kernel_size = 2, padding = 'causal') # rec: 4
        self.DilBlock8 = DilBlock(features, features, dilation = 128, kernel_size = 2, padding = 'causal') # rec: 2
        self.DilBlock9 = DilBlock(features, features, dilation = 256, kernel_size = 2, padding = 'causal') # rec: 1

        self.receptivefield = 256

        self.end = EndBlock(features, features)
        self.device = device

        
    def forward(self, input):
        x = input.to(self.device)
        x, skip1 = self.DilBlock1(x)
        x, skip2 = self.DilBlock2(x)
        x, skip3 = self.DilBlock3(x)
        x, skip4 = self.DilBlock4(x)
        x, skip5 = self.DilBlock5(x)
        x, skip6 = self.DilBlock6(x)
        x, skip7 = self.DilBlock7(x)
        x, skip8 = self.DilBlock8(x)
        x, skip9 = self.DilBlock9(x)

        skips = skip1 + skip2 + skip3 + skip4 + skip5 + skip6 + skip7 + skip8 + skip9

        x = self.end(skips)[:, :, -1:]

        return x


# Custom dataset for SP Denoiser
class SPDataset(Dataset):

    def __init__(self, trainN_dir, trainC_dir, transform=None, target_transform=None, segments=8, receptivefield = 256):
        print("Directories: ", trainN_dir, "|", trainC_dir)
        self.receptivefield = receptivefield
        
        nTrain_dirlist = os.listdir(trainN_dir)
        cTrain_dirlist = os.listdir(trainC_dir)

        self.nTrain_samples = []
        self.cTrain_samples = []

        print("Noisy file amount: ", len(nTrain_dirlist))
        print("Clean file amount: ", len(cTrain_dirlist))

        with tqdm(total=len(nTrain_dirlist), desc='Loading files to dataset') as pbar:
            for noisy, clean in zip(nTrain_dirlist, cTrain_dirlist):
                noisy_path = os.path.join(trainN_dir, noisy)
                clean_path = os.path.join(trainC_dir, clean)

                noisy_file = np.load(noisy_path).astype(np.float32)
                clean_file = np.load(clean_path).astype(np.float32)

                

                # meannumb = torch.mean(torch.cat((noisy_file, clean_file), dim=0))

                # noisy_file = nn.functional.normalize(from_numpy(noisy_file))
                

                # clean_file = nn.functional.normalize(from_numpy(clean_file))

                clean_file, noisy_file = normalize2(clean_file, noisy_file)
                

                assert len(clean_file) == len(noisy_file)
                splitamounts = decomposition(len(clean_file))
                timeamount = 0
                i = 0
                for time in splitamounts:
                    amount = time + timeamount
                    self.nTrain_samples.append(noisy_file[timeamount:amount])
                    self.cTrain_samples.append(clean_file[timeamount:amount])
                    timeamount += time

                pbar.update(1)

    def __len__(self):
        return len(self.nTrain_samples)

    def __getitem__(self, idx):
        noisy = self.nTrain_samples[idx]
        clean = self.cTrain_samples[idx]

        noisy = from_numpy(noisy).transpose(0,1)
        clean = from_numpy(clean).transpose(0,1)

        if len(noisy) < self.receptivefield:
            padamount = self.receptivefield - len(noisy)
            noisy = torch.nn.ConstantPad1d((padamount, 0), 0)(noisy)
            clean = torch.nn.ConstantPad1d((padamount, 0), 0)(clean)

        noisy = noisy.unsqueeze(0)
        # cleanog = from_numpy(clean).transpose(0,1).unsqueeze(0)
        clean = clean[:, -1:].unsqueeze(0)

        # noisy = nn.functional.normalize(noisy)
        # clean = nn.functional.normalize(clean)
        return noisy, clean
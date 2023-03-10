from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np

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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 'same', use_bn=True, skip=False):
        super().__init__()

        self.conv1D = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, stride=(1,1), padding = padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.use_bn = use_bn
        self.skip = skip

    def forward(self, x, oldskip = None):
        skip = self.conv1D(x.to('cuda'))
        # print("Convstep size: ", skip.size())

        if oldskip is not None:
            skip = skip + oldskip

        x = self.relu(skip)
        if self.use_bn:
            x = self.bn(x)
        if self.skip:
            return x, skip
        else:
            return x

class ConvSeq(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.Conv1 = ConvBlock(8, 18, (9,1))
        self.Conv2 = ConvBlock(18, 30, (5,1), skip=True)
        self.Conv3 = ConvBlock(30, 8, (9,1))
        self.device = device

    def forward(self, x, skip=None):
        x = self.Conv1(x.to(self.device))
        x, skip = self.Conv2(x, skip)
        x = self.Conv3(x)
        return x, skip
    

# Denoiser will get tensors of 60 features (frequency) by 8 segments (timesteps)
class SPDenoiser(nn.Module):
    def __init__(self, features, segments, device):
        super().__init__()

        self.zeroPadding = nn.ZeroPad2d((0, 0, 4, 4)).to(device)
        self.firstConv = ConvBlock(1, 18, (9,segments), padding='valid').to(device)
        self.secondConv = ConvBlock(18, 30, (5,1), skip=True).to(device)
        self.thirdConv = ConvBlock(30, 8, (9,1)).to(device)

        self.ConvSeq1 = ConvSeq(device).to(device)
        self.ConvSeq2 = ConvSeq(device).to(device)
        self.ConvSeq3 = ConvSeq(device).to(device)
        self.ConvSeq4 = ConvSeq(device).to(device)
        # self.fourthConv = ConvBlock(8, 18, (9,1))
        # self.fifthConv = ConvBlock(18, 30, (5,1), skip=True)
        # self.sixthConv = ConvBlock(30, 8, (9,1))
        # self.seventhConv = ConvBlock(8, 18, (9,1))
        # self.eightConv = ConvBlock(18, 30, (5,1), skip=True)
        # self.ninthConv = ConvBlock(30, 8, (9,1))

        self.dropout = nn.Dropout2d(p=0.2).to(device)
        self.finalConv = nn.Conv2d(8, 1, kernel_size=(features, 1), bias=False, padding = 'same', stride=(1,1)).to(device)
        self.device = device

        
    def forward(self, input):
        x = input.to(self.device)
        x = self.zeroPadding(x)
        x = self.firstConv(x)
        x, skip0 = self.secondConv(x)
        x = self.thirdConv(x)
        x, skip1 = self.ConvSeq1(x)
        x, red = self.ConvSeq2(x)
        x, red = self.ConvSeq3(x, skip1)
        x, red = self.ConvSeq4(x, skip0)
        x = self.dropout(x)
        x = self.finalConv(x)
        return x


# Custom dataset for SP Denoiser
class SPDataset(Dataset):

    def __init__(self, trainN_dir, trainC_dir, transform=None, target_transform=None, segments=8):
        print("Directories: ", trainN_dir, "|", trainC_dir)
        
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

                i = 0
                while i < len(clean_file) - segments:
                    self.nTrain_samples.append(noisy_file[i:i+segments])
                    self.cTrain_samples.append(clean_file[i:i+segments])
                    i += 1

                pbar.update(1)

    def __len__(self):
        return len(self.nTrain_samples)

    def __getitem__(self, idx):
        noisy = self.nTrain_samples[idx]
        clean = self.cTrain_samples[idx]

        noisy = from_numpy(noisy).transpose(0,1).unsqueeze(0)
        # cleanog = from_numpy(clean).transpose(0,1).unsqueeze(0)
        clean = from_numpy(clean).transpose(0,1)[:, -1:].unsqueeze(0)

        # noisy = nn.functional.normalize(noisy)
        # clean = nn.functional.normalize(clean)
        return noisy, clean
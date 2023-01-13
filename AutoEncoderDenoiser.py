from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import math
import sys

class ConvLayer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()

        padding = 'same'
        if stride > 1:
            padding = math.floor((kernel_size-1)/stride)

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(output_channels)

    def forward(self, input):
        x = input

        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)

        return x


class ConvTransLayer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, out_pad):
        super().__init__()

        padding = (1,1)
        # out_pad = 0
        if stride > 1:
            padding = (1,1)
            # out_pad = 1


        self.padding = padding

        self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding, output_padding=out_pad)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(output_channels)

    def forward(self, input):
        x = input
        # print(self.padding)

        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)

        return x

class Encoder(nn.Module):

    def __init__(self, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, dense_size):
        super().__init__()

        layeramt = len(layer_channels)
        self.convlayers = nn.ModuleList()

        for i in range(layeramt):
            input_channels = 1 if i == 0 else layer_channels[i - 1]
            convlayer = ConvLayer(input_channels, layer_channels[i], layer_kernel_sizes[i], layer_strides[i])
            self.convlayers.append(convlayer)

        self.flatten = nn.Flatten(start_dim=1)

        
        print('Encoder dense size: ', dense_size, latent_dim)
        self.dense = nn.Linear(dense_size, latent_dim)

        
    def forward(self, input):
        x = input

        for layer in self.convlayers:
            x = layer(x)
            # print(x.size())

        x = self.flatten(x)
        # print('Flattened size', x.size())
        x = self.dense(x)
        # print(x.size())
        
        return x


class Decoder(nn.Module):

    def __init__(self, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, dense_size, small_shape, output_pads):
        super().__init__()

        layeramt = len(layer_channels)
        self.convlayers = nn.ModuleList()
        self.dense = nn.Linear(latent_dim, dense_size)
        self.reshape = nn.Unflatten(1, small_shape)

        for i in reversed(range(1, layeramt)):
            input_channels = layer_channels[-1] if i == layeramt - 1 else layer_channels[i + 1]
            convlayer = ConvTransLayer(input_channels, layer_channels[i], layer_kernel_sizes[i], layer_strides[i], output_pads[i])
            self.convlayers.append(convlayer)

        self.finaltranspose = nn.ConvTranspose2d(layer_channels[1], 1, layer_kernel_sizes[0], layer_strides[0], padding=1, output_padding=1)
        self.sig = nn.Sigmoid()
        

    def forward(self, input):
        x = self.dense(input)
        # print(x.size())
        x = self.reshape(x)
        # print(x.size())

        for layer in self.convlayers:
            x = layer(x)
            # print(x.size())

        x = self.finaltranspose(x)
        # print(x.size())
        x = self.sig(x)
        # print(x.size())

        return x


class SPDenoiser(nn.Module):
    def __init__(self, features, rec_field, layer_channels, layer_kernel_sizes, layer_strides, latent_dim, output_pads, device):
        super().__init__()
        # Input size = [c, features (60), receptive field (256)]

        dim_div = np.prod(np.array(layer_strides))
        # print("Dim div = ", dim_div)
        # small_shape = (layer_channels[-1], math.ceil(features / dim_div), math.ceil(rec_field / dim_div))
        small_shape = (layer_channels[-1], math.ceil(features / dim_div), 16)
        print('Small shape: ', small_shape)
        # small_shape = 512
        dense_size = 2048
        

        self.encoder = Encoder(layer_channels, layer_kernel_sizes, layer_strides, latent_dim, dense_size)
        self.decoder = Decoder(layer_channels, layer_kernel_sizes, layer_strides, latent_dim, dense_size, small_shape, output_pads)


        self.device = device

        
    def forward(self, input):
        latent_rep = self.encoder(input)
        clean_rep = self.decoder(latent_rep)


        return clean_rep


class TimbreDataset(torch.utils.data.Dataset):
    def __init__(self,
                 trainN_dir, trainC_dir,
                 receptive_field,
                 target_length=256,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             | | | | | | | | |
        self._receptive_field = receptive_field
        self.target_length = target_length
        self.item_length = self.target_length + 1
        self.noise = 1.2

        sp_folder = trainC_dir
        condi_folder = trainN_dir

        # store every data length
        self.data_lengths = []

        self.spectral_array = []
        self.condition_array = []
        Cdirlist = os.listdir(sp_folder)
        Ndirlist = os.listdir(condi_folder)

        self.sp_max = (-sys.maxsize - 1)
        self.sp_min = sys.maxsize

        with tqdm(total=len(Ndirlist), desc='Loading files to dataset') as pbar:
            for condi, sp in zip(Cdirlist, Ndirlist):

                
                sp = np.load(os.path.join(sp_folder, sp)).astype(np.float16)
                condition = np.load(os.path.join(condi_folder, condi)).astype(np.float16)

                assert len(sp) == len(condition)

                self.data_lengths.append(math.ceil(len(sp)/target_length))

                # pad zeros(_receptive_field, 60) ahead for each data
                sp = np.pad(sp, ((1, 0), (0, 0)), 'constant', constant_values=0)

                _sp_max = max(np.max(condition), np.max(sp))
                _sp_min = min(np.min(condition), np.min(sp))

                if _sp_min < self.sp_min:
                    self.sp_min = _sp_min
                if _sp_max > self.sp_max:
                    self.sp_max = _sp_max

                self.condition_array.append(condition)
                self.spectral_array.append(sp)

                pbar.update(1)

        self._length = 0
        self.calculate_length()
        self.train = train


    def calculate_length(self):
        self._length = 0
        for _len in self.data_lengths:
            self._length += _len

    def __getitem__(self, idx):
        # find witch file it require
        sp, condition = None, None
        current_files_idx = 0
        total_len = 0
        for fid, _len in enumerate(self.data_lengths):
            current_files_idx = idx - total_len
            total_len += _len
            if idx < total_len:
                sp = self.spectral_array[fid]
                condition = self.condition_array[fid]
                break

        target_index = current_files_idx*self.target_length
        short_sample = self.target_length - (len(sp) - target_index)
        if short_sample > 0:
            target_index -= short_sample


        condition = (condition[target_index:target_index+self.target_length, :] - self.sp_min) / (self.sp_max - self.sp_min) - 0.5
        sp = (sp[target_index:target_index+self.item_length, :] - self.sp_min) / (self.sp_max - self.sp_min) - 0.5
        

        item_condition = torch.Tensor(condition).transpose(0, 1)

        # notice we pad 1 before so
        sp_sample = torch.Tensor(sp).transpose(0, 1)
        # sp_item = sp_sample[:, :self.target_length]
        # sp_target = sp_sample[:, -self.target_length:]

        item_condition = item_condition.unsqueeze(0)
        sp_sample = sp_sample.unsqueeze(0)

        print(item_condition.size()[2], sp_sample.size()[2])

        assert item_condition.size()[2] == sp_sample.size()[2] == 256
        
        return item_condition, sp_sample


    def __len__(self):

        return self._length


class SPDataset(Dataset):

    def __init__(self, trainN_dir, trainC_dir, transform=None, target_transform=None, segments=256):
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

                sp_max = max(np.max(clean_file), np.max(noisy_file))
                sp_min = min(np.min(clean_file), np.min(noisy_file))

                

                # meannumb = torch.mean(torch.cat((noisy_file, clean_file), dim=0))

                # noisy_file = nn.functional.normalize(from_numpy(noisy_file))
                

                # clean_file = nn.functional.normalize(from_numpy(clean_file))

                clean_file = (clean_file - sp_min) / (sp_max - sp_min) - 0.5
                noisy_file = (noisy_file - sp_min) / (sp_max - sp_min) - 0.5
                

                assert len(clean_file) == len(noisy_file)

                i = 0
                while i < len(clean_file) - segments:
                    self.nTrain_samples.append(noisy_file[i:i+segments])
                    self.cTrain_samples.append(clean_file[i:i+segments])
                    i += 128

                pbar.update(1)

    def __len__(self):
        return len(self.nTrain_samples)

    def __getitem__(self, idx):
        noisy = self.nTrain_samples[idx]
        clean = self.cTrain_samples[idx]

        noisy = from_numpy(noisy).transpose(0,1).unsqueeze(0)
        # cleanog = from_numpy(clean).transpose(0,1).unsqueeze(0)
        clean = from_numpy(clean).transpose(0,1).unsqueeze(0)

        # noisy = nn.functional.normalize(noisy)
        # clean = nn.functional.normalize(clean)
        return noisy, clean
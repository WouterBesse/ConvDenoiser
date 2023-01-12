from torch import nn, from_numpy
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.distributions.normal import Normal
from util import sample_from_CGM
import os
import math
from tqdm import tqdm
import numpy as np
import random
import sys


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

    noisy_file = noisy_file
    clean_file = clean_file

    return clean_file, noisy_file

# Torch implementation of https://github.com/sthalles/cnn_denoiser
class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)

class CausalDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding = 1):
        super().__init__()

        # self.dilpad = nn.ConstantPad1d((dilation, 0), 0)
        # self.pad = (kernel_size - 1) * dilation
        self.conv1D = nn.Conv1d(in_channels, out_channels, kernel_size, dilation = dilation, bias=True)
        nn.init.xavier_uniform_(self.conv1D.weight, gain=nn.init.calculate_gain('linear'))
        # self.ignoreOutIndex = (kernel_size-1) * dilation

    def forward(self, x):
        # x = self.dilpad(x)
        return self.conv1D(x)

class DilBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, condition_channels, skip_channels, dilation, kernel_size):
        super().__init__()
        
        self.resconv = nn.Conv1d(residual_channels, residual_channels, kernel_size = 1, bias=True)
        nn.init.xavier_uniform_(self.resconv.weight, gain=nn.init.calculate_gain('linear'))

        self.dilconv = CausalConv1d(residual_channels, dilation_channels * 2, dilation = dilation, kernel_size = kernel_size)

        self.condiconv = nn.Conv1d(condition_channels, dilation_channels * 2, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.condiconv.weight, gain=nn.init.calculate_gain('linear'))

        self.skipConv1D = nn.Conv1d(residual_channels, skip_channels, kernel_size = 1, bias=True)
        nn.init.xavier_uniform_(self.skipConv1D.weight, gain=nn.init.calculate_gain('linear'))

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, condition, skip):
        
        xdil = self.dilconv(x)
        condi = self.condiconv(condition)

        dilated = xdil + condi

        filter, gate = torch.chunk(dilated, 2, dim=1)

        xf = self.tanh(filter)
        xg = self.sig(gate)

        z = xf * xg
        res = self.resconv(z)

        res = res + x[:, :, -z.size(2):]

        s = self.skipConv1D(z)
        try:
            skip = skip[:, :, -z.size(2):]
        except:
            skip = 0
        
        skip = s + skip

        return res, skip

class EndBlock(nn.Module):

    def __init__(self, skip_channels, condition_channels, out_channels):
        super().__init__()

        self.tanh = nn.Tanh()
        self.lastconv = nn.Conv1d(skip_channels, out_channels, kernel_size = 1, bias=True)
        nn.init.xavier_uniform_(self.lastconv.weight, gain=nn.init.calculate_gain('linear'))
        self.dropout = nn.Dropout1d(p=0.2)

        self.condiendconv = nn.Conv1d(condition_channels, skip_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.condiendconv.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, condition):
        condi = self.condiendconv(condition)
        x = x + condi

        out = self.tanh(x)
        out = self.dropout(out)
        out = self.lastconv(out)

        return out
    

# Denoiser will get tensors of 129 features (frequency) by 8 segments (timesteps)
class SPWavenetDenoiser(nn.Module):
    def __init__(   self, 
                    in_channels = 60, 
                    stacks = 4, 
                    layers = 3,
                    dilation_channels = 130,
                    residual_channels = 130,
                    skip_channels = 240, 
                    condition_channels = 60,
                    output_channels = 240,
                    initial_kernel = 10,
                    kernel_size = 2,
                    device = 'cuda'):

        super().__init__()

        self.resblocks = nn.ModuleList()
        dilation = 1
        receptive_field = 1

        for b in range(stacks):
            additional_scope = 2 - 1
            new_dilation = 1

            actual_layer = layers
            if b == stacks-1:
                actual_layer = layers - 1

            for layer in range(actual_layer):
                self.resblocks.append(DilBlock(residual_channels, dilation_channels, condition_channels, skip_channels, dilation = new_dilation, kernel_size = kernel_size))
                additional_scope *= 2
                receptive_field += additional_scope
                new_dilation *= 2

        
        self.preconv = nn.Conv1d(in_channels, residual_channels, kernel_size = initial_kernel, bias=True)
        nn.init.xavier_uniform_(self.preconv.weight, gain=nn.init.calculate_gain('linear'))

        self.start_pad = nn.ConstantPad1d((initial_kernel-1, 0), 0)

        self.receptivefield = 256
        self.noise_lambda = 0.2

        self.end = EndBlock(skip_channels = skip_channels, condition_channels = condition_channels, out_channels = output_channels)
        self.device = device

        self.receptive_field = receptive_field + 10 - 1
        print("Receptive field =", self.receptivefield)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def generate(self, condition):

        self.eval()

        cat_input = None
        num_samples = condition.shape[1]
        generated = torch.zeros(60, num_samples).to(self.device)

        model_input = torch.zeros(1, 60, 1).to(self.device)

        from tqdm import tqdm
        for i in tqdm(range(num_samples)):
            if i < self.receptive_field:
                condi = condition[:, :i + 1]
            else:
                condi = condition[:, i - self.receptive_field + 1:i + 1]
            condi = condi.unsqueeze(0)

            x = self.forward(model_input, condi, True)
            x = x[:, :, -1].squeeze()

            t = 0.05
            x_sample = sample_from_CGM(x.detach(), t)

            generated[:, i] = x_sample.squeeze(0)

            # set new input
            if i < self.receptive_field - 1:
                model_input = generated[:, :i + 1]
                if cat_input is not None:
                    to_cat = cat_input[:, :i + 1]
                    model_input = torch.cat((to_cat, model_input), 0)

                model_input = torch.Tensor(np.pad(model_input.cpu(), ((0, 0), (1, 0)), 'constant', constant_values=0)).to(self.device)
            else:
                model_input = generated[:, i - self.receptive_field + 1:i + 1]
                if cat_input is not None:
                    to_cat = cat_input[:, i - self.receptive_field + 1:i + 1]
                    model_input = torch.cat((to_cat, model_input), 0)

            model_input = model_input.unsqueeze(0)

        self.train()

        return generated
        
    def forward(self, input, condition, generate = False):
        inputty = input.to(self.device)
        condition = condition.to(self.device)
        if generate:
            dist = inputty
        else:
            dist = Normal(inputty, self.noise_lambda)
            dist = dist.sample()

        x = self.start_pad(dist)
        x = self.preconv(x)
        skip = 0

        for resblock in self.resblocks:
            x, skip = resblock(x, condition, skip)

        x = self.end(skip, condition)

        return x


# Custom dataset for SP Denoiser
class WavenetDataset(Dataset):

    def __init__(self, trainN_dir, trainC_dir, transform=None, target_transform=None, segments=8, receptivefield = 256, target_length = 210):
        print("Directories: ", trainN_dir, "|", trainC_dir)
        self.receptivefield = receptivefield
        self.item_length = target_length + 1
        self.target_length = target_length

        averagelength = 256
        lengthcount = 0
        
        nTrain_dirlist = os.listdir(trainN_dir)
        cTrain_dirlist = os.listdir(trainC_dir)

        self.nTrain_samples = []
        self.cTarTrain_samples = []
        self.cInTrain_samples = []

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
                extra = 0
                i = 0

                while i < len(clean_file) - self.item_length:
                    cleansample = clean_file[i:i+self.item_length].transpose(0,1)
                    noisesample = noisy_file[i:i+self.target_length].transpose(0,1)
                    # # print("Size before: ", madesample.size())
                    # padamount = self.receptivefield - noisesample.size()[1]
                    # # print("Padamount: ", padamount)
                    # if padamount <= 0:
                    #     extra += 256
                    # else:
                    #     cleansample = torch.nn.ConstantPad1d((padamount, 0), 0)(cleansample)
                    #     noisesample = torch.nn.ConstantPad1d((padamount, 0), 0)(noisesample)
                    self.nTrain_samples.append(noisesample)
                    self.cInTrain_samples.append(cleansample[:, :self.target_length])
                    self.cTarTrain_samples.append(cleansample[:, -self.target_length:])
                    averagelength = (averagelength + noisesample.size()[1]) / 2

                    i += 4

                # for time in splitamounts:
                #     amount = time + timeamount
                #     self.nTrain_samples.append(noisy_file[timeamount:amount])
                #     self.cTrain_samples.append(clean_file[timeamount:amount])
                #     timeamount += time

                pbar.update(1)

            print('Average length: ', averagelength)

    def __len__(self):
        return len(self.nTrain_samples)

    def __getitem__(self, idx):
        noisy = self.nTrain_samples[idx]
        cleanTar = self.cTarTrain_samples[idx]
        cleanIn = self.cInTrain_samples[idx]

        # if length < self.receptivefield:
        #     padamount = self.receptivefield - length
        #     noisy = torch.nn.ConstantPad1d((padamount, 0), 0)(noisy)
        #     clean = torch.nn.ConstantPad1d((padamount, 0), 0)(clean)

        # cleanog = from_numpy(clean).transpose(0,1).unsqueeze(0)

        # noisy = nn.functional.normalize(noisy)
        # clean = nn.functional.normalize(clean)
        return (cleanIn, noisy), cleanTar


class TimbreDataset(torch.utils.data.Dataset):
    def __init__(self,
                 trainN_dir, trainC_dir,
                 receptive_field,
                 target_length=210,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             | | | | | | | | |
        self._receptive_field = receptive_field
        self.target_length = target_length
        self.item_length = 1+self.target_length
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
        short_sample = self.target_length - (len(sp) - 1 - target_index)
        if short_sample > 0:
            target_index -= short_sample


        condition = (condition[target_index:target_index+self.target_length, :] - self.sp_min) / (self.sp_max - self.sp_min) - 0.5
        sp = (sp[target_index:target_index+self.item_length, :] - self.sp_min) / (self.sp_max - self.sp_min) - 0.5
        

        item_condition = torch.Tensor(condition).transpose(0, 1)

        # notice we pad 1 before so
        sp_sample = torch.Tensor(sp).transpose(0, 1)
        sp_item = sp_sample[:, :self.target_length]
        sp_target = sp_sample[:, -self.target_length:]
        
        return (sp_item, item_condition), sp_target


    def __len__(self):

        return self._length
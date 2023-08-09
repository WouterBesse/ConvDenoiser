import math
import torch
from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC, Resample
from torchaudio.functional import compute_deltas
import torchaudio
import models.VAEWavenet.WaveVaeOperations as WOP
import models.VAEWavenet.WaveVaeWavenet as WaveNet
from tqdm import tqdm
import random
import numpy as np
import os

class Decoder(nn.Module):
    """
    VAE Decoder
    """    
    def __init__(self, input_size, hidden_dim, dilation_rates, out_channels, upsamples, zsize = 128, num_cycles = 10, num_cycle_layers = 3, kernel_size = 3, pre_kernel_size = 32, use_jitter = True, jitter_probability = 0.12, use_kaiming_normal = False):
        super().__init__()

        self.receptive_field = sum(dilation_rates) * (kernel_size - 1) + pre_kernel_size

        self.use_jitter = use_jitter
        if use_jitter:
            self.jitter = WOP.Jitter(jitter_probability)

        self.conv_1 = WOP.Conv1dWrap(
            in_channels=zsize,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.wavenet = WaveNet.Wavenet(
            out_channels = 1,
            layers = 13,
            stacks = 3,
            res_channels = 256,
            skip_channels = 256,
            gate_channels = 512,
            condition_channels = 256,
            kernel_size = 3,
            upsample_conditional_features=True,
            upsample_scales = upsamples, # 768
            #upsample_scales=[2, 2, 2, 2, 12]
        )

    def forward(self, x, cond, jitter):
        """Forward step
        Args:
            x (Tensor): Mono audio signal, shape (B x 1 x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            xsize (Tuple): Size of condition before flattening
            jitter (Bool): Argument deciding if we should jitter our condition or not
        Returns:
            X (Tensor): Denoised result, shape (B x 1 x T)
        """
        if self.use_jitter and jitter:
            condition = self.jitter(cond)

        condition = self.conv_1(cond)
        x = self.wavenet(x, condition)

        return x

class Encoder(nn.Module):
    """
    VAE Encoder
    """    
    def __init__(self, input_size, hidden_dim = 768, zsize = 128, resblocks = 2, relublocks = 4):
        super().__init__()

        features, timesteps = input_size
        self.prenet = WOP.Conv1dWrap(in_channels = features, 
                                     out_channels = hidden_dim, 
                                     kernel_size = 3, 
                                     padding = 'same',
                                     bias = False)
        
        self.preconv = WOP.Conv1dWrap(in_channels = features, 
                                      out_channels = hidden_dim, 
                                      kernel_size = 3, 
                                      padding='same',
                                      bias = False)
        self.ReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.zsize = zsize

        self.downsample = WOP.Conv1dWrap(in_channels = hidden_dim, 
                                         out_channels = hidden_dim, 
                                         kernel_size = 4, 
                                         stride = 2, 
                                         padding = 1,
                                         bias = False)
        
        self.resblocks = nn.ModuleList()
        for _ in range(resblocks):
            self.resblocks.append(WOP.Conv1dWrap(in_channels = hidden_dim, 
                                                 out_channels = hidden_dim, 
                                                 kernel_size = 3, 
                                                 padding='same'))

        self.relublocks = nn.ModuleList()
        for _ in range(relublocks):
            self.relublocks.append(WOP.Conv1dWrap(in_channels = hidden_dim, 
                                                  out_channels = hidden_dim, 
                                                  kernel_size = 3, 
                                                  padding='same'))

        self.linear = WOP.Conv1dWrap(in_channels = hidden_dim, 
                                     out_channels = zsize * 2, 
                                     kernel_size = 1)


    def forward(self, x):
        """Forward step
        Args:
            x (Tensor): Noisy MFCC, shape (B x features x timesteps)
        Returns:
            zcomb[:, :self.zsize] (Tensor): Latent space mean, shape (B x zsize)
            zcomb[:, self.zsize:] (Tensor): Latent space variance, shape (B x zsize)
            x_size (Tuple): Size of condition before flattening, shape (B x hidden_dim x timesteps)
        """
        # Preprocessing conv with residual connections
        net = self.prenet(x)
        conv = self.preconv(x)
        x = self.ReLU(net) + self.ReLU(conv)

        # Downsample
        x = self.ReLU(self.downsample(x))

        # Residual convs
        for resblock in self.resblocks:
            xres = self.ReLU(resblock(x))
            x = xres + x

        # Relu blocks
        for relblock in self.relublocks:
            xrelu = self.ReLU(relblock(x))
            x = x + xrelu
        x = self.ReLU(x)

        zcomb = self.linear(x)
        mu, log_var = torch.split(zcomb, self.zsize, dim=1)

        return mu, log_var
    

class WaveNetVAE(nn.Module):

    def __init__(self, input_size, device, num_hiddens, dil_rates, zsize = 128, resblocks = 2, out_channels = 256):
        super(WaveNetVAE, self).__init__()

        self.encoder = Encoder(
            input_size = input_size,
            hidden_dim = 768,
            zsize = zsize,
            resblocks = resblocks,   
        )

        self.decoder = Decoder(
            input_size = input_size,
            hidden_dim = 768,
            dilation_rates = dil_rates,
            out_channels = out_channels,
            upsamples = dil_rates,
            zsize = zsize
        )
        
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        
    
    def sample(self, mu, logvar):
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
        else:
            z = mu
            
        return z

    def forward(self, xau, xspec, jitter):
        """Forward step
        Args:
            xau (Tensor): Noisy audio, shape (B x 1 x T)
            xspec (Tensor): Noisy MFCC, shape (B x features x timesteps)
            jitter (Bool): To jitter the latent space condition or not
        Returns:
            x_hat (Tensor): Denoised audio, shape (B x 1 x T)
            mean (Tensor): Mean of latent space, shape (B x zsize)
            var (Tensor): Variance of latent space, shape (B x zsize)
        """
        mean, var = self.encoder(xspec)
        z = self.sample(mean, var)

        x_hat = self.decoder(xau, z, jitter)
        
        return x_hat, mean, var

class WaveVaeDataset(Dataset):

    def __init__(self, clean_folder, noisy_folder, clip_length = 512, clips = 1, sr = 44100):
        super(WaveVaeDataset, self).__init__()
        """
        Dataset for the WaveVAE model
        Will return clean and noisy audio pairs, as well as an MFCC of the clean/noisy audio
        """
        
        self.clip_length = clip_length
        self.clean_folder = clean_folder

        self.clean_files = []
        self.noisy_files = []
        self.mfccs = []
        
        print("Collecting filepaths")
        clean_filepaths = os.listdir(clean_folder)
        clean_filepaths = clean_filepaths[:clips]
        
        self.noisy_filepaths = [os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if os.path.isfile(os.path.join(noisy_folder, f))]
        
        # Min and Max values for normalisation
        self.au_min = 99999999999
        self.au_max = -99999999999
        
        self.sp_min = 99999999999
        self.sp_max = -99999999999

        _, clean_samplerate = torchaudio.load(os.path.join(clean_folder, clean_filepaths[5]))
        _, noisy_samplerate = torchaudio.load(os.path.join(noisy_folder, self.noisy_filepaths[5]))
        self.clean_resampler = Resample(clean_samplerate, sr).cuda()
        self.noisy_resampler = Resample(noisy_samplerate, sr).cuda()
        
        self.mfcc_trans = MFCC(sr, 20, log_mels = True, melkwargs={"hop_length": 125}).cuda() # Create MFCC in the right samplerate

        with tqdm(total=len(clean_filepaths), desc=f'Loading files to dataset. Len clean_files =  {len(self.clean_files)}') as pbar:
            
            for f in clean_filepaths:
                if os.path.isfile(os.path.join(clean_folder, f)):
                    noise_indices = []
                    noiserange = 1
                    
                    # Load clean file
                    clean_audiopath = os.path.join(clean_folder, f)
                    
                    # CURRENTLY NOT USED - WILL ONLY USE 1 NOISE FILE
                    for r in range(noiserange): # Choose a random noise file 3 times, to create 3 copies of the same voice with different noise profiles
                        noise_selector = random.randint(0, len(self.noisy_filepaths) - 1)
                        if noise_selector not in noise_indices:
                            noise_indices.append(noise_selector)
                        else:
                            noiserange += 1
                    # MIGHT ENABLE IN THE FUTURE AGAIN

                    noisy_audiopath = self.noisy_filepaths[noise_indices[0]]                        

                    noisy_audiofile, clean_audiofile_og = self.loadFiles(clean_audiopath, noisy_audiopath)

                    clean_audiofile, noisy_audiofile = self.processAudio(clean_audiofile_og.cuda(), noisy_audiofile.cuda())      

                    for i in range(clip_length, clean_audiofile.size()[-1] - 5120, 8192*2):

                        clean_audio = clean_audiofile[:, i - clip_length:i + 8192 + clip_length]
                        noisy_audio = noisy_audiofile[:, i - clip_length:i + 8192 + clip_length]
                        if clean_audio.size()[0] > 1 or noisy_audio.size()[0] > 1:
                            print("ERROR: Audio file has more than 1 channel")

                        mfcc = self.getMFCC(clean_audio)
                        
                        audiosize = clip_length * 2 + 8192
                        if clean_audio.size()[-1] == audiosize and noisy_audio.size()[-1] == audiosize:

                            # Add to lists
                            self.clean_files.append(clean_audio.cpu())
                            self.noisy_files.append(noisy_audio.cpu())
                            self.mfccs.append(mfcc.cpu())

                            # Get data for normalisation
                            au_min = min(torch.min(noisy_audio), torch.min(clean_audio))
                            au_max = max(torch.max(noisy_audio), torch.max(clean_audio))
                            self.au_min, self.au_max = self.getMinMax(au_min, au_max, self.au_min, self.au_max)

                            sp_min = torch.min(mfcc)
                            sp_max = torch.max(mfcc)
                            self.sp_min, self.sp_max = self.getMinMax(sp_min, sp_max, self.sp_min, self.sp_max)


                    pbar.set_description(f'Loading files to dataset. Len clean_files =  {len(self.clean_files)}. ')
                    pbar.update(1)
        
        print(self.noisy_files[0].size())
        print(self.clean_files[0].size())
        self.audiomean = torch.cat((torch.stack(self.noisy_files), torch.stack(self.clean_files)), 0).mean()
        self.audiovar = torch.cat((torch.stack(self.noisy_files), torch.stack(self.clean_files)), 0).var()
        
    def processAudio(self, clean_audiofile, noisy_audiofile):
        snr_dbs = random.randint(2, 15) # Random signal to noise ratio between 2 and 13

        # Make samplerate the same
        
        noisy_audiofile = self.noisy_resampler(noisy_audiofile)
        clean_audiofile = self.clean_resampler(clean_audiofile)

        # Make stereo file mono
        if noisy_audiofile.size()[0] == 2:
            noisy_audiofile = torch.mean(noisy_audiofile, dim=0).unsqueeze(0)

        if clean_audiofile.size()[0] == 2:
            clean_audiofile = torch.mean(clean_audiofile, dim=0).unsqueeze(0)

        # Make sure both files are same length, if not use one of the two and make the other one zeros to create a fully clean or fully noisy datapoint
        if noisy_audiofile.size()[-1] <= clean_audiofile.size()[-1]:
            clean_audiofile = clean_audiofile[:, :noisy_audiofile.size()[-1]]
            noisy_audiofile = noisy_audiofile[:, :clean_audiofile.size()[-1]]
            noisy_audiofile = WOP.add_noise(clean_audiofile, noisy_audiofile[:, :clean_audiofile.size()[-1]], torch.Tensor([snr_dbs]))
        else: # In exceptions add some instances where it's just noise or just clean voice
            noise_or_clean = random.randint(0, 1)
            if noise_or_clean == 0:
                clean_audiofile = torch.zeros(noisy_audiofile.size()).cuda()
            else:
                noisy_audiofile = clean_audiofile
                
        return clean_audiofile, noisy_audiofile
    
    def loadFiles(self, clean_path, noisy_path):
        clean_audio, _ = torchaudio.load(clean_path)
        noisy_audio, _ = torchaudio.load(noisy_path)

        return clean_audio, noisy_audio
    
    def getMFCC(self, audio):
        mfcc = self.mfcc_trans(audio).squeeze()
        mfcc_delta = compute_deltas(mfcc)
        mfcc_delta2 = compute_deltas(mfcc_delta)
        mfcc = torch.concatenate((mfcc, mfcc_delta, mfcc_delta2), dim=0)

        return mfcc

    def getMinMax(self, sp_min, sp_max, total_min, total_max):
        total_min = total_min
        total_max = total_max

        if sp_min < total_min:
            total_min = sp_min
        if sp_max > total_max:
            total_max = sp_max

        return total_min.cpu(), total_max.cpu()

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_audio = self.clean_files[idx]
        noisy_audio = self.noisy_files[idx]
        mfcc = self.mfccs[idx]

        noisy_audio = (noisy_audio - self.audiomean) / math.sqrt(self.audiovar)
        clean_audio = (clean_audio - self.audiomean) / math.sqrt(self.audiovar)
        
        mfcc = (mfcc - self.sp_min) / (self.sp_max - self.sp_min) # Normalise the spectrum to be between 0 and 1
        
        return noisy_audio, mfcc.squeeze(), clean_audio, noisy_audio.squeeze() - clean_audio.squeeze()#, self.pure_noise[idx]#, waveform_noisy_unquantized

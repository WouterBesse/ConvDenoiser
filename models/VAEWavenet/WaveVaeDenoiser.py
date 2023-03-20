import math
import torch
from torch import nn, from_numpy
from torch.utils.data import Dataset
from torchaudio.transforms import MuLawEncoding, MFCC, Resample
import torchaudio.functional as AF
import torchaudio
import models.VAEWavenet.WaveVaeOperations as WOP
import models.VAEWavenet.WaveVaeWavenet as WaveNet
from tqdm import tqdm
import librosa
import random
import numpy as np
import soundfile as sf
import os

class Decoder(nn.Module):
    """
    VAE Decoder
    """    
    def __init__(self, input_size, hidden_dim, dilation_rates, out_channels, upsamples, zsize = 128, num_cycles = 10, num_cycle_layers = 3, kernel_size = 3, pre_kernel_size = 32, use_jitter = True, jitter_probability = 0.12, use_kaiming_normal = False):
        super().__init__()

        # assert len(dilation_rates) == num_cycles * num_cycle_layers
        self.receptive_field = sum(dilation_rates) * (kernel_size - 1) + pre_kernel_size

        self.use_jitter = use_jitter
        if use_jitter:
            self.jitter = WOP.Jitter(jitter_probability)

        self.mulaw = MuLawEncoding(256)
        self.linear = nn.Linear(int(zsize), int(input_size[1] // 2 * 256))

        # self.conv_1 = nn.Conv1d(
        #     in_channels=64,
        #     out_channels=768,
        #     kernel_size=2,
        #     stride=1,
        #     padding=0
        # )

        if use_kaiming_normal:
            self.conv_1 = nn.utils.weight_norm(self.conv_1)
            nn.init.kaiming_normal_(self.conv_1.weight)

        self.wavenet = WaveNet.Wavenet(
            out_channels = 1,
            layers = 9,
            stacks = 3,
            res_channels = 256,
            skip_channels = 256,
            gate_channels = 256,
            condition_channels = 256,
            kernel_size = 3,
            upsample_conditional_features=True,
            upsample_scales = upsamples, # 768
            timesteps = 512
            #upsample_scales=[2, 2, 2, 2, 12]
        )


        features, timesteps = input_size

    def forward(self, x, cond, xsize, jitter):
        cond = self.linear(cond)
        condition = cond.view(xsize)
        if self.use_jitter and jitter:
            condition = self.jitter(condition)
        # print(x.size())

        # condition = self.conv_1(cond)

        # labels = self.mulaw(x)
        x = self.wavenet(x, condition)

        return x

class Encoder(nn.Module):
    """
    VAE Encoder
    """    
    def __init__(self, input_size, hidden_dim = 256, zsize = 128, resblocks = 2, relublocks = 4):
        super().__init__()

        features, timesteps = input_size
        self.prenet = nn.Conv1d(features, hidden_dim, kernel_size = 3, padding='same')
        self.preconv = nn.Conv1d(features, hidden_dim, kernel_size = 3, padding='same')
        self.ReL = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.ReLu = nn.Sequential(
            nn.ReLU(inplace = True),
            # nn.BatchNorm1d(hidden_dim)
        )
        self.zsize = zsize

        self.downsample = nn.Conv1d(hidden_dim, hidden_dim, kernel_size = 4, stride = 2, padding = 1)
        
        self.resblocks = nn.ModuleList()
        for _ in range(resblocks):
            self.resblocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size = 3, padding='same'))

        self.relublocks = nn.ModuleList()
        for _ in range(relublocks):
            self.relublocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size = 3, padding='same'))

        self.linear = nn.Linear(int(timesteps // 2 * hidden_dim), int(zsize * 2))
        # print(WOP.dimensionSize(timesteps, 2) * hidden_dim)
        self.flatton = WOP.Flatten()


    def forward(self, x):

        # Preprocessing conv with residual connections
        net = self.prenet(x)
        conv = self.preconv(x)
        x = self.ReL(net) + self.ReL(conv)
        # x = self.batchnorm(x)
        
        

        # Downsample
        x = self.ReLu(self.downsample(x))

        # Residual convs
        for resblock in self.resblocks:
            xres = self.ReLu(resblock(x))
            x = xres + x

        # Relu blocks
        for relblock in self.relublocks:
            xrelu = self.ReLu(relblock(x))
            x = xrelu + xrelu

        # Flatten into latent space
        # self.
        x_size = x.size()

        flatx = self.flatton(x)
        zcomb = self.linear(flatx)

        return zcomb[:, :self.zsize], zcomb[:, self.zsize:], x_size
    

class WaveNetVAE(nn.Module):

    def __init__(self, input_size, device, num_hiddens, dil_rates, zsize = 128, resblocks = 2, out_channels = 256):
        super(WaveNetVAE, self).__init__()

        self.encoder = Encoder(
            input_size = input_size,
            hidden_dim = 256,
            zsize = zsize,
            resblocks = resblocks,   
        )

        self.decoder = Decoder(
            input_size = input_size,
            hidden_dim = num_hiddens,
            dilation_rates = dil_rates,
            out_channels = out_channels,
            upsamples = dil_rates,
            zsize = zsize
        )
        
        self.mulaw = MuLawEncoding()
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        
        
    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        # eps = torch.randn(*mu.size()).to(mu.get_device())
        eps = self.N.sample(mu.shape).to(mu.get_device())
        z = mu + std * eps
        return z

    def forward(self, xau, xspec, jitter):
        mean, var, xsize = self.encoder(xspec)
        z = self.sample(mean, var)
        
        # xau = self.mulaw(xau)
        # xau = torch.nn.functional.one_hot(xau)
        # xau = xau.permute(0, 2, 1)

        x_hat = self.decoder(xau, z, xsize, jitter)
        
        return x_hat, mean, var

class WaveVaeDataset(Dataset):

    def __init__(self, clean_folder, noisy_folder, clip_length = 512, clips = 1, one_hot=True):
        super(WaveVaeDataset, self).__init__()
        
        self.clip_length = clip_length
        self.clean_folder = clean_folder
        self.mulaw = MuLawEncoding()

        # print("Loading clean files")
        self.clean_files = []
        self.noisy_files = []
        self.pure_noise = []
        self.mfccs = []
        self.samplerate = 0
        self.one_hot = one_hot
        
        
        clean_filepaths = os.listdir(clean_folder)
        clean_filepaths = clean_filepaths[:clips]
        
        self.noisy_filepaths = [os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if os.path.isfile(os.path.join(noisy_folder, f))]
        
        self.au_min = 99999999999
        self.au_max = -99999999999
        
        self.sp_min = 99999999999
        self.sp_max = -99999999999

        with tqdm(total=len(clean_filepaths), desc=f'Loading files to dataset. Len clean_files =  {len(self.clean_files)}') as pbar:
            # for noisy, clean in zip(nTrain_dirlist, cTrain_dirlist):
            for f in clean_filepaths:
                if os.path.isfile(os.path.join(clean_folder, f)):
                    audiopath = os.path.join(clean_folder, f)
                    audiofile, samplerate = torchaudio.load(audiopath)
                    # audiofile = self.remove_silent_frames(audiofile.squeeze().numpy())
                    # audiofile = torch.from_numpy(audiofile).unsqueeze(0)
                    
                    mfcc_trans = MFCC(samplerate, 40, log_mels = True, melkwargs={"hop_length": 33})
                    
                    
                    
                    # Load and create noisy file
                    noise_selector = random.randint(0, len(self.noisy_filepaths) - 1) # Choose a random noise file each time
                    snr_dbs = random.randint(2, 13) # Signal to noise ratio
                    # print("SNR DBS:", snr_dbs)
        
                    noise_path = self.noisy_filepaths[noise_selector]
                    # print(noise_path)
                    noise_waveform, noise_samplerate = torchaudio.load(noise_path)
                    resampler = Resample(noise_samplerate, samplerate)
                    noise_waveform = resampler(noise_waveform)
                    while noise_waveform.size()[-1] < 128000:
                        noise_selector = random.randint(0, len(self.noisy_filepaths) - 1) # Choose a random noise file each time

                        noise_path = self.noisy_filepaths[noise_selector]
                        # print(noise_path)
                        noise_waveform, noise_samplerate = torchaudio.load(noise_path)
                        resampler = Resample(noise_samplerate, samplerate)
                        noise_waveform = resampler(noise_waveform)
                        
                        
                    if noise_waveform.size()[0] == 2:
                        noise_waveform = torch.mean(noise_waveform, dim=0).unsqueeze(0)

                    audiofile = audiofile[:, :noise_waveform.size()[-1]]
                    noise_waveform = noise_waveform[:, :audiofile.size()[-1]]
                    
                    noise_waveform = WOP.add_noise(audiofile, noise_waveform[:, :audiofile.size()[-1]], torch.Tensor([snr_dbs]))

                    i = 256
                    # Make sure there are plenty of zero lists to symbolise data where you havent generated anything yet
                    while i < audiofile.size()[-1] - 256:
                    # while i < 66000:
#                         if i < clip_length:
#                             clean_audio = audiofile[:, 0:i + clip_length + 1]
#                             noisy_audio = WOP.add_noise(clean_audio, noise_waveform[:, : clean_audio.size()[-1]], torch.Tensor([snr_dbs]))
                            
#                             # MFCC generations
#                             # mfcc = mfcc_trans(clean_audio).squeeze()
                            
#                             # Create Zeros
#                             audio_zeros = torch.zeros(1, clip_length - i)
#                             # mfcc_zeros = torch.zeros(64, clip_length // 32 - mfcc.size()[-1])
                            
#                             # Add Zeros                    
#                             clean_audio = torch.cat((audio_zeros, clean_audio), 1)
#                             noisy_audio = torch.cat((audio_zeros, noisy_audio), 1)
#                             mfcc = mfcc_trans(clean_audio).squeeze() # Noisy audio spectrum
#                             # mfcc = torch.cat((mfcc_zeros, mfcc), 1) 
                            
#                             if torch.isnan(torch.min(mfcc)) or torch.isnan(torch.max(mfcc)):
#                                 print("NAN!", i)
                            
#                             self.clean_files.append(clean_audio)
#                             self.noisy_files.append(noisy_audio)
#                             self.mfccs.append(mfcc)
                                                        
#                             au_min = min(torch.min(noisy_audio), torch.min(clean_audio))
#                             au_max = max(torch.max(noisy_audio), torch.max(clean_audio))
#                             au_max_abs = max(au_max, abs(au_min))
#                             self.au_min, self.au_max = self.getMinMax(au_min, au_max_abs, self.au_min, self.au_max)
                            
#                             sp_min = torch.min(mfcc)
#                             sp_max = torch.max(mfcc)
#                             self.sp_min, self.sp_max = self.getMinMax(sp_min, sp_max, self.sp_min, self.sp_max)
                            
#                             i += 16
#                             # print(i)
#                         else:
                        clean_audio = audiofile[:, i - clip_length:i + clip_length]
                        # self.pure_noise.append(noise_waveform[:, : clean_audio.size()[-1]])
                        noisy_audio = noise_waveform[:, i - clip_length:i + clip_length]
                        mfcc = mfcc_trans(clean_audio).squeeze()

                        self.clean_files.append(clean_audio)
                        self.noisy_files.append(noisy_audio)
                    
                        self.mfccs.append(mfcc)

                        au_min = min(torch.min(noisy_audio), torch.min(clean_audio))
                        au_max = max(torch.max(noisy_audio), torch.max(clean_audio))
                        au_max_abs = max(au_max, abs(au_min))
                        self.au_min, self.au_max = self.getMinMax(au_min, au_max, self.au_min, self.au_max)

                        sp_min = torch.min(mfcc)
                        sp_max = torch.max(mfcc)
                        self.sp_min, self.sp_max = self.getMinMax(sp_min, sp_max, self.sp_min, self.sp_max)

                        i += 400

                    self.samplerate = samplerate
                    
                    
                    pbar.set_description(f'Loading files to dataset. Len clean_files =  {len(self.clean_files)}. ')
                    pbar.update(1)
                    
        self.spmean = torch.stack(self.mfccs).mean()
        self.spvar = torch.stack(self.mfccs).std()
        
        self.audiomean = torch.cat((torch.stack(self.noisy_files), torch.stack(self.clean_files)), 0).mean()
        self.audiovar = torch.cat((torch.stack(self.noisy_files), torch.stack(self.clean_files)), 0).var()
        
                    
        print(self.samplerate)
        print("Sp mean:", self.spmean, "Sp var:", self.spvar)
        print("Au mean:", self.audiomean, "Au var:", self.audiovar)
        
        # print("Loaded ", len(self.clean_files), " clean files <3")
        
    def remove_silent_frames(self, audio, overlap = 256):
        
        if len(audio.shape) > 1:
            audio = np.ascontiguousarray((audio[:, 0]+audio[:, 1])/2)
        
        trimmed_audio = []
        indices = librosa.effects.split(audio, hop_length=overlap, top_db=30)

        for index in indices:
            trimmed_audio.extend(audio[index[0]: index[1]])
            
        return np.array(trimmed_audio)
    
    def getMinMax(self, sp_min, sp_max, total_min, total_max):
        total_min = total_min
        total_max = total_max

        if sp_min < total_min:
            total_min = sp_min
        if sp_max > total_max:
            total_max = sp_max

        return total_min, total_max

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_audio = self.clean_files[idx]
        noisy_audio = self.noisy_files[idx]
        mfcc = self.mfccs[idx]
        
#         if clean_audio.size()[-1] != self.clip_length + 1:
#             print('clean audio', clean_audio.size())
            
#         if noisy_audio.size()[-1] != self.clip_length + 1:
#             print('clean audio', noisy_audio.size())
            
        # if torch.min(mfcc) == nan
        # noisy_audio = noisy_audio / self.au_max # Normalise the audio to be between -1 and 1
        # clean_audio = clean_audio / self.au_max # Normalise the audio to be between -1 and 1
        
        # noisy_audio = (noisy_audio - self.audiomean) / math.sqrt(self.audiovar)
        # noisy_audio_for_noise = (noisy_audio - self.cleanmean) / self.cleanvar
        # clean_audio = (clean_audio - self.audiomean) / math.sqrt(self.audiovar)
        noisy_audio = (noisy_audio - self.au_min) / (self.au_max - self.au_min)
        clean_audio = (clean_audio - self.au_min) / (self.au_max - self.au_min)
        
        mfcc = (mfcc - self.sp_min) / (self.sp_max - self.sp_min) # Normalise the spectrum to be between 0 and 1
        
        # Create clean target
#         waveform_norm_mult_c = clean_audio
#         waveform_target = waveform_norm_mult_c
        # waveform_quantized_target = self.mulaw(clean_audio)
        
#         # Create clean input
#         waveform_norm_mult_n = noisy_audio
#         waveform_noisy = waveform_norm_mult_n
#         waveform_noisy_unquantized = waveform_noisy

        # waveform_quantized_noisy = self.mulaw(noisy_audio)

        # if torch.min(waveform_quantized_noisy) < 0:
        #     print(torch.min(waveform_noisy_unquantized), torch.max(waveform_noisy_unquantized))
#         if self.one_hot:
#             waveform_noisy = torch.nn.functional.one_hot(waveform_quantized_noisy, num_classes = 256).squeeze().float()
#             waveform_quantized_noisy = waveform_noisy.permute(1, 0)
        
        return noisy_audio, mfcc.squeeze(), clean_audio, noisy_audio.squeeze() - clean_audio.squeeze()#, self.pure_noise[idx]#, waveform_noisy_unquantized

# Ways of denoising Mel Cepstrums using convolutional networks.

In this repo, I'm trying to denoise noisy Mel Cepstrums for resynthesis with a vocoder. For this I have three legacy experiments, and one main architecture using a **WaveNet VAE** setup.

Each method has its' own notebook, ~~they all share the same preprocessing functions.~~
The data needs to be provided as separate clean speech files and ambience noise files, which it stitches together to make new noisy speech files.

**Audio results are available in the results folder.**

## WaveNet VAE Denoiser
This denoiser is an architecture inspired by three research papers:
1. [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810)
2. [A Wavenet for Speech Denoising](https://arxiv.org/abs/1706.07162)
3. [Variational Autoencoder for Speech Enhancement with a Noise-Aware Encoder](https://arxiv.org/abs/2102.08706)

<img src="https://github.com/WouterBesse/ConvDenoiser/raw/master/results/WavenetVAE_Diagram.png" height="600" align="left"/>

### Shortly summarized 
This model works by extracting a latent representation of speech from an mfcc of a noisy audio recording, then it uses this representation to guide a WaveNet model as a condition in denoising the original audio.

For this, I implement a WaveNet variational autoencoder model from the first paper. In this model, the WaveNet decoder is modified with some changes from paper 2. Important modifications include removing the causal padding from WaveNet, rendering direct samples and not mu-law quantised samples, and using a padding size of 3 in the WaveNet.

Then for the training method I first train it on clean cepstra with a target of clean audio, then lock the decoder and train it on noisy cepstra with a target of clean audio, like the third paper. This is to ensure it tries to compress only the clean speech data from the cepstrum. This last training part may need some closer inspection to better match the paper though.

### Results
I'm pretty with the results so far, they are way better than all my other experiments.

It is important to note, however, that I am still in the testing phase of this model. This means that I still have to fully validate it's working as intended, and if that's the case that there are likely improvements to be made.


![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/WaveNetVAE_Results.png "WaveNetVAE Results")
Just as a little legend I forgot to add: the left MFCC is the noisy audio, the centre MFCC is the denoised result, and the right MFCC is the original clean MFCC.

## Legacy Experiments

### Standard Convolutional Network with encoder decoder archetecture

This is a PyTorch implementation of a denoising architecture mimicking the one described in [Thalles Santhos Silva's blog](https://sthalles.github.io/practical-deep-learning-audio-denoising/)
It generates new Cepstrum frames based on the previous 7 frames + the current frame.

Unfortunately, this doesn't sound that good.


#### Results

Noisy MFCC             |  CNN Denoised MFCC
:-------------------------:|:-------------------------:
![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/NoisySP.png "Noisy MFCC") |  ![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/DenoisedSP_CNN.png "CNN Denoised MFCC")

---
### 2D WaveNet archetecture
This is a PyTorch implementation of a WaveNet architecture like the one described in [the paper from Merlijn Blaauw](https://arxiv.org/abs/1704.03809) and [the implementation from Seanie Zhao](https://github.com/seaniezhao/torch_npss).
It generates a mixed Gaussian distribution for every frame and stitches these together.
It uses the noisy Mel Cepstrum as condition that is fed into the system at every step.

This also doesn't sound too good.

#### Results
Noisy MFCC             |  WaveNet Denoised MFCC
:-------------------------:|:-------------------------:
![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/NoisySP.png "Noisy MFCC") |   ![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/DenoisedSP_WaveNet.png "WaveNet Denoised MFCC")

---
### AutoEncoder
This is a simple variational autoencoder model.

#### Results
Noisy MFCC             |  AE Denoised MFCC
:-------------------------:|:-------------------------:
![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/NoisySP.png "Noisy MFCC") |   ![alt text](https://github.com/WouterBesse/ConvDenoiser/raw/master/results/DenoisedSP_AE.png "AE Denoised MFCC")

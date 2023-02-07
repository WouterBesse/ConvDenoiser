# ConvDenoiser
Ways of denoising Mel Cepstrums using convolutional networks.
We are trying to denoise noisy Mel Cepstrums for resynthesis with a vocoder. We are currently working with three methods.
Each method has its' own notebook, they all share the same preprocessing functions. 
The data needs to be provided as seperate clean speach files and ambience noise files, which it stitches together to make new noisy speech files.

## Standard Convolutional Network with encoder decoder archetecture
This is a pytorch implementation of a denoising archetecture mimicking the one described in [Thalles Santhos Silva's blog](https://sthalles.github.io/practical-deep-learning-audio-denoising/)
It generates new Cepstrum frames based on the previous 7 frames + the current frame.

## 2D WaveNet archetecture
This is a pytorch implementation of a WaveNet archetecture like the one described in [the paper from Merlijn Blaauw](https://arxiv.org/abs/1704.03809) and [the implementation from Seanie Zhao](https://github.com/seaniezhao/torch_npss).
It generates a mixed gaussian distribution for every frame and stitches these together.
It uses the noisy Mel Cepstrum as condition that is feeded into the system at every step.

## AutoEncoder
This one is still in construction :(

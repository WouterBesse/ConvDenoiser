import os

verbose = {
    "preprocess": True
}

hop = 160  # sample_rate * 0.005 in which world default_frame_period is 5 millisecond == 0.005 second
sample_rate = 32000
# data_sample_rate = 44100

fft_size = 2048
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0

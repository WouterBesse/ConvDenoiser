import librosa
import pyworld as pw
import numpy as np
import soundfile as sf
import pysptk
import copy
import math
from os import listdir, walk, makedirs
from os.path import isfile, join, basename, isdir
import argparse
import pathlib
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import tqdm
from config import *
import random

import warnings
warnings.filterwarnings('ignore')

# Code and decode parameters
gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)

def checkDir(path):
    if not isdir(path):
        makedirs(path)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def code_harmonic(sp, order):

    #get mcep
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    #do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc

def decode_harmonic(mfsc, fftlen):
    # get mcep back
    mceps_mirror = np.fft.irfft(mfsc)
    mceps_back = mceps_mirror[:, :60]
    mceps_back[:, 0] /= 2
    mceps_back[:, -1] /= 2

    #get sp
    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, mceps_back, alpha, gamma, fftlen=fftlen).real)

    return spSm

def import_wav(wav_path):

    audioSamples, originalSampleRate = sf.read(wav_path)

    if len(audioSamples.shape) > 1:
        audioSamples = np.ascontiguousarray((audioSamples[:, 0]+audioSamples[:, 1])/2)

    sampleRate = sample_rate
    # print("Samplerate: ", sampleRate, " | OGSamplerate: ", originalSampleRate)
    if originalSampleRate != sampleRate:
        # print('resampling')
        audioSamples = librosa.resample(audioSamples, orig_sr=originalSampleRate, target_sr=sampleRate)

    return audioSamples, sampleRate

def extract_f0(audioSamples, sampleRate):

    # _f0, t = pw.dio(audioSamples, sampleRate, f0_floor=f0_min, f0_ceil=f0_max,
    #                     frame_period=pw.default_frame_period)
    _f0, t = pw.dio(audioSamples, sampleRate)
    _f0 = pw.stonemask(audioSamples, _f0, t, sampleRate)
    # _f0[_f0 > f0_max] = f0_max

    return _f0, t

def extract_sp(audioSamples, sampleRate, f0, t):
    sp = pw.cheaptrick(audioSamples, f0, t, sampleRate)

    code_sp = code_harmonic(sp, 60)

    return sp, code_sp

def extract_ap(audioSamples, sampleRate, f0, t):
    ap = pw.d4c(audioSamples, f0, t, sampleRate)

    code_ap = pw.code_aperiodicity(ap, sampleRate)

    return ap, code_ap

def remove_silent_frames(audio, overlap):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

def audio_random_crop(audio, duration):
        audio_duration_secs = librosa.core.get_duration(y = audio, sr = sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            # print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return audio

        audio_duration_ms = math.floor(audio_duration_secs * sample_rate)
        duration_ms = math.floor(duration * sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx: idx + duration_ms]

def add_noise_to_clean_audio(clean_audio, noise_signal):
        if len(clean_audio) >= len(noise_signal):
            # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
            while len(clean_audio) >= len(noise_signal):
                noise_signal = np.append(noise_signal, noise_signal)

        ## Extract a noise segment from a random location in the noise file
        ind = np.random.randint(0, noise_signal.size - clean_audio.size)

        noiseSegment = noise_signal[ind: ind + clean_audio.size]

        speech_power = np.sum(clean_audio ** 2)
        noise_power = np.sum(noiseSegment ** 2)
        noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
        return noisyAudio

def cut_combine_audio(clean_audio, noise_audio):

    windowLength = 256
    overlap = round(0.25 * windowLength)
    fs = 16000
    audio_max_duration = 2

    # remove silent frame from clean audio
    # clean_audio = remove_silent_frames(clean_audio, overlap)

    # remove silent frame from noise audio
    # noise_audio = remove_silent_frames(noise_audio, overlap)
    # print(len(noise_audio))
    # print(noise_audio.shape)

    # sample random fixed-sized snippets of audio
    # clean_audio = audio_random_crop(clean_audio, duration=2)

    # Add noise to clean audio
    combined_audio = add_noise_to_clean_audio(clean_audio, noise_audio)

    return combined_audio, clean_audio

def printInfo(f0, sp, code_sp, ap, code_ap):
    
    print("######## PREPROCESS INFORMATION ########")
    print("\n")
    print("F0 shape: ", f0.shape)
    print("\n")
    print("Spectrum shape: ", sp.shape)
    print("Coded Spectrum shape: ", code_sp.shape)
    print("\n")
    print("Aperiodicty shape: ", ap.shape)
    print("Coded aperiodicity shape: ", code_ap.shape)
    print("\n")

def process_wav(input_data, preMerged = False):

    # print('swag')
    
    clean_audio_path = input_data[0]
    noise_audio_path = input_data[1]

    cAudioSamples, cSampleRate = import_wav(clean_audio_path)
    nAudioSamples, nSampleRate = import_wav(noise_audio_path)

    
    nAudioSamples, cAudioSamples = cut_combine_audio(cAudioSamples, nAudioSamples)


    sampleRate = sample_rate

    # if len(save_path) > 1:
    #     sf.write(save_path, audioSamples, sampleRate)

    f0, sp, ap = pw.wav2world(nAudioSamples, sampleRate)
    code_sp = code_harmonic(sp, 60)
    code_ap = pw.code_aperiodicity(ap, sampleRate)

    f0c, spc, apc = pw.wav2world(cAudioSamples, sampleRate)
    code_spc = code_harmonic(spc, 60)
    code_apc = pw.code_aperiodicity(apc, sampleRate)

    return [f0, sp, code_sp, ap, code_ap, f0c, spc, code_spc, apc, code_apc, cAudioSamples, sampleRate, basename(clean_audio_path), nAudioSamples]

def startProcessing(save_path, cleanfolder, noisefolder, outfolder, exporttype):

    cleanfiles = [join(cleanfolder, f) for f in listdir(cleanfolder) if isfile(join(cleanfolder, f))]

    # noisefiles = [val for sublist in [[join(i[0], j) for j in i[2] if j.endswith('.wav')] for i in walk(noisefolder)] for val in sublist]
    noisefiles = [join(noisefolder, f) for f in listdir(noisefolder) if isfile(join(noisefolder, f))]

    # Get the minimum of files to get
    fileamount = min(len(cleanfiles), len(noisefiles))
    # fileamount = 10

    combinedfiles = zip(cleanfiles[:fileamount], noisefiles[:fileamount])
    combinedfiles = list(combinedfiles)

    combinedfiles = [list(ele) for ele in combinedfiles]

    checkDir(outfolder + "csp/")
    checkDir(outfolder + "cap/")
    checkDir(outfolder + "sp2/")
    checkDir(outfolder + "ap2/")
    checkDir(outfolder + "csp2/")
    checkDir(outfolder + "cap2/")
    checkDir(outfolder + "sp/")
    checkDir(outfolder + "ap/")
    checkDir(outfolder + "f0c/")
    checkDir(outfolder + "f0/")

    random.shuffle(combinedfiles)

    print(combinedfiles[0:20])

    # print("### Small file sample ###")
    # print(combinedfiles)
    # for i in range(10):
    #     print("Sample ", i, ": ", combinedfiles[i], "\n")
    # print(cleanfiles[:50])
    # print(noisefiles[:50])

    # exit()

    print("Number of cpu's : ", cpu_count())

    with Pool(cpu_count()) as p:
        for output in tqdm.tqdm(p.imap(process_wav, combinedfiles[0:2000]), total=2000):
            f0 = output[0].astype(np.double)
            sp = output[1].astype(np.double)
            code_sp = output[2].astype(np.double)
            ap = output[3].astype(np.double)
            code_ap = output[4].astype(np.double)

            f0c = output[5].astype(np.double)
            spc = output[6].astype(np.double)
            code_spc = output[7].astype(np.double)
            apc = output[8].astype(np.double)
            code_apc = output[9].astype(np.double)

            audiosamples = output[10]
            samplerate = output[11]
            basername = output[12]

            # print('Did one!')

            # if len(save_path) > 1:
            #     sf.write(join(save_path, basername), audiosamples, samplerate)

            # np.save(join(args.Output + "ccondi/", basername) + '_condi.npy', spc)
            np.save(join(outfolder + "csp/", basername) + '_sp.npy', code_spc)
            np.save(join(outfolder + "cap/", basername) + '_ap.npy', code_apc)

            np.save(join(outfolder + "csp2/", basername) + '_sp.npy', spc)
            np.save(join(outfolder + "cap2/", basername) + '_ap.npy', apc)

            np.save(join(outfolder + "sp2/", basername) + '_sp.npy', sp)
            np.save(join(outfolder + "ap2/", basername) + '_ap.npy', ap)

            np.save(join(outfolder + "sp/", basername) + '_sp.npy', code_sp)
            np.save(join(outfolder + "ap/", basername) + '_ap.npy', code_ap)

            np.save(join(outfolder + "f0c/", basername) + '_f0.npy', f0c)
            np.save(join(outfolder + "f0/", basername) + '_f0.npy', f0)

    
    p.close()
    p.join()

    print("\n")
    print("Enjoy your freshly processed files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'FAD-GAN preprocesser',
        description="Cleans, cuts, combines and processes clean and noise audio.")

    parser.add_argument('--AudioSavePath', type=str, 
                        help='Where processed audio files can be saved. If left empty no actual audiofiles will be saved.')

    parser.add_argument('--CleanFolder', type=str,
                        help="Folder where all your clean voice files are stored.")

    parser.add_argument('--NoiseFolder', type=str,
                        help="Folder where all your noise files are stored.")

    parser.add_argument('--Output', type=str,
                        help="Output folder")

    parser.add_argument('--Type', type=str,
                        help="All, F0, SP or AP")

    args = parser.parse_args()

    startProcessing(args.AudioSavePath, args.CleanFolder, args.NoiseFolder, args.Output, args.Type)

    
    

    # printProgressBar(0, fileamount, prefix = 'Progress', suffix = 'Complete', length = 50)

    # for i in range(fileamount):

    #     # f0, sp, code_sp, ap, code_ap = process_wav(join(cleanfolder + cleanfiles[i]), noisefiles[i], join(save_path + cleanfiles[i]))

    #     f0 = f0.astype(np.double)
    #     sp = sp.astype(np.double)
    #     code_sp = code_sp.astype(np.double)
    #     ap = ap.astype(np.double)
    #     code_ap = code_ap.astype(np.double)

    #     np.save("G:/Projects/2022-2023/cool snapshots/data/condition" + '/' + cleanfiles[i] + '_condi.npy', sp)
    #     np.save("G:/Projects/2022-2023/cool snapshots/data/sp" + '/' + cleanfiles[i] + '_sp.npy', code_sp)
    #     np.save("G:/Projects/2022-2023/cool snapshots/data/ap" + '/' + cleanfiles[i] + '_ap.npy', code_ap)
    #     printProgressBar(i + 1, fileamount, prefix = 'Progress', suffix = 'Complete', length = 50)


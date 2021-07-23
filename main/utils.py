import librosa as lib
import torch
import torch.nn.functional as F
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train import Separation
import torchaudio
import os
import numpy as np
from speechbrain.dataio.dataio import read_audio
from scipy import signal
from make_log import signaltonoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

def padding(wav, max_len):
    res = max_len - wav.shape[0]
    wav_padding = F.pad(input=wav, pad=(0,res), mode='constant', value=0)
    return wav_padding

def _process(file_wav, model):
    wav = load_wav(file_wav)
    wav = torch.FloatTensor(wav)
    wav = wav.unsqueeze(0)
    num_wav = torch.Tensor((2))
    wav = (wav, num_wav)
    
    s1 = torch.Tensor((0,0))
    s2 = s1
    zero_targer = [s1,s2]
    model.modules.eval()
    with torch.no_grad():
        separated, zero_targer = model.compute_forward(wav, zero_targer, stage = sb.Stage.TEST)
    return separated

def _load_model():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    checkpoint = hparams["checkpointer"]
    checkpoint.recover_if_possible()
    model = Separation(modules=hparams["modules"],
                        hparams=hparams,
                        run_opts=run_opts,
                        checkpointer=checkpoint,)

    return model

def prepare_mixed(s1_path, s2_path):
    from oct2py import octave

    filedir = os.getcwd()
    filedir = filedir.split("/")
    filedir = "/".join(filedir[:-2])

    octave.addpath(
        filedir +"/SpeechSeparation/main/meta"
    )  # add the matlab functions to octave dir here

    fs_read = 16000
    _, fs1 = torchaudio.load(s1_path)
    _, fs2 = torchaudio.load(s2_path)
    s1 = read_audio(s1_path)
    s2 = read_audio(s2_path)

    inwav1_snr = signaltonoise(s1)
    inwav2_snr = signaltonoise(s2)

    # resample, determine levels for source 1
    s1_16k = signal.resample(s1, int((fs_read / fs1) * len(s1)))
    out = octave.activlev(s1_16k, fs_read, "n")
    s1_16k, lev1 = out[:-1].squeeze(), out[-1]
    # print('lev1 {}'.format(lev1))

    # resample, determine levels for source 2
    s2_16k = signal.resample(s2, int((fs_read / fs2) * len(s2)))
    out = octave.activlev(s2_16k, fs_read, "n")
    s2_16k, lev2 = out[:-1].squeeze(), out[-1]

    weight_1 = 10 ** (float(inwav1_snr) / 20)
    weight_2 = 10 ** (float(inwav2_snr) / 20)
    s1 = weight_1 * s1_16k / np.sqrt(lev1)
    s2 = weight_2 * s2_16k / np.sqrt(lev2)

    mix_len = max(s1.shape[0], s2.shape[0])

    s1 = np.pad(
        s1, (0, mix_len - s1.shape[0]), "constant", constant_values=(0, 0),
    )
    s2 = np.pad(
        s2, (0, mix_len - s2.shape[0]), "constant", constant_values=(0, 0),
    )

    mix = s1 + s2

    max_amp = max(np.abs(mix).max(), np.abs(s1).max(), np.abs(s2).max(),)
    mix_scaling = 1 / max_amp * 0.9
    s1 = mix_scaling * s1
    s2 = mix_scaling * s2
    mix = mix_scaling * mix

    return mix

if __name__ == "__main__":
    # s3 = load_wav("/home/SpeechSeparation/test_wav/noise.wav")
    # s1_2 = load_wav("/home/SpeechSeparation/test_wav/mix.wav")
    # print(s3.shape)
    # print(s1_2.shape)
    
    # s3 = s3[300:300 + s1_2.shape[0]]
    # s3 = s3
    # wav_file = s3+s1_2
    model = _load_model()
    # torchaudio.save("/home/SpeechSeparation/test_wav/mix_3_speakers.wav" , wav_file.unsqueeze(0), 16000)
    wav_file = "/home/SpeechSeparation/test_wav/text.wav"
    separated = _process(wav_file, model)
    signal = separated[0, :, 0]
    signal = signal / signal.abs().max()
    torchaudio.save("/home/SpeechSeparation/test_wav/mix_spk1.wav" , signal.unsqueeze(0).cpu(), 16000)
    signal = separated[0, :, 1]
    signal = signal / signal.abs().max()
    torchaudio.save("/home/SpeechSeparation/test_wav/mix_spk2.wav" , signal.unsqueeze(0).cpu(), 16000)


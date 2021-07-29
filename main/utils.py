import librosa as lib
import torch
import torch.nn.functional as F
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train import Separation
import torchaudio
torchaudio.set_audio_backend("sox_io")
import os
import numpy as np
from speechbrain.dataio.dataio import read_audio
from scipy import signal
from make_log import signaltonoise
import auditok

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

def padding(wav, max_len):
    res = max_len - wav.shape[0]
    wav_padding = F.pad(input=wav, pad=(0,res), mode='constant', value=0)
    return wav_padding

def _process(file_dir, model):
    wav = load_wav(file_dir)
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
    separated[:, :, 0] = separated[:, :, 0] / separated[:, :, 0].abs().max()
    separated[:, :, 1] = separated[:, :, 1] / separated[:, :, 1].abs().max()

    separated = separated.detach().cpu()

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
    s1, fs1 = torchaudio.load(s1_path)
    s2, fs2 = torchaudio.load(s2_path)

    if s1.shape[0] > 1:
        s1 = s1[0,:].unsqueeze(0)
    if s2.shape[0] > 1:
        s2 = s2[0,:].unsqueeze(0)
    s1 = s1.squeeze(0)
    s2 = s2.squeeze(0)
    print(s1.shape)
    print(s2.shape)
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

def VAD(file_path: str, model):

        audio, sr = lib.load(file_path, sr=16000, mono=True)
        duration = len(audio)/sr
        print("load done, duration=",duration)

        separated = []
        if duration <= 30:

            separated.append(_process(file_dir= file_path, model= model))

        else:
            audio_regions = auditok.split(
                    file_path,
                    min_dur=0.5,     # minimum duration of a valid audio event in seconds
                    max_dur=10,      # maximum duration of an event
                    max_silence=0.5, # maximum duration of tolerated continuous silence within an event
                    energy_threshold=55 # threshold of detection
                )

            for i, r in enumerate(audio_regions):
                tmp_audio = audio[int(r.meta.start*sr):int(r.meta.end*sr)].astype('float32') 

                tmp_audio = torch.Tensor(tmp_audio)
                tmp_path = file_path.split('/')
                tmp_path = '/'.join(tmp_path[:-1])
                tmp_path = os.path.join(tmp_path, 'tmp.wav')

                torchaudio.save(tmp_path, tmp_audio.unsqueeze(0), sample_rate= 16000)
                tmp_audio = _process(file_dir= tmp_path, model= model)
                separated.append(tmp_audio)

        return separated

def add_noise(data):
    wn = np.random.normal(0,1,len(data))
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.02 * wn, 0.0).astype(np.float32)
    return data_noise

def add_rebv(data):
    pass

from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .phaser()
    .delay()
    .lowshelf()
)


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
    s1_path = "/home/SpeechSeparation/test_wav/spk1.wav"

    s2_path = "/home/SpeechSeparation/test_wav/spk2.wav"
    mix = prepare_mixed(s1_path, s2_path)
    mix = mix[:-16000*5]
    mixed_filepath = "/home/SpeechSeparation/test_wav/mixed.wav"

    mix = torch.tensor(mix)
    mix = mix.to('cpu').numpy()

    # mix = add_noise(mix)

    # mix = fx(mix)

    # print('*****************************************')
    mix = torch.Tensor(mix)

    torchaudio.save(mixed_filepath, mix.unsqueeze(0), sample_rate= 16000, )

    separated = VAD(mixed_filepath, model)

    for i in range(len(separated)):
        signal = separated[i][0, :, 0]
        torchaudio.save("/home/SpeechSeparation/test_wav/"+ str(i) +"_mix_spk1.wav" , signal.unsqueeze(0).cpu(), 16000)
        signal = separated[i][0, :, 1]
        torchaudio.save("/home/SpeechSeparation/test_wav/"+ str(i) +"_mix_spk2.wav" , signal.unsqueeze(0).cpu(), 16000)


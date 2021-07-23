import librosa as lib
import torch
import torch.nn.functional as F
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train import Separation
import torchaudio
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


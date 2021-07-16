import librosa as lib
import torch
import torch.nn.functional as F
import sys
import speechbrain as sb
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
from train import Separation
# class Separation(sb.Brain):
#     def compute_forward(self, mix, targets, stage, noise=None):
#         """Forward computations from the mixture to the separated signals."""
#         # Unpack lists and put tensors in the right device
#         mix, mix_lens = mix
#         mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

#         # Convert targets to tensor
#         targets = torch.cat(
#             [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
#             dim=-1,
#         ).to(self.device)
                
#         # Separation
        
#         mix_w = self.hparams.Encoder(mix)

#         est_mask = self.hparams.MaskNet(mix_w)
       
#         # LARGE PARAMS HERE

#         mix_w = torch.stack([mix_w] * self.hparams.num_spks)
#         sep_h = mix_w * est_mask
        
#         # Decoding
#         est_source = torch.cat(
#             [
#                 self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
#                 for i in range(self.hparams.num_spks)
#             ],
#             dim=-1,
#         )

#         # T changed after conv1d in encoder, fix it here
#         T_origin = mix.size(1)
#         T_est = est_source.size(1)
#         if T_origin > T_est:
#             est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
#         else:
#             est_source = est_source[:, :T_origin, :]

#         return est_source, targets

#     def compute_objectives(self, predictions, targets):
#         """Computes the sinr loss"""
#         return self.hparams.loss(targets, predictions)

#     def cut_signals(self, mixture, targets):
#         """This function selects a random segment of a given length within the mixture.
#         The corresponding targets are selected accordingly"""
#         randstart = torch.randint(
#             0,
#             1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
#             (1,),
#         ).item()
#         targets = targets[
#             :, randstart : randstart + self.hparams.training_signal_len, :
#         ]
#         mixture = mixture[
#             :, randstart : randstart + self.hparams.training_signal_len
#         ]
#         return mixture, targets

#     def save_audio(self, snt_id, mixture, targets, predictions):
#         "saves the test audio (mixture, targets, and estimated sources) on disk"

#         # Create outout folder
#         save_path = os.path.join(self.hparams.save_folder, "audio_results")
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)

#         for ns in range(self.hparams.num_spks):

#             # Estimated source
#             signal = predictions[0, :, ns]
#             signal = signal / signal.abs().max()
#             save_file = os.path.join(
#                 save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
#             )
#             torchaudio.save(
#                 save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
#             )

#             # Original source
#             signal = targets[0, :, ns]
#             signal = signal / signal.abs().max()
#             save_file = os.path.join(
#                 save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
#             )
#             torchaudio.save(
#                 save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
#             )

#         # Mixture
#         signal = mixture[0][0, :]
#         signal = signal / signal.abs().max()
#         save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
#         torchaudio.save(
#             save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
#         )

# Pre-process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

max_len = 3*16000

def padding(noisy_wav):
    res = max_len - noisy_wav.shape[0]
    noisy_wav_padding = F.pad(input=noisy_wav, pad=(0,res), mode='constant', value=0)
    return noisy_wav_padding

def _process(file_wav, model):
    wav = load_wav(file_wav)
    len_input = wav.shape[0]
    if len_input > max_len :
        wav = wav[: max_len]
    elif len_input < max_len :
        wav = padding(wav)
    wav = (wav, 2)

    zero_targer = [[0],[0]]
    with torch.no_grad():
        separated, zero_targer = model.compute_forward(wav, zero_targer)
    return separated

def _load_model():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    model = Separation(modules=hparams["modules"],
                        hparams=hparams,
                        run_opts=run_opts,
                        checkpointer=hparams["checkpointer"],)

    model.modules.eval()

    return model

if __name__ == "__main__":
    model = _load_model()
    wav_file = ""
    print(_process(wav_file, model).shape)
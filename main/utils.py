import librosa as lib
import torch
import os
import torch.nn.functional as F
import math
import torchaudio
import sys
import speechbrain as sb
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
import matplotlib.pyplot as plt
import time
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""
        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)
                
        # Separation
        
        mix_w = self.hparams.Encoder(mix)

        est_mask = self.hparams.MaskNet(mix_w)
       
        # LARGE PARAMS HERE

        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask
        
        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list

        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if "wham" in self.hparams.data_folder:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)
            
        predictions, targets = self.compute_forward(
            mixture, targets, sb.Stage.TRAIN, noise
        )

        loss = self.compute_objectives(predictions, targets)
        
        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            loss_to_keep = loss[loss > th]
            if loss_to_keep.nelement() > 0:
                loss = loss_to_keep.mean()
        else:
            loss = loss.mean()

        if (
            loss < self.hparams.loss_upper_lim and loss.nelement() > 0
        ):  # the fix for computational problems
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

# Pre-process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

max_len = 5*16000

def padding(noisy_wav):
    res = max_len - noisy_wav.shape[0]
    noisy_wav_padding = F.pad(input=noisy_wav, pad=(0,res), mode='constant', value=0)
    return noisy_wav_padding

def _process(file_wav, model):
    wav = load_wav(file_wav)
    batch = []
    miniWav = []
    len_input = wav.shape[0]
    # chia do dai cho 5*16000, duoc so luong cac doan nho do dai bang 5s
    num_miniWav = math.ceil(wav.shape[0] / max_len) 
    # gop 8 cai 1 de dua vao mang
    num_batch = math.ceil(num_miniWav / 8)
    # padding 0 vec to fill up batch
    res = 8 - (num_miniWav % 8) 
    padding_batch = torch.zeros(max_len)

    if num_miniWav > 1 :
        for j in range(num_miniWav-1):
            miniWav.append(wav[j*max_len : (j+1)*max_len].unsqueeze(0)) 
    need_add = wav[(num_miniWav-1)*max_len:]
    miniWav.append(padding(need_add).unsqueeze(0))

    for i in range(res):
        miniWav.append(padding_batch.unsqueeze(0))

    tmp_1 = torch.cat((miniWav[0],miniWav[1]))
    tmp_2 = torch.cat((miniWav[2],miniWav[3]))
    tmp_3 = torch.cat((miniWav[4],miniWav[5]))
    tmp_4 = torch.cat((miniWav[6],miniWav[7]))
    tmp12 = torch.cat((tmp_1,tmp_2))
    tmp34 = torch.cat((tmp_3,tmp_4))
    tmp = torch.cat((tmp12,tmp34)).to(device)
    with torch.no_grad():
        denoise_flt = model(tmp).to('cpu').reshape(1,640000)

    if num_batch > 1:
        for i in range(1,num_batch):
            tmp_1 = torch.cat((miniWav[i*8+0],miniWav[i*8+1]))
            tmp_2 = torch.cat((miniWav[i*8+2],miniWav[i*8+3]))
            tmp_3 = torch.cat((miniWav[i*8+4],miniWav[i*8+5]))
            tmp_4 = torch.cat((miniWav[i*8+6],miniWav[i*8+7]))
            tmp12 = torch.cat((tmp_1,tmp_2))
            tmp34 = torch.cat((tmp_3,tmp_4))
            tmp = torch.cat((tmp12,tmp34)).to(device)
            with torch.no_grad():
                denoise = model(tmp).to('cpu').reshape(1,640000)
            
            denoise_flt = torch.cat((denoise_flt,denoise), -1)

    return denoise_flt[:,:len_input]

def _load_model(model_path):
    #model_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/logs"
    #model_path = "/home/hieule/DeepDenoise"
    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256], batch_size= 8)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

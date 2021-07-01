import os
import glob
import random
import tqdm
from scipy.io import wavfile
from ultis.dataio import read_audio
import torch
import torchaudio

root_path = os.path.dirname(os.path.realpath(__file__))

path_wav = [os.path.join(root_path, 'dataset', 'zalo2spk', 's2')]
for path in path_wav:
    files = glob.glob(path + '/*.wav', recursive=True)

    for i in tqdm.tqdm(range(len(files))):
        fs = read_audio(files[i])
        if len(fs.size()) > 1:
            wav, _ = torchaudio.load(files[i])

            wav_new = torch.Tensor(1, wav.shape[1])

            for j in range(wav.shape[1]):
                wav_new[0][j] = (wav[0][j] + wav[1][j]) /2 
            # print(wav_new.shape)
            # torchaudio.save(files[i], wav_new,_ )
        
    #     count = count + 1
    #     os.rename(files[i],os.path.join(path, '2mixed_' + str(count-1) +'.wav'))


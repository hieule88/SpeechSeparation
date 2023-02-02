from genericpath import exists
import os
import torchaudio
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm
import shutil
from scipy import signal
import torch 

root_path = os.getcwd().split('/')
root_path = '/'.join(root_path[:-1])
root_path = os.path.join(root_path, 'dataset', 'data_vd', 's1')
speakers = [os.path.join(root_path, path) for path in os.listdir(root_path)]
sum = 4152.658928287987
cnt = 0

for i in tqdm(range(len(speakers)), total= len(speakers)):
    newspeaker = speakers[i].split('/')
    # newspeaker.remove('all')
    tr_tt_ratio = cnt / sum
    
    if tr_tt_ratio <= 0.5: 
        newspeaker.insert(6,'s1')
    else:
        newspeaker.insert(6,'s2')
    newspeaker = '/'.join(newspeaker)

    audios = os.listdir(speakers[i])
    for audio in audios:
        name = os.path.join(speakers[i], audio)
        spec_name = audio.split('.')[0]
        s1 , fs = torchaudio.load(name)
        len_s = s1.shape[1] / fs
        cnt = cnt + len_s
    # os.rename(speakers[i],newspeaker)    
# FOR PUBLIC-TEST
        if len_s > 0.65:
            if s1.shape[0] == 2:
                s1 = s1.mean(axis=0)
                s1 = torch.unsqueeze(s1, 0)
                
            if len_s > 3 :
                num_rep = int((len_s // 3) +  1)
                for rep in range(num_rep):
                    start = rep * fs * 3
                    end = (rep + 1) * fs * 3 if (rep + 1) * 3 < len_s else None
                    data_frame = s1[ :, start : end ]
                    torchaudio.save(os.path.join(speakers[i], spec_name) + '*' + str(rep) + '.wav', data_frame, fs)
            else:
                torchaudio.save(os.path.join(speakers[i], spec_name) + '*.wav', s1, fs)


    # torchaudio.save(os.path.join(root_path,'temp_0.wav'), torch.unsqueeze(s1[0], 0), _)
    # torchaudio.save(os.path.join(root_path,'temp_1.wav'), torch.unsqueeze(s1[1], 0), _)
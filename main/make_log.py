import os
import numpy as np
from numpy.lib.npyio import load
import librosa
import tqdm
import glob

def load_wav(path):
    signal, sr = librosa.load(path)
    return signal

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis = axis, ddof = ddof)
    return np.where(sd == 0, 0, m / sd)

if __name__ == "__main__":
    root_path = os.getcwd()
    root_path = root_path.split("/")
    root_path = "/".join(root_path[:-2])

    type = ['tr', 'vd']
    
    for i in type:
            
        data_path = os.path.join(root_path, 'dataset', 'data_' + i)
        log_path = os.path.join(root_path, 'dataset', 'mix_2_spk_' + i +'.txt')
        s1_files = glob.glob(os.path.join(data_path,'s1', '*', '*.wav'))
        s2_files = glob.glob(os.path.join(data_path,'s2', '*', '*.wav'))

        with open(log_path, "w") as fid:
            for i in tqdm.tqdm(range(len(s1_files))):
                s1_path = s1_files[i]
                s2_path = s2_files[i]
                s1_wav = load_wav(s1_path)
                s2_wav = load_wav(s2_path)
                s1_snr = signaltonoise(s1_wav)
                s2_snr = signaltonoise(s2_wav)
                fid.write("{} {} {} {}\n".format(s1_path, s1_snr, s2_path, s2_snr))

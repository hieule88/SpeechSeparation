import os
import glob
import numpy as np
from numpy.lib.npyio import load
import scipy.io
import librosa
import tqdm

def load_wav(path):
    signal, sr = librosa.load(path)
    return signal

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis = axis, ddof = ddof)
    return np.where(sd == 0, 0, m / sd)

root_path = os.getcwd()
root_path = root_path.split("/")
root_path = "/".join(root_path[:-2])

data_path = os.path.join(root_path, 'dataset', 'zalo2spk')
log_path = os.path.join(root_path, 'dataset', 'mix_2_spk.txt')
s1_files = os.listdir(os.path.join(data_path, 's1'))

type_file = ['s1', 's2']
with open(log_path, "w") as fid:
    for i in tqdm.tqdm(range(len(s1_files))):
        s1_path = os.path.join(data_path, 's1', '2mixed_'+ str(i) +'.wav')
        s2_path = os.path.join(data_path, 's2', '2mixed_'+ str(i) +'.wav')
        s1_wav = load_wav(s1_path)
        s2_wav = load_wav(s2_path)
        s1_snr = signaltonoise(s1_wav)
        s2_snr = signaltonoise(s2_wav)
        fid.write("{} {} {} {}\n".format(s1_path, s1_snr, s2_path, s2_snr))

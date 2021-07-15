from genericpath import exists
import os
import torchaudio
from speechbrain.dataio.dataio import read_audio
import tqdm
import shutil
from scipy import signal

root_path = os.getcwd()
root_path = root_path.split('/')
root_path = '/'.join(root_path[:-2])

root_path = os.path.join(root_path, 'dataset', 'test')
mix_csv = os.path.join(root_path, 'public-test.csv')
wav_file = os.path.join(root_path, 'public-test')

# os.makedirs(os.path.join(root_path,'s1'), exist_ok= True)
# os.makedirs(os.path.join(root_path,'s2'), exist_ok= True)
s1_files = os.path.join(root_path, 's1')
s2_files = os.path.join(root_path, 's2')

s1_paths = os.listdir(s1_files)
s2_paths = os.listdir(s2_files)

paths_s1 = [os.path.join(s1_files, p) for p in s1_paths]
paths_s2 = [os.path.join(s2_files, p) for p in s2_paths]

fs_read = 16000
for i in range(len(paths_s2)):
    s1 , _ = torchaudio.load(paths_s2[i])
    if (s1.shape[0] == 2):
        print(paths_s2[i])
        exit(0)
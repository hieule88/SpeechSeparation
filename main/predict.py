import utils 
import torchaudio
import os 
import torch.nn.functional as F
import numpy as np
# khoi tao model
import torch
global model 
import argparse
import speechbrain as sb
import sys

model = None

def predict(file1, file2):
    device = torch.device('cuda')
    dns_home = "/home/hieule/SpeechSeparation"
    
    s1_path = os.path.join(dns_home,'static/upload/') + file1
    s2_path = os.path.join(dns_home,'static/upload/') + file2

    file1_name = file1.split('.')[-2]
    file2_name = file2.split('.')[-2]
    mix_name = file1_name + '_' + file2_name
    
    mix = utils.prepare_mixed(s1_path, s2_path)

    mixed_filepath = os.path.join(dns_home,'static/upload/', mix_name + '.wav')
    mix = torch.tensor(mix)
    mix.to('cpu')
    torchaudio.save(mixed_filepath, mix.unsqueeze(0), sample_rate= 16000)

    # mixed_filepath = os.path.join(dns_home,'static/upload/', 'nu1_nu2_fix1.wav')
    separated = utils._process(mixed_filepath, model)
    
    out_file1_path = os.path.join(dns_home,'static/upload/', mix_name + '_speaker1.wav')
    out_file2_path = os.path.join(dns_home,'static/upload/', mix_name + '_speaker2.wav')

    torchaudio.save(out_file1_path, separated[:, :, 0], sample_rate= 16000)
    torchaudio.save(out_file2_path, separated[:, :, 1], sample_rate= 16000)

    return 'Done'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', action='store', type=str, default='')
    parser.add_argument('--file1', action='store', type=str, default='')
    parser.add_argument('--file2', action='store', type=str, default='')

    args = parser.parse_args()
    hparams = args.hparams
    file1 = args.file1
    file2 = args.file2

    print("App run!")
	#load model
    model = utils._load_model(hparams)    
    print("load model done!")
    print(predict(file1, file2))
    
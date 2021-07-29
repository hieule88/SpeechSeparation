from flask import Flask, request, render_template, send_file
import utils 
import torchaudio
import os 
import torch.nn.functional as F
import numpy as np
from mps_storage import MPS
# khoi tao model
import torch
global model 

mps_upload = MPS()
model = None

# khoi tao flask app
app = Flask(__name__)

# Khai báo các route 1 cho API

@app.route("/", methods=["GET","POST"])
def index():
    print("hihi")
    if request.method == "GET":
        print("get")
        return render_template("upload.html")
    else:
        device = torch.device('cuda')
        print("post")
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        # print(file1, file2)
        dns_home = "/home/SpeechSeparation"
        print(file1)
        print(file2)
        
        s1_path = os.path.join(dns_home,'static/upload') + file1.filename
        s2_path = os.path.join(dns_home,'static/upload') + file2.filename

        file1_name = file1.filename.split('.')[-2]
        file2_name = file2.filename.split('.')[-2]
        mix_name = file1_name + '_' + file2_name
        
        # luu file
        file1.save(s1_path)
        file2.save(s2_path)

        mix = utils.prepare_mixed(s1_path, s2_path)

        mixed_filepath = os.path.join(dns_home,'static/upload', mix_name + '.wav')
        mix = torch.tensor(mix)
        mix.to('cpu')
        torchaudio.save(mixed_filepath, mix.unsqueeze(0), sample_rate= 16000)

        separated = utils._process(mixed_filepath, model)
        
        out_file1_path = os.path.join(dns_home,'static/upload', mix_name + '_speaker1.wav')
        out_file2_path = os.path.join(dns_home,'static/upload', mix_name + '_speaker2.wav')

        mix_dir = mps_upload.upload(mixed_filepath)

        print(separated.shape)

        torchaudio.save(out_file1_path, separated[:, :, 0], sample_rate= 16000)
        torchaudio.save(out_file2_path, separated[:, :, 1], sample_rate= 16000)

        spk1_dir = mps_upload.upload(out_file1_path)
        spk2_dir = mps_upload.upload(out_file2_path)

        return render_template("upload.html", mixed_filepath=mix_dir, out_file1_path=spk1_dir, out_file2_path=spk2_dir)

if __name__ == "__main__":
    print("App run!")

	#load model
    port = 5005
    host = '0.0.0.0'
    model = utils._load_model()    
    print("load model done!")
    app.run(debug=False, port = port, host=host)
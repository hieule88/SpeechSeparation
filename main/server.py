from flask import Flask, request, render_template, send_file
import utils 
import torchaudio
import os 
import torch.nn.functional as F
import numpy as np
# khoi tao model
global model 

model = None

# khoi tao flask app
app = Flask(__name__)

# Khai báo các route 1 cho API

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "GET":
        return render_template("upload.html")
    else:
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        # print(file1, file2)
        dns_home = "/home/SpeechSeparation"
        
        s1_path = os.path.join(dns_home,'static/upload') + file1.filename
        s2_path = os.path.join(dns_home,'static/upload') + file2.filename
        # luu file
        file1.save(s1_path)
        file2.save(s2_path)

        mix = utils.prepare_mixed(s1_path, s2_path)

        mixed_filepath = os.path.join(dns_home,'static/upload', 'mixed_wav.wav')

        torchaudio.save(mixed_filepath, mix.unsqueeze(0), sample_rate= 16000)

        separated = utils._process(mixed_filepath, model)
        
        out_file1_path = os.path.join(dns_home,'static/upload', 'speaker1.wav')
        out_file2_path = os.path.join(dns_home,'static/upload', 'speaker2.wav')

        torchaudio.save(out_file1_path, separated[:, :, 0].unsqueeze(0), sample_rate= 16000)
        torchaudio.save(out_file2_path, separated[:, :, 1].unsqueeze(0), sample_rate= 16000)
        
        return render_template("upload.html", mixed_filepath=mixed_filepath, out_file1_path=out_file1_path, out_file2_path=out_file2_path)

if __name__ == "__main__":
    print("App run!")

	#load model
    port = 5010
    host = '0.0.0.0'
    model = utils._load_model()    
    app.run(debug=False, port = port, host= host, threaded=False)
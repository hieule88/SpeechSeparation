from flask import Flask, request, render_template, send_file
import flask
import utils 
import torchaudio
import os 
import zipfile
# khoi tao model
global model 

model = None

# khoi tao flask app
app = Flask(__name__)

# Khai báo các route 1 cho API

@app.route('/')
def upload_form():
	return render_template('upload_templates/upload.html', audio_path = 'select file to predict!')

@app.route('/separate', methods=['POST'])
def get_prediction():
    print('SEPARATE MODE')
    dns_home = "/home/SpeechSeparation"
    #dns_home = "/home/hieule/DeepDenoise"
    if request.method == 'POST':
        _file = request.files['file']
        if _file.filename == '':
            return upload_form()
        print('\n\nfile uploaded:',_file.filename) 
        _file.save(os.path.join(dns_home,'static/upload', _file.filename)) 
        print('Write file success!')

        separated = utils._process(os.path.join(dns_home,'static/upload',_file.filename), model)

        zipfolder = zipfile.ZipFile(os.path.join(dns_home,'static/upload', 'Audiofiles.zip')
                                    ,'w', compression = zipfile.ZIP_STORED)

        for i in range(2):
            signal = separated[0, :, i]
            signal = signal / signal.abs().max()
            torchaudio.save(os.path.join(dns_home, 'static/separated',_file.filename + '_Speaker_' + str(i+1)), 
                                signal.unsqueeze(0).cpu(), sample_rate = 16000)
            zipfolder.write(os.path.join(dns_home, 'static/separated',_file.filename + '_Speaker_' + str(i+1)))
        zipfolder.close() 
        
        os.remove(os.path.join(dns_home,'static/upload',_file.filename))
        os.remove(os.path.join(dns_home, 'static/separated',_file.filename + '_Speaker_1'))
        os.remove(os.path.join(dns_home, 'static/separated',_file.filename + '_Speaker_2'))

        print('Done')        
        try :
            return send_file(os.path.join(dns_home,'static/upload', 'Audiofiles.zip'),
                            mimetype = 'zip',
                            attachment_filename= 'Audiofiles.zip',
                            as_attachment = True)
        except Exception as e:
            return str(e)

        os.remove(os.path.join(dns_home,'static/upload', 'Audiofiles.zip'))
if __name__ == "__main__":
    print("App run!")

	#load model
    port = 5010
    host = '0.0.0.0'
    model = utils._load_model()    
    app.run(debug=False, port = port, host= host, threaded=False)
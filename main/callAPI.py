import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

multipart_data = MultipartEncoder(
    fields={
            # a file upload field
            'file': ('Mixed_Wav', open('videoplayback.wav', 'rb'))
           }
    )
    
response = requests.post('http://localhost:5010/', data=multipart_data,
                  headers={'Content-Type': multipart_data.content_type})

with open("Speaker_1.wav", "wb") as f:
    f.write(response.content[0])
with open("Speaker_2.wav", "wb") as f:
    f.write(response.content[1])

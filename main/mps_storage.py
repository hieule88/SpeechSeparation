import requests
import subprocess
import sys

sys.path.insert(0, '../')
sys.path.insert(0, '.')
from config.config import mps_config


class MPS:
    def __init__(self):
        self.host = mps_config['host']
        self.namespace = mps_config['namespace']
        self.secret_key = mps_config['secret_key']
        self.file_host = mps_config['file_host']

    def ls(self, dir):
        API_ENDPOINT = 'http://' + self.host + '/_/ls?secret_key=' + self.secret_key + '&dirpath=' + dir
        ls = requests.get(url=API_ENDPOINT).json()
        return ls

    def upload(self, file):
        path = file
        API_ENDPOINT = self.host + '/_/upload'
        query = 'curl ' + API_ENDPOINT + ' -F convert=true -F filename=' + path + ' -F secret_key=' + \
                self.secret_key + ' -F filedata=@' + path
        res = subprocess.call(query, shell=True)
        return 'https://' + self.file_host + '/' + path

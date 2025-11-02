import os
import sys
from abc import ABC, abstractmethod

import torch
import yaml

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

from helper.image import Image


class Template():

    def __init__(self, device: str=None):
        
        self.name = None
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.device = None
        self.token = None
        self.cache_dir = os.path.join(repo_dir, 'model_cache')

        self._set_device(device=device)
        self._get_token()


    @abstractmethod
    def load_model(self):
        pass


    @abstractmethod
    def ask(prompt: str=None, images: list[Image]=None):
        pass


    @abstractmethod
    def embedd(self, text: str=None, images: list[Image]=None):
        pass


    @abstractmethod
    def _tokenize(self, text: str=None, images: list[Image]=None):
        pass


    @abstractmethod
    def _load_tokenizer(self):
        pass


    def _set_device(self, device: str=None):

        if device:
            if device == 'gpu':
                device_index = 0
                device = torch.device(f'cuda:{device_index}')
            elif device == 'cpu':
                device = torch.device('cpu')
            else:
                raise ValueError(f'Input arg "device" must be "cpu" or "gpu" but was {device}!')
            
        else:
            if torch.cuda.is_available():
                device_index = 0
                device = torch.device(f'cuda:{device_index}')

                gpu_name = torch.cuda.get_device_name(device_index)

                print(f'GPU detected: {gpu_name}.')
            else:
                print('No GPU detected. Using CPU.')
                device = torch.device('cpu')

        self.device = device

        return
    

    def _get_token(self):
        config_file = os.path.join(repo_dir, 'config.yml')

        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)

        self.token = config_data['token']

        return
    

if __name__ == '__main__':
    pass
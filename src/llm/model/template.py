from abc import ABC, abstractmethod

import torch

from llm.helper.image import Image


class Template(ABC):

    def __init__(self, hf_token: str=None):

        self.name: str = None
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.device: torch.device = None
        self.attention: str = None
        self.hf_token: str = hf_token
        self.location: str = None
        self.quantization: str = None


    @abstractmethod
    def load_model(self):
        pass


    @abstractmethod
    def _load_tokenizer(self):
        pass


    @abstractmethod
    def _tokenize(self, text: str=None, images: list[Image]=None):
        pass


    @abstractmethod
    def ask(prompt: str=None, images: list[Image]=None):
        pass


    @abstractmethod
    def embedd(self, text: str=None, images: list[Image]=None):
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
    

if __name__ == '__main__':
    pass
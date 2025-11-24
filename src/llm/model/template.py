from abc import ABC, abstractmethod

import torch


class Template(ABC):

    def __init__(self, hf_token: str=None):

        self.hf_token: str = hf_token
        self.name: str = None

        self.location: str = None
        self.remote: bool = False
        self.commit: str = None
        self.quantization: str = None

        self.device: torch.device = None
        self.model = None


    @abstractmethod
    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        quantization: str=None):

        pass


    @abstractmethod
    def ask(
        prompt: str=None,
        max_tokens: int=256,
        temperature: float=0.0) -> str:

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
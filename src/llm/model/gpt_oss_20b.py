from transformers import pipeline
import torch

from .template import Template


class GptOss20b(Template):

    def __init__(self, name: str, hf_token: str=None):
        super().__init__(name=name, hf_token=hf_token)

        self.name = 'openai/gpt-oss-20b'


    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        quantization: str=None):

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self.model = pipeline(
            task='text-generation',
            model=self.name,
            torch_dtype='auto',
            device_map=self.device,
            token=self.hf_token,
            cache_dir=self.location,
            revision=self.commit,
            trust_remote_code=self.remote)

        return


    def ask(self,
        prompt: str=None,
        max_tokens: int=256,
        temperature: float=0.0):

        messages = [{
            "role": "user",
            "content": prompt}]
        
        response = self.model(messages, max_new_tokens=max_tokens)[0]["generated_text"][-1]

        return response


if __name__ == '__main__':

    # model = GptOss20b(hf_token=<hf_token>)
    # model.load(location=<save dir>, remote=True)
    # model.ask('Tell me a joke.')

    pass
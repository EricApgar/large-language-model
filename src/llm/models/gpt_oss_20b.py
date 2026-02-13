from transformers import pipeline

from llm.models.template import Template


class GptOss20b(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name = 'openai/gpt-oss-20b'


    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        quantization: str=None,
        device: str=None):

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)

        model_kwargs = {
            'cache_dir': self.location,
            'local_files_only': not self.remote}

        self.model = pipeline(
            task="text-generation",
            model=self.name,
            dtype="auto",
            device_map=self.device,
            token=self.hf_token,
            revision=self.commit,
            model_kwargs=model_kwargs)

        return


    def ask(self,
        prompt: str,
        max_tokens: int=256,
        temperature: float=0.1):

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')

        messages = [{
            "role": "user",
            "content": prompt}]
        
        kwargs = {}
        if temperature == 0:
            kwargs['do_sample'] = False
        else:
            kwargs['temperature'] = temperature

        full_response = self.model(
            messages,
            max_new_tokens=max_tokens,
            **kwargs)

        response = full_response[0]["generated_text"][-1]['content']

        # Remove thinking process from response. If it didn't
        # have enough tokens to finish thinking, the response
        # will come out half complete (with thinking included).
        if 'assistantfinal' in response:
            response = response.split("assistantfinal", 1)[-1]

        return response


if __name__ == '__main__':

    pass
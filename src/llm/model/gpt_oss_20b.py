from transformers import pipeline

from llm.model.template import Template


class GptOss20b(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

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

        # self.model = pipeline(
        #     task='text-generation',
        #     model=self.name,
        #     torch_dtype='auto',
        #     device_map=self.device,
        #     token=self.hf_token,
        #     cache_dir=self.location,
        #     revision=self.commit,
        #     trust_remote_code=self.remote)

        model_kwargs = {
            'cache_dir': self.location,
            'local_files_only': not self.remote}

        self.model = pipeline(
            "text-generation",
            model="openai/gpt-oss-20b",
            dtype="auto",
            device_map=self.device,
            token=self.hf_token,
            revision=self.commit,
            model_kwargs=model_kwargs)

        return


    def ask(self,
        prompt: str=None,
        max_tokens: int=256,
        temperature: float=0.1):

        messages = [{
            "role": "user",
            "content": prompt}]

        full_response = self.model(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature)

        response = full_response[0]["generated_text"][-1]['content']

        # Remove thinking process from response.
        if 'assistantfinal' in response:
            response = response.split("assistantfinal", 1)[-1]

        return response


if __name__ == '__main__':

    pass
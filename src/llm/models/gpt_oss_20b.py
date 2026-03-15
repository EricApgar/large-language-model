import os

from transformers import pipeline

from llm.models.template import Template
from llm_conversation import Conversation


class GptOss20b(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name = 'openai/gpt-oss-20b'
        self.model: pipeline = None


    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        quantization: str=None,
        device: str=None):

        if (not remote) and (not os.path.isdir(location)):
            raise ValueError(f'Nonexistant location ({location}) - fix or set remote=True.')

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)

        model_kwargs = {
            'cache_dir': self.location,
            'local_files_only': not self.remote}
        
        self.model = pipeline(
            task='text-generation',
            model=self.name,
            dtype='auto',
            device_map=self.device,
            token=self.hf_token,
            revision=self.commit,
            # trust_remote_code=self.remote,
            model_kwargs=model_kwargs)

        return
    

    def ask(self,
        prompt: str | Conversation,
        max_tokens: int=512,
        temperature: float=0.9,
        reasoning_level: str=None):

        formatted_messages = self._format_prompt(prompt=prompt, reasoning_level=reasoning_level)

        kwargs = {}
        if temperature == 0:
            kwargs['do_sample'] = False
        else:
            kwargs['temperature'] = temperature

        model_output = self.model(
            formatted_messages,
            max_new_tokens=max_tokens,
            **kwargs)
        
        full_text_response = model_output[0]['generated_text'][-1]['content']

        if 'assistantfinal' in full_text_response:
            text = full_text_response.split("assistantfinal", 1)[1].strip()
        else:
            raise ValueError(f'Mangled LLM output. Could not find expected end marker "assistantfinal" in generated text: {full_text_response}')

        return text


    @staticmethod
    def _format_prompt(prompt: str | Conversation, reasoning_level: str=None) -> list[dict]:
        '''
        Structure the input convo and images into the expected format
        to get a good clean LLM response. Embedd it and prepare for LLM
        token generation.
        '''

        if isinstance(prompt, str):
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        system_pieces = []
        formatted_messages = []

        if reasoning_level:
            system_pieces.append(f'Reasoning level: {reasoning_level}.')

        if convo.overall_prompt:
            system_pieces.append(convo.overall_prompt)

        if convo.context:
            for context in convo.context:
                system_pieces.append(context)

        if system_pieces:  # Merge background context pieces.
            formatted_messages.append({'role': 'system', 'content': ' '.join(system_pieces)})

        if convo.history:
            for response in convo.history:
                formatted_messages.append({'role': response.role, 'content': response.text})

        return formatted_messages


if __name__ == '__main__':

    # model = GptOss20b()
    # model.load(location=<path to model cache>)  # NOTE: set <path to model cache>.
    # response = model.ask(prompt='Name a primary color.')

    # print(response)

    pass
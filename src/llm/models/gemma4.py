from __future__ import annotations
from typing import TPYE_CHECKING

import os
from transformers import AutoProcessor, AutoModelForMultimodalLM

if TYPE_CHECKING:
      from PIL import Image as PillowImage

from llm.models.template import template
from llm_conversation import Conversation


class Gemma4(Template):

    def __init__(self, name: str='google/gemma-4-E2B-it', hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name = name


    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        quantization: str=None,
        device: str=None):
        '''
        '''

        if (not remote) and (not os.path.isdir(location)):
            raise ValueError(f'Nonexistant location ({location}) - fix or set remote=True.')

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)
        self._load_processor()

        self.model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            token=self.hf_token,
            cache_dir=self.location,
            device_map=self.device)
        
        return


    def ask(self,
        prompt: str | Conversation,
        images: list[PillowImage.Image]=None,
        max_tokens: int=1024) -> str:

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')

        formatted_messages = self._format_prompt(prompt=prompt, images=images)

        text = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=True)

        inputs = self.processor(text=text, images=images, return_tensors='pt').to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]

        # Generate new tokens from the input via an LLM.
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
      
        response = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

        return response
    
    @staticmethod
    def _format_prompt(
        prompt: str | Conversation,
        images: list[PillowImage.Image]=None) -> list[dict]:  # TODO: check type hint of result.
        '''
        '''

        if isinstance(prompt, str):
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        system_pieces = []
        formatted_messages = []

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

        if images:
            last_text = formatted_messages[-1]['content']
            formatted_messages[-1]['content'] = [{'type': 'image', 'image': i} for i in images] + [{'type': 'text', 'text': last_text]

        return formatted_messages


    def _load_processor(self):

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.name)

        return
    

if __name__ == '__main__':

    # model = Phi4MultimodalInstruct()
    # model.load(location=<path to model cache>)  # NOTE: set <path to model cache>.

    # response = model.ask(prompt='Name a primary color. Be brief.', max_tokens=256)
    # print(f'{response}\n')

    # convo = Conversation()
    # convo.set_overall_prompt(text='You are a helpful assistant.')
    # convo.add_context(text='Your favorite color is red.')
    # convo.add_context(text='Your favorite shape is the hexagon.')
    # convo.add_response(role='user', text='What is your favorite color-shape combination?')
    # response = model.ask(prompt=convo, max_tokens=256)
    # print(f'{response}\n')

    # from PIL import Image as PillowImage
    # image = PillowImage.open(r'/home/eric/Desktop/monkey.png')  # NOTE: Point to existing image.
    # response = model.ask(prompt='Describe the image.', images=[image], max_tokens=256)
    # print(f'{response}\n')

    pass


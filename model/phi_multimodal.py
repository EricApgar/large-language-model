import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

from helper.image import Image
from model.template import Template


class PhiMultimodal(Template):
    def __init__(self, device: str=None, version: str='4'):
        super().__init__(device=device)

        SUPPORTED_VERSIONS = {
            '4': 'microsoft/Phi-4-multimodal-instruct'}
        
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f'Input arg "version" must be one of {SUPPORTED_VERSIONS}. Was {version}!')
        else:
            self.name = SUPPORTED_VERSIONS[version]

        self.is_quantized = None
        self.version = version


    def load_model(self, quantize: bool=True, attention: str=None, update: bool=False):
        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16)
            self.is_quantized = True
        else:
            quantization_config = None
            self.is_quantized = False

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            token=self.token,
            cache_dir=self.cache_dir,
            local_files_only=not update,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=True,
            _attn_implementation=self._get_attention(attention=attention),
            torch_dtype=torch.float16)

        return
    

    def ask(self, prompt: str, images: list[Image]=None, max_tokens: int=100):

        if self.is_quantized and images is not None and self.version == '4':
            raise ValueError('Quantized Phi-4-Multimodal does not support images!')

        tokens = self._tokenize(text=prompt, images=images)

        generation_args = {
            'max_new_tokens': max_tokens,
            'temperature': 0.0,
            'do_sample': False}
        
        generate_ids = self.model.generate(
            **tokens,
            eos_token_id=self.tokenizer.tokenizer.eos_token_id,
            **generation_args)
        
        generate_ids = generate_ids[:, tokens['input_ids'].shape[1]:]

        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        
        return response
    

    def embed(self, text: str=None, images: list[Image]=None):
        pass


    def _load_tokenizer(self, update: bool=False):
        
        self.tokenizer = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.name,
            local_files_only=not update,
            trust_remote_code=True,
            num_crops=4)

        return
    
    def _tokenize(self, text: str=None, images: list[Image]=None) -> dict:
        
        if images is not None:
            images = [image.data for image in images]
            image_tags = [''.join([f'<|image_{i+1}|>\n' for i, _ in enumerate(images)])]
            content_wrap = '<|user|>\n' + image_tags + f'{text}<|end|>/n<|assistant|>\n'
        else:
            content_wrap = f'<|user|>\n{text}<|end|>/n<|assistant|>\n'
            
        messages = [{
            'role': 'user',
            'content': content_wrap}]
        
        structured_prompt = self.tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        
        tokens = self.tokenizer(
            images=images,
            text=structured_prompt,
            return_tensors='pt').to(self.device)

        return tokens
    

    def _get_attention(self, attention: str=None):
        ATTENTION_OPTIONS = ['flash_attention_2', 'eager']
        if attention:
            if attention not in ATTENTION_OPTIONS:
                raise ValueError(f'Input arg "attention" must be in {ATTENTION_OPTIONS}. Was {attention}!')
        else:
            try:
                import flash_attn
                attention = 'flash_attention_2'
            except ImportError:
                attention = 'eager'

        return attention
    

if __name__ == '__main__':
    model = PhiMultimodal()
    model.load_model(update=True)

    response = model.ask(prompt='Tell me a joke.')

    print(response)
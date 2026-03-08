from __future__ import annotations
from typing import TYPE_CHECKING

from transformers import AutoProcessor, AutoModelForCausalLM

from llm.models.template import Template
from llm_conversation import Conversation

if TYPE_CHECKING:
    from PIL import Image as PillowImage


class Phi4MultimodalInstruct(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name: str = 'microsoft/Phi-4-multimodal-instruct'

        self.processor = None


    def load(self,
        location: str,
        remote: bool=False,
        commit: str='0cb22ab20b10ac01c49ecd8b7138dcd98bc00548',
        quantization: str=None,
        device: str=None):
        '''
        The commit is locked to this because this is the version of Phi4 with patches
        needed to run. There's a whole mess regarding Phi-4 getting out of date with
        transformers and breaking, and no one updating the HF Phi-4. See the discussions
        (https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions) for details.

        Just know that it's huge hassle to try to get Phi4 working as the transformers lib updates.
        '''

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)
        self._patch_dynamic_cache()

        # TODO: Turn on quantization at some point.
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16)

        self._load_processor()

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            token=self.hf_token,
            cache_dir=self.location,
            local_files_only=not self.remote,
            revision=self.commit,
            low_cpu_mem_usage=True,
            # quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=True,  # self.remote,
            _attn_implementation='eager',
            torch_dtype='auto')
        
        return


    def ask(self,
        prompt: str | Conversation,
        images: list[PillowImage.Image]=None,
        max_tokens: int=1024,
        temperature: float=0.5,
        reasoning_level: str='low',
        repetition_penalty: float=1.12,
        top_p: float=0.95) -> str:

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')
        
        if isinstance(prompt, str):  # Create a structured conversation from input.
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        embedding = self.structure_inputs(convo=convo, images=images)

        generation_args = {
            'max_new_tokens': max_tokens,
            'do_sample': False}

        # Generate new tokens from the input via an LLM.
        generated_tokens = self.model.generate(
            **embedding,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            **generation_args)

        # Extract the explicit response tokens (sans thinking and original question).
        response_tokens = generated_tokens[:, embedding['input_ids'].shape[1]:]

        # Translate the tokens to text (essentially a look up table).
        response = self.processor.batch_decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return response
    

    def structure_inputs(self, convo: Conversation, images: list=None) -> dict:
        '''
        Structure the input convo and images into the expected format
        to get a good clean LLM response. Embedd it and prepare for LLM
        token generation.
        '''

        if not convo.overall_prompt:
            convo.set_overall_prompt(text='')

        system_prompt = convo.overall_prompt + ' '.join(convo.context)
        messages = [{
            'role': 'system',
            'content': system_prompt}] + [{
                'role': i.role,
                'content': i.text} for i in convo.history]

        if images:  # Modify last item in convo to carry image tags.
            image_tags = ''.join([f'<|image_{i+1}|>' for i, _ in enumerate(images)])
            last_role = messages[-1]['role']
            messages[-1] = {'role': last_role, 'content': image_tags + messages[-1]['content']}

        structured_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        embedding = self.processor(
            images=images,
            text=structured_prompt,
            return_tensors='pt').to(self.device)

        return embedding


    def _load_processor(self, num_images: int=1):

        if num_images == 1:
            num_crops = 16
        else:
            num_crops = 4

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.name,
            trust_remote_code=True,  # self.remote,
            num_crops=num_crops)

        return


    def _patch_dynamic_cache(self):
        '''
        Phi-4 lags the latest version of transformers. So the transformers
        library introduces breaking changes. To fix this, we point Phi4 at
        the commit that fixes the "missing prepare_inputs_for_generation()"
        function and then we manually create the needed method for the dynamic
        cache to fix the error.
        '''

        from transformers.cache_utils import Cache

        def get_usable_length(self, new_seq_length: int, layer_idx: int=0) -> int:
            prev_len = self.get_seq_length(layer_idx)

            max_len = None
            if hasattr(self, 'get_max_length'):
                try:
                    max_len = self.get_max_length()
                except TypeError:
                    max_len = self.get_max_length(layer_idx)
            elif hasattr(self, 'get_max_cache_shape'):
                try:
                    shape = self.get_max_cache_shape(layer_idx)
                    max_len = shape[2] if shape is not None else None
                except Exception:
                    max_len = None

            if max_len is not None and prev_len + new_seq_length > max_len:
                return max_len - new_seq_length
            
            return prev_len
        
        Cache.get_usable_length = get_usable_length

        return
    

if __name__ == '__main__':

    model = Phi4MultimodalInstruct()
    model.load(location=r'/home/eric/Repos/model_cache')  # <path to model cache>

    response = model.ask(prompt='Name a primary color.', max_tokens=256)
    print(f'{response}\n')

    convo = Conversation()
    convo.set_overall_prompt(text='You are a helpful assistant.')
    convo.add_context(text='Your favorite color is red.')
    convo.add_context(text='Your favorite shape is the hexagon.')
    convo.add_response(role='user', text='What is your favorite color-shape combination?')
    response = model.ask(prompt=convo, max_tokens=256)
    print(f'{response}\n')

    from PIL import Image as PillowImage
    image = PillowImage.open(r'/home/eric/Desktop/monkey.png')
    response = model.ask(prompt='Describe the image.', images=[image], max_tokens=256)
    print(f'{response}\n')

    pass
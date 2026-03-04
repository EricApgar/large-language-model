from transformers import AutoProcessor, AutoModelForCausalLM

from llm.models.template import Template

# from PIL import image as PillowImage


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

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)
        self._patch_dynamic_cache()

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
        prompt: str,
        images: list=None,  # list[PillowImage.Image]
        max_tokens: int=256,
        temperature: float=0.1):

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')

        embedding = self.embed(text=prompt, images=images)

        generation_args = {
            'max_new_tokens': max_tokens,
            # temperature: 0.0,
            'do_sample': False}
        
        generate_ids = self.model.generate(
            **embedding,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args)
        
        # Decode the output (un-embed the output to convert to text).
        generate_ids = generate_ids[:, embedding['input_ids'].shape[1]:]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        
        return response
    

    def embed(self, text: str=None, images: list=None) -> dict:
        '''
        Background:
        <|user|>\n<|image_1|>\n<|image_2|\n<|image_3|\n{prompt}<|end|>\n<|assistant|>\n
        '''

        # TODO: Image list might not need chat template applied again.
        if images is not None:
            image_tags = ''.join([f'<|image_{i+1}|>\n' for i, _ in enumerate(images)])
            content_wrap = '<|user|\n>' + image_tags + f'{text}<|end|>\n<|assistant|>\n'
        else:
            # content_wrap = f'<|user|>\n{text}<|end|>\n<|assistant|>\n'
            content_wrap = text

        messages = [{'role': 'user', 'content': content_wrap}]

        structured_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        embedding = self.processor(
            images=images,
            text=structured_prompt,
            return_tensors='pt').to(self.device)

        return embedding


    def _load_processor(self):

        # TODO: For best performance, supposed to use num_crops=4 for multi
        # frame and 16 for single frame.
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.name,
            trust_remote_code=True,  # self.remote,
            num_crops=4)
        
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
    response = model.ask(prompt='Name a primary color.')
    print(response)

    pass
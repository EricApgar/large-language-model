import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation as HarmonyConversation,
    Role,
    RenderConversationConfig)

from llm.models.template import Template
from llm.other.conversations import Conversation


class GptOss20b(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name = 'openai/gpt-oss-20b'
        self.tokenizer = None


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

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            token=self.hf_token,
            cache_dir=self.location,
            local_files_only=not self.remote,
            revision=self.commit,
            low_cpu_mem_usage=True,
            # quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=True,  # self.remote, TODO
            _attn_implementation='eager',
            torch_dtype='auto')  # Might be obsolete. Change to "dtype"?

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

        return


    def ask(self,
        prompt: str | HarmonyConversation,
        max_tokens: int=256,
        temperature: float=0.5,
        repetition_penalty: float=1.12,
        top_p: float=0.95):

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')
        
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        if isinstance(prompt, str):
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        render_cfg = RenderConversationConfig(auto_drop_analysis=True)
        prefill_ids = encoding.render_conversation_for_completion(
            convo.harmony_convo,
            Role.ASSISTANT,
            config=render_cfg)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()

        input_ids = torch.tensor([prefill_ids], device=self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id)

        completion_ids = out[0, input_ids.shape[-1]:].tolist()

        parsed = encoding.parse_messages_from_completion_tokens(completion_ids, role=Role.ASSISTANT)

        final_msg = next(m for m in parsed if m.channel == "final")
        response = final_msg.content[0].text

        return response


if __name__ == '__main__':

    model = GptOss20b()
    model.load(location=r'/home/eric/Repos/model_cache')
    response = model.ask(prompt='What is the capital of France?')

    print(response)

    pass
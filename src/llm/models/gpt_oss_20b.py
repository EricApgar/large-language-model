import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai_harmony import (
    Conversation as HarmonyConversation,
    RenderConversationConfig,
    load_harmony_encoding,
    HarmonyEncodingName,
    DeveloperContent,
    ReasoningEffort,
    SystemContent,
    Message,
    Role)

from llm.models.template import Template
from llm_conversation import Conversation


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
        prompt: str | Conversation,
        max_tokens: int=256,
        temperature: float=0.5,
        reasoning_level: str='low',
        repetition_penalty: float=1.12,
        top_p: float=0.95):

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')
        
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        if isinstance(prompt, str):  # Create a structured conversation from 
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        convo = self._to_harmony(conversation=convo, reasoning_level=reasoning_level)

        render_cfg = RenderConversationConfig(auto_drop_analysis=True)
        prefill_ids = encoding.render_conversation_for_completion(
            convo,
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
    

    def _make_conversation(text: str):

        return
    

    @staticmethod
    def _to_harmony(conversation: Conversation, reasoning_level: str) -> HarmonyConversation:
        '''
        Build a Harmony-Conversation object from a Generic Conversation object.
        '''

        if reasoning_level == 'low':
            reasoning_level = ReasoningEffort.LOW
        elif reasoning_level == 'medium':
            reasoning_level = ReasoningEffort.MEDIUM
        elif reasoning_level == 'high':
            reasoning_level = ReasoningEffort.HIGH

        # System Details about the Overall Conversation.
        system_msg = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(reasoning_level))
        
        developer_msg = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(conversation.overall_prompt))

        msgs = [system_msg, developer_msg]

        # Background Context Information.
        if conversation.context:
            context_block = '\n'.join([
                "BACKGROUND CONTEXT (not part of the dialogue):",
                '\n'.join(conversation.context),
                "END BACKGROUND CONTEXT"])

            msgs.append(Message.from_role_and_content(Role.USER, context_block))

        # Conversation history between user and AI.
        for turn in conversation.history:
            if turn.role == "user":
                msgs.append(Message.from_role_and_content(Role.USER, turn.text))
            else:
                msgs.append(Message.from_role_and_content(Role.ASSISTANT, turn.text))

        harmony_convo = HarmonyConversation.from_messages(msgs)

        return harmony_convo


if __name__ == '__main__':

    model = GptOss20b()
    model.load(location=r'/home/eric/Repos/model_cache')
    response = model.ask(prompt='Name a primary color.')

    print(response)

    pass
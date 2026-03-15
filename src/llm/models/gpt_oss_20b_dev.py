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


class GptOss20bDev(Template):
    '''
    This development version of the GPT-OSS-20B model uses a lower level
    method for generating tokens compared to the transformers.pipeline()
    method.
    '''

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)

        self.name = 'openai/gpt-oss-20b'
        self.tokenizer = None


    def load(self,
        location: str,
        remote: bool=False,
        commit: str=None,
        device: str=None):

        self.location = location
        self.remote = remote
        self.commit = commit

        self._set_device(device=device)

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            token=self.hf_token,
            cache_dir=self.location,
            local_files_only=not self.remote,
            revision=self.commit,
            low_cpu_mem_usage=True,
            device_map=self.device,
            trust_remote_code=True,  # self.remote, TODO
            _attn_implementation='eager',
            dtype='auto')  # Might be obsolete. Change to "dtype"?

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = 0  # use a dedicated ID that isn't EOS

        return


    def ask(self,
        prompt: str | Conversation,
        max_tokens: int=256,
        temperature: float=1.0,
        reasoning_level: str='low',
        repetition_penalty: float=1.15,
        top_p: float=0.95):
        '''
        Call an LLM with a prompt and generate a response.

        This model works best when the input is formatted into an
        openai-harmony conversation structure, so all inputs are converted
        into a generic Conversation structure (if not already one) and then
        converted into the harmony structure.
        '''

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        if isinstance(prompt, str):  # Create a structured conversation from input.
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        convo_harmony = self._to_harmony(conversation=convo, reasoning_level=reasoning_level)

        render_cfg = RenderConversationConfig(auto_drop_analysis=True)
        prefill_ids = encoding.render_conversation_for_completion(
            convo_harmony,
            Role.ASSISTANT,
            config=render_cfg)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()

        input_ids = torch.tensor([prefill_ids], device=self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # Generate new tokens from the LLM.
        generated_tokens = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,  # NOTE: Debatable usefulness.
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_ids,
            attention_mask=attention_mask,  # NOTE: Debatable usefulness.
            pad_token_id=self.tokenizer.pad_token_id)  # NOTE: Debatable usefulness. self.tokenizer.eos_token_id

        # Shape the generated tokens into final form.
        generated_tokens = generated_tokens[0, input_ids.shape[-1]:].tolist()

        # Translate tokens (which are numbers) directly to output. Basically a look up table.
        # This is an tangent we take to check if the output is mangled and needs to be adjusted.
        text_tokens = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        # I can patch the mangled harmony token problem by assuming that the message is in the generated tokens, 
        # and just wiping out everything up until the first occurrence of [..., '', 'analysis', ...].
        generated_tokens = generated_tokens[get_good_token_start(token_list=text_tokens):]

        # Transform tokens into the text equivalent (will contain model reasoning and thinking).
        full_response = encoding.parse_messages_from_completion_tokens(generated_tokens, role=Role.ASSISTANT)

        # Extract the actual response (sans reasoning) from the full set of generated text.
        final_response = next(m for m in full_response if m.channel == "final")
        text_response = final_response.content[0].text

        return text_response


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


def get_good_token_start(token_list: list[str]) -> int:
    '''
    This is a patch for handling bad generated tokens which break
    the openai-harmony prompt formatter.

    It finds the start of rational thought in the generated output, skipping
    over the nonsense content that's generated.
    '''
    i_start = None

    for i in range(len(token_list) - 1):
        if token_list[i] == "" and token_list[i + 1] == "analysis":
            i_start = i
            break

    if i_start is None:
        raise ValueError(f'Mangled LLM output. Could not find expected start tokens ["", "analysis"] in generated tokens: {token_list}')

    return i_start


if __name__ == '__main__':

    # model = GptOss20bDev()
    # model.load(location=<path to model cache>)  # NOTE: set <path to model cache>.
    # response = model.ask(prompt='Name a primary color.')

    # print(response)

    pass
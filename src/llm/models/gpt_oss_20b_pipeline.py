from transformers import pipeline

from llm.models.template import Template
from llm_conversation import Conversation


class GptOss20bPipeline(Template):

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

        self.location = location
        self.remote = remote
        self.commit = commit
        self.quantization = quantization

        self._set_device(device=device)

        # self.pipeline = pipeline(
        #     task='text-generation',
        #     pretrained_model_name_or_path=self.name,
        #     cache_dir=self.location,
        #     local_files_only=not self.remote,
        #     # model=model_path,
        #     dtype='auto',
        #     device=device)
        
        self.model = pipeline(
            task='text-generation',
            model=self.name,
            dtype='auto',
            device_map=self.device,
            token=self.hf_token,
            revision=self.commit,
            trust_remote_code=self.remote)

        return
    

    def ask(self,
        prompt: str | Conversation,
        max_tokens: int=1024,
        temperature: float=0.5,
        reasoning_level: str='low'):

        if isinstance(prompt, str):  # Create a structured conversation from input.
            convo = Conversation()
            convo.add_response(role='user', text=prompt)
        else:
            convo = prompt

        messages = self._structure_prompt(
            convo=convo,
            reasoning_level=reasoning_level)

        output = self.model(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature != 1.0)

        text = self._parse_final_output(output=output)

        return text


    def _structure_prompt(self, convo: Conversation, reasoning_level: str='low') -> list[dict]:
        '''
        Structure the input convo and images into the expected format
        to get a good clean LLM response. Embedd it and prepare for LLM
        token generation.
        '''

        messages = []

        system_prompt = f'Reasoning: {reasoning_level}. {convo.overall_prompt} {' '.join(convo.context)}' 
        messages.append({'role': 'system', 'content': system_prompt})

        for i in convo.history:
            messages.append({'role': i.role, 'content': i.text})

        return messages


    def _parse_final_output(self, output: list[dict]) -> str:

        generated_text = output[0]["generated_text"][-1]['content']

        if 'assistantfinal' not in generated_text:
            raise ValueError(f'Mangled LLM output. Could not find expected end marker "assistantfinal" in generated text: {generated_text}')

        text = generated_text.split("assistantfinal", 1)[1].strip()

        return text


if __name__ == '__main__':

    model = GptOss20bPipeline()
    model.load(location=r'/home/eric/Repos/model_cache')
    response = model.ask(prompt='Name a primary color.')

    print(response)

    pass
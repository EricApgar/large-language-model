from transformers import AutoProcessor, AutoModelForCausalLM

from llm.model.template import Template


class Phi4MultimodalInstruct(Template):

    def __init__(self, hf_token: str=None):
        super().__init__(hf_token=hf_token)
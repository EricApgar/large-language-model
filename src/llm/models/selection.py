from typing import Literal

from .gpt_oss_20b import GptOss20b
from .phi4_multimodal_instruct import Phi4MultimodalInstruct


SUPPORTED_LLMS = {
    GptOss20b().name: GptOss20b,
    Phi4MultimodalInstruct().name: Phi4MultimodalInstruct
}

SUPPORTED_EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']


def model(name: str, hf_token: str=None):

    if name not in SUPPORTED_LLMS:
        raise ValueError(f'Model "{name}" not supported! Call "list_models()" for options.')

    return SUPPORTED_LLMS[name](hf_token=hf_token)


def list_models(kind: Literal['llm', 'embed']):

    if kind == 'llm':
        supported_models = list(SUPPORTED_LLMS.keys())
    elif kind == 'embed':
        supported_models = SUPPORTED_EMBEDDING_MODELS
    else:
        raise ValueError(f'Unsupported "kind" ({kind}). Must be "llm" or "embed".')
    
    return supported_models
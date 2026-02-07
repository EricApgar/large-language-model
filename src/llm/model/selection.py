from .gpt_oss_20b import GptOss20b
from .phi4_multimodal_instruct import Phi4MultimodalInstruct


MODEL_DICT = {
    GptOss20b().name: GptOss20b,
    Phi4MultimodalInstruct().name: Phi4MultimodalInstruct
}


def model(name: str, hf_token: str=None):

    if name not in MODEL_DICT:
        raise ValueError(f'Model "{name}" not supported!')

    return MODEL_DICT[name](hf_token=hf_token)
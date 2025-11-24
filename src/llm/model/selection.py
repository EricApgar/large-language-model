from .gpt_oss_20b import GptOss20b


MODEL_DICT = {
    'openai/gpt-oss-20b': GptOss20b
}


def model(name: str, hf_token: str=None):

    if name not in MODEL_DICT:
        raise ValueError(f'Model "{name}" not supported!')

    return MODEL_DICT[name](hf_token=hf_token)
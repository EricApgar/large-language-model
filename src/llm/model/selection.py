'''
Import the specific model named by the user.
'''

def model(name: str, hf_token: str=None):

    if name == 'openai/gpt-oss-20b':
        try:
            from .gpt_oss_20b import GptOss20b
        except ImportError:
            raise ImportError(f'Missing dependencies for "{name}"! Install with "llm[openai]" or "llm[all]".')
        return GptOss20b(hf_token=hf_token)

    elif name == 'microsoft/Phi-4-multimodal-instruct':
        try:
            from .phi4_multimodal_instruct import Phi4MultimodalInstruct
        except ImportError:
            raise ImportError(f'Missing dependencies for "{name}"! Install with "llm[microsoft]" or "llm[all]".')
        return Phi4MultimodalInstruct(hf_token=hf_token)

    else:
        raise ValueError(f'Model "{name}" not supported!')
'''
Desired interface:

import llm

model = llm.model(
    name='openai/gpt-oss-20b',
    hf_token=<hf token>)
model.load(
    location=<path to save dir>,
    remote=true, 
    commit=<git commit>,
    quantization='4-bit')
response = model.ask(prompt='Tell me a joke.')
'''

from typing import TYPE_CHECKING


# Accessible namespaces.
__all__ = [
    'model',
    'embedding'
]


def __getattr__(name: str):
    if name == 'model':
        from .model.selection import model
        return model
    if name == 'embedding':
        from .model.embed import EmbeddingModel
        return EmbeddingModel
    else:
        raise AttributeError(f'Module "llm" has no attribute {name!r}!')
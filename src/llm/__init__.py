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

from __future__ import annotations
from typing import TYPE_CHECKING, Any


__all__ = (
    "model",
    "embedding")


if TYPE_CHECKING:
    from .models.selection import model
    from .models.embed import EmbeddingModel


def __getattr__(name: str) -> Any:

    if name == "model":
        from .models.selection import model
        globals()[name] = model
        return model

    if name == "embedding":
        from .models.embed import EmbeddingModel
        globals()[name] = EmbeddingModel
        return EmbeddingModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))

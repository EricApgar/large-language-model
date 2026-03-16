from __future__ import annotations
from typing import TYPE_CHECKING, Any


__all__ = (
    'model',
    'list_models',
    'embedding')


if TYPE_CHECKING:
    from .models.selection import model
    from .models.selection import list_models
    from .models.embed import EmbeddingModel


def __getattr__(name: str) -> Any:

    if name == 'model':
        from .models.selection import model
        globals()[name] = model
        return model
    
    if name == 'list_models':
        from .models.selection import list_models
        globals()[name] = list_models
        return list_models

    if name == 'embedding':
        from .models.embed import EmbeddingModel
        globals()[name] = EmbeddingModel
        return EmbeddingModel

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))

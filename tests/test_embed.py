"""
Tests for EmbeddingModel.

These tests load a real model and call embed() and get_similarity(), so they
require local model weights. Sync the environment first, then pass
LLM_MODEL_CACHE inline when invoking pytest so it only exists for that one
command and does not persist in your shell:

    uv sync --extra openai
    LLM_MODEL_CACHE=<path to model cache dir> pytest

Tests are skipped automatically if LLM_MODEL_CACHE is not set.

Make sure the virtual environment is active.
"""

import os

import pytest
import torch

from llm.models.embed import EmbeddingModel


MODEL_CACHE = os.environ.get('LLM_MODEL_CACHE')


@pytest.fixture(scope='module')
def model():
    if not MODEL_CACHE:
        pytest.skip('LLM_MODEL_CACHE environment variable not set.')
    m = EmbeddingModel()
    m.load(location=MODEL_CACHE)
    yield m
    del m
    torch.cuda.empty_cache()


def test_get_similarity(model):
    e1 = model.embed(text='What shape is best?')
    e2 = model.embed(text='Hexagons are the bestagons.')
    similarity = model.get_similarity(v1=e1, v2=e2)
    assert isinstance(similarity, (int, float))

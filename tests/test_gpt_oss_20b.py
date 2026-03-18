"""
Tests for GptOss20b.

These tests load a real model and call ask(), so they require local model
weights. Sync the environment first, then pass LLM_MODEL_CACHE inline when
invoking pytest so it only exists for that one command and does not persist
in your shell:

    uv sync --extra dev --extra openai
    LLM_MODEL_CACHE=<path to model cache dir> pytest

Tests are skipped automatically if LLM_MODEL_CACHE is not set.

Make sure the virtual environment is active.
"""

import os

import pytest
import torch

from llm.models.gpt_oss_20b import GptOss20b


MODEL_CACHE = os.environ.get('LLM_MODEL_CACHE')


@pytest.fixture(scope='module')
def model():
    if not MODEL_CACHE:
        pytest.skip('LLM_MODEL_CACHE environment variable not set.')
    m = GptOss20b()
    m.load(location=MODEL_CACHE)
    yield m
    del m
    torch.cuda.empty_cache()


def test_ask(model):
    response = model.ask(prompt='Name a primary color.')
    assert isinstance(response, str)
    assert len(response) > 0

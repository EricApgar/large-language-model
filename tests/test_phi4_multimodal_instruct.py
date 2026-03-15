"""
Tests for Phi4MultimodalInstruct.

These tests load a real model and call ask(), so they require local model
weights. Sync the environment first, then pass LLM_MODEL_CACHE inline when
invoking pytest so it only exists for that one command and does not persist
in your shell:

    uv sync --extra microsoft
    LLM_MODEL_CACHE=/home/yourname/Repos/model_cache pytest

Tests are skipped automatically if LLM_MODEL_CACHE is not set.
"""

import os

import pytest
import torch
from PIL import Image as PillowImage

from llm.models.phi4_multimodal_instruct import Phi4MultimodalInstruct


MODEL_CACHE = os.environ.get('LLM_MODEL_CACHE')


@pytest.fixture(scope='module')
def model():
    if not MODEL_CACHE:
        pytest.skip('LLM_MODEL_CACHE environment variable not set.')
    m = Phi4MultimodalInstruct()
    m.load(location=MODEL_CACHE)
    yield m
    del m
    torch.cuda.empty_cache()


@pytest.fixture(scope='module')
def image():
    return PillowImage.new('RGB', (64, 64), color=(255, 0, 0))


def test_ask_with_image(model, image):
    response = model.ask(prompt='Describe the image.', images=[image], max_tokens=256)
    assert isinstance(response, str)
    assert len(response) > 0

"""
Tests for llm.list_models.
"""

import pytest

import llm


def test_list_models_llm():
    result = llm.list_models(kind='llm')
    assert isinstance(result, list)
    assert len(result) > 0


def test_list_models_embed():
    result = llm.list_models(kind='embed')
    assert isinstance(result, list)
    assert len(result) > 0


def test_list_models_invalid_kind():
    with pytest.raises(ValueError):
        llm.list_models(kind='invalid')

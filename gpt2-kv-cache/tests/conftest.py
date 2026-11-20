"""Shared test fixtures: the tokenizer and my GPT-2 with OpenAI's weights, loaded once."""

import pytest
from transformers import GPT2TokenizerFast

from gpt2_kv.load_weights import load_gpt2


@pytest.fixture(scope="session")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def ours():
    return load_gpt2()

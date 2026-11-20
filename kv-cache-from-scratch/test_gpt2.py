"""Correctness checks. Run: uv run pytest kv-cache-from-scratch"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from load_weights import load_gpt2
from model import KVCache

PROMPTS = [
    "The cat sat on the",
    "A KV cache stores the keys and values of earlier tokens so that",
    "Hello",
]


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained("gpt2").eval()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_hugging_face(prompt, tokenizer, ours, hf):
    """Our GPT-2's output scores match Hugging Face's GPT-2 within 1e-4."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        our_logits = ours(idx)
        hf_logits = hf(idx, use_cache=False).logits
    max_diff = (our_logits - hf_logits).abs().max().item()
    assert max_diff < 1e-4, f"largest difference {max_diff:.2e}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_text_with_and_without_cache(prompt, tokenizer, ours):
    """Greedy generation gives identical tokens with and without our KV cache."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    no_cache = ours.generate(idx, max_new_tokens=30, use_cache=False)
    with_cache = ours.generate(idx, max_new_tokens=30, use_cache=True)
    assert torch.equal(no_cache, with_cache), (
        f"\nno cache:   {tokenizer.decode(no_cache[0])!r}\nwith cache: {tokenizer.decode(with_cache[0])!r}"
    )


def test_cached_step_matches_full_recompute(tokenizer, ours):
    """Feeding one token with the cache gives the same scores as re-reading the whole text."""
    idx = tokenizer(PROMPTS[1], return_tensors="pt").input_ids
    with torch.no_grad():
        full = ours(idx)[:, -1, :]
        cache = KVCache(ours.config.n_layer)
        ours(idx[:, :-1], cache)  # prefill everything except the last token
        step = ours(idx[:, -1:], cache)[:, -1, :]  # then the last token on its own
    assert torch.allclose(full, step, atol=1e-4)


def test_cache_size_matches_formula(tokenizer, ours):
    """2 (K and V) x 12 layers x 12 heads x 64 numbers x 4 bytes = 73,728 bytes per token."""
    idx = tokenizer(PROMPTS[1], return_tensors="pt").input_ids
    cache = KVCache(ours.config.n_layer)
    with torch.no_grad():
        ours(idx, cache)
    c = ours.config
    per_token = 2 * c.n_layer * c.n_head * (c.n_embd // c.n_head) * 4
    assert per_token == 73_728
    assert len(cache) == idx.shape[1]
    assert cache.nbytes() == per_token * idx.shape[1]

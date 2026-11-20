"""KV cache checks: same output as without the cache, exact maths, expected memory.
Run from gpt2-kv-cache/: uv run pytest"""

import pytest
import torch

from gpt2_kv.kv_cache import KVCache, generate_with_cache

PROMPTS = [
    "The cat sat on the",
    "A KV cache stores the keys and values of earlier tokens so that",
    "Hello",
]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_text_with_and_without_cache(prompt, tokenizer, ours):
    """Greedy generation gives identical tokens with and without the KV cache."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    no_cache = ours.generate(idx, max_new_tokens=30)
    with_cache = generate_with_cache(ours, idx, max_new_tokens=30)
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

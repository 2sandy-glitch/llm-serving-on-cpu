"""Evicting KV cache checks: with enough budget nothing changes, the budget is respected, each policy
keeps the tokens it should, and positions keep counting. Run from gpt2-kv-cache-eviction/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.evicting_kv_cache import POLICIES, EvictingKVCache
from gpt2lab.kv_cache import generate_with_cache
from gpt2lab.load_weights import load_gpt2

PROMPT = "A KV cache stores the keys and values of earlier tokens so that"


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


@pytest.mark.parametrize("policy", POLICIES)
def test_budget_bigger_than_text_changes_nothing(policy, tokenizer, ours):
    idx = tokenizer(PROMPT, return_tensors="pt").input_ids
    cache = EvictingKVCache(ours.config.n_layer, budget=200, policy=policy)
    assert torch.equal(generate_with_cache(ours, idx, 40, cache=cache), generate_with_cache(ours, idx, 40))


@pytest.mark.parametrize("policy", POLICIES[1:])
def test_budget_respected_and_positions_keep_counting(policy, tokenizer, ours):
    idx = tokenizer(PROMPT, return_tensors="pt").input_ids
    cache = EvictingKVCache(ours.config.n_layer, budget=8, policy=policy)
    out = generate_with_cache(ours, idx, 30, cache=cache)
    assert out.shape[1] == idx.shape[1] + 30
    assert all(cache.stored(layer) == 8 for layer in range(ours.config.n_layer))
    assert len(cache) == idx.shape[1] + 29  # tokens seen, not stored
    assert cache.nbytes() == 8 * 73_728


def fake_cache(policy, budget, n=10, weights=None):
    """One layer, one head: n tokens, optional attention weights for observe()."""
    cache = EvictingKVCache(1, budget=budget, policy=policy)
    x = torch.arange(n, dtype=torch.float).view(1, 1, n, 1)
    cache.update(0, x, x)
    w = torch.zeros(1, 1, 1, n) if weights is None else weights
    cache.observe(0, w)
    return cache


def test_recent_keeps_the_last_tokens():
    assert fake_cache("recent", 4).pos[0].flatten().tolist() == [6, 7, 8, 9]


def test_sink_keeps_the_first_tokens_and_the_last():
    cache = EvictingKVCache(1, budget=6, policy="sink", sinks=2)
    x = torch.arange(10, dtype=torch.float).view(1, 1, 10, 1)
    cache.update(0, x, x)
    cache.observe(0, torch.zeros(1, 1, 1, 10))
    assert cache.pos[0].flatten().tolist() == [0, 1, 6, 7, 8, 9]


def test_h2o_keeps_heavy_hitters_and_recent_tokens():
    w = torch.zeros(1, 1, 1, 10)
    w[..., 2], w[..., 5] = 0.6, 0.3  # tokens 2 and 5 got the most attention
    cache = fake_cache("h2o", 4, weights=w)  # 2 recent (8, 9) + 2 heavy hitters
    assert cache.pos[0].flatten().tolist() == [2, 5, 8, 9]
    assert torch.equal(cache.k[0].flatten(), torch.tensor([2.0, 5.0, 8.0, 9.0]))  # K moved with them


def test_random_keeps_the_newest_and_is_repeatable():
    a, b = fake_cache("random", 4), fake_cache("random", 4)
    assert 9 in a.pos[0].flatten().tolist()
    assert torch.equal(a.pos[0], b.pos[0])


def test_heads_choose_their_own_tokens():
    cache = EvictingKVCache(1, budget=2, policy="h2o", recent=1)
    x = torch.zeros(1, 2, 5, 1)
    cache.update(0, x, x)
    w = torch.zeros(1, 2, 1, 5)
    w[0, 0, 0, 1], w[0, 1, 0, 3] = 1.0, 1.0  # head 0 likes token 1, head 1 likes token 3
    cache.observe(0, w)
    assert cache.pos[0][0, 0].tolist() == [1, 4] and cache.pos[0][0, 1].tolist() == [3, 4]

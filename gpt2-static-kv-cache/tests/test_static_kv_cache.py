"""Reserved (static) KV cache checks: same text as the growing cache and as no cache, nothing copied,
memory taken up front. Run from gpt2-static-kv-cache/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.kv_cache import KVCache, generate_with_cache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.static_kv_cache import StaticKVCache

PROMPTS = [
    "The cat sat on the",
    "A KV cache stores the keys and values of earlier tokens so that",
    "Hello",
]
BYTES_PER_TOKEN = 73_728  # 2 (K and V) x 12 layers x 12 heads x 64 numbers x 4 bytes


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_text_as_growing_cache_and_no_cache(prompt, tokenizer, ours):
    """The key test: the reserved cache changes where K and V are stored, not what they are."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    no_cache = ours.generate(idx, max_new_tokens=30)
    growing = generate_with_cache(ours, idx, 30)
    reserved = generate_with_cache(ours, idx, 30, cache=StaticKVCache.for_model(ours))
    assert torch.equal(reserved, growing)
    assert torch.equal(reserved, no_cache)


def test_same_scores_step_by_step(tokenizer, ours):
    """Prefill, then 5 single-token steps: the scores match the growing cache at every step."""
    idx = tokenizer(PROMPTS[1], return_tensors="pt").input_ids
    growing, reserved = KVCache(ours.config.n_layer), StaticKVCache.for_model(ours, max_tokens=64)
    with torch.no_grad():
        a, b = ours(idx, growing), ours(idx, reserved)
        for _ in range(5):
            assert torch.allclose(a, b, atol=1e-5)
            next_id = a[:, -1:].argmax(dim=-1)
            a, b = ours(next_id, growing), ours(next_id, reserved)
    assert len(growing) == len(reserved) == idx.shape[1] + 5


def test_memory_is_reserved_up_front(ours):
    """All the memory is taken at the start and never grows: 1,024 tokens = 72 MiB for GPT-2 small."""
    cache = StaticKVCache.for_model(ours)
    assert cache.max_tokens == 1024
    assert cache.nbytes() == 1024 * BYTES_PER_TOKEN == 72 * 2**20
    assert len(cache) == 0 and cache.used_bytes() == 0
    with torch.no_grad():
        ours(torch.tensor([[464, 3797, 3332]]), cache)  # 3 tokens
    assert cache.nbytes() == 72 * 2**20  # unchanged
    assert cache.used_bytes() == 3 * BYTES_PER_TOKEN


def test_no_copy_of_old_tokens(ours):
    """Each step writes into the same reserved memory: the storage never moves, and the returned K
    is a view into it, not a new tensor."""
    cache = StaticKVCache.for_model(ours, max_tokens=16)
    address = cache.k.data_ptr()
    with torch.no_grad():
        ours(torch.tensor([[464, 3797]]), cache)
        ours(torch.tensor([[3332]]), cache)
    assert cache.k.data_ptr() == address
    k, _ = cache.update(0, torch.zeros(1, 12, 1, 64), torch.zeros(1, 12, 1, 64))
    assert k.data_ptr() == address  # layer 0's slice starts at the very start of the big tensor
    assert k.shape[2] == 4


def test_full_cache_says_so(ours):
    cache = StaticKVCache.for_model(ours, max_tokens=4)
    with torch.no_grad():
        ours(torch.tensor([[464, 3797, 3332, 319]]), cache)  # exactly full
        with pytest.raises(ValueError, match="holds 4 tokens"):
            ours(torch.tensor([[262]]), cache)

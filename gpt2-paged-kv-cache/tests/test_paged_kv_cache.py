"""Paged KV cache checks: same text as the other caches, blocks taken only when needed and returned
when done, requests sharing one pool don't disturb each other. Run from gpt2-paged-kv-cache/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.kv_cache import generate_with_cache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.paged_kv_cache import BlockPool, PagedKVCache

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


@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_text_as_growing_cache(prompt, block_size, tokenizer, ours):
    """Paging changes where K and V live, not what they are."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    pool = BlockPool.for_model(ours, num_blocks=64, block_size=block_size)
    paged = generate_with_cache(ours, idx, 40, cache=PagedKVCache(pool))
    assert torch.equal(paged, generate_with_cache(ours, idx, 40))


def test_blocks_taken_only_when_needed_and_returned(ours):
    pool = BlockPool.for_model(ours, num_blocks=10, block_size=16)
    cache = PagedKVCache(pool)
    assert cache.nbytes() == 0
    with torch.no_grad():
        ours(torch.arange(16).unsqueeze(0), cache)  # exactly one full block
        assert len(cache.blocks) == 1
        ours(torch.tensor([[5]]), cache)  # token 17 needs a second block
        assert len(cache.blocks) == 2 and pool.used_blocks == 2
    assert cache.nbytes() == 2 * pool.block_bytes() == 2 * 16 * 73_728
    cache.free()
    assert pool.used_blocks == 0 and len(cache) == 0


def test_two_requests_share_one_pool_without_mixing(tokenizer, ours):
    """Two requests decoded in turns, one token each, sharing the pool: each gets the same text as
    when run alone, and they never hold the same block."""
    pool = BlockPool.for_model(ours, num_blocks=32, block_size=4)
    ids = [tokenizer(p, return_tensors="pt").input_ids for p in PROMPTS[:2]]
    caches = [PagedKVCache(pool), PagedKVCache(pool)]
    outs = [i.clone() for i in ids]
    with torch.no_grad():
        logits = [ours(i, c) for i, c in zip(ids, caches)]
        for _ in range(25):
            for r in range(2):  # take turns
                next_id = logits[r][:, -1:].argmax(dim=-1)
                outs[r] = torch.cat([outs[r], next_id], dim=1)
                logits[r] = ours(next_id, caches[r])
    assert not set(caches[0].blocks) & set(caches[1].blocks)
    for r in range(2):
        alone = generate_with_cache(ours, ids[r], 25)
        assert torch.equal(outs[r], alone)


def test_freed_blocks_are_reused(ours):
    pool = BlockPool.for_model(ours, num_blocks=4, block_size=8)
    first = PagedKVCache(pool)
    with torch.no_grad():
        ours(torch.arange(20).unsqueeze(0), first)  # 3 blocks
    taken = set(first.blocks)
    first.free()
    second = PagedKVCache(pool)
    with torch.no_grad():
        ours(torch.arange(20).unsqueeze(0), second)
    assert set(second.blocks) == taken


def test_full_pool_says_so(ours):
    pool = BlockPool.for_model(ours, num_blocks=2, block_size=4)
    cache = PagedKVCache(pool)
    with torch.no_grad():
        ours(torch.arange(8).unsqueeze(0), cache)  # fills both blocks
        with pytest.raises(MemoryError, match="2 free blocks|0 free blocks"):
            ours(torch.tensor([[1]]), cache)

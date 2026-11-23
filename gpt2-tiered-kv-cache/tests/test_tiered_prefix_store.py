"""Tiered prefix store checks: entries pushed out of RAM land on disk, come back identical, give the
same text, and both tiers respect their limits. Run from gpt2-tiered-kv-cache/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.kv_cache import KVCache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.prefix_cache import PrefixStore, prefill_with_prefix_cache
from gpt2lab.tiered_prefix_store import TieredPrefixStore


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


def fake(ids, n_layer=2):
    cache = KVCache(n_layer)
    for layer in range(n_layer):
        x = torch.randn(1, 2, len(ids), 4)
        cache.update(layer, x, x + 1)
    return cache


def test_evicted_entry_goes_to_disk_and_comes_back_identical(tmp_path):
    store = TieredPrefixStore(ram_entries=1, disk_dir=tmp_path)
    a = fake([1, 2, 3])
    store.add([1, 2, 3], a)
    store.add([4, 5, 6], fake([4, 5, 6]))  # pushes [1, 2, 3] to disk
    assert list(store.entries) == [(4, 5, 6)] and list(store.on_disk) == [(1, 2, 3)]
    assert len(list(tmp_path.glob("*.pt"))) == 1

    key, n = store.lookup([1, 2, 3, 9])  # found on disk, loaded back into RAM
    assert (key, n, store.last_hit) == ((1, 2, 3), 3, "disk")
    ks, vs = store.entries[(1, 2, 3)]
    assert all(torch.equal(k, a.k[i]) for i, k in enumerate(ks))
    assert all(torch.equal(v, a.v[i]) for i, v in enumerate(vs))
    assert list(store.on_disk) == [(4, 5, 6)]  # loading pushed the other one down


def test_ram_hit_preferred_and_no_duplicates(tmp_path):
    store = TieredPrefixStore(ram_entries=2, disk_dir=tmp_path)
    store.add([1, 2], fake([1, 2]))
    store.lookup([1, 2, 3])
    assert store.last_hit == "ram"
    store.add([1, 2], fake([1, 2]))  # same key again: still one entry, no file
    assert len(store.entries) == 1 and not store.on_disk


def test_disk_tier_limit_deletes_oldest_file(tmp_path):
    store = TieredPrefixStore(ram_entries=1, disk_dir=tmp_path, disk_entries=2)
    for i in range(5):
        store.add([i, i], fake([i, i]))
    assert list(store.on_disk) == [(2, 2), (3, 3)]  # 4 in RAM, 2 and 3 on disk, 0 and 1 deleted
    assert len(list(tmp_path.glob("*.pt"))) == 2
    store.clear_disk()
    assert not list(tmp_path.glob("*.pt"))


def test_look_only_lookup_does_not_load(tmp_path):
    store = TieredPrefixStore(ram_entries=1, disk_dir=tmp_path)
    store.add([1, 2], fake([1, 2]))
    store.add([3, 4], fake([3, 4]))
    assert store.lookup([1, 2, 5], touch=False) == ((1, 2), 2)
    assert list(store.on_disk) == [(1, 2)]  # still on disk


@torch.no_grad()
def test_same_text_after_a_round_trip_through_disk(tmp_path, tokenizer, ours):
    """A prompt pushed to disk and loaded back gives the same scores and text as a plain prefill."""
    store = TieredPrefixStore(ram_entries=1, disk_dir=tmp_path)
    doc = tokenizer("The library opens at nine and closes at six. Members may borrow five books. ").input_ids
    p1 = torch.tensor([doc + tokenizer("Question: When does it open?").input_ids])
    p2 = torch.tensor([tokenizer("Something completely different to fill the RAM tier.").input_ids])
    prefill_with_prefix_cache(ours, p1, store)
    prefill_with_prefix_cache(ours, p2, store)  # p1 goes to disk
    assert tuple(p1[0].tolist()) in store.on_disk

    logits, cache, reused = prefill_with_prefix_cache(ours, p1, store)
    assert store.last_hit == "disk" and reused == p1.shape[1] - 1
    plain_cache = KVCache(ours.config.n_layer)
    plain = ours(p1, plain_cache)
    assert torch.allclose(logits[:, -1], plain[:, -1], atol=1e-4)
    a, b = [], []
    for out, lg, c in ((a, logits, cache), (b, plain, plain_cache)):
        for _ in range(15):
            nxt = lg[:, -1:].argmax(-1)
            out.append(nxt.item())
            lg = ours(nxt, c)
    assert a == b


def test_plain_prefix_store_still_drops():
    store = PrefixStore(max_entries=1)
    store.add([1], fake([1]))
    store.add([2], fake([2]))
    assert list(store.entries) == [(2,)]

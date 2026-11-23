"""Prefix cache checks: same scores and text with and without it, the right number of tokens reused,
stored entries never change, least-recently-used eviction. Run from gpt2-prefix-cache/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.kv_cache import KVCache, generate_with_cache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.prefix_cache import PrefixStore, prefill_with_prefix_cache, shared_prefix_length
from gpt2lab.static_kv_cache import StaticKVCache

DOC = ("The library opens at nine in the morning and closes at six in the evening. Members may borrow "
       "up to five books at a time for three weeks. Late returns cost ten cents per day. ")


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


def ids_of(tokenizer, *parts):
    """Tokenize parts separately and join them, so part boundaries never merge into one token."""
    return torch.tensor([sum((tokenizer(p).input_ids for p in parts), [])])


def generate_after_prefill(model, logits, cache, n):
    out = []
    for _ in range(n):
        next_id = logits[:, -1:].argmax(dim=-1)
        out.append(next_id.item())
        logits = model(next_id, cache)
    return out


@torch.no_grad()
def test_four_cases_same_scores_and_text(tokenizer, ours):
    """Cold, identical, extended, changed beginning, same document + new question: each gives the
    same last-token scores and the same 20 tokens as a plain prefill, and reuses the expected count."""
    store = PrefixStore()
    q1 = "Question: When does the library open?"
    cases = [
        ("cold", ids_of(tokenizer, DOC, q1), 0),
        ("identical", ids_of(tokenizer, DOC, q1), None),  # all but the last token
        ("extended", ids_of(tokenizer, DOC, q1, " Answer briefly."), "q1"),
        ("changed beginning", ids_of(tokenizer, "Note: ", DOC, q1), 0),
        ("new question", ids_of(tokenizer, DOC, "Question: How many books can I borrow?"), "doc"),
    ]
    doc_len = len(tokenizer(DOC).input_ids)
    q1_len = doc_len + len(tokenizer(q1).input_ids)
    for name, idx, expected in cases:
        logits, cache, reused = prefill_with_prefix_cache(ours, idx, store)
        plain_cache = KVCache(ours.config.n_layer)
        plain = ours(idx, plain_cache)
        assert torch.allclose(logits[:, -1], plain[:, -1], atol=1e-4), name
        assert generate_after_prefill(ours, logits, cache, 20) == generate_after_prefill(ours, plain, plain_cache, 20), name
        # "doc": the document plus "Question:", which both questions start with.
        shared_q = len(tokenizer("Question:").input_ids)
        want = {None: idx.shape[1] - 1, "q1": q1_len, "doc": doc_len + shared_q}.get(expected, expected)
        assert reused == want, f"{name}: reused {reused}, expected {want}"


@torch.no_grad()
def test_works_with_the_reserved_cache(tokenizer, ours):
    store = PrefixStore()
    idx = ids_of(tokenizer, DOC, "Question: What do late returns cost?")
    prefill_with_prefix_cache(ours, idx, store)  # fills the store
    logits, cache, reused = prefill_with_prefix_cache(ours, idx, store, cache=StaticKVCache.for_model(ours))
    assert reused == idx.shape[1] - 1 and len(cache) == idx.shape[1]
    assert torch.allclose(logits[:, -1], ours(idx)[:, -1], atol=1e-4)


@torch.no_grad()
def test_stored_entries_never_change(tokenizer, ours):
    """Generating after a reuse must not write into the stored K and V."""
    store = PrefixStore()
    idx = ids_of(tokenizer, DOC, "Question: When does it close?")
    prefill_with_prefix_cache(ours, idx, store)
    before = [k.clone() for k in store.entries[tuple(idx[0].tolist())][0]]
    logits, cache, _ = prefill_with_prefix_cache(ours, idx, store)
    generate_after_prefill(ours, logits, cache, 10)
    after = store.entries[tuple(idx[0].tolist())][0]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_least_recently_used_is_dropped():
    store = PrefixStore(max_entries=2)
    fake = KVCache(1)
    fake.update(0, torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))
    store.add([1, 2, 3], fake)
    store.add([4, 5, 6], fake)
    store.lookup([1, 2, 9])  # uses [1, 2, 3], so [4, 5, 6] is now the least recently used
    store.add([7, 8, 9], fake)
    assert list(store.entries) == [(1, 2, 3), (7, 8, 9)]


def test_shared_prefix_length():
    assert shared_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2
    assert shared_prefix_length([1, 2], [1, 2, 3]) == 2
    assert shared_prefix_length([9], [1]) == 0


def test_appended_text_can_change_the_last_token(tokenizer):
    """GPT-2's tokenizer can merge characters across the point where text is appended, so the
    shared token prefix can be one shorter than the shared text suggests."""
    a = tokenizer("The answer is 4").input_ids
    b = tokenizer("The answer is 42").input_ids
    assert len(a) == len(b)  # " 4" became " 42": still one token, but a different one
    assert shared_prefix_length(a, b) == len(a) - 1

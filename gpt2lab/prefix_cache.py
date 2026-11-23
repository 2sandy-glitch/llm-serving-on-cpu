"""Prefix caching: reuse the saved Keys and Values of a prompt start that was seen before.

Many requests begin with the same text (a system prompt, a shared document). Their K and V for
that shared start are identical, because a token's K and V depend only on the tokens before it.
So keep the K and V of past prompts; for a new prompt, find the longest token prefix it shares with
a stored one, load that many tokens' K and V into a fresh cache, and run the model only on the rest.

  new prompt:   [ shared start ................ | new part ]
                  loaded from the store            computed now

The last prompt token is always computed, even if the whole prompt is stored: its scores are what
pick the first new token, and scores are not stored.

This is the simple version: a list of stored prompts, searched one by one, each keeping its own
copy of K and V. Real engines (SGLang's RadixAttention, vLLM's automatic prefix caching) keep a
tree of token blocks so shared starts are stored once and found quickly.

Works with any cache that has update() (KVCache, StaticKVCache, QuantizedKVCache): the shared
tokens are loaded by calling update() with them, as if they had just been computed.
"""

from collections import OrderedDict

import torch

from gpt2lab.kv_cache import KVCache


class PrefixStore:
    """Saved K and V of earlier prompts. Holds at most max_entries; the least recently used goes first."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        # prompt token ids (tuple) -> (per-layer K list, per-layer V list), each (1, H, T, D)
        self.entries: OrderedDict[tuple, tuple[list, list]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, ids: list[int], touch: bool = True) -> tuple[tuple | None, int]:
        """The stored prompt sharing the longest token prefix with `ids`, and that length.
        touch=False only looks (a router checking where a prompt is cached), without counting
        it as a use."""
        best, best_len = None, 0
        for key in self.entries:
            n = shared_prefix_length(key, ids)
            if n > best_len:
                best, best_len = key, n
        if best is not None and touch:
            self.entries.move_to_end(best)  # used just now
        return best, best_len

    def add(self, ids: list[int], cache: KVCache) -> None:
        """Save the K and V of `ids` from a cache whose first len(ids) tokens are exactly `ids`."""
        key = tuple(ids)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        n = len(ids)

        def keep(t):  # copy when keeping only the start, so the rest of the prompt can be freed
            return t if t.shape[2] == n else t[:, :, :n].clone()

        self.entries[key] = ([keep(k) for k in cache.k], [keep(v) for v in cache.v])
        self._make_room()

    def _make_room(self) -> None:
        while len(self.entries) > self.max_entries:
            key, value = self.entries.popitem(last=False)  # the least recently used
            self._evict(key, value)

    def _evict(self, key: tuple, value: tuple[list, list]) -> None:
        """What happens to an entry pushed out of the store: here it is simply dropped.
        TieredPrefixStore overrides this to save it to disk instead."""

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for ks, vs in self.entries.values() for t in ks + vs)


def shared_prefix_length(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def prefill_with_prefix_cache(model, idx: torch.Tensor, store: PrefixStore, cache=None, store_upto: int | None = None):
    """Prefill prompt idx (1, T), reusing stored K and V for its longest stored prefix.

    Returns (scores for the prompt's new tokens, the filled cache, how many tokens were reused).
    The prompt is then saved in the store for later requests: all of it, or only its first
    `store_upto` tokens (a "cache breakpoint": keep the shared document, not each question).
    """
    ids = idx[0].tolist()
    if cache is None:
        cache = KVCache(model.config.n_layer)
    key, shared = store.lookup(ids)
    reuse = min(shared, len(ids) - 1)  # always compute the last token: its scores pick the next one
    if reuse:
        ks, vs = store.entries[key]
        for layer in range(model.config.n_layer):  # load the shared start, as if just computed
            cache.update(layer, ks[layer][:, :, :reuse], vs[layer][:, :, :reuse])
    logits = model(idx[:, reuse:], cache)
    if isinstance(cache, KVCache):  # store needs plain K and V tensors: only the growing cache has them
        store.add(ids[:store_upto] if store_upto else ids, cache)
    return logits, cache, reuse

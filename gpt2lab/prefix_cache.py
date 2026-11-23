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

    def lookup(self, ids: list[int]) -> tuple[tuple | None, int]:
        """The stored prompt sharing the longest token prefix with `ids`, and that length."""
        best, best_len = None, 0
        for key in self.entries:
            n = shared_prefix_length(key, ids)
            if n > best_len:
                best, best_len = key, n
        if best is not None:
            self.entries.move_to_end(best)  # used just now
        return best, best_len

    def add(self, ids: list[int], cache: KVCache) -> None:
        """Save the K and V of prompt `ids` from a cache that holds exactly that prompt."""
        key = tuple(ids)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        self.entries[key] = ([k for k in cache.k], [v for v in cache.v])
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)  # drop the least recently used

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
def prefill_with_prefix_cache(model, idx: torch.Tensor, store: PrefixStore, cache=None):
    """Prefill prompt idx (1, T), reusing stored K and V for its longest stored prefix.

    Returns (scores for the prompt's new tokens, the filled cache, how many tokens were reused).
    The prompt itself is then saved in the store for later requests.
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
        store.add(ids, cache)
    return logits, cache, reuse

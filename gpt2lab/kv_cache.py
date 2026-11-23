"""My KV cache for GPT-2: everything the cache needs, kept apart from the model code.

Why it works: once a token has been processed, its Keys (K) and Values (V) never change, and only
the newest token needs a Query (Q). So we keep every layer's K and V, and on each new step compute
Q, K and V for the new token only.

Two pieces live here:
  KVCache               stores K and V for every layer and grows by one token per step
  generate_with_cache   greedy generation: read the prompt once (prefill), then feed only the
                        newest token (decode)

The model (model.py) only needs three small hooks to use it, each marked "KV cache hook":
  1. attention adds the new K and V to the cache and attends over everything cached
  2. the causal mask is shifted by the number of cached tokens
  3. positions continue after the cached tokens instead of restarting at 0

Shapes: K and V are stored per layer as (B, H, tokens so far, D):
  B = batch size, H = heads (12), D = size of one head (64).
"""

import torch


class KVCache:
    """Keeps every layer's Keys and Values from earlier steps so they are never recomputed."""

    def __init__(self, n_layer: int):
        # One slot per layer. Each layer only ever reads and writes its own slot.
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    def __len__(self) -> int:
        """How many tokens are cached. Read it BEFORE a forward pass: during a pass, layer 0
        is updated first, so later layers must count from their own K instead."""
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the new tokens' K and V to this layer's slot, and return K and V for ALL tokens so far."""
        if self.k[layer] is not None:
            # dim 2 is the "tokens" dimension. torch.cat copies the whole cache each step;
            # real serving engines reserve the memory once and fill it in place.
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def nbytes(self) -> int:
        """Memory the cache really uses, in bytes (GPT-2 small, float32: 73,728 per token)."""
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


@torch.no_grad()
def generate_with_cache(model, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy generation using the KV cache. Gives the same tokens as model.generate(), faster.

    model: a GPT2 from model.py. idx: prompt token ids (B, T).
    """
    cache = KVCache(model.config.n_layer)
    logits = model(idx, cache)  # prefill: the whole prompt goes in once and fills the cache
    for step in range(max_new_tokens):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if step < max_new_tokens - 1:  # the last token needs no further pass
            logits = model(next_id, cache)  # decode: only the 1 new token goes in
    return idx

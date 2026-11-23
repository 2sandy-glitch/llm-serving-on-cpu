"""A reserved ("static" or "preallocated") KV cache: grab all the memory once, then fill it in place.

Why: the first KV cache (kv_cache.py) grows with torch.cat, which builds a new, bigger tensor and
copies every cached token into it on every step. Adding token 400 copies all 400 tokens, in every
layer. Here the full space for max_tokens is reserved at the start, and each new token's K and V
are written into the next free slot. Nothing is ever copied, and the tensors never change shape.

The trade-off: the memory for max_tokens is taken up front, even if the answer turns out short.
For GPT-2 small at 1,024 tokens that is 1,024 x 73,728 bytes = 72 MiB from the first step.
Paged caches (vLLM) exist to fix exactly that waste.

Same interface as KVCache (update, len, nbytes), so model.py needs no changes: the three
"KV cache hook" lines already work with it. Hugging Face calls this idea StaticCache.

Shapes: K and V for all layers live in one tensor each, (layers, B, H, max_tokens, D):
  B = batch size, H = heads, D = size of one head.
"""

import torch


class StaticKVCache:
    """Reserves K and V for max_tokens tokens up front and fills them in place, one step at a time."""

    def __init__(self, n_layer: int, n_head: int, head_dim: int, max_tokens: int, batch_size: int = 1,
                 dtype: torch.dtype = torch.float32, device: torch.device | str | None = None):
        shape = (n_layer, batch_size, n_head, max_tokens, head_dim)
        # All the memory, taken now. torch.zeros (not empty) so unused slots are clearly blank.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_tokens = max_tokens
        # Where the next token goes, per layer. Each layer advances its own position during a pass.
        self.pos = [0] * n_layer

    @classmethod
    def for_model(cls, model, max_tokens: int | None = None, batch_size: int = 1) -> "StaticKVCache":
        """A cache sized for this model. max_tokens defaults to the longest text it can read (1,024)."""
        c = model.config
        p = model.lm_head.weight
        return cls(c.n_layer, c.n_head, c.n_embd // c.n_head, max_tokens or c.block_size, batch_size,
                   dtype=p.dtype, device=p.device)

    def __len__(self) -> int:
        """How many tokens are cached. Like KVCache, read it BEFORE a forward pass: during a pass,
        layer 0 moves forward first."""
        return self.pos[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new tokens' K and V into the next free slots, and return K and V for ALL tokens
        so far. The result is a view into the reserved memory, not a copy."""
        start = self.pos[layer]
        end = start + k.shape[2]  # dim 2 is the "tokens" dimension
        if end > self.max_tokens:
            raise ValueError(f"cache holds {self.max_tokens} tokens; this step needs {end}")
        self.k[layer, :, :, start:end] = k  # copy only the NEW tokens in (usually just 1)
        self.v[layer, :, :, start:end] = v
        self.pos[layer] = end
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def nbytes(self) -> int:
        """Memory the cache has reserved, in bytes. Fixed from the start, however much is used."""
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    def used_bytes(self) -> int:
        """Bytes actually holding cached tokens (the rest is reserved but still empty)."""
        return self.nbytes() * len(self) // self.max_tokens

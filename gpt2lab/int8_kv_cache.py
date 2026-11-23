"""A quantized KV cache: K and V stored as 8-bit integers with scales instead of 32-bit floats.

Simply: instead of writing every number with full precision (4 bytes), write a whole number from
-127 to 127 (1 byte) plus a "scale" that says how big one step is. Reading back:
value ~= int8 x scale. About 4x less memory per token, slightly fuzzier numbers.

The important choice is which numbers share one scale:
  per_token    each token's numbers (one head, 64 of them) share one scale. Easy to stream: every
               token is quantized on its own, the moment it arrives.
  per_channel  each of the 64 channels gets its own scale, shared by a group of tokens (32 here).
               Keys have a few "outlier" channels with much bigger numbers; a per-token scale must
               stretch to fit them, which wastes precision for all the other channels. Per channel
               avoids that. The newest tokens wait in float32 (the "residual") until a group of 32
               is full, then the group is quantized together. This is the KIVI recipe (Liu et al.
               2024, arXiv 2402.02750: keys per channel, values per token).
  none         plain float32, to switch quantization off for K or V separately.

Memory is reserved once for max_tokens, like StaticKVCache. Same interface as the other caches
(update, len, nbytes), so model.py needs no change. update() returns float K and V (dequantized),
because the model's attention is ordinary float maths; real engines fuse the dequantization into
the attention kernel instead.

bits=4 is allowed for quality experiments, but it is still stored one value per byte (not packed),
so it saves no extra memory here.
"""

import torch

MODES = ("none", "per_token", "per_channel")


def quantize(x: torch.Tensor, dim: int, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization: one scale per slice along `dim` (the numbers sharing a scale).
    Returns (int8 values in [-qmax, qmax], float32 scales with `dim` kept as size 1)."""
    qmax = 2 ** (bits - 1) - 1  # 127 for 8 bits, 7 for 4 bits
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.float()


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


class _Store:
    """K or V for every layer, kept in one of the MODES. Shapes: (layers, B, H, max_tokens, D)."""

    def __init__(self, mode: str, n_layer: int, batch_size: int, n_head: int, max_tokens: int,
                 head_dim: int, bits: int, group_size: int, device):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.bits, self.G = mode, bits, group_size
        shape = (n_layer, batch_size, n_head, max_tokens, head_dim)
        if mode == "none":
            self.data = torch.zeros(shape, device=device)
            return
        self.q = torch.zeros(shape, dtype=torch.int8, device=device)
        if mode == "per_token":
            self.scale = torch.zeros(shape[:-1] + (1,), device=device)  # one per token
        else:  # per_channel: one scale per channel for each group of G tokens, plus a float residual
            self.scale = torch.zeros((n_layer, batch_size, n_head, max_tokens // group_size, head_dim), device=device)
            self.residual = torch.zeros((n_layer, batch_size, n_head, group_size, head_dim), device=device)
            self.res_len = [0] * n_layer  # tokens waiting in the residual
            self.q_len = [0] * n_layer  # tokens already quantized (always a multiple of G)

    def write(self, layer: int, start: int, x: torch.Tensor) -> None:
        end = start + x.shape[2]
        if self.mode == "none":
            self.data[layer, :, :, start:end] = x
        elif self.mode == "per_token":
            q, s = quantize(x, dim=-1, bits=self.bits)  # scale over the 64 numbers of each token
            self.q[layer, :, :, start:end] = q
            self.scale[layer, :, :, start:end] = s
        else:
            self._write_per_channel(layer, x)

    def _write_per_channel(self, layer: int, x: torch.Tensor) -> None:
        G, r, done = self.G, self.res_len[layer], self.q_len[layer]
        pending = torch.cat([self.residual[layer, :, :, :r], x], dim=2)  # waiting tokens + new ones
        full = pending.shape[2] // G  # how many complete groups we can quantize now
        if full:
            B, H, _, D = pending.shape
            groups = pending[:, :, : full * G].reshape(B, H, full, G, D)
            q, s = quantize(groups, dim=3, bits=self.bits)  # scale over the G tokens of each channel
            self.q[layer, :, :, done : done + full * G] = q.reshape(B, H, full * G, D)
            self.scale[layer, :, :, done // G : done // G + full] = s.squeeze(3)
            self.q_len[layer] = done + full * G
        rest = pending[:, :, full * G :]
        self.residual[layer, :, :, : rest.shape[2]] = rest
        self.res_len[layer] = rest.shape[2]

    def read(self, layer: int, end: int) -> torch.Tensor:
        """Float values for tokens [0, end) of this layer."""
        if self.mode == "none":
            return self.data[layer, :, :, :end]
        if self.mode == "per_token":
            return dequantize(self.q[layer, :, :, :end], self.scale[layer, :, :, :end])
        done, G = self.q_len[layer], self.G
        q = self.q[layer, :, :, :done]
        B, H, _, D = q.shape
        old = dequantize(q.reshape(B, H, done // G, G, D), self.scale[layer, :, :, : done // G].unsqueeze(3))
        return torch.cat([old.reshape(B, H, done, D), self.residual[layer, :, :, : self.res_len[layer]]], dim=2)

    def tensors(self) -> list[torch.Tensor]:
        if self.mode == "none":
            return [self.data]
        return [self.q, self.scale] + ([self.residual] if self.mode == "per_channel" else [])


class QuantizedKVCache:
    """Reserved KV cache that keeps K and V as int8 + scales (or float32, per tensor)."""

    def __init__(self, n_layer: int, n_head: int, head_dim: int, max_tokens: int, batch_size: int = 1,
                 key_mode: str = "per_token", value_mode: str = "per_token", bits: int = 8,
                 group_size: int = 32, device=None):
        max_tokens = -(-max_tokens // group_size) * group_size  # round up to whole groups
        args = (n_layer, batch_size, n_head, max_tokens, head_dim, bits, group_size, device)
        self.k = _Store(key_mode, *args)
        self.v = _Store(value_mode, *args)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.pos = [0] * n_layer

    @classmethod
    def for_model(cls, model, max_tokens: int | None = None, batch_size: int = 1, **kwargs) -> "QuantizedKVCache":
        c = model.config
        return cls(c.n_layer, c.n_head, c.n_embd // c.n_head, max_tokens or c.block_size, batch_size,
                   device=model.lm_head.weight.device, **kwargs)

    def __len__(self) -> int:
        """Tokens cached. Read it BEFORE a forward pass (layer 0 moves first during a pass)."""
        return self.pos[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store the new tokens' K and V (quantized), return float K and V for ALL tokens so far."""
        start = self.pos[layer]
        end = start + k.shape[2]
        if end > self.max_tokens:
            raise ValueError(f"cache holds {self.max_tokens} tokens; this step needs {end}")
        self.k.write(layer, start, k)
        self.v.write(layer, start, v)
        self.pos[layer] = end
        return self.k.read(layer, end), self.v.read(layer, end)

    def nbytes(self) -> int:
        """All memory the cache has reserved, in bytes."""
        return sum(t.numel() * t.element_size() for t in self.k.tensors() + self.v.tensors())

    def bytes_per_token(self) -> float:
        """Reserved bytes per token of one sequence, not counting the fixed float residual
        (per_channel only)."""
        fixed = sum(s.residual.numel() * 4 for s in (self.k, self.v) if s.mode == "per_channel")
        return (self.nbytes() - fixed) / (self.max_tokens * self.batch_size)

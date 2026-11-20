"""GPT-2 small written from scratch in plain PyTorch (nanoGPT style).

The KV cache lives in kv_cache.py. This file only has three small hooks where the cache plugs in;
search for "KV cache hook" to see all three.

Shape names used in comments:
  B = batch size, T = number of tokens, C = embedding size (768),
  H = number of heads (12), D = size of one head (C / H = 64).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_kv.kv_cache import KVCache


@dataclass
class GPT2Config:
    vocab_size: int = 50257  # number of different tokens
    block_size: int = 1024  # longest text the model can read
    n_layer: int = 12  # transformer blocks stacked on top of each other
    n_head: int = 12  # attention heads per block
    n_embd: int = 768  # numbers used to describe one token
    layer_norm_eps: float = 1e-5


class CausalSelfAttention(nn.Module):
    """Each token looks back at earlier tokens and mixes in what it finds useful."""

    def __init__(self, config: GPT2Config, layer: int):
        super().__init__()
        self.layer = layer  # which block this is, so it finds its own slot in the cache
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # One layer makes Q, K and V together: 768 -> 3 * 768.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Mixes the heads' results back together: 768 -> 768.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        # x holds only the NEW tokens. With a cache, that is the whole prompt at first
        # (prefill), then just 1 token per step (decode).
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # each (B, T, C)

        # Split 768 into 12 heads of 64: (B, T, C) -> (B, H, T, D).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache hook 1/3: add the new K and V to this layer's cache, then attend over all of
        # them. Q is never cached: only the new tokens ask questions.
        if cache is not None:
            k, v = cache.update(self.layer, k, v)  # now (B, H, past + T, D)
        # Count earlier tokens from THIS layer's K. (len(cache) reads layer 0, which has
        # already added this step's tokens by the time later layers run.) Without a cache, 0.
        past = k.shape[2] - T

        # How well each new Query matches every Key: (B, H, T, past + T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: a token may not look at tokens after it.
        # KV cache hook 2/3: new token i sits at position past + i, so the mask is shifted by
        # `past`. During decode (T = 1) nothing is in the future, so it blocks nothing.
        future = torch.ones(T, past + T, dtype=torch.bool, device=x.device).triu(diagonal=past + 1)
        scores = scores.masked_fill(future, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # each row sums to 1
        out = weights @ v  # weighted mix of Values: (B, H, T, D)

        # Join the heads back together: (B, H, T, D) -> (B, T, C).
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    """Each token thinks on its own: widen to 3072, apply GELU, shrink back to 768."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT-2 uses the tanh approximation of GELU ("gelu_new").
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """One transformer block: look at other tokens, then think. Each step adds to x (residual)."""

    def __init__(self, config: GPT2Config, layer: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config, layer)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token -> vector
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position -> vector
                h=nn.ModuleList([Block(config, layer) for layer in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps),
            )
        )
        # Turns a vector back into a score for every token in the vocabulary.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # GPT-2 reuses the token embedding here ("tied weights").
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """idx: token ids (B, T). Returns scores for the next token at every position (B, T, vocab).

        With a cache, idx holds only the NEW tokens; earlier ones are already in the cache.
        """
        B, T = idx.shape
        # KV cache hook 3/3: positions continue after the cached tokens.
        # Starting again at 0 here is the classic bug.
        past = 0 if cache is None else len(cache)
        if past + T > self.config.block_size:
            raise ValueError(f"text is {past + T} tokens; GPT-2 reads at most {self.config.block_size}")
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, cache)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation WITHOUT a cache: re-reads the whole text at every step (the slow way).
        For the fast way, see generate_with_cache() in kv_cache.py."""
        for _ in range(max_new_tokens):
            logits = self(idx)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

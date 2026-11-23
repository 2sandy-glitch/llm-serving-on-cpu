"""A KV cache with a fixed budget: when it is full, drop the tokens that matter least.

Every cache so far keeps every token, so memory grows with the text. Here the cache holds at most
`budget` tokens per head; after each step, anything beyond that is dropped. Which tokens to keep is
the whole question. Policies (each gives every stored token a priority; the top `budget` stay):

  recent   the last `budget` tokens: a sliding window. Simplest.
  sink     the first `sinks` tokens plus the most recent ones. StreamingLLM (Xiao et al. 2023,
           arXiv 2309.17453) found models put a lot of attention on the very first tokens ("attention
           sinks"); dropping them breaks the model.
  h2o      the tokens that have received the most attention so far ("heavy hitters"), plus the
           most recent half of the budget. H2O (Zhang et al. 2023, arXiv 2306.14048).
  random   the newest token plus random others: a control, to show the rules beat luck.
  full     never drop anything (the normal cache), for comparison.

Each head chooses its own tokens, so different heads can keep different tokens; the count per head
is always the same, so K and V stay one tensor of shape (B, H, budget, D).

GPT-2 adds the position to each token before computing K, so dropped tokens leave no gap to fix.
But positions must keep counting the tokens SEEN, not the tokens stored: __len__ returns the number
seen, which is what model.py uses for the next position.

H2O needs to know how much attention each cached token received. model.py calls observe() with the
attention weights if the cache has that method (an optional hook; other caches don't have it).
"""

import torch

POLICIES = ("full", "recent", "sink", "h2o", "random")


class EvictingKVCache:
    def __init__(self, n_layer: int, budget: int, policy: str = "h2o", recent: int | None = None,
                 sinks: int = 4, seed: int = 0):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.budget, self.policy, self.sinks = budget, policy, sinks
        self.recent = budget // 2 if recent is None else recent  # h2o: share of the budget kept for recent tokens
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer
        self.score: list[torch.Tensor | None] = [None] * n_layer  # attention received, (B, H, stored)
        self.pos: list[torch.Tensor | None] = [None] * n_layer  # position of each stored token, (B, H, stored)
        self.seen = [0] * n_layer
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        """Tokens SEEN so far (not stored), so new tokens get the right position."""
        return self.seen[0]

    def stored(self, layer: int = 0) -> int:
        return 0 if self.k[layer] is None else self.k[layer].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, T, _ = k.shape
        start = self.seen[layer]
        pos = torch.arange(start, start + T).expand(B, H, T)
        zeros = torch.zeros(B, H, T)
        if self.k[layer] is None:
            self.k[layer], self.v[layer], self.score[layer], self.pos[layer] = k, v, zeros, pos
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
            self.score[layer] = torch.cat([self.score[layer], zeros], dim=2)
            self.pos[layer] = torch.cat([self.pos[layer], pos], dim=2)
        self.seen[layer] = start + T
        return self.k[layer], self.v[layer]

    def observe(self, layer: int, weights: torch.Tensor) -> None:
        """Called by the model after attention. weights: (B, H, new tokens, stored tokens).
        Adds up the attention each stored token received, then drops tokens over the budget."""
        self.score[layer] += weights.sum(dim=2).float()
        if self.policy == "full" or self.stored(layer) <= self.budget:
            return
        keep = self._choose(layer).sort(dim=-1).values  # (B, H, budget), kept in position order
        D = self.k[layer].shape[-1]
        idx = keep.unsqueeze(-1).expand(-1, -1, -1, D)
        self.k[layer] = self.k[layer].gather(2, idx)
        self.v[layer] = self.v[layer].gather(2, idx)
        self.score[layer] = self.score[layer].gather(2, keep)
        self.pos[layer] = self.pos[layer].gather(2, keep)

    def _choose(self, layer: int) -> torch.Tensor:
        """Indices of the `budget` stored tokens to keep, per (batch row, head)."""
        pos, score = self.pos[layer].float(), self.score[layer]
        newest = self.seen[layer] - 1
        big = 1e9
        if self.policy == "recent":
            priority = pos
        elif self.policy == "sink":
            priority = pos + big * (pos < self.sinks)
        elif self.policy == "h2o":
            priority = score + big * (pos > newest - self.recent)
        else:  # random
            priority = torch.rand(pos.shape, generator=self.generator) + big * (pos == newest)
        return priority.topk(self.budget, dim=-1).indices

    def nbytes(self) -> int:
        """Bytes of K and V stored (scores and positions are bookkeeping, counted separately)."""
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)

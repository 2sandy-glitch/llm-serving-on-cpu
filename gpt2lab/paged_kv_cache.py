"""A paged KV cache: memory handed out in small blocks as a request needs it (vLLM's PagedAttention idea).

The reserved cache (static_kv_cache.py) takes room for the longest possible text up front: 72 MiB per
request for GPT-2's 1,024 tokens, even if the answer is 20 tokens long. With many users, most of
that memory sits empty. Paging fixes it, the way an operating system manages memory:

  - one shared POOL of fixed-size blocks (16 tokens each by default), reserved once for the server;
  - each request has a BLOCK TABLE: the list of pool blocks holding its tokens, in order;
  - a new block is taken from the pool only when the request's last block is full;
  - when a request ends, its blocks go back to the pool for the next request.

So a request wastes at most one partly filled block, and blocks of different requests can sit
anywhere in the pool. To attend, the request's blocks are gathered back into token order. vLLM's
attention kernel reads the blocks directly instead; here the gather is a separate copy, which costs
time (measured in the benchmark).

Same interface as the other caches (update, len, nbytes), so model.py needs no change.
"""

import torch


class BlockPool:
    """All K and V memory for every request, as num_blocks blocks of block_size tokens.
    Shapes: (layers, H, num_blocks, block_size, D). Heads come BEFORE blocks on purpose: gathering a
    request's blocks then gives (H, blocks, block_size, D), already in token order per head, so it
    needs one copy. With blocks first, a second copy is needed to move heads to the front."""

    def __init__(self, n_layer: int, n_head: int, head_dim: int, num_blocks: int, block_size: int = 16,
                 dtype: torch.dtype = torch.float32, device=None):
        shape = (n_layer, n_head, num_blocks, block_size, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks - 1, -1, -1))  # a stack: pop() hands out block 0 first

    @classmethod
    def for_model(cls, model, num_blocks: int, block_size: int = 16) -> "BlockPool":
        c = model.config
        p = model.lm_head.weight
        return cls(c.n_layer, c.n_head, c.n_embd // c.n_head, num_blocks, block_size, dtype=p.dtype, device=p.device)

    def allocate(self, n: int) -> list[int]:
        if n > len(self.free_blocks):
            raise MemoryError(f"pool has {len(self.free_blocks)} free blocks; {n} needed")
        return [self.free_blocks.pop() for _ in range(n)]

    def release(self, blocks: list[int]) -> None:
        self.free_blocks.extend(reversed(blocks))

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def block_bytes(self) -> int:
        """Bytes of one block across all layers, K and V."""
        per = self.k[:, :, 0].numel() * self.k.element_size()
        return 2 * per


class PagedKVCache:
    """One request's view of the pool: its block table and how many tokens it has written."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.blocks: list[int] = []  # the block table: pool block ids in token order
        self.index = torch.empty(0, dtype=torch.long)  # the same, as a tensor for gathering
        self.pos = [0] * pool.k.shape[0]

    def __len__(self) -> int:
        """Tokens cached. Read it BEFORE a forward pass (layer 0 moves first during a pass)."""
        return self.pos[0]

    def _ensure_room(self, tokens: int) -> None:
        need = -(-tokens // self.pool.block_size) - len(self.blocks)  # blocks still missing
        if need > 0:
            self.blocks += self.pool.allocate(need)
            self.index = torch.tensor(self.blocks, dtype=torch.long)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new tokens into this request's blocks (taking new blocks when needed), and
        return K and V for ALL its tokens, gathered back into order."""
        if k.shape[0] != 1:
            raise ValueError("a PagedKVCache holds one request (batch size 1)")
        bs = self.pool.block_size
        start = self.pos[layer]
        end = start + k.shape[2]
        if layer == 0:  # every layer shares one block table; layer 0 runs first in each pass
            self._ensure_room(end)
        t = start
        while t < end:  # copy the new tokens in, block by block
            block, offset = divmod(t, bs)
            n = min(bs - offset, end - t)
            b = self.blocks[block]
            self.pool.k[layer, :, b, offset : offset + n] = k[0, :, t - start : t - start + n]
            self.pool.v[layer, :, b, offset : offset + n] = v[0, :, t - start : t - start + n]
            t += n
        self.pos[layer] = end
        return self._gather(self.pool.k[layer], end), self._gather(self.pool.v[layer], end)

    def _gather(self, blocks: torch.Tensor, end: int) -> torch.Tensor:
        """(H, num_blocks, bs, D) pool slice -> (1, H, end, D) for this request, in token order."""
        used = -(-end // self.pool.block_size)
        mine = blocks[:, self.index[:used]]  # (H, used, bs, D): the one copy
        H, D = mine.shape[0], mine.shape[3]
        return mine.reshape(1, H, used * self.pool.block_size, D)[:, :, :end]  # a view, no copy

    def free(self) -> None:
        """The request is done: give its blocks back to the pool."""
        self.pool.release(self.blocks)
        self.blocks = []
        self.index = torch.empty(0, dtype=torch.long)
        self.pos = [0] * len(self.pos)

    def nbytes(self) -> int:
        """Memory this request holds: its blocks, including the unused part of the last one."""
        return len(self.blocks) * self.pool.block_bytes()

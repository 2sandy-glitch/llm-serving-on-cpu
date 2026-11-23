"""Tiered prefix store: when RAM is full, move cached prompts to disk instead of throwing them away.

The prefix store (prefix_cache.py) keeps a few prompts' K and V in RAM and DELETES the least recently
used one when it's full. If that prompt comes back, it must be recomputed from scratch. But reading
saved K and V from a fast disk can be much quicker than recomputing them, so keep a second, bigger,
slower tier:

  RAM tier    a few prompts, instant to use          (least recently used moves down to disk)
  disk tier   many more prompts, one file each       (least recently used is deleted)

A lookup checks RAM first, then disk. A disk hit loads the file back into RAM (and may push another
entry down to disk). This is the idea behind LMCache, NVIDIA Dynamo's KV block manager and
"CPU/disk offloading" in serving engines.

Honest disk timing: Linux keeps recently written files in its page cache (RAM), so a file written a
moment ago would be read back from memory, not disk. After writing each file, it is flushed to disk
and the OS is told to drop it from the page cache (posix_fadvise DONTNEED; no root needed), so later
reads really come from the disk. drop_page_cache=False turns this off.
"""

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import torch

from gpt2lab.kv_cache import KVCache
from gpt2lab.prefix_cache import PrefixStore, shared_prefix_length


class TieredPrefixStore(PrefixStore):
    """A PrefixStore whose evicted entries go to files in `disk_dir` instead of being deleted."""

    def __init__(self, ram_entries: int, disk_dir: str | Path, disk_entries: int = 64, drop_page_cache: bool = True):
        super().__init__(max_entries=ram_entries)
        self.disk_dir = Path(disk_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.disk_entries = disk_entries
        self.drop_page_cache = drop_page_cache
        self.on_disk: OrderedDict[tuple, Path] = OrderedDict()  # key -> file, least recently used first
        self.last_hit = None  # "ram", "disk" or None: where the last lookup found its prefix

    def lookup(self, ids: list[int], touch: bool = True) -> tuple[tuple | None, int]:
        """Longest shared prefix across RAM and disk. A disk hit (with touch=True) is loaded into RAM."""
        ram_key, ram_len = super().lookup(ids, touch=touch)
        disk_key, disk_len = None, 0
        for key in self.on_disk:
            n = shared_prefix_length(key, ids)
            if n > disk_len:
                disk_key, disk_len = key, n
        if disk_len > ram_len:
            if touch:
                self._load(disk_key)
            self.last_hit = "disk"
            return disk_key, disk_len
        self.last_hit = "ram" if ram_key is not None else None
        return ram_key, ram_len

    def add(self, ids: list[int], cache: KVCache) -> None:
        key = tuple(ids)
        if key in self.on_disk:  # already saved: no need to keep a second copy on disk
            self._delete_file(key)
        super().add(ids, cache)

    def _evict(self, key: tuple, value: tuple[list, list]) -> None:
        """Pushed out of RAM: save to disk instead of dropping."""
        self._save(key, *value)

    def _save(self, key: tuple, ks: list, vs: list) -> None:
        path = self.disk_dir / (hashlib.sha1(str(key).encode()).hexdigest() + ".pt")
        torch.save({"ids": list(key), "k": torch.stack(ks), "v": torch.stack(vs)}, path)
        if self.drop_page_cache:
            with open(path, "rb+") as f:
                os.fsync(f.fileno())  # make sure it's on the disk, then drop it from RAM's page cache
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.on_disk[key] = path
        while len(self.on_disk) > self.disk_entries:
            old, _ = next(iter(self.on_disk.items()))
            self._delete_file(old)

    def _load(self, key: tuple) -> None:
        path = self.on_disk.pop(key)
        data = torch.load(path)
        path.unlink()
        self.entries[key] = (list(data["k"].unbind(0)), list(data["v"].unbind(0)))
        self.entries.move_to_end(key)
        self._make_room()  # may push another entry down to disk

    def _delete_file(self, key: tuple) -> None:
        path = self.on_disk.pop(key)
        path.unlink(missing_ok=True)

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.on_disk.values())

    def clear_disk(self) -> None:
        for key in list(self.on_disk):
            self._delete_file(key)


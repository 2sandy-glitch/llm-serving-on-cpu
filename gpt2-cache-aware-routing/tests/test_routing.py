"""Cache-aware routing checks: each policy picks the right worker, the accounting is right, and the
prefix store's look-only lookup and cache breakpoint behave. Run from gpt2-cache-aware-routing/: uv run pytest"""

import torch

from gpt2lab.kv_cache import KVCache
from gpt2lab.prefix_cache import PrefixStore
from gpt2lab.routing import CacheAware, LeastLoaded, Request, RoundRobin, Worker, simulate


def store_with(*prompts, size=4):
    store = PrefixStore(max_entries=size)
    for ids in prompts:
        cache = KVCache(1)
        cache.update(0, torch.zeros(1, 1, len(ids), 2), torch.zeros(1, 1, len(ids), 2))
        store.add(list(ids), cache)
    return store


def test_round_robin_takes_turns():
    rr, workers = RoundRobin(), [Worker(PrefixStore()) for _ in range(3)]
    assert [rr.choose(workers, Request([1], 1)) for _ in range(5)] == [0, 1, 2, 0, 1]


def test_least_loaded_picks_least_busy():
    workers = [Worker(PrefixStore(), busy_s=s) for s in (3.0, 1.0, 2.0)]
    assert LeastLoaded().choose(workers, Request([1], 1)) == 1


def test_cache_aware_goes_where_the_prefix_is():
    workers = [Worker(store_with([9, 9, 9])), Worker(store_with([1, 2, 3, 4])), Worker(store_with([1, 2]))]
    assert CacheAware().choose(workers, Request([1, 2, 3, 4, 7], 4)) == 1  # longest match


def test_cache_aware_balances_when_too_busy_or_no_match():
    workers = [Worker(store_with([1, 2, 3]), busy_s=5.0), Worker(store_with([8]), busy_s=0.5)]
    assert CacheAware(max_lead_s=1.0).choose(workers, Request([1, 2, 3, 4], 3)) == 1  # match is overloaded
    assert CacheAware(max_lead_s=10.0).choose(workers, Request([1, 2, 3, 4], 3)) == 0  # within the limit
    assert CacheAware().choose(workers, Request([6, 6], 2)) == 1  # no match: least busy


def test_cache_aware_lookup_does_not_count_as_use():
    """A router only looks; it must not change which entry is least recently used."""
    store = store_with([1, 1], [2, 2], size=2)
    CacheAware().choose([Worker(store)], Request([1, 1, 5], 2))
    assert list(store.entries) == [(1, 1), (2, 2)]


def test_cache_breakpoint_stores_only_the_shared_part():
    cache = KVCache(1)
    cache.update(0, torch.randn(1, 1, 10, 2), torch.randn(1, 1, 10, 2))
    store = PrefixStore()
    store.add(list(range(6)), cache)  # keep only the first 6 of 10 tokens
    k, _ = store.entries[tuple(range(6))]
    assert k[0].shape[2] == 6 and torch.equal(k[0], cache.k[0][:, :, :6])
    assert store.nbytes() == 2 * 6 * 2 * 4  # K and V, 6 tokens, 2 numbers, 4 bytes


def test_simulate_accounting():
    """Fake costs: a request costs 1 s if its document wasn't cached on that worker, else 0.1 s."""
    def run(store, request):
        _, shared = store.lookup(request.ids)
        reused = min(shared, len(request.ids) - 1)
        store_with_cache = KVCache(1)
        store_with_cache.update(0, torch.zeros(1, 1, len(request.ids), 2), torch.zeros(1, 1, len(request.ids), 2))
        store.add(request.ids[: request.shared_len], store_with_cache)
        return (0.1 if reused else 1.0), reused

    docs = {"a": [1, 1, 1], "b": [2, 2, 2]}
    requests = [Request(docs[d] + [q], 3) for d, q in ("a5", "b6", "a7", "b8")]  # a, b, a, b
    rr = simulate(requests, RoundRobin(), n_workers=2, store_size=1, run=run)
    ca = simulate(requests, CacheAware(max_lead_s=5), n_workers=2, store_size=1, run=run)
    # Round robin happens to alternate in step with the documents here, so it hits too:
    assert rr.reused_tokens == ca.reused_tokens == 6
    assert ca.finish_s == 1.1 and ca.total_compute_s == 2.2
    assert ca.hit_rate == 6 / 16
    assert ca.balance == 1.0

    # Now the order breaks round robin's luck: a, a, b, b.
    requests = [Request(docs[d] + [q], 3) for d, q in ("a5", "a6", "b7", "b8")]
    rr = simulate(requests, RoundRobin(), n_workers=2, store_size=1, run=run)
    ca = simulate(requests, CacheAware(max_lead_s=5), n_workers=2, store_size=1, run=run)
    assert rr.reused_tokens == 0 and rr.total_compute_s == 4.0
    assert ca.reused_tokens == 6 and ca.total_compute_s == 2.2

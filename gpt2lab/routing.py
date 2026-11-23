"""Cache-aware routing: with several workers, send each request where its prefix is already cached.

A server runs several workers (model copies), each with its own prefix store. Prefix caching only
helps if a request lands on a worker that already holds its shared start. Round robin spreads
requests evenly but ignores the caches, so with many different documents each worker keeps
re-reading them. A cache-aware router checks which worker already has the request's prefix and
sends it there, unless that worker is much busier than the rest (then balance wins).
This is the idea behind SGLang's cache-aware load balancer.

Here it is a simulation on one CPU: each request is really run through the model on its chosen
worker's store, and the MEASURED time is added to that worker's busy time. All requests are queued
at the start, and workers would run in parallel, so the time to finish everything is the busiest
worker's total. It is not a real multi-machine cluster.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from gpt2lab.prefix_cache import PrefixStore


@dataclass
class Worker:
    store: PrefixStore
    busy_s: float = 0.0  # total measured time of the requests it ran
    requests: int = 0
    prompt_tokens: int = 0
    reused_tokens: int = 0


@dataclass
class Request:
    ids: list[int]  # the whole prompt
    shared_len: int  # how much of it is the shared document (the cache breakpoint)
    label: str = ""


@dataclass
class RoundRobin:
    """Take turns: 0, 1, 2, 3, 0, 1, ..."""
    next: int = 0

    def choose(self, workers: list[Worker], request: Request) -> int:
        i = self.next % len(workers)
        self.next += 1
        return i


@dataclass
class LeastLoaded:
    """The worker with the least work so far."""

    def choose(self, workers: list[Worker], request: Request) -> int:
        return min(range(len(workers)), key=lambda i: workers[i].busy_s)


@dataclass
class CacheAware:
    """The worker holding the longest prefix of this request, unless it is more than `max_lead_s`
    busier than the least busy worker; with no useful match, the least busy worker."""
    max_lead_s: float = 1.0
    min_match: int = 1

    def choose(self, workers: list[Worker], request: Request) -> int:
        least = min(range(len(workers)), key=lambda i: workers[i].busy_s)
        matches = [w.store.lookup(request.ids, touch=False)[1] for w in workers]
        best = max(range(len(workers)), key=lambda i: (matches[i], -workers[i].busy_s))
        if matches[best] < self.min_match:
            return least
        if workers[best].busy_s - workers[least].busy_s > self.max_lead_s:
            return least  # too unbalanced: spread the load instead
        return best


@dataclass
class Report:
    policy: str
    requests: int
    prompt_tokens: int
    reused_tokens: int
    total_compute_s: float
    finish_s: float  # busiest worker's total = time to finish everything
    worker_busy_s: list[float] = field(default_factory=list)
    worker_requests: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Share of prompt tokens that came from a cache instead of being computed."""
        return self.reused_tokens / self.prompt_tokens

    @property
    def requests_per_s(self) -> float:
        return self.requests / self.finish_s

    @property
    def balance(self) -> float:
        """Busiest worker / average worker (1.0 = perfectly even)."""
        return max(self.worker_busy_s) / (sum(self.worker_busy_s) / len(self.worker_busy_s))


def simulate(requests: list[Request], policy, n_workers: int, store_size: int,
             run: Callable[[PrefixStore, Request], tuple[float, int]], name: str = "") -> Report:
    """Route every request with `policy`; run(store, request) runs it on that worker's store and
    returns (measured seconds, tokens reused)."""
    workers = [Worker(PrefixStore(max_entries=store_size)) for _ in range(n_workers)]
    for request in requests:
        w = workers[policy.choose(workers, request)]
        seconds, reused = run(w.store, request)
        w.busy_s += seconds
        w.requests += 1
        w.prompt_tokens += len(request.ids)
        w.reused_tokens += reused
    return Report(
        policy=name or type(policy).__name__,
        requests=len(requests),
        prompt_tokens=sum(w.prompt_tokens for w in workers),
        reused_tokens=sum(w.reused_tokens for w in workers),
        total_compute_s=sum(w.busy_s for w in workers),
        finish_s=max(w.busy_s for w in workers),
        worker_busy_s=[w.busy_s for w in workers],
        worker_requests=[w.requests for w in workers],
    )

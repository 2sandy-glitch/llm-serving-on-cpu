"""Measure the paged KV cache: memory waste, how many requests fit, and the speed cost of gathering.

Run from gpt2-paged-kv-cache/:   uv run python benchmark/run.py      (~10 minutes on a 6-core CPU)

1. Capacity. A workload of 200 requests with realistic lengths (most short, a few long; log-normal,
   median ~200 tokens, 16-1,024). With a 512 MiB cache budget:
   - reserved cache: every request takes room for 1,024 tokens (72 MiB);
   - paged: requests are really allocated in a BlockPool of that size, in order, until it is full;
   - ideal: exactly the tokens needed, no waste (a lower bound no real cache reaches).
   Memory waste = reserved-but-unused / reserved, over all 200 requests.
2. Block size 4 ... 128: waste, requests that fit, and the cache step alone (write 1 token into all
   12 layers and gather everything back) with 100 / 400 / 800 tokens cached, vs the reserved cache.
   No model around it, so other programs disturb it far less (median of 15 repeats x 20 steps).
3. Speed: growing vs reserved vs paged (16-token blocks) at 100 / 400 / 800 new tokens (gpt2lab.bench
   method: warm-up, 3 timed runs, median and min-max, interleaved).

The allocator in part 1 and the waste in part 2 use synthetic K and V (the numbers don't matter for
memory); speed uses the real model. Correctness is in the tests.

Writes to results/: capacity.csv, block_sizes.csv, cache_step.csv, speed.csv, environment.json, chart.png
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, summarize, timed_generate, write_csv, write_json
from gpt2lab.kv_cache import KVCache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.paged_kv_cache import BlockPool, PagedKVCache
from gpt2lab.static_kv_cache import StaticKVCache

RESULTS = Path(__file__).parent.parent / "results"
BYTES_PER_TOKEN = 73_728
MAX_TOKENS = 1024
BUDGET_MIB = 512
N_REQUESTS = 200
BLOCK_SIZE = 16
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
PROMPT = "In a small village by the sea, there lived an old fisherman who"
SPEED_TOKENS = [100, 400, 800]
STEP_CACHED = [100, 400, 800]
STEP_STEPS, STEP_REPEATS = 20, 15
TIMED_RUNS = 3


def workload(seed=0):
    """Request lengths (prompt + answer): log-normal, median ~200 tokens, clipped to 16-1,024."""
    rng = np.random.default_rng(seed)
    return [int(x) for x in np.clip(rng.lognormal(mean=math.log(200), sigma=0.8, size=N_REQUESTS), 16, MAX_TOKENS)]


def pool_for_budget(block_size, n_layer=12, n_head=12, head_dim=64):
    block_bytes = 2 * n_layer * n_head * block_size * head_dim * 4
    return BlockPool(n_layer, n_head, head_dim, BUDGET_MIB * 2**20 // block_bytes, block_size)


def fill(pool, lengths):
    """Allocate requests in order until the pool is full. Returns (requests that fit, caches)."""
    caches = []
    for n in lengths:
        cache = PagedKVCache(pool)
        try:
            x = torch.zeros(1, 12, n, 64)
            for layer in range(12):
                cache.update(layer, x, x)
        except MemoryError:
            cache.free()
            break
        caches.append(cache)
    return len(caches), caches


def paged_waste(lengths, block_size):
    blocks = sum(-(-n // block_size) for n in lengths)
    return 1 - sum(lengths) / (blocks * block_size)


def capacity(lengths):
    budget = BUDGET_MIB * 2**20
    reserved_each = MAX_TOKENS * BYTES_PER_TOKEN
    fit_ideal, used = 0, 0
    for n in lengths:
        if used + n * BYTES_PER_TOKEN > budget:
            break
        used += n * BYTES_PER_TOKEN
        fit_ideal += 1
    pool = pool_for_budget(BLOCK_SIZE)
    fit_paged, caches = fill(pool, lengths)
    measured_waste = 1 - sum(len(c) for c in caches) / (pool.used_blocks * BLOCK_SIZE)
    return [
        dict(scheme="reserved (1,024 tokens each)", requests_fit=budget // reserved_each,
             memory_waste_pct=round(100 * (1 - sum(lengths) / (len(lengths) * MAX_TOKENS)), 1)),
        dict(scheme=f"paged ({BLOCK_SIZE}-token blocks)", requests_fit=fit_paged,
             memory_waste_pct=round(100 * paged_waste(lengths, BLOCK_SIZE), 1),
             waste_of_requests_that_fit_pct=round(100 * measured_waste, 1)),
        dict(scheme="ideal (exact tokens)", requests_fit=fit_ideal, memory_waste_pct=0.0),
    ]


def time_cache_step(make_cache, cached):
    """Median ms to write 1 token into all 12 layers and get every token's K and V back, with
    `cached` tokens already in the cache. Only the cache work is timed."""
    import statistics
    import time
    old, new = torch.randn(1, 12, cached, 64), torch.randn(1, 12, 1, 64)
    runs = []
    for _ in range(STEP_REPEATS):
        cache = make_cache()
        for layer in range(12):
            cache.update(layer, old, old)
        times = []
        for _ in range(STEP_STEPS):
            start = time.perf_counter()
            for layer in range(12):
                cache.update(layer, new, new)
            times.append(time.perf_counter() - start)
        runs.append(statistics.median(times))
        if isinstance(cache, PagedKVCache):
            cache.free()
    return 1000 * statistics.median(runs)


def speed_row(name, n, totals):
    s = summarize(totals)
    return dict(cache=name, new_tokens=n, median_s=round(s["median"], 3), min_s=round(s["min"], 3),
                max_s=round(s["max"], 3), tokens_per_s=round(n / s["median"], 2))


def draw_chart(cap, blocks, speed, title):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#7f8c8d", "#1e8449", "#2471a3"]
    bars = a.bar([r["scheme"].replace(" (", "\n(") for r in cap], [r["requests_fit"] for r in cap], color=colors)
    for bar, r in zip(bars, cap):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f"{int(r['requests_fit'])} requests\n{r['memory_waste_pct']:.0f}% wasted", ha="center", va="bottom")
    a.set(ylabel=f"requests that fit in {BUDGET_MIB} MiB", title="Capacity", ylim=(0, max(r["requests_fit"] for r in cap) * 1.25))

    xs = [r["block_size"] for r in blocks]
    b.plot(xs, [r["memory_waste_pct"] for r in blocks], "o-", color="#c0392b", label="memory wasted (%)")
    b.set(xscale="log", xlabel="tokens per block", ylabel="memory wasted (%)", title="Block size trade-off")
    b.set_xticks(xs, [str(int(x)) for x in xs])
    b2 = b.twinx()
    b2.plot(xs, [r["step_ms_at_800"] for r in blocks], "s--", color="#2471a3", label="cache step, 800 cached (ms)")
    b2.set_ylabel("ms per cache step, 800 tokens cached")
    b2.set_ylim(0, max(r["step_ms_at_800"] for r in blocks) * 1.3)
    lines = b.get_legend_handles_labels()[0] + b2.get_legend_handles_labels()[0]
    b.legend(lines, [l.get_label() for l in lines], loc="upper left")
    b.grid(alpha=0.3)

    names = ["growing", "reserved", "paged"]
    width = 0.27
    for i, name in enumerate(names):
        rows = [r for r in speed if r["cache"] == name]
        c.bar([x + (i - 1) * width for x in range(len(SPEED_TOKENS))], [r["tokens_per_s"] for r in rows], width,
              label=name, color=["#7f8c8d", "#2471a3", "#1e8449"][i])
    c.set_xticks(range(len(SPEED_TOKENS)), [str(n) for n in SPEED_TOKENS])
    c.set(xlabel="new tokens generated", ylabel="tokens per second (median)", title="Speed")
    c.set_ylim(0, max(r["tokens_per_s"] for r in speed) * 1.25)
    c.legend(loc="upper center", ncol=3)
    c.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)


def read_csv(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            out.append(row)
    return out


def main():
    title = "GPT-2 small, paged KV cache, on {}, {} threads"
    if "--chart-only" in sys.argv:
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "capacity.csv"), read_csv(RESULTS / "block_sizes.csv"),
                   read_csv(RESULTS / "speed.csv"), title.format(env["cpu"], env["threads"]))
        return
    load_at_start = setup()
    lengths = workload()
    print(f"CPU load at start: {load_at_start}. Workload: {N_REQUESTS} requests, median {int(np.median(lengths))} "
          f"tokens, max {max(lengths)}, total {sum(lengths):,}", flush=True)

    # 1. Capacity.
    cap = capacity(lengths)
    for r in cap:
        print(f"{r['scheme']:30s} fits {r['requests_fit']:4d} requests, waste {r['memory_waste_pct']}%", flush=True)
    write_csv(RESULTS / "capacity.csv", [{**r, "waste_of_requests_that_fit_pct": r.get("waste_of_requests_that_fit_pct", "")} for r in cap])

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    idx = tokenizer(PROMPT, return_tensors="pt").input_ids

    def paged_factory(block_size):
        p = BlockPool.for_model(model, num_blocks=-(-MAX_TOKENS // block_size), block_size=block_size)
        return lambda: PagedKVCache(p)

    # 2. Block sizes: waste and fit, and the cache step alone.
    block_rows, step_rows = [], []
    pools = {bs: BlockPool(12, 12, 64, -(-(max(STEP_CACHED) + STEP_STEPS) // bs), bs) for bs in BLOCK_SIZES}
    for bs in BLOCK_SIZES:
        fit, _ = fill(pool_for_budget(bs), lengths)
        block_rows.append(dict(block_size=bs, memory_waste_pct=round(100 * paged_waste(lengths, bs), 1), requests_fit=fit))
    for cached in STEP_CACHED:
        row = dict(cached_tokens=cached,
                   reserved_ms=round(time_cache_step(lambda: StaticKVCache(12, 12, 64, MAX_TOKENS), cached), 3))
        for bs in BLOCK_SIZES:
            row[f"paged_{bs}_ms"] = round(time_cache_step(lambda b=bs: PagedKVCache(pools[b]), cached), 3)
        step_rows.append(row)
        print(f"cache step, {cached} cached: " + "  ".join(f"{k} {v}" for k, v in row.items() if k != "cached_tokens"), flush=True)
    for r in block_rows:
        r["step_ms_at_800"] = step_rows[-1][f"paged_{r['block_size']}_ms"]
        print(f"block {r['block_size']:4d}: waste {r['memory_waste_pct']:5.1f}%  fits {r['requests_fit']:4d}  "
              f"cache step at 800 cached {r['step_ms_at_800']} ms", flush=True)
    write_csv(RESULTS / "block_sizes.csv", block_rows)
    write_csv(RESULTS / "cache_step.csv", step_rows)

    # 3. Speed.
    caches = {
        "growing": lambda: KVCache(model.config.n_layer),
        "reserved": lambda: StaticKVCache.for_model(model, max_tokens=MAX_TOKENS),
        "paged": paged_factory(BLOCK_SIZE),
    }
    for f in caches.values():
        timed_generate(model, idx, 20, f)
    speed = []
    for n in SPEED_TOKENS:
        totals = {name: [] for name in caches}
        outs = {}
        for _ in range(TIMED_RUNS):
            for name, f in caches.items():
                out, steps, cache = timed_generate(model, idx, n, f)
                totals[name].append(sum(steps))
                outs[name] = out
                if isinstance(cache, PagedKVCache):
                    cache.free()
        assert torch.equal(outs["growing"], outs["paged"]) and torch.equal(outs["growing"], outs["reserved"])
        for name in caches:
            speed.append(speed_row(name, n, totals[name]))
            print(f"speed {n:4d} tokens {name:9s} {speed[-1]['median_s']:6.2f} s ({speed[-1]['min_s']}-{speed[-1]['max_s']})", flush=True)
    write_csv(RESULTS / "speed.csv", speed)

    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        workload=f"{N_REQUESTS} request lengths, log-normal (median 200, sigma 0.8), clipped 16-{MAX_TOKENS}, seed 0",
        workload_median_tokens=int(np.median(lengths)),
        workload_total_tokens=sum(lengths),
        budget_mib=BUDGET_MIB,
        block_size=BLOCK_SIZE,
        block_sizes=BLOCK_SIZES,
        cache_step_cached_tokens=STEP_CACHED,
        cache_step_repeats=f"{STEP_REPEATS} x {STEP_STEPS} steps, median",
        speed_prompt=PROMPT,
        speed_new_tokens=SPEED_TOKENS,
        timed_runs=TIMED_RUNS,
        statistic="median",
    )
    write_json(RESULTS / "environment.json", env)
    draw_chart(cap, block_rows, speed, title.format(env["cpu"], env["threads"]))
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

"""Measure the growing KV cache (torch.cat) against the reserved one (StaticKVCache) on the CPU.

Run from gpt2-static-kv-cache/:   uv run python benchmark/run.py

Two measurements:
  1. Whole generation (gpt2lab.bench method): 6 threads, 1 warm-up run thrown away, 3 timed runs,
     median and min-max, only the model's work timed. The two caches take turns run by run
     (growing, reserved, growing, ...), so a slow moment on the machine hits both equally.
     Both caches must produce exactly the same text, or it stops.
  2. The cache update alone (the only part that differs): time to add 1 token to all 12 layers
     when the cache already holds 100 ... 1,000 tokens. No model around it, so far less noise.

Writes to results/:
  results.csv        time and tokens/s for each cache and length (median, min, max of 3 runs)
  per_token.csv      time for each new token by its position (longest length, median of 3 runs)
  cache_update.csv   time to add 1 token to all layers, by how many tokens are already cached
  environment.json   the machine, versions, CPU load and settings
  chart.png          the charts used in the README
"""

import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, summarize, timed_generate, write_csv, write_json
from gpt2lab.kv_cache import KVCache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.static_kv_cache import StaticKVCache

RESULTS = Path(__file__).parent.parent / "results"
PROMPT = "In a small village by the sea, there lived an old fisherman who"  # same as gpt2-kv-cache
NEW_TOKENS = [100, 200, 400, 800]
TIMED_RUNS = 3
WARMUP_TOKENS = 20
UPDATE_SIZES = [100, 200, 400, 800, 1000]  # cached tokens for the cache-update measurement
UPDATE_STEPS = 20  # tokens added per measurement
UPDATE_REPEATS = 15
MAX_TOKENS = 1024  # the reserved cache holds GPT-2's whole context, as an engine would reserve it


def n_bytes_growing(model, tokens):
    """Bytes the growing cache holds for `tokens` tokens (it grows to exactly what it needs)."""
    c = model.config
    return 2 * c.n_layer * c.n_head * (c.n_embd // c.n_head) * 4 * tokens


def time_cache_update(cache, n_layer, n_head, head_dim, cached):
    """Fill a cache with `cached` tokens of random K and V, then time adding 1 token to every layer,
    UPDATE_STEPS times. Returns the median ms per step. Only the cache work is timed."""
    old = torch.randn(1, n_head, cached, head_dim)
    new = torch.randn(1, n_head, 1, head_dim)
    for layer in range(n_layer):
        cache.update(layer, old, old)
    times = []
    for _ in range(UPDATE_STEPS):
        start = time.perf_counter()
        for layer in range(n_layer):
            cache.update(layer, new, new)
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


def main():
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    idx = tokenizer(PROMPT, return_tensors="pt").input_ids
    caches = {
        "growing": lambda: KVCache(model.config.n_layer),
        "reserved": lambda: StaticKVCache.for_model(model, max_tokens=MAX_TOKENS),
    }
    print(f"Prompt: {idx.shape[1]} tokens. CPU load at start: {load_at_start}", flush=True)

    for factory in caches.values():  # warm-up, not counted
        timed_generate(model, idx, WARMUP_TOKENS, factory)

    rows, per_token = [], {}
    for n in NEW_TOKENS:
        texts = {}
        totals = {name: [] for name in caches}
        steps_per_run = {name: [] for name in caches}
        for run in range(1, TIMED_RUNS + 1):
            for name, factory in caches.items():  # take turns: growing, reserved, growing, ...
                out, steps, cache = timed_generate(model, idx, n, factory)
                totals[name].append(sum(steps))
                steps_per_run[name].append(steps)
                texts[name] = out
                print(f"{n:4d} tokens  {name:8s}  run {run}: {sum(steps):6.2f} s", flush=True)
        for name, factory in caches.items():
            s = summarize(totals[name])
            rows.append(dict(
                cache=name,
                new_tokens=n,
                median_s=round(s["median"], 3),
                min_s=round(s["min"], 3),
                max_s=round(s["max"], 3),
                tokens_per_s=round(n / s["median"], 2),
                ms_per_token=round(1000 * s["median"] / n, 2),
                cache_mib_at_end=round(factory().nbytes() / 2**20, 2) if name == "reserved"
                else round(n_bytes_growing(model, idx.shape[1] + n - 1) / 2**20, 2),
            ))
            if n == max(NEW_TOKENS):
                per_token[name] = [statistics.median(t) for t in zip(*steps_per_run[name])]
        if not torch.equal(texts["growing"], texts["reserved"]):
            raise AssertionError(f"the two caches wrote different text at {n} tokens")
        print(f"{n:4d} tokens: same text with both caches", flush=True)

    c = model.config
    head_dim = c.n_embd // c.n_head
    update_rows = []
    for cached in UPDATE_SIZES:
        ms = {"growing": [], "reserved": []}
        for _ in range(UPDATE_REPEATS):  # take turns again
            ms["growing"].append(time_cache_update(KVCache(c.n_layer), c.n_layer, c.n_head, head_dim, cached))
            ms["reserved"].append(time_cache_update(StaticKVCache.for_model(model, MAX_TOKENS), c.n_layer, c.n_head, head_dim, cached))
        g, r = statistics.median(ms["growing"]), statistics.median(ms["reserved"])
        update_rows.append(dict(cached_tokens=cached, growing_ms=round(g, 3), reserved_ms=round(r, 3)))
        print(f"cache update with {cached:4d} tokens cached: growing {g:.3f} ms, reserved {r:.3f} ms", flush=True)
    write_csv(RESULTS / "cache_update.csv", update_rows)

    write_csv(RESULTS / "results.csv", rows)
    write_csv(RESULTS / "per_token.csv", [
        dict(position=i, growing_ms=round(1000 * a, 3), reserved_ms=round(1000 * b, 3))
        for i, (a, b) in enumerate(zip(per_token["growing"], per_token["reserved"]), start=1)
    ])
    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        decoding="greedy",
        prompt=PROMPT,
        prompt_tokens=idx.shape[1],
        new_tokens=NEW_TOKENS,
        timed_runs=TIMED_RUNS,
        statistic="median",
        reserved_cache_max_tokens=MAX_TOKENS,
        runs_order="interleaved (growing, reserved, growing, ...)",
        cache_update_cached_tokens=UPDATE_SIZES,
        cache_update_steps=UPDATE_STEPS,
        cache_update_repeats=UPDATE_REPEATS,
    )
    write_json(RESULTS / "environment.json", env)

    # Charts: time per token by position, and total time per length.
    colors = {"growing": "#2471a3", "reserved": "#1e8449"}
    labels = {"growing": "growing cache (torch.cat)", "reserved": "reserved cache"}
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, times in per_token.items():
        left.plot(range(1, len(times) + 1), [1000 * t for t in times], label=labels[name], color=colors[name], lw=1)
    left.set(xlabel="new token number", ylabel="ms for this token", title=f"Time for each new token ({max(NEW_TOKENS)} tokens)")
    left.legend()
    left.grid(alpha=0.3)

    width = 0.38
    xs = range(len(NEW_TOKENS))
    for offset, name in ((-width / 2, "growing"), (width / 2, "reserved")):
        r = [row for row in rows if row["cache"] == name]
        med = [row["median_s"] for row in r]
        err = [[row["median_s"] - row["min_s"] for row in r], [row["max_s"] - row["median_s"] for row in r]]
        right.bar([x + offset for x in xs], med, width, yerr=err, capsize=3, label=labels[name], color=colors[name])
    right.set_xticks(list(xs), [str(n) for n in NEW_TOKENS])
    right.set(xlabel="new tokens generated", ylabel="seconds (median, bars show min-max)", title="Total generation time")
    right.legend()
    right.grid(axis="y", alpha=0.3)

    xs_u = [row["cached_tokens"] for row in update_rows]
    for name in ("growing", "reserved"):
        middle.plot(xs_u, [row[f"{name}_ms"] for row in update_rows], "o-", label=labels[name], color=colors[name])
    middle.set(xlabel="tokens already in the cache", ylabel="ms to add 1 token (all 12 layers)",
               title="The cache update alone")
    middle.legend()
    middle.grid(alpha=0.3)

    fig.suptitle(f"GPT-2 small, growing vs reserved KV cache, on {env['cpu']}, {env['threads']} threads")
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)
    print(f"CPU load at end: {env['load_average_1min_at_end']}. Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

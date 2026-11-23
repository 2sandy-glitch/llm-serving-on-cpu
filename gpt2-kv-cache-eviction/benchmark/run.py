"""Measure KV cache eviction: quality at a fixed cache budget for each policy, and speed.

Run from gpt2-kv-cache-eviction/:   uv run python benchmark/run.py      (~12 minutes on a 6-core CPU)
Speed again only (keeps quality.csv):  uv run python benchmark/run.py --speed-only

1. Quality vs budget. WikiText-2 test text, 16 windows of 512 tokens, fed ONE token at a time
   through each cache (teacher forcing). Budgets 256 / 128 / 64 / 32 tokens (50% ... 6% of the
   window) for the recent, sink, h2o and random policies, plus the full cache. All run side by side,
   so each is compared with the full cache at every position: perplexity, and how often the top
   choice is the same.
2. Speed: 800 new tokens, full cache vs a 128-token budget (gpt2lab.bench method: warm-up, 3 timed
   runs, median and min-max, interleaved).

Writes to results/: quality.csv, speed.csv, environment.json, chart.png
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, summarize, timed_generate, write_csv, write_json
from gpt2lab.evicting_kv_cache import EvictingKVCache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import wikitext2_test_ids

RESULTS = Path(__file__).parent.parent / "results"
WINDOWS, WINDOW_LEN = 16, 512
BUDGETS = [256, 128, 64, 32]
POLICIES = ["recent", "sink", "h2o", "random"]
SPEED_PROMPT = "In a small village by the sea, there lived an old fisherman who"
SPEED_TOKENS = 800
SPEED_BUDGET = 128
TIMED_RUNS = 3
COLORS = {"recent": "#7f8c8d", "sink": "#2471a3", "h2o": "#1e8449", "random": "#c0392b"}


@torch.no_grad()
def quality(model, ids):
    batch = ids[0, : WINDOWS * WINDOW_LEN].reshape(WINDOWS, WINDOW_LEN)
    settings = [("full", WINDOW_LEN)] + [(p, b) for b in BUDGETS for p in POLICIES]
    L = model.config.n_layer
    caches = {s: EvictingKVCache(L, budget=s[1], policy=s[0]) for s in settings}
    nll = {s: 0.0 for s in settings}
    agree = {s: 0 for s in settings}
    for t in range(WINDOW_LEN - 1):
        target = batch[:, t + 1]
        logits = {s: model(batch[:, t : t + 1], caches[s])[:, -1].float() for s in settings}
        ref_top = logits[("full", WINDOW_LEN)].argmax(dim=-1)
        for s in settings:
            nll[s] += F.cross_entropy(logits[s], target, reduction="sum").item()
            agree[s] += (logits[s].argmax(dim=-1) == ref_top).sum().item()
        if (t + 1) % 64 == 0:
            print(f"  quality: position {t + 1} of {WINDOW_LEN - 1}", flush=True)
    scored = WINDOWS * (WINDOW_LEN - 1)
    base = math.exp(nll[("full", WINDOW_LEN)] / scored)
    rows = []
    for policy, budget in settings:
        ppl = math.exp(nll[(policy, budget)] / scored)
        rows.append(dict(policy=policy, budget_tokens=budget, budget_pct=round(100 * budget / WINDOW_LEN, 1),
                         cached_perplexity=round(ppl, 3), perplexity_change_pct=round(100 * (ppl / base - 1), 2),
                         top1_same_as_full_pct=round(100 * agree[(policy, budget)] / scored, 2)))
    return rows, scored


def draw_chart(rows, speed, title):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8]))
    full = next(r for r in rows if r["policy"] == "full")
    for p in POLICIES:
        rs = sorted([r for r in rows if r["policy"] == p], key=lambda r: r["budget_tokens"])
        xs = [r["budget_tokens"] for r in rs]
        a.plot(xs, [r["cached_perplexity"] for r in rs], "o-", color=COLORS[p], label=p)
        b.plot(xs, [r["top1_same_as_full_pct"] for r in rs], "o-", color=COLORS[p], label=p)
    a.axhline(full["cached_perplexity"], color="black", ls=":", label="full cache (512)")
    a.set(xscale="log", yscale="log", xlabel="cache budget (tokens kept, of 512)", ylabel="perplexity (log scale)",
          title="Quality vs cache budget (lower is better)")
    b.set(xscale="log", xlabel="cache budget (tokens kept, of 512)", ylabel="top choice same as full cache (%)",
          title="Agreement with the full cache", ylim=(0, 100))
    for ax in (a, b):
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # only the budget ticks
        ax.set_xticks(BUDGETS, [str(x) for x in BUDGETS])
        ax.legend()
        ax.grid(alpha=0.3)
    names = [r["cache"] for r in speed]
    bars = c.bar(names, [r["tokens_per_s"] for r in speed], color=["#7f8c8d", "#1e8449", "#2471a3"][: len(speed)])
    for bar, r in zip(bars, speed):
        c.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{r['tokens_per_s']:.1f}", ha="center", va="bottom")
    c.set(ylabel="tokens per second (median)", title=f"Speed, {SPEED_TOKENS} new tokens")
    c.tick_params(axis="x", labelsize=8)
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
    title = "GPT-2 small, KV cache eviction, on {}, {} threads"
    if "--chart-only" in sys.argv:
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "quality.csv"), read_csv(RESULTS / "speed.csv"), title.format(env["cpu"], env["threads"]))
        return
    speed_only = "--speed-only" in sys.argv  # re-measure speed, keep the saved quality results
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    print(f"CPU load at start: {load_at_start}", flush=True)

    if speed_only:
        rows = read_csv(RESULTS / "quality.csv")
        scored = json.loads((RESULTS / "environment.json").read_text())["quality_tokens_scored"]
    else:
        rows, scored = quality(model, wikitext2_test_ids(tokenizer))
        for r in rows:
            print(f"{r['policy']:7s} budget {r['budget_tokens']:4d}: ppl {r['cached_perplexity']:9.3f} "
                  f"({r['perplexity_change_pct']:+8.2f}%)  top-1 same {r['top1_same_as_full_pct']:6.2f}%", flush=True)
        write_csv(RESULTS / "quality.csv", rows)

    idx = tokenizer(SPEED_PROMPT, return_tensors="pt").input_ids
    L = model.config.n_layer
    factories = {
        "full": lambda: EvictingKVCache(L, budget=4096, policy="full"),
        f"h2o, budget {SPEED_BUDGET}": lambda: EvictingKVCache(L, budget=SPEED_BUDGET, policy="h2o"),
        f"sink, budget {SPEED_BUDGET}": lambda: EvictingKVCache(L, budget=SPEED_BUDGET, policy="sink"),
    }
    for f in factories.values():
        timed_generate(model, idx, 20, f)
    totals = {n: [] for n in factories}
    for _ in range(TIMED_RUNS):
        for n, f in factories.items():
            _, steps, _ = timed_generate(model, idx, SPEED_TOKENS, f)
            totals[n].append(sum(steps))
    speed = []
    for n in factories:
        s = summarize(totals[n])
        speed.append(dict(cache=n, new_tokens=SPEED_TOKENS, median_s=round(s["median"], 3), min_s=round(s["min"], 3),
                          max_s=round(s["max"], 3), tokens_per_s=round(SPEED_TOKENS / s["median"], 2)))
        print(f"speed {n:18s} {s['median']:6.2f} s ({s['min']:.2f}-{s['max']:.2f})", flush=True)
    write_csv(RESULTS / "speed.csv", speed)

    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        quality_eval=f"WikiText-2 raw test, first {WINDOWS} x {WINDOW_LEN} tokens, teacher-forced one token at a time",
        quality_tokens_scored=scored,
        budgets=BUDGETS,
        policies=POLICIES,
        h2o_recent_share="half the budget",
        sink_tokens=4,
        speed_prompt=SPEED_PROMPT,
        speed_new_tokens=SPEED_TOKENS,
        speed_budget=SPEED_BUDGET,
        timed_runs=TIMED_RUNS,
        statistic="median",
    )
    if speed_only:  # keep the quality run's start load, add the speed run's
        old = json.loads((RESULTS / "environment.json").read_text())
        env["load_average_1min_at_start"] = old["load_average_1min_at_start"]
        env["speed_rerun"] = dict(date=env["date"], load_average_1min_at_start=load_at_start,
                                  load_average_1min_at_end=env["load_average_1min_at_end"])
    write_json(RESULTS / "environment.json", env)
    draw_chart(rows, speed, title.format(env["cpu"], env["threads"]))
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

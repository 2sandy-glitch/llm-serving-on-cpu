"""Measure cache-aware routing against round robin and least-loaded, with measured request costs.

Run from gpt2-cache-aware-routing/:   uv run python benchmark/run.py      (~5 minutes on a 6-core CPU)

Simulation on one CPU (see gpt2lab/routing.py): 4 workers, each with its own prefix store. The
workload is 8 shared documents (500 tokens each, from the WikiText-2 test set) x 12 questions =
96 requests in random order. Every request really runs through the model on its chosen worker's
store (only the document is stored: a cache breakpoint), and the measured time is added to that
worker. Finish time = the busiest worker's total, as if the workers ran in parallel.

1. Three policies x three random orders (seeds), stores holding 2 documents per worker.
2. Store size 1, 2 or 4 documents per worker (seed 0).

Writes to results/: results.csv, store_sizes.csv, environment.json, chart.png
"""

import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, write_csv, write_json
from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import wikitext2_test_ids
from gpt2lab.prefix_cache import prefill_with_prefix_cache
from gpt2lab.routing import CacheAware, LeastLoaded, Request, RoundRobin, simulate

RESULTS = Path(__file__).parent.parent / "results"
N_WORKERS = 4
N_DOCS = 8
DOC_TOKENS = 500
DOC_SPACING = 5_000  # documents start this many tokens apart in the test set
STORE_SIZE = 2  # documents each worker can keep
STORE_SIZES = [1, 2, 4]
SEEDS = [0, 1, 2]
MAX_LEAD_S = 1.0  # cache-aware: give up the match if that worker is this much busier than the least busy
QUESTIONS = [
    "Summarise the passage above in one sentence.",
    "What is the main subject of this text?",
    "List three facts from the passage.",
    "Who is mentioned most often?",
    "When did these events happen?",
    "Where does this take place?",
    "Explain the passage to a ten-year-old.",
    "What happened first?",
    "What is the most surprising detail?",
    "Give the passage a title.",
    "What question does the passage leave open?",
    "Write one follow-up question about it.",
]
POLICIES = {
    "round robin": RoundRobin,
    "least loaded": LeastLoaded,
    "cache-aware": lambda: CacheAware(max_lead_s=MAX_LEAD_S),
}


def build_requests(tokenizer, wiki, seed):
    docs = [wiki[i * DOC_SPACING : i * DOC_SPACING + DOC_TOKENS] for i in range(N_DOCS)]
    requests = [
        Request(doc + tokenizer(f"\n\nQuestion: {q}").input_ids, len(doc), f"doc {d}")
        for d, doc in enumerate(docs) for q in QUESTIONS
    ]
    random.Random(seed).shuffle(requests)
    return requests


def make_runner(model):
    @torch.no_grad()
    def run(store, request):
        start = time.perf_counter()
        logits, _, reused = prefill_with_prefix_cache(model, torch.tensor([request.ids]), store,
                                                      store_upto=request.shared_len)
        logits[0, -1].argmax().item()  # the first token: part of the time to first token
        return time.perf_counter() - start, reused
    return run


def row(report, seed, store_size):
    return dict(
        seed=seed,
        store_size=store_size,
        policy=report.policy,
        hit_rate_pct=round(100 * report.hit_rate, 1),
        total_compute_s=round(report.total_compute_s, 2),
        finish_s=round(report.finish_s, 2),
        requests_per_s=round(report.requests_per_s, 2),
        balance=round(report.balance, 2),
        worker_requests=" ".join(map(str, report.worker_requests)),
    )


def draw_chart(rows, sizes, title):
    fig, (hit, thr, size) = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"round robin": "#7f8c8d", "least loaded": "#2471a3", "cache-aware": "#1e8449"}
    names = list(colors)
    for ax, key, label in ((hit, "hit_rate_pct", "cache hit rate (% of prompt tokens reused)"),
                           (thr, "requests_per_s", "requests per second")):
        meds, lo, hi = [], [], []
        for n in names:
            vals = [r[key] for r in rows if r["policy"] == n]
            m = statistics.median(vals)
            meds.append(m)
            lo.append(m - min(vals))
            hi.append(max(vals) - m)
        bars = ax.bar(names, meds, yerr=[lo, hi], capsize=4, color=[colors[n] for n in names])
        for b, m, h in zip(bars, meds, hi):
            ax.text(b.get_x() + b.get_width() / 2, m + h, f"{m:.1f}", ha="center", va="bottom")
        ax.set(ylabel=label, title=f"{label.split(' (')[0].capitalize()} (median of {len(SEEDS)} orders, bars min-max)")
        ax.grid(axis="y", alpha=0.3)
    for n in names:
        rs = [r for r in sizes if r["policy"] == n]
        size.plot([r["store_size"] for r in rs], [r["hit_rate_pct"] for r in rs], "o-", color=colors[n], label=n)
    size.set(xlabel="documents each worker's store can hold", ylabel="cache hit rate (%)", ylim=(0, 100),
             title=f"Store size ({N_DOCS} documents, {N_WORKERS} workers)")
    size.set_xticks(STORE_SIZES)
    size.legend(loc="center right")
    size.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)


def read_csv(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({k: (float(v) if k not in ("policy", "worker_requests") else v) for k, v in r.items()})
    return out


def main():
    title = "GPT-2 small, cache-aware routing (simulated {} workers, measured costs), on {}"
    if "--chart-only" in sys.argv:
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "results.csv"), read_csv(RESULTS / "store_sizes.csv"), title.format(N_WORKERS, env["cpu"]))
        return
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    wiki = wikitext2_test_ids(tokenizer)[0].tolist()
    run = make_runner(model)
    print(f"CPU load at start: {load_at_start}", flush=True)

    warm = build_requests(tokenizer, wiki, 99)[:3]  # warm-up, not counted
    simulate(warm, RoundRobin(), N_WORKERS, STORE_SIZE, run)

    rows = []
    for seed in SEEDS:
        requests = build_requests(tokenizer, wiki, seed)
        for name, policy in POLICIES.items():  # policies take turns within each order
            r = row(simulate(requests, policy(), N_WORKERS, STORE_SIZE, run, name), seed, STORE_SIZE)
            rows.append(r)
            print(f"order {seed}  {name:12s}  hit {r['hit_rate_pct']:5.1f}%  compute {r['total_compute_s']:6.2f} s  "
                  f"finish {r['finish_s']:6.2f} s  {r['requests_per_s']:5.2f} req/s  balance {r['balance']}  "
                  f"per worker {r['worker_requests']}", flush=True)
    write_csv(RESULTS / "results.csv", rows)

    sizes = []
    requests = build_requests(tokenizer, wiki, 0)
    for store_size in STORE_SIZES:
        for name, policy in POLICIES.items():
            r = row(simulate(requests, policy(), N_WORKERS, store_size, run, name), 0, store_size)
            sizes.append(r)
            print(f"store {store_size}  {name:12s}  hit {r['hit_rate_pct']:5.1f}%  {r['requests_per_s']:5.2f} req/s", flush=True)
    write_csv(RESULTS / "store_sizes.csv", sizes)

    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        simulation="one CPU; each request measured on its worker's store; finish time = busiest worker's total",
        workers=N_WORKERS,
        documents=N_DOCS,
        document_tokens=DOC_TOKENS,
        questions=len(QUESTIONS),
        requests=N_DOCS * len(QUESTIONS),
        store_size_documents=STORE_SIZE,
        store_sizes_swept=STORE_SIZES,
        seeds=SEEDS,
        cache_aware_max_lead_s=MAX_LEAD_S,
    )
    write_json(RESULTS / "environment.json", env)
    draw_chart(rows, sizes, title.format(N_WORKERS, env["cpu"]))
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

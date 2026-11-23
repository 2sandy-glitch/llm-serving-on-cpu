"""Measure prefix caching: time to first token with and without a reusable prompt start.

Run from gpt2-prefix-cache/:   uv run python benchmark/run.py      (~3 minutes on a 6-core CPU)

Prompts are a document (a slice of the WikiText-2 test set) followed by a question. Time to first
token (TTFT) = looking up the store + loading the shared K and V + running the model on the rest +
picking the first token. Method from gpt2lab.bench: 6 threads, warm-up, 7 timed runs, median and
min-max; the cases take turns run by run. Before each timed request the store is set up (untimed)
with exactly the earlier requests that case needs.

1. Five cases with a ~600-token prompt: first request, identical request, same start + extra text,
   changed beginning, same document + new question.
2. Document length 100 / 300 / 600 tokens: TTFT cold vs reusing the document (new question).

Writes to results/: cases.csv, lengths.csv, environment.json, chart.png
"""

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, summarize, write_csv, write_json
from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import wikitext2_test_ids
from gpt2lab.prefix_cache import PrefixStore, prefill_with_prefix_cache

RESULTS = Path(__file__).parent.parent / "results"
DOC_TOKENS = 580  # document part of the ~600-token prompt
LENGTHS = [100, 300, 600]
Q1 = "\n\nQuestion: Summarise the passage above in one sentence."
Q2 = "\n\nQuestion: What is the main subject of this text?"
EXTRA = " Use simple words, and mention the most important name, place or date from the passage."
CHANGE = "Note: "
TIMED_RUNS = 7  # each run is under 2 s, so more runs are cheap and steady the median


def tokens(tokenizer, text):
    return tokenizer(text).input_ids


@torch.no_grad()
def ttft(model, prompt, earlier):
    """Time to first token for `prompt`, with a store that has already seen `earlier` prompts.
    Returns (seconds, tokens reused, first token)."""
    store = PrefixStore()
    for ids in earlier:  # set-up, not timed
        prefill_with_prefix_cache(model, torch.tensor([ids]), store)
    start = time.perf_counter()
    logits, _, reused = prefill_with_prefix_cache(model, torch.tensor([prompt]), store)
    first = logits[0, -1].argmax().item()
    return time.perf_counter() - start, reused, first


def measure(model, cases):
    """cases: {name: (prompt, earlier prompts)}. Interleaved timed runs; returns one row per case."""
    for prompt, earlier in cases.values():  # warm-up
        ttft(model, prompt, earlier)
    times = {name: [] for name in cases}
    info = {}
    for _ in range(TIMED_RUNS):
        for name, (prompt, earlier) in cases.items():
            seconds, reused, first = ttft(model, prompt, earlier)
            times[name].append(seconds)
            info[name] = (len(prompt), reused, first)
    rows = []
    for name in cases:
        s = summarize(times[name])
        n, reused, first = info[name]
        rows.append(dict(case=name, prompt_tokens=n, reused_tokens=reused, computed_tokens=n - reused,
                         ttft_median_ms=round(1000 * s["median"], 1), ttft_min_ms=round(1000 * s["min"], 1),
                         ttft_max_ms=round(1000 * s["max"], 1), first_token=first))
    return rows


def draw_chart(cases, lengths, title):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#c0392b" if r["reused_tokens"] == 0 else "#1e8449" for r in cases]
    bars = left.barh([r["case"] for r in cases][::-1], [r["ttft_median_ms"] for r in cases][::-1], color=colors[::-1])
    for bar, r in zip(bars, cases[::-1]):
        left.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                  f"  {r['ttft_median_ms']:.0f} ms, reused {r['reused_tokens']}/{r['prompt_tokens']}", va="center")
    left.set(xlabel="time to first token, ms (median)", title=f"~{cases[0]['prompt_tokens']}-token prompt")
    left.set_xlim(0, max(r["ttft_median_ms"] for r in cases) * 1.6)
    left.grid(axis="x", alpha=0.3)

    for mode, color, label in (("cold", "#c0392b", "no reuse"), ("warm", "#1e8449", "document reused")):
        rows = [r for r in lengths if r["mode"] == mode]
        right.plot([r["prompt_tokens"] for r in rows], [r["ttft_median_ms"] for r in rows], "o-", color=color, label=label)
    right.set(xlabel="prompt tokens (document + question)", ylabel="time to first token, ms (median)",
              title="Time to first token vs prompt length")
    right.legend()
    right.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)


def read_csv(path):
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v.replace(".", "", 1).isdigit() else v) for k, v in row.items()})
    return rows


def main():
    if "--chart-only" in sys.argv:
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "cases.csv"), read_csv(RESULTS / "lengths.csv"),
                   f"GPT-2 small, prefix caching, on {env['cpu']}, {env['threads']} threads")
        return
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    wiki = wikitext2_test_ids(tokenizer)[0].tolist()
    print(f"CPU load at start: {load_at_start}", flush=True)

    # 1. The five cases.
    doc = wiki[:DOC_TOKENS]
    q1, q2, extra, change = (tokens(tokenizer, t) for t in (Q1, Q2, EXTRA, CHANGE))
    p1 = doc + q1
    cases = {
        "first request": (p1, []),
        "identical request": (p1, [p1]),
        "same start + extra text": (p1 + extra, [p1]),
        "changed beginning": (change + doc + q1, [p1]),
        "same document, new question": (doc + q2, [p1]),
    }
    case_rows = measure(model, cases)
    # Same prompt must give the same first token whether reused or not.
    by_name = {r["case"]: r for r in case_rows}
    assert by_name["first request"]["first_token"] == by_name["identical request"]["first_token"]
    for r in case_rows:
        print(f"{r['case']:30s} {r['prompt_tokens']:4d} tokens, reused {r['reused_tokens']:4d}: "
              f"TTFT {r['ttft_median_ms']:7.1f} ms ({r['ttft_min_ms']}-{r['ttft_max_ms']})", flush=True)
    write_csv(RESULTS / "cases.csv", case_rows)

    # 2. Document length sweep: cold vs warm (same document seen before with another question).
    length_rows = []
    for n in LENGTHS:
        d = wiki[:n]
        rows = measure(model, {"cold": (d + q2, []), "warm": (d + q2, [d + q1])})
        for r in rows:
            length_rows.append(dict(mode=r["case"], document_tokens=n, prompt_tokens=r["prompt_tokens"],
                                    reused_tokens=r["reused_tokens"], ttft_median_ms=r["ttft_median_ms"],
                                    ttft_min_ms=r["ttft_min_ms"], ttft_max_ms=r["ttft_max_ms"]))
        cold, warm = rows
        print(f"document {n:4d} tokens: cold {cold['ttft_median_ms']:7.1f} ms, warm {warm['ttft_median_ms']:6.1f} ms "
              f"({cold['ttft_median_ms'] / warm['ttft_median_ms']:.1f}x)", flush=True)
    write_csv(RESULTS / "lengths.csv", length_rows)

    # Memory and the tokenizer boundary example.
    store = PrefixStore()
    prefill_with_prefix_cache(model, torch.tensor([p1]), store)
    a, b = tokenizer("The answer is 4").input_ids, tokenizer("The answer is 42").input_ids
    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        document="WikiText-2 raw test set, first tokens",
        questions=[Q1, Q2],
        extra_text=EXTRA,
        changed_beginning=CHANGE,
        store_bytes_for_one_prompt=store.nbytes(),
        store_prompt_tokens=len(p1),
        tokenizer_boundary_example={"The answer is 4": a, "The answer is 42": b},
        timed_runs=TIMED_RUNS,
        statistic="median",
    )
    write_json(RESULTS / "environment.json", env)
    print(f"Store: {store.nbytes() / 2**20:.1f} MiB for one {len(p1)}-token prompt", flush=True)
    draw_chart(case_rows, length_rows, f"GPT-2 small, prefix caching, on {env['cpu']}, {env['threads']} threads")
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

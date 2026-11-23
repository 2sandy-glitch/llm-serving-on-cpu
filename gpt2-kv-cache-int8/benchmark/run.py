"""Measure the int8 KV cache: memory, quality through the cache, greedy text, and speed.

Run from gpt2-kv-cache-int8/:   uv run python benchmark/run.py      (~20 minutes on a 6-core CPU)
Redraw the chart only:          uv run python benchmark/run.py --chart-only

1. Memory: bytes per cached token for each setting.
2. Quality through the cache: WikiText-2 test text, 32 windows of 512 tokens, fed ONE token at a
   time through each cache (teacher forcing: always the real next token, never the model's guess).
   Every prediction is scored, giving a "cached perplexity". All settings run side by side, so each
   one is also compared with the float32 cache at every position: how often the top choice
   matches, and how far the scores move. (gpt2lab.perplexity runs whole windows without a cache,
   so it can't see rounding inside the cache.)
3. Greedy text: 8 prompts x 200 new tokens per int8 setting. Where does the text first differ from
   the float32 cache's text?
4. Speed: tokens/s for 100 / 400 / 800 new tokens, float32 vs int8 (gpt2lab.bench method: warm-up,
   3 timed runs, median and min-max; the settings take turns run by run).

Writes to results/: quality.csv, greedy.csv, speed.csv, environment.json, chart.png
"""

import csv
import json
import math
import sys
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, setup, summarize, timed_generate, write_csv, write_json
from gpt2lab.int8_kv_cache import QuantizedKVCache
from gpt2lab.kv_cache import generate_with_cache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import wikitext2_test_ids

RESULTS = Path(__file__).parent.parent / "results"

# (name, key mode, value mode, bits). 4-bit rows test sensitivity only: stored one value per byte.
SETTINGS = [
    ("float32", "none", "none", 8),
    ("int8 K+V per token", "per_token", "per_token", 8),
    ("int8 K per channel + V per token", "per_channel", "per_token", 8),
    ("int8 K only, per token", "per_token", "none", 8),
    ("int8 V only, per token", "none", "per_token", 8),
    ("4-bit K only, per token", "per_token", "none", 4),
    ("4-bit K only, per channel", "per_channel", "none", 4),
    ("4-bit V only, per token", "none", "per_token", 4),
    ("4-bit K per channel + V per token", "per_channel", "per_token", 4),
]
WINDOWS, WINDOW_LEN = 32, 512
GREEDY_TOKENS = 200
GREEDY_PROMPTS = [
    "In a small village by the sea, there lived an old fisherman who",
    "The history of the Roman Empire",
    "Scientists have discovered a new species of",
    "def fibonacci(n):",
    "The best way to learn a new language is",
    "Once upon a time, in a land far away,",
    "The stock market fell sharply on Monday after",
    "Photosynthesis is the process by which",
]
SPEED_PROMPT = GREEDY_PROMPTS[0]
SPEED_TOKENS = [100, 400, 800]
SPEED_SETTINGS = ["float32", "int8 K+V per token", "int8 K per channel + V per token"]
TIMED_RUNS = 3


def make_cache(model, name, batch_size=1, max_tokens=None):
    _, key_mode, value_mode, bits = next(s for s in SETTINGS if s[0] == name)
    return QuantizedKVCache.for_model(model, max_tokens=max_tokens, batch_size=batch_size,
                                      key_mode=key_mode, value_mode=value_mode, bits=bits)


@torch.no_grad()
def quality_through_cache(model, ids):
    """Teacher-forced decoding of WINDOWS x WINDOW_LEN tokens, one position at a time, for every
    setting side by side. Returns one row per setting."""
    batch = ids[0, : WINDOWS * WINDOW_LEN].reshape(WINDOWS, WINDOW_LEN)
    names = [s[0] for s in SETTINGS]
    caches = {n: make_cache(model, n, batch_size=WINDOWS, max_tokens=WINDOW_LEN) for n in names}
    nll = {n: 0.0 for n in names}
    agree = {n: 0 for n in names}
    max_diff = {n: 0.0 for n in names}
    sum_diff = {n: 0.0 for n in names}
    scored = 0
    for t in range(WINDOW_LEN - 1):
        target = batch[:, t + 1]
        logits = {n: model(batch[:, t : t + 1], caches[n])[:, -1].float() for n in names}
        ref = logits["float32"]
        ref_top = ref.argmax(dim=-1)
        for n in names:
            nll[n] += F.cross_entropy(logits[n], target, reduction="sum").item()
            agree[n] += (logits[n].argmax(dim=-1) == ref_top).sum().item()
            diff = (logits[n] - ref).abs()
            max_diff[n] = max(max_diff[n], diff.max().item())
            sum_diff[n] += diff.mean(dim=-1).sum().item()
        scored += WINDOWS
        if (t + 1) % 64 == 0:
            print(f"  quality: position {t + 1} of {WINDOW_LEN - 1}", flush=True)
    base = math.exp(nll["float32"] / scored)
    rows = []
    for name, key_mode, value_mode, bits in SETTINGS:
        ppl = math.exp(nll[name] / scored)
        cache = caches[name]
        rows.append(dict(
            setting=name,
            key_mode=key_mode,
            value_mode=value_mode,
            bits=bits,
            bytes_per_token=round(cache.bytes_per_token()) if bits == 8 else "",  # 4-bit isn't packed
            cached_perplexity=round(ppl, 3),
            perplexity_change_pct=round(100 * (ppl / base - 1), 3),
            top1_same_as_float_pct=round(100 * agree[name] / scored, 2),
            max_score_diff=round(max_diff[name], 3),
            mean_score_diff=round(sum_diff[name] / scored, 4),
        ))
    return rows, scored


@torch.no_grad()
def no_cache_perplexity(model, ids):
    """The same windows read whole, without any cache: float32 cached perplexity must equal this."""
    batch = ids[0, : WINDOWS * WINDOW_LEN].reshape(WINDOWS, WINDOW_LEN)
    nll = 0.0
    for row in batch.split(8):
        logits = model(row)
        nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), row[:, 1:].reshape(-1), reduction="sum").item()
    return math.exp(nll / (WINDOWS * (WINDOW_LEN - 1)))


def draw_chart(quality, speed, title):
    """Three charts: memory per token, quality cost, speed."""
    fig, (mem, qual, spd) = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw=dict(width_ratios=[0.8, 1.2, 1]))
    eight = [r for r in quality if r["bits"] == 8 and r["key_mode"] != "none" and r["value_mode"] != "none"]
    eight = [quality[0]] + eight
    short = {"float32": "float32", "int8 K+V per token": "int8\nper token",
             "int8 K per channel + V per token": "int8\nK per channel"}
    mem.bar([short[r["setting"]] for r in eight], [r["bytes_per_token"] / 1024 for r in eight],
            color=["#7f8c8d", "#2471a3", "#1e8449"])
    mem.set(ylabel="KiB per cached token", title="Memory per token")
    for i, r in enumerate(eight):
        mem.text(i, r["bytes_per_token"] / 1024, f"{r['bytes_per_token']:,} B", ha="center", va="bottom")

    labels = [r["setting"] for r in quality[1:]]
    qual.barh(labels[::-1], [r["perplexity_change_pct"] for r in quality[1:]][::-1],
              color=["#c0392b" if r["bits"] == 4 else "#2471a3" for r in quality[1:]][::-1])
    qual.set(xlabel="perplexity change vs float32 cache (%)", title="Quality cost (lower is better)")
    qual.set_xscale("symlog", linthresh=0.1)
    qual.grid(axis="x", alpha=0.3)

    width = 0.27
    for i, name in enumerate(SPEED_SETTINGS):
        rows = [r for r in speed if r["setting"] == name]
        spd.bar([x + (i - 1) * width for x in range(len(SPEED_TOKENS))], [r["tokens_per_s"] for r in rows], width,
                label=name, color=["#7f8c8d", "#2471a3", "#1e8449"][i])
    spd.set_xticks(range(len(SPEED_TOKENS)), [str(n) for n in SPEED_TOKENS])
    spd.set(xlabel="new tokens generated", ylabel="tokens per second (median)", title="Speed")
    spd.legend(fontsize=8, loc="lower left")
    spd.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)


def read_csv(path):
    """Rows from a results CSV, with numbers turned back into numbers."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    row[key] = float(value) if "." in value else int(value)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def main():
    if "--chart-only" in sys.argv:  # redraw chart.png from the saved results, no measuring
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "quality.csv"), read_csv(RESULTS / "speed.csv"),
                   f"GPT-2 small, int8 KV cache, on {env['cpu']}, {env['threads']} threads")
        return
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    print(f"CPU load at start: {load_at_start}", flush=True)

    # 1 + 2. Memory and quality through the cache.
    ids = wikitext2_test_ids(tokenizer)
    quality, scored = quality_through_cache(model, ids)
    check = no_cache_perplexity(model, ids)
    float_ppl = quality[0]["cached_perplexity"]
    print(f"Check: float32 cached perplexity {float_ppl} vs no-cache {check:.3f}", flush=True)
    if abs(float_ppl - check) > 1e-2:
        raise AssertionError("float32 through the cache should match the no-cache perplexity")
    for row in quality:
        print(f"{row['setting']:36s} ppl {row['cached_perplexity']:8.3f} ({row['perplexity_change_pct']:+.3f}%)  "
              f"top-1 same {row['top1_same_as_float_pct']:6.2f}%  max diff {row['max_score_diff']}", flush=True)
    write_csv(RESULTS / "quality.csv", quality)

    # 3. Greedy text.
    greedy = []
    for prompt in GREEDY_PROMPTS:
        idx = tokenizer(prompt, return_tensors="pt").input_ids
        ref = generate_with_cache(model, idx, GREEDY_TOKENS, cache=make_cache(model, "float32"))[0, idx.shape[1] :]
        for name, *_ in SETTINGS[1:5]:  # the int8 settings
            out = generate_with_cache(model, idx, GREEDY_TOKENS, cache=make_cache(model, name))[0, idx.shape[1] :]
            differ = (out != ref).nonzero()
            first = differ[0].item() + 1 if len(differ) else None
            greedy.append(dict(setting=name, prompt=prompt, identical=first is None,
                               first_different_token=first or "", same_positions_pct=round(100 * (out == ref).float().mean().item(), 1)))
        print(f"greedy done: {prompt!r}", flush=True)
    write_csv(RESULTS / "greedy.csv", greedy)

    # 4. Speed.
    idx = tokenizer(SPEED_PROMPT, return_tensors="pt").input_ids
    for name in SPEED_SETTINGS:  # warm-up, not counted
        timed_generate(model, idx, 20, lambda n=name: make_cache(model, n))
    speed = []
    for n in SPEED_TOKENS:
        totals = {name: [] for name in SPEED_SETTINGS}
        for run in range(TIMED_RUNS):
            for name in SPEED_SETTINGS:  # take turns
                _, steps, _ = timed_generate(model, idx, n, lambda nm=name: make_cache(model, nm))
                totals[name].append(sum(steps))
        for name in SPEED_SETTINGS:
            s = summarize(totals[name])
            speed.append(dict(setting=name, new_tokens=n, median_s=round(s["median"], 3), min_s=round(s["min"], 3),
                              max_s=round(s["max"], 3), tokens_per_s=round(n / s["median"], 2)))
            print(f"speed {n:4d} tokens  {name:34s} {s['median']:6.2f} s ({s['min']:.2f}-{s['max']:.2f})", flush=True)
    write_csv(RESULTS / "speed.csv", speed)

    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32 model; cache as listed",
        quality_eval=f"WikiText-2 raw test, first {WINDOWS} x {WINDOW_LEN} tokens, teacher-forced one token at a time",
        quality_tokens_scored=scored,
        no_cache_perplexity_same_windows=round(check, 3),
        per_channel_group_size=32,
        greedy_prompts=GREEDY_PROMPTS,
        greedy_new_tokens=GREEDY_TOKENS,
        speed_prompt=SPEED_PROMPT,
        speed_new_tokens=SPEED_TOKENS,
        timed_runs=TIMED_RUNS,
        statistic="median",
    )
    write_json(RESULTS / "environment.json", env)

    draw_chart(quality, speed, f"GPT-2 small, int8 KV cache, on {env['cpu']}, {env['threads']} threads")
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

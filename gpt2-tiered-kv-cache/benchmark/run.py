"""Measure tiered offloading: time to first token when a cached prompt is in RAM, on disk, or gone.

Run from gpt2-tiered-kv-cache/:   uv run python benchmark/run.py      (~8 minutes on a 6-core CPU)

Offload files go to ./offload_tmp/ inside this folder (on the SSD; /tmp is RAM-backed tmpfs on this
machine), and are deleted at the end.

1. One prompt, four situations, at document lengths 100 / 300 / 600 / 1,000 tokens (+ a question):
   - recompute: nothing cached;
   - RAM hit: the document's K and V are in RAM;
   - disk hit: they were pushed to disk and dropped from the OS page cache, so they're read from the SSD;
   - disk hit, page cache warm: same file, but still in the OS page cache (what a naive test measures).
   The timed request only reads (no entry is saved during it); the time to SAVE an entry to disk is
   measured separately. gpt2lab.bench method: warm-up, 7 timed runs, median, min-max.
2. A workload: 16 documents x 6 questions = 96 requests in random order, RAM holding 4 documents.
   RAM-only store (evicted = recomputed later) vs tiered (evicted = saved to disk, up to 16).
   Where each request's document came from (RAM / disk / recomputed), and total time to first token.
   Saving to disk happens inside the request that causes it (synchronous here), and is counted.

Writes to results/: lengths.csv, workload.csv, environment.json, chart.png
"""

import csv
import json
import random
import shutil
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
from gpt2lab.kv_cache import KVCache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import wikitext2_test_ids
from gpt2lab.prefix_cache import PrefixStore, prefill_with_prefix_cache
from gpt2lab.tiered_prefix_store import TieredPrefixStore

HERE = Path(__file__).parent.parent
RESULTS = HERE / "results"
OFFLOAD = HERE / "offload_tmp"
LENGTHS = [100, 300, 600, 1000]
TIMED_RUNS = 7
Q1 = "\n\nQuestion: Summarise the passage above in one sentence."
Q2 = "\n\nQuestion: What is the main subject of this text?"
FILLER = "An unrelated short request that pushes the document out of the RAM tier."
N_DOCS, DOC_TOKENS, QUESTIONS, RAM_DOCS, DISK_DOCS = 16, 500, 6, 4, 16
MODES = ["recompute", "RAM hit", "disk hit", "disk hit, page cache warm"]


def first_token(model, ids, store, store_upto=None):
    start = time.perf_counter()
    logits, _, reused = prefill_with_prefix_cache(model, torch.tensor([ids]), store, store_upto=store_upto)
    logits[0, -1].argmax().item()
    return time.perf_counter() - start, reused


def one_case(model, mode, doc, q1, q2, filler):
    """Set up the store for `mode` (untimed), then time the request doc + q2."""
    shutil.rmtree(OFFLOAD, ignore_errors=True)
    if mode == "recompute":
        store = PrefixStore()
    elif mode == "RAM hit":
        store = PrefixStore()
        first_token(model, doc + q1, store, store_upto=len(doc))
    else:
        store = TieredPrefixStore(ram_entries=1, disk_dir=OFFLOAD, drop_page_cache=(mode == "disk hit"))
        first_token(model, doc + q1, store, store_upto=len(doc))
        first_token(model, filler, store)  # pushes the document to disk
        assert tuple(doc) in store.on_disk
        store.max_entries = 2  # room for the reloaded document: time the read, not another save
    # Store only the document (as it already is), so the timed request never adds or evicts anything.
    seconds, reused = first_token(model, doc + q2, store, store_upto=len(doc))
    if isinstance(store, TieredPrefixStore):
        assert store.last_hit == "disk"
        store.clear_disk()
    return seconds, reused


def save_time(model, doc):
    """Seconds to write one document's K and V to disk (flushed, page cache dropped)."""
    shutil.rmtree(OFFLOAD, ignore_errors=True)
    cache = KVCache(model.config.n_layer)
    with torch.no_grad():
        model(torch.tensor([doc]), cache)
    store = TieredPrefixStore(ram_entries=1, disk_dir=OFFLOAD)
    start = time.perf_counter()
    store._save(tuple(doc), cache.k, cache.v)
    seconds = time.perf_counter() - start
    size = store.disk_bytes()
    store.clear_disk()
    return seconds, size


def lengths_part(model, wiki, q1, q2, filler):
    rows = []
    for n in LENGTHS:
        doc = wiki[:n]
        for mode in MODES:  # warm-up
            one_case(model, mode, doc, q1, q2, filler)
        times = {m: [] for m in MODES}
        saves, size, reused = [], 0, {}
        for _ in range(TIMED_RUNS):
            for m in MODES:  # take turns
                s, r = one_case(model, m, doc, q1, q2, filler)
                times[m].append(s)
                reused[m] = r
            s, size = save_time(model, doc)
            saves.append(s)
        for m in MODES:
            st = summarize(times[m])
            rows.append(dict(document_tokens=n, prompt_tokens=n + len(q2), mode=m, reused_tokens=reused[m],
                             ttft_median_ms=round(1000 * st["median"], 1), ttft_min_ms=round(1000 * st["min"], 1),
                             ttft_max_ms=round(1000 * st["max"], 1), file_mib=round(size / 2**20, 1),
                             save_median_ms=round(1000 * statistics.median(saves), 1)))
        print(f"document {n:5d} tokens ({size / 2**20:5.1f} MiB): " + "  ".join(
            f"{r['mode']} {r['ttft_median_ms']} ms" for r in rows[-4:]) + f"  save {rows[-1]['save_median_ms']} ms", flush=True)
    return rows


def workload_part(model, wiki, tokenizer):
    docs = [wiki[i * 5000 : i * 5000 + DOC_TOKENS] for i in range(N_DOCS)]
    qs = [tokenizer(f"\n\nQuestion {j}: tell me something about the passage.").input_ids for j in range(QUESTIONS)]
    requests = [(d, j) for d in range(N_DOCS) for j in range(QUESTIONS)]
    random.Random(0).shuffle(requests)
    rows = []
    for name in ("RAM only", "RAM + disk"):
        shutil.rmtree(OFFLOAD, ignore_errors=True)
        store = PrefixStore(max_entries=RAM_DOCS) if name == "RAM only" else \
            TieredPrefixStore(ram_entries=RAM_DOCS, disk_dir=OFFLOAD, disk_entries=DISK_DOCS)
        where = {"RAM": 0, "disk": 0, "recomputed": 0}
        total = 0.0
        for d, j in requests:
            if isinstance(store, TieredPrefixStore):
                store.last_hit = None
            _, shared = store.lookup(docs[d] + qs[j], touch=False)
            seconds, reused = first_token(model, docs[d] + qs[j], store, store_upto=DOC_TOKENS)
            total += seconds
            if reused >= DOC_TOKENS:
                where["disk" if isinstance(store, TieredPrefixStore) and store.last_hit == "disk" else "RAM"] += 1
            else:
                where["recomputed"] += 1
        rows.append(dict(store=name, requests=len(requests), from_ram=where["RAM"], from_disk=where["disk"],
                         recomputed=where["recomputed"], total_ttft_s=round(total, 2),
                         mean_ttft_ms=round(1000 * total / len(requests), 1)))
        print(f"{name:10s}: RAM {where['RAM']:3d}  disk {where['disk']:3d}  recomputed {where['recomputed']:3d}  "
              f"total {total:6.2f} s", flush=True)
        if isinstance(store, TieredPrefixStore):
            store.clear_disk()
    return rows


def draw_chart(lengths, workload, title):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    colors = {"recompute": "#c0392b", "RAM hit": "#1e8449", "disk hit": "#2471a3", "disk hit, page cache warm": "#85c1e9"}
    for m in MODES:
        rs = [r for r in lengths if r["mode"] == m]
        a.plot([r["document_tokens"] for r in rs], [r["ttft_median_ms"] for r in rs], "o-", color=colors[m], label=m)
    a.set(xlabel="document tokens (K and V cached or not)", ylabel="time to first token, ms (median)",
          title="Where the cached document comes from")
    a.legend()
    a.grid(alpha=0.3)
    names = [r["store"] for r in workload]
    bottom = [0] * len(workload)
    for key, label, color in (("from_ram", "from RAM", "#1e8449"), ("from_disk", "from disk", "#2471a3"),
                              ("recomputed", "recomputed", "#c0392b")):
        vals = [r[key] for r in workload]
        b.bar(names, vals, bottom=bottom, color=color, label=label)
        bottom = [x + y for x, y in zip(bottom, vals)]
    for i, r in enumerate(workload):
        b.text(i, r["requests"] + 1, f"total {r['total_ttft_s']:.1f} s", ha="center", va="bottom")
    b.set(ylabel="requests", title=f"{N_DOCS} documents, RAM holds {RAM_DOCS}", ylim=(0, workload[0]["requests"] * 1.35))
    b.legend(loc="upper center", ncol=3)
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
    title = "GPT-2 small, tiered KV offloading (RAM -> SSD), on {}, {} threads"
    if "--chart-only" in sys.argv:
        env = json.loads((RESULTS / "environment.json").read_text())
        draw_chart(read_csv(RESULTS / "lengths.csv"), read_csv(RESULTS / "workload.csv"), title.format(env["cpu"], env["threads"]))
        return
    load_at_start = setup()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2()
    wiki = wikitext2_test_ids(tokenizer)[0].tolist()
    q1, q2, filler = (tokenizer(t).input_ids for t in (Q1, Q2, FILLER))
    print(f"CPU load at start: {load_at_start}", flush=True)
    try:
        lengths = lengths_part(model, wiki, q1, q2, filler)
        write_csv(RESULTS / "lengths.csv", lengths)
        workload = workload_part(model, wiki, tokenizer)
        write_csv(RESULTS / "workload.csv", workload)
    finally:
        shutil.rmtree(OFFLOAD, ignore_errors=True)
    env = environment(
        load_at_start,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32 (also on disk)",
        disk="SATA SSD (EVM25 512 GB); files written with torch.save, fsync, then posix_fadvise DONTNEED",
        lengths=LENGTHS,
        workload=f"{N_DOCS} documents x {QUESTIONS} questions, shuffled (seed 0); RAM {RAM_DOCS} documents, disk {DISK_DOCS}",
        timed_runs=TIMED_RUNS,
        statistic="median",
    )
    write_json(RESULTS / "environment.json", env)
    draw_chart(lengths, workload, title.format(env["cpu"], env["threads"]))
    print("Wrote results/ files.", flush=True)


if __name__ == "__main__":
    main()

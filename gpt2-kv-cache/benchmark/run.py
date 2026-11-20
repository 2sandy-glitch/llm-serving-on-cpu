"""Measure our GPT-2 with and without our KV cache on the CPU.

Run from gpt2-kv-cache/:       uv run python -m benchmark.run
Watch it live (optional, in another terminal): uv run python benchmark/dashboard/serve.py

Writes:
  results/results.csv        total time and tokens/sec for each setting (median of the timed runs)
  results/per_token.csv      time for each new token by its position (longest setting)
  results/environment.json   the machine, versions and settings these numbers came from
  results/chart.png          the two charts used in the README
  benchmark/dashboard/progress.json   live progress for the dashboard (not committed)
"""

import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import GPT2TokenizerFast

from gpt2_kv.load_weights import load_gpt2
from gpt2_kv.kv_cache import KVCache

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
PROMPT = "In a small village by the sea, there lived an old fisherman who"
NEW_TOKENS = [50, 100, 200, 400]
MODES = ["no_cache", "with_cache"]
TIMED_RUNS = 3
WARMUP_TOKENS = 20
THREADS = 6


@torch.no_grad()
def timed_generate(model, idx, max_new_tokens, use_cache, on_step=None):
    """Same greedy loop as model.generate(), but records how long each new token takes.

    Returns (all token ids, list of seconds per new token, cache bytes at the end).
    on_step(step_number, seconds) is called after every token, for live progress.
    """
    step_times = []
    cache = KVCache(model.config.n_layer) if use_cache else None
    next_input = idx
    for step in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(next_input, cache) if use_cache else model(idx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seconds = time.perf_counter() - start
        step_times.append(seconds)
        idx = torch.cat([idx, next_id], dim=1)
        next_input = next_id  # with a cache, only the newest token goes in next time
        if on_step:
            on_step(step + 1, seconds)
    return idx, step_times, (cache.nbytes() if cache else 0)


class LiveProgress:
    """Writes benchmark/dashboard/progress.json while the benchmark runs. Only for watching; it never
    changes what is measured (the file is written outside the timed part of each step)."""

    def __init__(self, prompt_tokens, previous):
        self.path = HERE / "dashboard" / "progress.json"
        self.path.parent.mkdir(exist_ok=True)
        self.prompt_tokens = prompt_tokens
        self.started = time.time()
        self.last_write = 0.0
        # Expected seconds for one run of each setting: from the previous results if we have
        # them, otherwise a rough guess from how much text the model must read.
        self.expected = {}
        for n in NEW_TOKENS:
            for mode in MODES:
                if (mode, n) in previous:
                    self.expected[(mode, n)] = previous[(mode, n)]
                elif mode == "no_cache":
                    self.expected[(mode, n)] = 0.0012 * sum(prompt_tokens + i for i in range(n))
                else:
                    self.expected[(mode, n)] = 0.03 * n
        self.expected[("warmup", 0)] = sum(
            self.expected.get((m, NEW_TOKENS[0]), 1) * WARMUP_TOKENS / NEW_TOKENS[0] for m in MODES
        )
        self.total_work = self.expected[("warmup", 0)] + TIMED_RUNS * sum(
            self.expected[(m, n)] for n in NEW_TOKENS for m in MODES
        )
        self.done_work = 0.0
        self.state = dict(
            status="starting",
            prompt=PROMPT,
            prompt_tokens=prompt_tokens,
            new_tokens=NEW_TOKENS,
            timed_runs=TIMED_RUNS,
            threads=THREADS,
            cpu=None,
            current=None,
            results=[],
            checks=[],
            curves={},
            log=[],
        )

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.state["log"].append(f"{stamp}  {text}")
        self.state["log"] = self.state["log"][-40:]
        print(text, flush=True)
        self.write(force=True)

    def start_run(self, label, mode, n, run, expected_key):
        self.run_key = expected_key
        self.run_work = self.expected[expected_key]
        self.run_n = n
        self.run_mode = mode
        self.state["status"] = "warm-up" if run == 0 else "measuring"
        self.state["current"] = dict(label=label, mode=mode, new_tokens=n, run=run, of_runs=TIMED_RUNS, step=0)
        self.curve_key = f"{mode}|{n}"
        if run > 0:
            self.state["curves"][self.curve_key] = []
        self.write(force=True)

    def _run_fraction(self, step):
        """Share of this run's work done after `step` tokens (later no-cache steps cost more)."""
        if self.run_mode == "no_cache":
            done = sum(self.prompt_tokens + i for i in range(step))
            total = sum(self.prompt_tokens + i for i in range(self.run_n))
            return done / total
        return step / self.run_n

    def step(self, step, seconds):
        self.state["current"]["step"] = step
        if self.state["status"] == "measuring":
            self.state["curves"][self.curve_key].append(round(1000 * seconds, 2))
        self.write()

    def end_run(self):
        self.done_work += self.run_work
        self.write(force=True)

    def write(self, force=False):
        now = time.time()
        if not force and now - self.last_write < 0.5:
            return
        self.last_write = now
        cur = self.state.get("current")
        done = self.done_work
        if cur and self.state["status"] in ("warm-up", "measuring"):
            done += self.run_work * self._run_fraction(cur["step"])
        elapsed = now - self.started
        fraction = min(done / self.total_work, 1.0) if self.total_work else 0
        if self.state["status"] == "done":
            fraction, eta = 1.0, 0
        elif done > 0:
            eta = elapsed / done * (self.total_work - done)  # pace so far x work left
        else:
            eta = self.total_work
        self.state.update(
            updated=datetime.now().isoformat(timespec="seconds"),
            elapsed_s=round(elapsed, 1),
            eta_s=round(eta, 1),
            finish_at=datetime.fromtimestamp(now + eta).strftime("%H:%M:%S"),
            overall_fraction=round(fraction, 4),
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)  # swap in one go, so the page never reads half a file


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def read_previous_results():
    """Totals from the last run, used only to estimate time left."""
    previous = {}
    try:
        with open(RESULTS / "results.csv") as f:
            for row in csv.DictReader(f):
                previous[(row["mode"], int(row["new_tokens"]))] = float(row["total_s"])
    except (OSError, KeyError, ValueError):
        pass
    return previous


def main():
    load_at_start = round(os.getloadavg()[0], 2)  # how busy the CPU was before we began
    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = load_gpt2().float().eval()
    idx = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_tokens = idx.shape[1]

    live = LiveProgress(prompt_tokens, read_previous_results())
    live.state["cpu"] = cpu_name()
    live.log(f"Loaded GPT-2 (own model). Prompt is {prompt_tokens} tokens. CPU load at start: {load_at_start}")

    # Warm-up (not counted): the first runs are slower while PyTorch sets things up.
    live.start_run("warm-up (not counted)", "warmup", WARMUP_TOKENS, 0, ("warmup", 0))
    for use_cache in (False, True):
        timed_generate(model, idx, WARMUP_TOKENS, use_cache, on_step=live.step)
    live.end_run()
    live.log("Warm-up done")

    rows, per_token = [], {}
    for n in NEW_TOKENS:
        outputs = {}
        for use_cache in (False, True):
            mode = "with_cache" if use_cache else "no_cache"
            totals, steps_per_run = [], []
            for run in range(1, TIMED_RUNS + 1):
                live.start_run(f"{n} tokens, {mode.replace('_', ' ')}, run {run} of {TIMED_RUNS}", mode, n, run, (mode, n))
                out, steps, cache_bytes = timed_generate(model, idx, n, use_cache, on_step=live.step)
                live.end_run()
                totals.append(sum(steps))
                steps_per_run.append(steps)
                live.log(f"{n} tokens {mode}: run {run} took {sum(steps):.2f} s")
            outputs[mode] = out
            total = statistics.median(totals)
            row = dict(
                mode=mode,
                new_tokens=n,
                total_s=round(total, 4),
                tokens_per_s=round(n / total, 2),
                ms_per_token=round(1000 * total / n, 2),
                cache_mb=round(cache_bytes / 2**20, 2),
            )
            rows.append(row)
            live.state["results"].append(row)
            live.log(f"{n:4d} tokens  {mode:10s}  median {total:7.2f} s  {n / total:6.1f} tok/s")
            if n == max(NEW_TOKENS):
                # median time of each position across the timed runs
                per_token[mode] = [statistics.median(s) for s in zip(*steps_per_run)]
        same = torch.equal(outputs["no_cache"], outputs["with_cache"])
        live.state["checks"].append(dict(new_tokens=n, same_text=same))
        live.log(f"{n} tokens: same text with and without the cache: {'yes' if same else 'NO'}")
        if not same:
            live.state["status"] = "failed"
            live.write(force=True)
            raise AssertionError(f"text differs with and without the cache at {n} tokens")

    with open(RESULTS / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULTS / "per_token.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "no_cache_ms", "with_cache_ms"])
        for i, (a, b) in enumerate(zip(per_token["no_cache"], per_token["with_cache"]), start=1):
            writer.writerow([i, round(1000 * a, 3), round(1000 * b, 3)])

    c = model.config
    environment = dict(
        date=datetime.now().isoformat(timespec="seconds"),
        cpu=cpu_name(),
        cpu_cores=os.cpu_count(),
        threads=torch.get_num_threads(),
        load_average_1min_at_start=load_at_start,
        load_average_1min_at_end=round(os.getloadavg()[0], 2),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        model="GPT-2 small (124M), own implementation, OpenAI weights",
        dtype="float32",
        decoding="greedy",
        prompt=PROMPT,
        prompt_tokens=prompt_tokens,
        new_tokens=NEW_TOKENS,
        timed_runs=TIMED_RUNS,
        statistic="median",
        kv_bytes_per_token_formula=2 * c.n_layer * c.n_head * (c.n_embd // c.n_head) * 4,
    )
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    # Charts: time per token by position, and total time per setting.
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = range(1, len(per_token["no_cache"]) + 1)
    left.plot(positions, [1000 * t for t in per_token["no_cache"]], label="no cache", color="#c0392b")
    left.plot(positions, [1000 * t for t in per_token["with_cache"]], label="with KV cache", color="#2471a3")
    left.set(xlabel="new token number", ylabel="ms for this token", title="Time for each new token")
    left.legend()
    left.grid(alpha=0.3)

    width = 0.38
    xs = range(len(NEW_TOKENS))
    no = [r["total_s"] for r in rows if r["mode"] == "no_cache"]
    yes = [r["total_s"] for r in rows if r["mode"] == "with_cache"]
    right.bar([x - width / 2 for x in xs], no, width, label="no cache", color="#c0392b")
    right.bar([x + width / 2 for x in xs], yes, width, label="with KV cache", color="#2471a3")
    right.set_xticks(list(xs), [str(n) for n in NEW_TOKENS])
    right.set(xlabel="new tokens generated", ylabel="seconds (median)", title="Total generation time")
    right.legend()
    right.grid(axis="y", alpha=0.3)

    fig.suptitle(f"GPT-2 small from scratch on {environment['cpu']}, {THREADS} threads, no GPU")
    fig.tight_layout()
    fig.savefig(RESULTS / "chart.png", dpi=150)

    live.state["status"] = "done"
    live.state["current"] = None
    live.log("Finished. Wrote results.csv, per_token.csv, environment.json, chart.png")


if __name__ == "__main__":
    main()

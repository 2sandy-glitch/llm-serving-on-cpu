"""The benchmark method every experiment uses, so all numbers in this repo are measured the same way.

  1. Fix the thread count and the random seed (setup).
  2. Warm up first and throw that run away: PyTorch is slower the first time it runs something.
  3. Do several timed runs and report the median (a typical run) and the min-max (how much it varied).
  4. Time only the model's work, with time.perf_counter().
  5. Save environment.json: the machine, versions and CPU load, so anyone can judge the numbers.

Typical use:
    load_at_start = setup()
    stats = repeat(lambda: sum(timed_generate(model, idx, 100, cache_factory)[1]))
    write_json(results / "environment.json", environment(load_at_start, prompt=PROMPT))
"""

import csv
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import transformers

THREADS = 6  # this machine has 6 cores (Intel i5-9500)


def setup(threads: int = THREADS, seed: int = 0) -> float:
    """Fix threads and seed. Returns the 1-minute CPU load before anything starts."""
    load_at_start = round(os.getloadavg()[0], 2)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    return load_at_start


def summarize(values: list[float]) -> dict:
    """Median, min and max of a list of timed runs."""
    return dict(median=statistics.median(values), min=min(values), max=max(values), runs=len(values))


def repeat(run: Callable[[], float], runs: int = 3, warmup: int = 1) -> dict:
    """Call run() `warmup` times (not counted), then `runs` times. run() returns the seconds it
    measured itself, so only the model's work is counted. Returns summarize() of the timed runs."""
    for _ in range(warmup):
        run()
    return summarize([run() for _ in range(runs)])


@torch.no_grad()
def timed_generate(model, idx: torch.Tensor, max_new_tokens: int, cache_factory=None, on_step=None):
    """Greedy generation that records how long each new token takes.

    cache_factory: None for no cache (re-read the whole text every step), or a function that
    returns a fresh, empty cache, e.g. lambda: KVCache(model.config.n_layer).
    on_step(step_number, seconds) is called after every token, outside the timed part.

    Returns (all token ids, seconds for each new token, the cache or None).
    """
    cache = cache_factory() if cache_factory else None
    step_times = []
    next_input = idx
    for step in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(next_input, cache) if cache is not None else model(idx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_times.append(time.perf_counter() - start)
        idx = torch.cat([idx, next_id], dim=1)
        next_input = next_id  # with a cache, only the newest token goes in next time
        if on_step:
            on_step(step + 1, step_times[-1])
    return idx, step_times, cache


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment(load_at_start: float, **settings) -> dict:
    """The machine, versions and CPU load behind a set of numbers, plus the experiment's settings."""
    return dict(
        date=datetime.now().isoformat(timespec="seconds"),
        cpu=cpu_name(),
        cpu_cores=os.cpu_count(),
        threads=torch.get_num_threads(),
        load_average_1min_at_start=load_at_start,
        load_average_1min_at_end=round(os.getloadavg()[0], 2),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        **settings,
    )


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

"""gpt2lab.bench checks: the shared benchmark helper counts, times and records correctly.
Run from the repo root: uv run pytest tests"""

import json

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.bench import environment, repeat, summarize, timed_generate, write_csv, write_json
from gpt2lab.kv_cache import KVCache, generate_with_cache
from gpt2lab.load_weights import load_gpt2


@pytest.fixture(scope="module")
def model_and_prompt():
    idx = GPT2TokenizerFast.from_pretrained("gpt2")("The cat sat on the", return_tensors="pt").input_ids
    return load_gpt2(), idx


def test_summarize():
    assert summarize([3.0, 1.0, 2.0]) == dict(median=2.0, min=1.0, max=3.0, runs=3)


def test_repeat_throws_away_warmup():
    """Warm-up runs happen but are not counted in the result."""
    calls = []

    def run():
        calls.append(1)
        return 100.0 if len(calls) == 1 else 1.0  # the first (warm-up) run is slow

    stats = repeat(run, runs=3, warmup=1)
    assert len(calls) == 4
    assert stats == dict(median=1.0, min=1.0, max=1.0, runs=3)


def test_timed_generate_matches_plain_generation(model_and_prompt):
    """Same tokens as model.generate() (no cache) and generate_with_cache(); one time per token."""
    model, idx = model_and_prompt
    n = 10
    no_cache, steps, cache = timed_generate(model, idx, n)
    assert cache is None and len(steps) == n
    assert torch.equal(no_cache, model.generate(idx, n))

    with_cache, steps, cache = timed_generate(model, idx, n, lambda: KVCache(model.config.n_layer))
    assert len(steps) == n and all(s > 0 for s in steps)
    assert torch.equal(with_cache, generate_with_cache(model, idx, n))
    assert len(cache) == idx.shape[1] + n - 1  # the last new token is never fed back in


def test_environment_and_files(tmp_path):
    env = environment(1.5, prompt="hi")
    for key in ("cpu", "threads", "torch", "load_average_1min_at_start", "load_average_1min_at_end"):
        assert key in env
    assert env["prompt"] == "hi"

    write_json(tmp_path / "env.json", env)
    assert json.loads((tmp_path / "env.json").read_text())["prompt"] == "hi"
    write_csv(tmp_path / "r.csv", [dict(a=1, b=2), dict(a=3, b=4)])
    assert (tmp_path / "r.csv").read_text().splitlines() == ["a,b", "1,2", "3,4"]

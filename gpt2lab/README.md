# gpt2lab: the shared code

Every experiment in this repo imports this package instead of copying the model. `uv sync` at the
repo root installs it.

| File | What it is |
|---|---|
| [`model.py`](model.py) | GPT-2 written from scratch in plain PyTorch, with 3 marked hooks for a KV cache |
| [`load_weights.py`](load_weights.py) | Loads OpenAI's weights for `distilgpt2`, `gpt2` or `gpt2-medium` (sizes read from the Hugging Face config) |
| [`kv_cache.py`](kv_cache.py) | My first KV cache: grows with `torch.cat` ([experiment](../gpt2-kv-cache/)) |
| [`static_kv_cache.py`](static_kv_cache.py) | Reserved KV cache: memory taken once, filled in place ([experiment](../gpt2-static-kv-cache/)) |
| [`int8_kv_cache.py`](int8_kv_cache.py) | Quantized KV cache: K and V as int8 + scales, per token or per channel ([experiment](../gpt2-kv-cache-int8/)) |
| [`prefix_cache.py`](prefix_cache.py) | Prefix caching: reuse stored K and V of a shared prompt start ([experiment](../gpt2-prefix-cache/)) |
| [`routing.py`](routing.py) | Routing simulation: round robin, least loaded and cache-aware policies over workers with prefix stores ([experiment](../gpt2-cache-aware-routing/)) |
| [`bench.py`](bench.py) | The benchmark method every experiment uses: warm-up, 3 timed runs, median and min-max, `environment.json` |
| [`perplexity.py`](perplexity.py) | Quality score on WikiText-2, used by every experiment that might change the output |

Tests for this package are in [`../tests/`](../tests/): `uv run pytest tests` from the repo root.

## Baseline quality: perplexity on WikiText-2

Perplexity is how surprised the model is by real text (lower is better). Measured on the whole
WikiText-2 test set (287,644 tokens), window 1,024, stride 512, float32:

| Model | Mine | Hugging Face | Same? |
|---|---|---|---|
| GPT-2 small (`gpt2`) | **25.170** | 25.170 | ✅ |

This is the number later experiments compare against: anything that changes the model (smaller
numbers, dropped cache entries) must report how far it moves this.

Run it: `uv run python -m gpt2lab.perplexity` (add `--hf` for Hugging Face's model). About 15
minutes per model on a 6-core CPU.

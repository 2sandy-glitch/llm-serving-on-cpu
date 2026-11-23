# llm-serving-on-cpu

How LLMs really run, rebuilt from scratch and measured on a CPU with no GPU (Intel i5-9500, 6 cores).
One folder per experiment. Each one has its own README with what, why, how to run, and the numbers.

The shared code lives in [`gpt2lab/`](gpt2lab/): GPT-2 written from scratch, the loader for OpenAI's
weights and my KV cache. Every experiment imports it instead of copying the model.

Setup, once: `uv sync` (installs PyTorch for CPU and `gpt2lab`).

| # | Experiment | Folder | Status |
|---|---|---|---|
| 1 | GPT-2 KV cache: own GPT-2 + own KV cache, cache on vs off | [gpt2-kv-cache/](gpt2-kv-cache/) | ✅ 7.6× faster at 400 tokens, 72 KiB per token |
| 2 | Reserved KV cache: memory taken once, filled in place | [gpt2-static-kv-cache/](gpt2-static-kv-cache/) | ✅ cache update 50× cheaper at 1,000 tokens, same text |
| 3 | Quantization: the same model at 16 / 8 / 4 bits | – | Next |
| 4 | BitNet 1.58-bit on CPU | – | Planned |
| 5 | Speculative decoding (draft + target) | – | Planned |
| 6 | Prompt / prefix caching | – | Planned |

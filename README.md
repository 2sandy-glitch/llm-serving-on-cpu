# llm-serving-on-cpu

How LLMs really run, rebuilt from scratch and measured on a CPU with no GPU (Intel i5-9500, 6 cores).
One folder per experiment. Each one has its own README with what, why, how to run, and the numbers.

The shared code lives in [`gpt2lab/`](gpt2lab/): GPT-2 written from scratch, the loader for OpenAI's
weights and my KV cache. Every experiment imports it instead of copying the model.

Setup, once: `uv sync` (installs PyTorch for CPU and `gpt2lab`).

The current series is **caching**: several caching techniques built into my own GPT-2, each
measured on this CPU. Other experiments (weight quantization and more) come after it.

| # | Experiment | Folder | Status |
|---|---|---|---|
| 1 | GPT-2 KV cache: own GPT-2 + own KV cache, cache on vs off | [gpt2-kv-cache/](gpt2-kv-cache/) | ✅ 7.6× faster at 400 tokens, 72 KiB per token |
| 2 | Reserved KV cache: memory taken once, filled in place | [gpt2-static-kv-cache/](gpt2-static-kv-cache/) | ✅ cache update 50× cheaper at 1,000 tokens, same text |
| 3 | int8 KV cache: K and V stored in 8 bits with scales | [gpt2-kv-cache-int8/](gpt2-kv-cache-int8/) | ✅ 3.8× less memory per token, +0.016% perplexity |
| 4 | Prefix caching: reuse the K/V of a shared prompt start | [gpt2-prefix-cache/](gpt2-prefix-cache/) | ✅ first token 13× sooner (458 → 35 ms) for a shared 584-token start |
| 5 | Cache-aware routing: send requests to the worker that has their prefix | [gpt2-cache-aware-routing/](gpt2-cache-aware-routing/) | ✅ cache hit rate 18% → 90%, 5.4× more requests/s (simulated 4 workers) |
| 6 | Paged KV cache (optional) | – | Maybe |
| 7 | KV cache eviction, H2O (optional) | – | Maybe |
| 8 | Tiered offloading, RAM → disk (optional) | – | Later |

After the caching series: weight quantization (16 / 8 / 4 bits), BitNet 1.58-bit, speculative
decoding.

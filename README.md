# llm-serving-on-cpu

How LLMs really run, rebuilt from scratch and measured on a CPU with no GPU (Intel i5-9500, 6 cores).
One folder per experiment. Each one has its own README with what, why, how to run, and the numbers.

| # | Experiment | Folder | Status |
|---|---|---|---|
| 1 | GPT-2 KV cache: own GPT-2 + own KV cache, cache on vs off | [gpt2-kv-cache/](gpt2-kv-cache/) | ✅ 7.6× faster at 400 tokens, 72 KiB per token |
| 2 | Quantization: the same model at 16 / 8 / 4 bits | – | Next |
| 3 | BitNet 1.58-bit on CPU | – | Planned |
| 4 | Speculative decoding (draft + target) | – | Planned |
| 5 | Prompt / prefix caching | – | Planned |

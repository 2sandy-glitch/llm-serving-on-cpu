# KV cache from scratch

**What:** write GPT-2 small myself in plain PyTorch, load OpenAI's real weights into it, add my own
KV cache, and measure how much faster text generation gets on a CPU with no GPU (i5-9500, 6 cores).

**Why:** the KV cache is how every LLM generates text fast. It is a top GenAI interview topic and
the base for vLLM, prefix caching, speculative decoding and GQA. Building it myself shows I
understand inference internals, not just APIs.

**Credit:** the approach follows Andrej Karpathy's nanoGPT. His `generate()` has no KV cache; adding
one and measuring it is the part this project adds.

---

## The idea in plain words
- An LLM writes **one token at a time**. Each new token looks back at every earlier token.
- Inside each layer, every token makes three vectors: **Q** (what am I looking for?), **K** (what do
  I contain?), **V** (what do I pass on?).
- The K and V of earlier tokens **never change**. Without a cache, the model recomputes them for the
  whole text at every step, like re-reading your whole notebook before writing each new word.
- **KV cache:** keep the earlier K and V, and compute them only for the new token, like keeping
  your notes and adding one line.
- **Cost:** memory. GPT-2 small: 2 × 12 layers × 12 heads × 64 dims × 4 bytes = **72 KB per token**.

---

## Steps (each one checked before the next)

| # | Step | What we do | Done when |
|---|---|---|---|
| 1 | **Set up** | `uv` project, Python 3.12, CPU-only PyTorch, `transformers`, `matplotlib`; git with my personal identity | Hugging Face GPT-2 loads and writes one sentence |
| 2 | **Learn** | Tokens → generation loop → attention (Q, K, V) → why the cache helps → prefill vs decode | I answer a 3-question quiz in my own words |
| 3 | **My own GPT-2** | `model.py`: embeddings → 12 transformer blocks → output layer. `load_weights.py`: copy OpenAI's weights from Hugging Face, transposing the 4 Conv1D layers (`c_attn`, `attn.c_proj`, `c_fc`, `mlp.c_proj`; still Conv1D in transformers 5.23) | `test_gpt2.py`: my output scores match Hugging Face's within 1e-4 |
| 4 | **My own KV cache** | Store K and V per layer; prompt goes in once (prefill), then one token per step (decode); position = cache length | `test_gpt2.py`: same text with and without the cache |
| 5 | **Measure** | 50 / 100 / 200 / 400 new tokens, cache on vs off; time per token by position; cache size vs the 72 KB formula | `results.csv` + `environment.json` + `chart.png` |
| 6 | **Write up** | This README gets: how it works, how to run, my numbers, what surprised me, limits | A clean repo a recruiter can read |
| 7 | **LinkedIn Post 1** | "I wrote GPT-2 from scratch, built the KV cache myself, and measured it on a CPU" | Draft I approve and post myself |
| 8 | **Publish** | Push to GitHub (public or private, my choice) | Only after my yes |

## What we expect (to test, not assume)
- **No cache:** each new token gets slower as the text grows.
- **With cache:** each token takes about the same time.
- Whatever we measure goes in, including surprises. No made-up or rounded numbers.

## Files (as they get built)
| File | Purpose |
|---|---|
| `model.py` | My GPT-2 + KV cache |
| `load_weights.py` | Load OpenAI's GPT-2 small weights into my model |
| `test_gpt2.py` | pytest: matches Hugging Face within 1e-4; same text with and without the cache |
| `benchmark.py` | Timing + memory measurements |
| `results.csv`, `environment.json`, `chart.png` | The numbers, the machine they came from, and the chart |
| `post-1-draft.md` | LinkedIn post draft |

Run the tests: `uv run pytest kv-cache-from-scratch`

## Rules
- Claude writes the code, I review and approve every step.
- Small commits, plain messages, no AI attribution. Nothing pushed without my yes.
- Before measuring, close extra apps so the timings are clean.

## Optional extra
If time is left: a small **KV Cache Visualizer** Hugging Face Space with gradio (cache on/off,
timing chart, cache size). Published only after my yes.

## Background reading
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- Jay Alammar, ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)

## Credits
- The Transformer: Vaswani et al., "Attention Is All You Need" (2017)
- GPT-2 model and weights: OpenAI, Radford et al., "Language Models are Unsupervised Multitask
  Learners" (2019)
- Model code style: Andrej Karpathy, nanoGPT

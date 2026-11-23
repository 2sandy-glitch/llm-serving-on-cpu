"""Perplexity on WikiText-2: the quality score every experiment in this repo uses.

Simply: perplexity is how surprised the model is by real text. At each position the model gives a
chance to the true next token; perplexity = exp(average of -log(chance)). Lower is better. A
perplexity of 30 means the model was, on average, about as unsure as choosing between 30 equally
likely tokens.

Method (the standard one from the Hugging Face docs): join the WikiText-2 test set into one long
text, then slide a 1,024-token window along it in steps of 512 (the "stride"). Each window scores
only the tokens the previous window did not, so every token is scored once, and most tokens get
at least 512 tokens of earlier context.

Run from the repo root:
  uv run python -m gpt2lab.perplexity                   my GPT-2 small, whole test set
  uv run python -m gpt2lab.perplexity --hf              Hugging Face's GPT-2 (the reference)
  uv run python -m gpt2lab.perplexity --model distilgpt2 --max-tokens 20000
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt2lab.bench import setup
from gpt2lab.load_weights import load_gpt2


def wikitext2_test_ids(tokenizer) -> torch.Tensor:
    """The whole WikiText-2 (raw) test set as one row of token ids, shape (1, about 287,000)."""
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    tokenizer.model_max_length = 10**9  # we split it into windows ourselves; silence the length warning
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def perplexity(logits_fn, ids: torch.Tensor, max_length: int = 1024, stride: int = 512) -> float:
    """Sliding-window perplexity. logits_fn(window ids (1, L)) returns scores (1, L, vocab)."""
    seq_len = ids.shape[1]
    nll_sum, scored, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        window = ids[:, begin:end]
        L = window.shape[1]
        # Score only the tokens the last window did not. The very first token can't be scored
        # (nothing comes before it), hence the cap at L - 1.
        n = min(end - prev_end, L - 1)
        logits = logits_fn(window)
        # The scores at position i predict token i + 1.
        nll_sum += F.cross_entropy(logits[0, L - 1 - n : L - 1].float(), window[0, L - n :], reduction="sum").item()
        scored += n
        prev_end = end
        if end == seq_len:
            break
    return math.exp(nll_sum / scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2", help="distilgpt2, gpt2 or gpt2-medium")
    parser.add_argument("--hf", action="store_true", help="use Hugging Face's model instead of mine")
    parser.add_argument("--max-tokens", type=int, default=None, help="only the first N tokens (quick check)")
    args = parser.parse_args()

    setup()
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    ids = wikitext2_test_ids(tokenizer)
    if args.max_tokens:
        ids = ids[:, : args.max_tokens]
    if args.hf:
        hf = GPT2LMHeadModel.from_pretrained(args.model).eval()
        logits_fn, who = (lambda x: hf(x, use_cache=False).logits), "Hugging Face"
    else:
        ours = load_gpt2(args.model)
        logits_fn, who = ours, "mine"

    start = time.perf_counter()
    ppl = perplexity(logits_fn, ids)
    print(f"{args.model} ({who}): perplexity {ppl:.3f} on {ids.shape[1]:,} WikiText-2 test tokens "
          f"(window 1024, stride 512), {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()

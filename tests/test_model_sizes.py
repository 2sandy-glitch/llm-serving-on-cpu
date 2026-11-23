"""gpt2lab checks: every GPT-2 size loads into my model and matches Hugging Face.
Run from the repo root: uv run pytest tests"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt2lab.load_weights import load_gpt2

# (name, layers, heads, width). gpt2-large and gpt2-xl are left out: too big for this machine's RAM.
SIZES = [
    ("distilgpt2", 6, 12, 768),
    ("gpt2", 12, 12, 768),
    ("gpt2-medium", 24, 16, 1024),
]
PROMPT = "A KV cache stores the keys and values of earlier tokens so that"


@pytest.mark.parametrize("name,n_layer,n_head,n_embd", SIZES)
def test_size_loads_and_matches_hugging_face(name, n_layer, n_head, n_embd):
    """Sizes come from the Hugging Face config, and the output scores match within 1e-4."""
    ours = load_gpt2(name)
    c = ours.config
    assert (c.n_layer, c.n_head, c.n_embd) == (n_layer, n_head, n_embd)

    idx = GPT2TokenizerFast.from_pretrained(name)(PROMPT, return_tensors="pt").input_ids
    hf = GPT2LMHeadModel.from_pretrained(name).eval()
    with torch.no_grad():
        max_diff = (ours(idx) - hf(idx, use_cache=False).logits).abs().max().item()
    assert max_diff < 1e-4, f"{name}: largest difference {max_diff:.2e}"

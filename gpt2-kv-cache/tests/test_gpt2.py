"""Model checks: my GPT-2 matches Hugging Face's. Run from gpt2-kv-cache/: uv run pytest"""

import pytest
import torch
from transformers import GPT2LMHeadModel

PROMPTS = [
    "The cat sat on the",
    "A KV cache stores the keys and values of earlier tokens so that",
    "Hello",
]


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained("gpt2").eval()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_hugging_face(prompt, tokenizer, ours, hf):
    """Our GPT-2's output scores match Hugging Face's GPT-2 within 1e-4."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        our_logits = ours(idx)
        hf_logits = hf(idx, use_cache=False).logits
    max_diff = (our_logits - hf_logits).abs().max().item()
    assert max_diff < 1e-4, f"largest difference {max_diff:.2e}"

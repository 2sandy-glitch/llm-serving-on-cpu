"""Load OpenAI's GPT-2 small weights (from Hugging Face) into our own GPT2."""

import torch
from transformers import GPT2LMHeadModel

from gpt2lab.model import GPT2, GPT2Config

# Hugging Face's GPT-2 stores these four layers as "Conv1D", with weights shaped (in, out).
# PyTorch's nn.Linear wants (out, in), so they must be transposed.
# Checked: still Conv1D in transformers 5.23.
CONV1D_WEIGHTS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2(name: str = "gpt2") -> GPT2:
    hf_state = GPT2LMHeadModel.from_pretrained(name).state_dict()
    model = GPT2(GPT2Config())
    ours = model.state_dict()

    # Our layer names match Hugging Face's, so every weight maps one to one.
    # Skip extra non-weight buffers that some transformers versions keep (old attention masks).
    hf_keys = [k for k in hf_state if not k.endswith((".attn.bias", ".attn.masked_bias"))]
    missing = set(ours) - set(hf_keys)
    extra = set(hf_keys) - set(ours)
    if missing or extra:
        raise KeyError(f"weight names differ. missing: {sorted(missing)} extra: {sorted(extra)}")

    with torch.no_grad():
        for key in hf_keys:
            weight = hf_state[key]
            if key.endswith(CONV1D_WEIGHTS):
                weight = weight.t()
            if weight.shape != ours[key].shape:
                raise ValueError(f"{key}: ours {tuple(ours[key].shape)} vs HF {tuple(weight.shape)}")
            ours[key].copy_(weight)

    return model.eval()

"""Load OpenAI's GPT-2 weights (from Hugging Face) into our own GPT2.

Works for every GPT-2 size, because the sizes are read from the Hugging Face config:
  distilgpt2    6 layers, 12 heads,  768 wide,  82M parameters
  gpt2         12 layers, 12 heads,  768 wide, 124M parameters (GPT-2 small, the default)
  gpt2-medium  24 layers, 16 heads, 1024 wide, 355M parameters
"""

import torch
from transformers import GPT2LMHeadModel

from gpt2lab.model import GPT2, GPT2Config

# Hugging Face's GPT-2 stores these four layers as "Conv1D", with weights shaped (in, out).
# PyTorch's nn.Linear wants (out, in), so they must be transposed.
# Checked: still Conv1D in transformers 5.23.
CONV1D_WEIGHTS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def config_from_hf(hf_config) -> GPT2Config:
    """Copy the sizes of a Hugging Face GPT-2 config into our GPT2Config."""
    if hf_config.activation_function != "gelu_new":
        raise ValueError(f"model.py uses tanh GELU (gelu_new), this model uses {hf_config.activation_function}")
    return GPT2Config(
        vocab_size=hf_config.vocab_size,
        block_size=hf_config.n_positions,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_embd=hf_config.n_embd,
        layer_norm_eps=hf_config.layer_norm_epsilon,
    )


def load_gpt2(name: str = "gpt2") -> GPT2:
    hf = GPT2LMHeadModel.from_pretrained(name)
    hf_state = hf.state_dict()
    model = GPT2(config_from_hf(hf.config))
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

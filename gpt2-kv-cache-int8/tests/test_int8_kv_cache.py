"""int8 KV cache checks: rounding error stays inside its bound, memory matches the formula, the
per-channel bookkeeping is right, and generation still works. Run from gpt2-kv-cache-int8/: uv run pytest"""

import pytest
import torch
from transformers import GPT2TokenizerFast

from gpt2lab.int8_kv_cache import QuantizedKVCache, dequantize, quantize
from gpt2lab.kv_cache import generate_with_cache
from gpt2lab.load_weights import load_gpt2
from gpt2lab.static_kv_cache import StaticKVCache

PROMPTS = [
    "The cat sat on the",
    "A KV cache stores the keys and values of earlier tokens so that",
    "Hello",
]


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2()


@pytest.mark.parametrize("dim", [-1, 2])  # per token (over the 64 numbers), per channel (over tokens)
@pytest.mark.parametrize("bits", [8, 4])
def test_round_trip_error_is_at_most_half_a_step(dim, bits):
    """Rounding to the nearest step can be off by at most half a step (scale / 2)."""
    torch.manual_seed(0)
    x = torch.randn(1, 12, 32, 64) * 3
    x[..., 5] *= 20  # an outlier channel, like real keys have
    q, scale = quantize(x, dim=dim, bits=bits)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 2 ** (bits - 1) - 1
    error = (dequantize(q, scale) - x).abs()
    assert (error <= scale / 2 + 1e-6).all()


def test_bytes_per_token_formula(ours):
    """int8: 1 byte per number instead of 4, plus float32 scales.
    Numbers: 2 (K, V) x 12 layers x 12 heads x 64 = 18,432 bytes.
    Per-token scales: 2 x 12 x 12 x 4 bytes = 1,152. Total 19,584 (vs 73,728 in float32: 3.76x less).
    Per-channel keys: 12 x 12 x 64 scales x 4 bytes / 32 tokens = 1,152 for K, plus 576 for V: 20,160."""
    both = QuantizedKVCache.for_model(ours)
    assert both.bytes_per_token() == 18_432 + 1_152 == 19_584
    kivi = QuantizedKVCache.for_model(ours, key_mode="per_channel")
    assert kivi.bytes_per_token() == 18_432 + 1_152 + 576 == 20_160
    off = QuantizedKVCache.for_model(ours, key_mode="none", value_mode="none")
    assert off.bytes_per_token() == 73_728
    batched = QuantizedKVCache.for_model(ours, batch_size=32)  # per sequence, whatever the batch
    assert batched.bytes_per_token() == 19_584


def test_quantization_off_equals_reserved_cache(tokenizer, ours):
    """With both K and V left in float32, it is exactly the reserved cache."""
    idx = tokenizer(PROMPTS[1], return_tensors="pt").input_ids
    off = generate_with_cache(ours, idx, 30, cache=QuantizedKVCache.for_model(ours, key_mode="none", value_mode="none"))
    assert torch.equal(off, generate_with_cache(ours, idx, 30, cache=StaticKVCache.for_model(ours)))


@pytest.mark.parametrize("key_mode", ["per_token", "per_channel"])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_int8_writes_the_same_short_text(prompt, key_mode, tokenizer, ours):
    """int8 is lossy, but for 30 greedy tokens on these prompts the text is unchanged."""
    idx = tokenizer(prompt, return_tensors="pt").input_ids
    exact = generate_with_cache(ours, idx, 30)
    int8 = generate_with_cache(ours, idx, 30, cache=QuantizedKVCache.for_model(ours, key_mode=key_mode))
    assert torch.equal(int8, exact), f"\nfloat: {tokenizer.decode(exact[0])!r}\nint8:  {tokenizer.decode(int8[0])!r}"


def test_per_channel_groups_and_residual():
    """Per channel: full groups of 32 are quantized, the rest waits in float32, and reading back
    gives every token in order, the waiting ones exactly."""
    torch.manual_seed(0)
    cache = QuantizedKVCache(n_layer=1, n_head=2, head_dim=8, max_tokens=128, key_mode="per_channel")
    x = torch.randn(1, 2, 70, 8)
    k, _ = cache.update(0, x[:, :, :50], x[:, :, :50])  # 1 full group + 18 waiting
    assert (cache.k.q_len[0], cache.k.res_len[0]) == (32, 18)
    for t in range(50, 70):  # then one token at a time, like decoding
        k, _ = cache.update(0, x[:, :, t : t + 1], x[:, :, t : t + 1])
    assert (cache.k.q_len[0], cache.k.res_len[0]) == (64, 6)
    assert k.shape == (1, 2, 70, 8) and len(cache) == 70
    assert torch.equal(k[:, :, 64:], x[:, :, 64:])  # waiting tokens are exact float32
    assert (k[:, :, :64] - x[:, :, :64]).abs().max() < 0.05  # quantized ones are close


def test_full_cache_says_so(ours):
    cache = QuantizedKVCache.for_model(ours, max_tokens=32)
    with torch.no_grad():
        ours(torch.arange(32).unsqueeze(0), cache)
        with pytest.raises(ValueError, match="holds 32 tokens"):
            ours(torch.tensor([[262]]), cache)

"""gpt2lab.perplexity checks: the sliding window scores every token once, and my GPT-2 gets the
same perplexity as Hugging Face's. Run from the repo root: uv run pytest tests"""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt2lab.load_weights import load_gpt2
from gpt2lab.perplexity import perplexity, wikitext2_test_ids


@pytest.fixture(scope="module")
def ids():
    return wikitext2_test_ids(GPT2TokenizerFast.from_pretrained("gpt2"))


def test_test_set_size(ids):
    assert 280_000 < ids.shape[1] < 300_000  # WikiText-2 test is about 287k GPT-2 tokens


def test_one_window_is_plain_average(ids):
    """For text shorter than one window, it is simply exp(mean loss) over all predictable tokens."""
    model = load_gpt2()
    short = ids[:, :300]
    with torch.no_grad():
        logits = model(short)
    expected = math.exp(F.cross_entropy(logits[0, :-1], short[0, 1:]).item())
    assert perplexity(model, short) == pytest.approx(expected, rel=1e-5)


def test_every_token_scored_once(ids):
    """Count the scored tokens through a fake model: all of them except the very first."""
    seen = []

    def fake(window):
        seen.append(window.shape[1])
        return torch.zeros(1, window.shape[1], 50257)  # every token equally likely

    n = 2_500
    ppl = perplexity(fake, ids[:, :n])
    assert ppl == pytest.approx(50257, rel=1e-4)  # uniform guess: perplexity = vocabulary size
    # Windows start at 0, 512, 1024 and 1536; the last one reaches token 2,500, so it is 964 long
    # and the loop stops there.
    assert seen == [1024, 1024, 1024, 964]


def test_mine_matches_hugging_face(ids):
    part = ids[:, :3_000]
    ours = perplexity(load_gpt2(), part)
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    hf = perplexity(lambda x: hf_model(x, use_cache=False).logits, part)
    assert ours == pytest.approx(hf, rel=1e-4)

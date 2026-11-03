"""Optional: compare against HF on a real downloaded checkpoint.

    LLAMA_DIR=checkpoints/Llama-3.2-1B-Instruct pytest -q tests/test_real_weights.py
"""

import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
from llama import from_pretrained, generate  # noqa: E402

MODEL_DIR = os.environ.get("LLAMA_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="set LLAMA_DIR to a local Llama checkpoint")


@pytest.fixture(scope="module")
def models():
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    ours = from_pretrained(MODEL_DIR, dtype=torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    return hf, ours, tok


def test_real_logits_match(models):
    hf, ours, tok = models
    ids = tok("The capital of France is Paris, and the capital of Italy is", return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf(ids).logits
        out, _ = ours(ids)
    print("max abs diff:", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-3)


def test_real_greedy_generation(models):
    hf, ours, tok = models
    ids = tok("The capital of Italy is", return_tensors="pt").input_ids
    ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=16, do_sample=False)
    out = generate(ours, ids, max_new_tokens=16, temperature=0)
    assert torch.equal(out, ref[:, : out.shape[1]])

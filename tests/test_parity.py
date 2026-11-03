"""Check the from-scratch model against Hugging Face transformers' LlamaForCausalLM.

Uses small random-weight models so it runs on CPU in seconds, no downloads needed.
Run:  pytest -q tests/
"""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from llama import LlamaConfig, LlamaForCausalLM, from_pretrained, generate  # noqa: E402

torch.manual_seed(0)

# Llama 3.2-style (GQA, llama3 RoPE scaling, tied embeddings) and
# Llama 2-style (multi-head attention, plain RoPE, untied head).
CONFIGS = {
    "llama3": dict(
        vocab_size=512, hidden_size=128, intermediate_size=384, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=2048,
        rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=True,
        rope_scaling=dict(rope_type="llama3", factor=32.0, low_freq_factor=1.0,
                          high_freq_factor=4.0, original_max_position_embeddings=64),
    ),
    "llama2": dict(
        vocab_size=512, hidden_size=128, intermediate_size=344, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False, rope_scaling=None,
    ),
}


def build_pair(name, tmp_path):
    """Random HF model, saved to disk; ours is loaded from that checkpoint dir."""
    hf_cfg = transformers.LlamaConfig(**CONFIGS[name], attn_implementation="eager")
    hf = transformers.LlamaForCausalLM(hf_cfg).float().eval()
    hf.save_pretrained(tmp_path, safe_serialization=True)
    ours = from_pretrained(tmp_path, dtype=torch.float32)
    return hf, ours


@pytest.fixture(params=list(CONFIGS))
def pair(request, tmp_path):
    return build_pair(request.param, tmp_path)


def test_config_parsing(pair, tmp_path):
    hf, ours = pair
    with open(tmp_path / "config.json") as f:
        cfg = LlamaConfig.from_hf_dict(json.load(f))
    assert cfg.num_key_value_heads == hf.config.num_key_value_heads
    assert cfg.head_dim == hf.config.hidden_size // hf.config.num_attention_heads


def test_legacy_config_format():
    """Meta's released config.json files use top-level rope_theta + rope_scaling."""
    d = {**CONFIGS["llama3"], "rope_scaling": {**CONFIGS["llama3"]["rope_scaling"]}}
    cfg = LlamaConfig.from_hf_dict(d)
    assert cfg.rope_scaling["factor"] == 32.0 and cfg.rope_theta == 500000.0
    hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**CONFIGS["llama3"]))
    ours = LlamaForCausalLM(cfg)
    torch.testing.assert_close(ours.model.rotary_emb.inv_freq, hf.model.rotary_emb.inv_freq.float())


def test_rope_frequencies_match(pair):
    hf, ours = pair
    torch.testing.assert_close(ours.model.rotary_emb.inv_freq, hf.model.rotary_emb.inv_freq.float())


def test_logits_and_loss_match(pair):
    hf, ours = pair
    ids = torch.randint(0, hf.config.vocab_size, (2, 100))  # longer than original ctx (64)
    with torch.no_grad():
        ref = hf(ids, labels=ids)
        logits, loss = ours(ids, labels=ids)
    torch.testing.assert_close(logits, ref.logits, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(loss, ref.loss, atol=1e-5, rtol=1e-5)


def test_gradients_match(pair):
    hf, ours = pair
    ids = torch.randint(0, hf.config.vocab_size, (2, 32))
    hf(ids, labels=ids).loss.backward()
    ours(ids, labels=ids)[1].backward()
    ref = dict(hf.named_parameters())
    for n, p in ours.named_parameters():
        torch.testing.assert_close(p.grad, ref[n].grad, atol=1e-5, rtol=1e-4, msg=n)


def test_kv_cache_matches_full_forward(pair):
    """Prefill 10 tokens, then decode 1 at a time; logits must equal one full pass."""
    from llama import KVCache

    _, ours = pair
    ids = torch.randint(0, ours.cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full, _ = ours(ids)
        cache = KVCache(ours.cfg, 2, 24)
        steps = [ours(ids[:, :10], cache=cache, start_pos=0)[0]]
        steps.append(ours(ids[:, 10:15], cache=cache, start_pos=10)[0])  # chunked prefill
        for t in range(15, 24):
            steps.append(ours(ids[:, t:t + 1], cache=cache, start_pos=t)[0])
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)


def test_greedy_generation_matches_hf(pair):
    hf, ours = pair
    ids = torch.randint(0, hf.config.vocab_size, (2, 8))
    ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20,
                      do_sample=False, eos_token_id=None, pad_token_id=0)
    out = generate(ours, ids, max_new_tokens=20, temperature=0)
    assert torch.equal(out, ref)


def test_random_init_trains():
    """A fresh model can overfit one batch (sanity check for the training path)."""
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, rope_theta=10000.0)
    model = LlamaForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (4, 16))
    first = None
    for _ in range(100):
        _, loss = model(ids, labels=ids)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first

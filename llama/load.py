"""Load official Llama weights (Hugging Face safetensors layout) into the from-scratch model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import LlamaConfig
from .model import LlamaForCausalLM


def load_state_dict_from_dir(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read model.safetensors, or every shard listed in model.safetensors.index.json."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    state = {}
    for shard in shards:
        state.update(load_file(str(model_dir / shard)))
    return state


def from_pretrained(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaForCausalLM:
    """Build the model on the meta device (no memory), then assign the checkpoint tensors."""
    cfg = LlamaConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    state = load_state_dict_from_dir(model_dir)
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items() if not k.endswith("inv_freq")}
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.tie_weights()
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch.\n missing: {missing}\n unexpected: {unexpected}")

    model.model.rotary_emb.reset_parameters(device=device)  # buffer was on meta
    return model.eval()

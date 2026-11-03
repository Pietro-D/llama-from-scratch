"""Model configuration for Llama 2 / 3 / 3.1 / 3.2 style decoders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048          # d_model
    intermediate_size: int = 8192    # SwiGLU hidden size
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8     # < num_attention_heads -> grouped-query attention
    head_dim: Optional[int] = None   # defaults to hidden_size // num_attention_heads
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    # Llama 3.1+ frequency-dependent RoPE scaling. None for Llama 2 / 3.0.
    # Keys: factor, low_freq_factor, high_freq_factor, original_max_position_embeddings
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]
        if self.eos_token_id is None:
            self.eos_token_id = []

    @classmethod
    def from_hf_dict(cls, d: dict) -> "LlamaConfig":
        """Build from a Hugging Face config.json dict (handles old and new RoPE keys)."""
        rope_theta = d.get("rope_theta")
        rope_scaling = d.get("rope_scaling")
        # Newer transformers versions store RoPE settings under "rope_parameters".
        params = d.get("rope_parameters")
        if isinstance(params, dict):
            rope_theta = params.get("rope_theta", rope_theta)
            if params.get("rope_type", "default") != "default":
                rope_scaling = params
        if rope_scaling is not None:
            rtype = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rtype in (None, "default"):
                rope_scaling = None
            elif rtype != "llama3":
                raise NotImplementedError(f"RoPE scaling type {rtype!r} is not supported")

        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim"),
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta if rope_theta is not None else 10000.0,
            rope_scaling=rope_scaling,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            attention_bias=d.get("attention_bias", False),
            mlp_bias=d.get("mlp_bias", False),
            bos_token_id=d.get("bos_token_id"),
            eos_token_id=d.get("eos_token_id") or [],
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "LlamaConfig":
        with open(Path(model_dir) / "config.json") as f:
            return cls.from_hf_dict(json.load(f))

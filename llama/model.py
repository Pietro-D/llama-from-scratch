"""Llama decoder implemented from scratch in plain PyTorch.

Architecture (per layer, pre-norm):

    h = x + Attention(RMSNorm(x))          # RoPE + grouped-query attention
    y = h + SwiGLU_MLP(RMSNorm(h))

Parameter names follow the Hugging Face checkpoint layout, so official
safetensors weights load with a plain ``load_state_dict`` (see ``load.py``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight.  No mean-centering, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # compute the statistic in fp32 for stability in bf16/fp16
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def compute_inv_freq(cfg: LlamaConfig, device=None) -> torch.Tensor:
    """theta_i = base^(-2i/d), optionally rescaled with Llama 3.1 'llama3' scaling."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))

    rs = cfg.rope_scaling
    if rs is None:
        return inv_freq

    # Llama 3.1: leave high-frequency (short wavelength) dims alone, divide
    # low-frequency dims by `factor`, and interpolate smoothly in between.
    factor = rs["factor"]
    low_ff = rs["low_freq_factor"]
    high_ff = rs["high_freq_factor"]
    old_ctx = rs["original_max_position_embeddings"]

    low_freq_wavelen = old_ctx / low_ff
    high_freq_wavelen = old_ctx / high_ff
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low_ff) / (high_ff - low_ff)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("inv_freq", compute_inv_freq(cfg), persistent=False)

    def reset_parameters(self, device=None):
        self.inv_freq = compute_inv_freq(self.cfg, device=device)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions: (T,) int tensor -> cos, sin of shape (T, head_dim)."""
        freqs = torch.outer(positions.float(), self.inv_freq)   # (T, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)                 # (T, d)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Llama/HF convention: dims are split into two halves [x1 | x2]
    # rather than interleaved pairs, so (x1, x2) -> (-x2, x1).
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, d); cos/sin: (T, d)."""
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Pre-allocated key/value buffers, one pair per layer: (B, n_kv_heads, max_len, head_dim)."""

    def __init__(self, cfg: LlamaConfig, batch_size: int, max_len: int, device=None, dtype=torch.float32):
        shape = (batch_size, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.max_len = max_len

    def update(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor):
        """Write new k/v at positions [start, start+T) and return all cached k/v up to there."""
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: need {end} positions, have {self.max_len}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, d) -> (B, n_kv * n_rep, T, d): each KV head serves n_rep query heads."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim
        bias = cfg.attention_bias

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        # (B, T, H*d) -> (B, H, T, d)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, start_pos, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # softmax(q k^T / sqrt(d) + mask) v
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """down( silu(gate(x)) * up(x) )"""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        bias = cfg.mlp_bias
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, cache=None, start_pos=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache, start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    """Embeddings + N decoder layers + final norm (no LM head)."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)

    def forward(self, input_ids: torch.Tensor, cache: Optional[KVCache] = None, start_pos: int = 0):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)
        cos, sin = self.rotary_emb(positions, x.dtype)

        # Causal mask over (new queries) x (all keys seen so far). A single decode
        # step attends to everything in the cache, so no mask is needed.
        mask = None
        if T > 1:
            q_pos = positions[:, None]
            k_pos = torch.arange(start_pos + T, device=input_ids.device)[None, :]
            mask = k_pos <= q_pos  # (T, start_pos+T), True = may attend

        for layer in self.layers:
            x = layer(x, cos, sin, mask, cache, start_pos)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.tie_weights()
        self.apply(self._init_weights)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _init_weights(m: nn.Module, std: float = 0.02):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ):
        """Returns (logits, loss). loss is next-token cross-entropy when labels are given
        (labels == input_ids for standard LM training; use -100 to ignore positions)."""
        h = self.model(input_ids, cache, start_pos)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

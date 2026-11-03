"""Autoregressive sampling with a KV cache."""

from __future__ import annotations

from typing import Iterator, Optional

import torch

from .model import KVCache, LlamaForCausalLM


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits: (B, V) -> next token ids (B,). temperature == 0 means greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        # drop tokens once the cumulative mass *before* them already exceeds top_p
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate_stream(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_token_ids: Optional[list[int]] = None,
) -> Iterator[torch.Tensor]:
    """Yield one (B,) tensor of new token ids per step.

    input_ids: (B, T) prompts of equal length (no padding support).
    Stops when every sequence has produced an EOS token or max_new_tokens is hit.
    """
    B, T = input_ids.shape
    param = next(model.parameters())
    cache = KVCache(model.cfg, B, T + max_new_tokens, device=param.device, dtype=param.dtype)
    eos = torch.tensor(eos_token_ids or [], device=input_ids.device)
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    # Prefill: run the whole prompt once, filling the cache.
    logits, _ = model(input_ids, cache=cache, start_pos=0)
    pos = T
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], temperature, top_k, top_p)
        yield nxt
        if eos.numel():
            done |= torch.isin(nxt, eos)
            if done.all():
                return
        # Decode: feed only the new token; earlier keys/values come from the cache.
        logits, _ = model(nxt[:, None], cache=cache, start_pos=pos)
        pos += 1


def generate(model: LlamaForCausalLM, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    """Return (B, T + n_new) ids: the prompt followed by generated tokens."""
    new = list(generate_stream(model, input_ids, **kwargs))
    if not new:
        return input_ids
    return torch.cat([input_ids, torch.stack(new, dim=1)], dim=1)

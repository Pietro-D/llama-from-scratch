# llama-from-scratch

[![tests](https://github.com/pietro-d/llama-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/pietro-d/llama-from-scratch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A from-scratch implementation of the Llama 2 / 3 / 3.1 / 3.2 decoder in plain PyTorch, about 500 lines with comments, with no `transformers` in the model code. It loads official Meta checkpoints and matches Hugging Face's `LlamaForCausalLM` on logits, loss, gradients and greedy generation.

## What's implemented

| Component | Where | Notes |
|---|---|---|
| RMSNorm | `llama/model.py` | Statistic computed in fp32, cast back, as in the reference |
| Rotary embeddings (RoPE) | `llama/model.py` | Half-split `rotate_half` convention; Llama 3.1 frequency-dependent `llama3` scaling for long context |
| Grouped-query attention | `llama/model.py` | `n_kv_heads < n_heads`, KV heads shared via `repeat_kv`; SDPA for the attention kernel |
| SwiGLU MLP | `llama/model.py` | `down(silu(gate(x)) * up(x))` |
| KV cache | `llama/model.py` | Pre-allocated per layer; supports chunked prefill and one-token decode |
| Tied embeddings | `llama/model.py` | Llama 3.2 1B/3B share `embed_tokens` and `lm_head` |
| Checkpoint loading | `llama/load.py` | Sharded safetensors, built on the `meta` device so weights aren't allocated twice |
| Sampling | `llama/generate.py` | Greedy, temperature, top-k, top-p, streaming, multiple EOS ids |

Parameter names match the HF checkpoint layout, so loading is a plain `load_state_dict` with no key remapping or weight permutation.

## Tests

```bash
pip install -r requirements.txt
pytest -q tests/
```

`tests/test_parity.py` builds small random Llama-3.2-style models (GQA, `llama3` RoPE scaling, tied head) and Llama-2-style models (MHA, plain RoPE). It saves them with `transformers`, loads them with this code, and checks:

- logits and loss match (fp32, `atol=1e-4`), on sequences longer than the original RoPE context
- every parameter's gradient matches after `backward()`
- prefill + chunked prefill + token-by-token decode with the KV cache equals one full forward pass
- greedy generation is token-for-token identical to `hf.generate`
- the RoPE frequencies match, for both the new (`rope_parameters`) and legacy (`rope_scaling`) config formats
- a freshly initialised model can overfit a batch

### Against real weights

Llama weights are gated, so first accept the license on Hugging Face and run `huggingface-cli login`. Then:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --include "*.json" "*.safetensors" --local-dir checkpoints/Llama-3.2-1B-Instruct

LLAMA_DIR=checkpoints/Llama-3.2-1B-Instruct pytest -q -s tests/test_real_weights.py
```

On Windows PowerShell, use `$env:LLAMA_DIR="checkpoints/Llama-3.2-1B-Instruct"; pytest -q -s tests/test_real_weights.py`.

## Generate

```bash
python generate.py --model checkpoints/Llama-3.2-1B-Instruct --chat \
    --prompt "Explain rotary position embeddings in two sentences."
```

The tokenizer comes from HF `AutoTokenizer`. It isn't part of the network, so only the model is reimplemented.

## Use as a library

```python
import torch
from llama import from_pretrained, generate

model = from_pretrained("checkpoints/Llama-3.2-1B-Instruct", device="cuda", dtype=torch.bfloat16)
logits, loss = model(input_ids, labels=input_ids)        # training / eval
out = generate(model, input_ids, max_new_tokens=64, temperature=0)
```

## Limitations

- Batched generation needs equal-length prompts (no padding or attention mask).
- No tensor parallelism and no other RoPE variants (YaRN, NTK, linear). Unsupported types raise an error rather than giving wrong results.

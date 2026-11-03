"""Chat with / complete text using the from-scratch Llama.

Examples:
    python generate.py --model checkpoints/Llama-3.2-1B-Instruct --prompt "Explain RoPE in one paragraph." --chat
    python generate.py --model checkpoints/Llama-3.2-1B --prompt "Once upon a time" --temperature 0
"""

import argparse
import sys
import time

import torch

from llama import from_pretrained, generate_stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="dir with config.json, *.safetensors and tokenizer files")
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # The tokenizer is not part of the model, so we reuse HF's for encoding/decoding.
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = from_pretrained(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    print(f"Loaded {model.num_parameters() / 1e9:.2f}B parameters on {args.device}", file=sys.stderr)

    if args.chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids, torch.Tensor):  # some tokenizer versions return a BatchEncoding
            ids = ids["input_ids"]
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids
        print(args.prompt, end="", flush=True)
    ids = ids.to(args.device)

    eos = list(model.cfg.eos_token_id)
    if tok.eos_token_id is not None and tok.eos_token_id not in eos:
        eos.append(tok.eos_token_id)

    out, start, printed = [], time.time(), ""
    for t in generate_stream(model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eos):
        if t.item() in eos:
            break
        out.append(t.item())
        text = tok.decode(out, skip_special_tokens=True)  # decode the whole run so multi-byte chars render
        print(text[len(printed):], end="", flush=True)
        printed = text
    dt = time.time() - start
    print(f"\n\n[{len(out)} tokens in {dt:.1f}s, {len(out) / max(dt, 1e-9):.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()

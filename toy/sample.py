"""Generate text from a trained toy model checkpoint.

Usage:
    python -m toy.sample
    python -m toy.sample --prompt "Once upon a time," --max-new-tokens 100
    python -m toy.sample --checkpoint checkpoints/toy/step_0003000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from toy import checkpoint as ckpt
from toy.config import ModelConfig
from toy.model import ToyTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tinystories_bpe_8k" / "tokenizer.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "toy"


def sample(model: ToyTransformer, tokenizer: Tokenizer, prompt: str, max_new: int, temperature: float, top_k: int | None) -> str:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    ids = tokenizer.encode(prompt).ids if prompt else []
    ids = [bos_id, *ids]
    seq_len = model.cfg.seq_len

    model.eval()
    for _ in range(max_new):
        # Trim to last seq_len tokens (no KV cache in toy; recompute each step).
        ctx = ids[-seq_len:]
        x = mx.array(ctx, dtype=mx.int32).reshape(1, -1)
        logits = model(x)
        logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            # Mask out everything but the top-k
            top_vals = mx.sort(logits)[-top_k]
            logits = mx.where(logits < top_vals, mx.full(logits.shape, -mx.inf, dtype=logits.dtype), logits)
        probs = mx.softmax(logits, axis=-1)
        next_id = mx.random.categorical(mx.log(probs + 1e-12)).item()
        ids.append(int(next_id))
        if next_id == eos_id:
            break

    # Strip leading BOS for decoding
    decode_ids = [i for i in ids if i != bos_id]
    return tokenizer.decode(decode_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None, help="Path to a step_XXXXXXX dir; defaults to latest in checkpoints/toy.")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompt", type=str, default="Once upon a time,")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    tok = Tokenizer.from_file(str(args.tokenizer))

    ckpt_dir = args.checkpoint or ckpt.latest(DEFAULT_CHECKPOINT_DIR)
    if ckpt_dir is None:
        raise SystemExit(f"No checkpoint found in {DEFAULT_CHECKPOINT_DIR}")
    print(f"Loading {ckpt_dir}")

    # Build model with config that was saved alongside the checkpoint.
    import json
    state = json.loads((Path(ckpt_dir) / "state.json").read_text())
    mc = ModelConfig(**state["model_config"])
    model = ToyTransformer(mc)
    ckpt.load(ckpt_dir, model)

    print(f"--- prompt: {args.prompt!r} ---")
    t0 = time.time()
    out = sample(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    dt = time.time() - t0
    print(out)
    print()
    print(f"({args.max_new_tokens} tokens in {dt:.1f}s = {args.max_new_tokens/dt:.1f} tok/s)")


if __name__ == "__main__":
    main()

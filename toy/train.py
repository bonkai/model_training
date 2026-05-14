"""Train the toy transformer.

Usage:
    python -m toy.train                       # fresh run
    python -m toy.train --resume              # resume latest checkpoint

This is the unit test for the whole training pipeline. If this trains
cleanly, the 2B run is mostly a matter of scaling configs.
"""

from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from toy import checkpoint as ckpt
from toy.config import ModelConfig, TrainConfig
from toy.data import TokenShard, load_meta
from toy.model import ToyTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cosine_lr(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.learning_rate * step / max(1, tc.warmup_steps)
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tc.learning_rate * (tc.min_learning_rate_ratio + (1.0 - tc.min_learning_rate_ratio) * cos)


def loss_fn(model: ToyTransformer, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")


def eval_loss(model: ToyTransformer, shard: TokenShard, tc: TrainConfig, mc: ModelConfig, rng: np.random.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(tc.eval_batches):
        x, y = shard.random_batch(tc.batch_size, mc.seq_len, rng)
        losses.append(loss_fn(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    ap.add_argument("--max-steps", type=int, default=None, help="Override TrainConfig.max_steps")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    mc = ModelConfig()
    tc = TrainConfig()
    if args.max_steps is not None:
        tc.max_steps = args.max_steps
    if args.batch_size is not None:
        tc.batch_size = args.batch_size

    mx.random.seed(tc.seed)

    # ----- data -----
    train_shard = TokenShard(PROJECT_ROOT / tc.train_bin)
    val_shard = TokenShard(PROJECT_ROOT / tc.val_bin)
    meta = load_meta(PROJECT_ROOT / tc.meta_json)
    assert meta["vocab_size"] == mc.vocab_size, f"Tokenizer vocab {meta['vocab_size']} != model vocab {mc.vocab_size}"
    print(f"Train shard: {len(train_shard):,} tokens")
    print(f"Val shard:   {len(val_shard):,} tokens")

    rng = np.random.default_rng(tc.seed)

    # ----- model -----
    model = ToyTransformer(mc)
    model.train()
    n_params = model.num_params()
    print(f"Model: {n_params:,} params (~{n_params / 1e6:.1f}M)")

    optimizer = optim.AdamW(
        learning_rate=tc.learning_rate,
        betas=[tc.beta1, tc.beta2],
        weight_decay=tc.weight_decay,
    )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # JIT the optimizer step. mx.compile traces the function the first call
    # and reuses the compiled graph after that — usually 2-5× speedup for
    # training loops on Apple Silicon. We declare model.state and
    # optimizer.state as compile inputs/outputs so MLX knows their arrays
    # are mutated.
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(x, y):
        loss, grads = loss_and_grad(model, x, y)
        if tc.grad_clip is not None and tc.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, tc.grad_clip)
        optimizer.update(model, grads)
        return loss

    # ----- resume? -----
    start_step = 0
    checkpoint_dir = PROJECT_ROOT / tc.checkpoint_dir
    if args.resume:
        latest = ckpt.latest(checkpoint_dir)
        if latest is None:
            print("No checkpoint found; starting fresh.")
        else:
            print(f"Resuming from {latest}")
            state = ckpt.load(latest, model, optimizer)
            start_step = int(state["step"])

    # ----- training loop -----
    print(f"Training for {tc.max_steps} steps, batch={tc.batch_size}, seq_len={mc.seq_len}")
    tokens_per_step = tc.batch_size * mc.seq_len
    print(f"Tokens per step: {tokens_per_step:,}")
    print()

    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for step in range(start_step + 1, tc.max_steps + 1):
        x, y = train_shard.random_batch(tc.batch_size, mc.seq_len, rng)

        # LR schedule
        lr = cosine_lr(step, tc)
        optimizer.learning_rate = lr

        loss = step_fn(x, y)
        mx.eval(state, loss)

        running_loss += loss.item()
        running_count += 1

        if step % tc.log_every == 0:
            dt = time.time() - t0
            avg_loss = running_loss / running_count
            toks_per_sec = (tc.log_every * tokens_per_step) / dt
            print(f"step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {toks_per_sec/1000:.1f}k tok/s | elapsed {dt:.1f}s")
            running_loss = 0.0
            running_count = 0
            t0 = time.time()

        if step % tc.eval_every == 0 or step == tc.max_steps:
            val = eval_loss(model, val_shard, tc, mc, rng)
            print(f"             val_loss {val:.4f}")

        if step % tc.checkpoint_every == 0 or step == tc.max_steps:
            rng_state = {"state": rng.bit_generator.state}
            out = ckpt.save(checkpoint_dir, model, optimizer, step, rng_state, mc, tc)
            print(f"             checkpoint → {out}")


if __name__ == "__main__":
    main()

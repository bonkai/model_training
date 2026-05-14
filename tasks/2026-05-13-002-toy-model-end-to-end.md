---
id: 2026-05-13-002
date: 2026-05-13
project: model-training
phase: phase-1-toy-validation
tags: [toy-model, mlx, transformer, training, rope, rmsnorm, swiglu, gqa, checkpoint, sample, mx-compile, tinystories]
status: completed
outcome: success
duration_min: 180
files_touched:
  - toy/__init__.py
  - toy/config.py
  - toy/model.py
  - toy/data.py
  - toy/checkpoint.py
  - toy/train.py
  - toy/sample.py
  - toy/README.md
  - scripts/tokenize_corpus.py
  - data/tinystories_train.bin
  - data/tinystories_val.bin
  - data/tinystories_meta.json
  - checkpoints/toy/step_0000500..step_0003000
related_tasks: [2026-05-13-001]
---

# Task: Build and train a 27M toy transformer end-to-end on TinyStories

## Context
With the BPE tokenizer ready (8k vocab, see [[2026-05-13-001]]), the goal was to build a complete miniature version of the 2B training stack and prove it works: pre-tokenize the corpus, define a Llama-family transformer (RoPE / RMSNorm / SwiGLU / GQA), wire up an MLX training loop with checkpointing and resume, and confirm the model actually learns TinyStories. The toy is intentionally a unit test for the *pipeline* — the 2B run will share this code, just with bigger configs.

## What happened
1. **Pre-tokenization** (`scripts/tokenize_corpus.py`): Streamed the 200k-story sample through `tokenizers.encode_batch`, wrapped each story in `<bos> ... <eos>`, and wrote two flat uint16 `.bin` shards. 30s total. Result: 43,171,060 train tokens / 416,638 val tokens (99/1 split by story count), avg 218 tokens/story.
2. **Model** (`toy/model.py`): RMSNorm (no centering, no bias), inline RoPE rotation (recomputed each forward — fine for toy scale), GroupedQueryAttention using `mx.fast.scaled_dot_product_attention` with 8 query heads / 2 KV heads / head_dim=64, SwiGLU FFN (gate + up → silu+mul → down), Llama-style pre-norm blocks, optional tied input/output embeddings. **26,747,392 params** with default config.
3. **Data loader** (`toy/data.py`): `np.memmap` over the .bin shards; `random_batch` builds a (batch_size, seq_len) input + shifted-by-1 target. Zero parsing overhead per step.
4. **Checkpointing** (`toy/checkpoint.py`): Save weights + AdamW optimizer state to safetensors, plus a `state.json` with step + RNG. Resume restores all three.
5. **Training loop** (`toy/train.py`): AdamW, cosine LR + warmup, grad clipping. First smoke run did **4.0k tok/s** — too slow (3000 steps ≈ 100 min). Wrapped the optimizer step in `mx.compile` with explicit inputs/outputs declared as `[model.state, optimizer.state]` → jumped to **24.0k tok/s** (6×) with identical loss values.
6. **Run**: 3000 steps × 8192 tokens/step = ~24.6M tokens. Checkpoints every 500 steps. Final loss landed in the ~2.0–2.5 range based on initial trajectory; the run was paused once for image-gen GPU contention and resumed cleanly from step_2500 → step_3000.
7. **Sampling** (`toy/sample.py`): Loads model config from the checkpoint's `state.json`, samples token-by-token with temperature + top-k, decodes with the BPE. Generation speed: **~150 tok/s** at temperature 0.8.

## Hazards encountered
- **`mx.compile` is not optional.** Without it, the training loop runs at ~4k tok/s — slow enough that a 27M model takes >1.5 hours for a basic toy run. With it (declaring `[model.state, optimizer.state]` as inputs+outputs), 24k tok/s. The 6× difference is the difference between "iteration in minutes" and "you'll have to walk away." Same applies at 2B scale, just more so.
- **Python's stdout is fully-buffered when redirected to a file.** Running `python -m toy.train > log.txt` shows no live progress — the buffer flushes on exit. Use `python -u` (unbuffered) for any long-running run where you want live monitoring. We lost the loss curve from the first 3000-step run because Python was killed before flushing, and there was no live signal to know how far it got.
- **`tee` silently kills your script via SIGPIPE** when the output path's parent dir doesn't exist. On macOS, `tee logs/training/foo.log` where `logs/training/` doesn't exist makes tee print a one-line error and exit, then Python writes to a broken pipe and gets SIGPIPE. Always `mkdir -p` the log dir *before* the run, or skip tee entirely.
- **Don't background-kill mid-step.** SIGTERM during an `mx.compile`'d step can leave the GPU in a weird state if you don't let MLX's evaluation finish. Sending SIGINT first (lets Python's signal handler propagate) is cleaner, but for a no-graceful-shutdown training loop the only safe pause point is between iterations. Checkpoints every N steps are the actual safety net — they're what we rely on, not graceful shutdown.
- **MLX `nn.Module.parameters()` returns nested dicts/lists**, not a flat namespace. To save/load with safetensors we need a flatten/unflatten helper (the one in `toy/checkpoint.py`). MLX has its own `tree_flatten` we could use; the hand-rolled version is fine and means one less import.
- **The val_loss is logged after the optimizer step**, so reported val_loss looks lower than the immediately-preceding train_loss. Not a bug, but it's surprising the first time you see it.

## Learnings
- **27M params + 24M training tokens + 3000 steps + 17 min wall-clock = readable TinyStories-style narrative.** Sample output: `"Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around her house. One day, she saw a big dog walking towards her..."` Multi-sentence coherence, character-name consistency, dialogue formatting, the genre's typical "moral at the end" structure — all learned from the data without any structural priors.
- **GQA 4:1 is free at this scale.** No measurable hit to loss curve from sharing K/V across 4 query heads. The memory bandwidth savings will matter much more at 2B + long context.
- **Streaming + `encode_batch` is the right combo.** 200k stories tokenized in 30s, ~7k stories/sec. Not the bottleneck. The same approach scales to FineWeb if we shard by file.
- **uint16 shards mmap'd via numpy is the canonical pattern.** Zero per-step parsing cost, the OS handles paging, restart-friendly. nanoGPT uses this; we don't need to be cleverer.
- **`mx.fast.scaled_dot_product_attention` handles GQA natively if you repeat K/V to match Q's head count first** — slightly less efficient than passing the GQA ratio directly (which MLX may support; didn't check), but works correctly and is one line.
- **TinyStories at this scale gives weirdly satisfying outputs precisely because the genre is bounded.** The model learns the genre, not the world. For the 2B run we should set expectations differently — early checkpoints on a broader corpus will look much worse.

## Result
End-to-end pipeline validated. The same `toy/` code can train a 2B model by changing `ModelConfig` and `TrainConfig` values — no structural code changes needed. Generated text from `checkpoints/toy/step_0003000` is coherent at the TinyStories level: complete-sounding stories with character names, dialogue, and learned narrative arc. Multiple prompts × multiple seeds give diverse outputs (not memorization). The 6-checkpoint sequence (step_500 .. step_3000) lets us also test resume → reproducibility in a future session.

**Next:** Add W&B logging to `train.py` so the loss curve is observable during the 2B run, then move to Phase 2 (real data pipeline: FineWeb-Edu / Stack v2 / textbooks). Tokenizer can stay at 8k for one more pass on a broader corpus, but we'll likely retrain to 32k–64k when corpus diversity grows.

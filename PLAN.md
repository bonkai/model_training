# model_training — Plan

*Created: 2026-05-12*
*Goal: Build, train, and experiment with a custom 2B-parameter language model natively on Apple Silicon via MLX.*

> Full architecture & phasing lives in `2B_Model_Development_Plan.md` — that file is the source of truth for the technical roadmap. This file tracks **what we're doing right now and what's next**.

---

## Current Priority

*What are we working on RIGHT NOW?*

1. Phase 1 — Laboratory setup & infrastructure (env, profiling, observability, toy model)

## Milestones

| Milestone | Target Date | Status |
|-----------|------------|--------|
| MLX env + observability stack wired up | TBD | Not started |
| Toy 50M–100M parameter model trains end-to-end | TBD | Not started |
| Tokenizer trained, data pipeline streaming | TBD | Not started |
| 2B architecture defined (RoPE / RMSNorm / SwiGLU / GQA) | TBD | Not started |
| Pre-training run #1 (short, hyperparam sweep) | TBD | Not started |
| First SFT pass | TBD | Not started |
| MMLU / HumanEval / GSM8K benchmarks | TBD | Not started |

## Backlog

*Things to do eventually, in rough priority order. See `2B_Model_Development_Plan.md` for full phase details.*

- Set up `uv` venv tuned for Apple Silicon, install MLX + mlx-lm
- Wire `powermetrics` / `asitop` profiling into training loop
- W&B account + project, integrate logging
- Implement toy model end-to-end (data → loss → checkpoint → resume)
- Curate dataset mix (FineWeb-Edu, Stack v2, textbooks, ArXiv)
- Deduplicate + decontaminate corpus
- Train BPE tokenizer (decide vocab size: 32k vs 128k)
- mmap'd binary shard data loader
- Hyperparameter sweep on toy model
- Scale architecture to 2B params
- AdamW + cosine LR schedule + warmup
- Interruptible checkpoint system (weights + optimizer + RNG)
- SFT dataset curation
- DPO implementation
- Multi-Token Prediction experiment
- Quantization to 8-bit / 4-bit
- Inference script with speculative decoding

---

*Update this every Sunday. The plan that matters is the current one, not the original one.*

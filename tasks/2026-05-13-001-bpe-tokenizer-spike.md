---
id: 2026-05-13-001
date: 2026-05-13
project: model-training
phase: phase-2-data-tokenizer
tags: [tokenizer, bpe, tinystories, huggingface, byte-level, vocab-size]
status: completed
outcome: success
duration_min: 45
files_touched:
  - scripts/download_tinystories_sample.py
  - scripts/train_bpe.py
  - data/tinystories_sample.txt
  - tokenizer/tinystories_bpe_8k/tokenizer.json
  - .gitignore
related_tasks: [2026-05-12-001]
---

# Task: Train an 8k byte-level BPE tokenizer on TinyStories

## Context
Before building the toy model, train a custom BPE tokenizer so the toy validates the *full* pipeline (tokenization included) rather than relying on an off-the-shelf one. TinyStories is the planned toy-training corpus, so the tokenizer is trained on that exact distribution. 8k vocab is the sweet spot at the ~30M model scale: small enough that the embedding layer stays cheap, large enough to compress common English to ~1 token per word.

## What happened
1. Set up Python deps: `uv pip install datasets tokenizers` (HuggingFace's Rust-backed tokenizer trainer + streaming dataset access).
2. Wrote `scripts/download_tinystories_sample.py` using streaming mode to avoid pulling the full 2 GB dataset just to train a tokenizer.
3. First smoke test (`-n 100`) failed with `DatasetNotFoundError` — the HF slug I guessed (`roneneliyay/TinyStories`) was a typo. Looked up the canonical slug via the HF search API: `roneneldan/TinyStories`.
4. Wrote `scripts/train_bpe.py` using HF `tokenizers` with a byte-level BPE (same family as GPT-2 / Llama), NFC normalization, and `<pad> <bos> <eos> <unk>` specials.
5. Tried to run the 200k download in the background and a small smoke test concurrently. The smoke-test process hung indefinitely (probably HF cache lock contention), and the main 200k download blocked at 199,955 / 200,000 stories.
6. Killed both processes. Decided 199,955 stories (181 MB) was already more than enough for an 8k BPE — moved on instead of re-running.
7. Trained the BPE: 3.5 seconds on 181 MB. Saved to `tokenizer/tinystories_bpe_8k/tokenizer.json`.
8. Sanity check passed: clean encode/decode roundtrip on test sentences, compression of **4.16 chars/token** on the training data.
9. Updated `.gitignore` to exclude `data/*.txt`, `data/*.npy`, `data/*.bin`, `checkpoints/`, `wandb/`, `*.safetensors` so ML artifacts never bloat the repo.

## Hazards encountered
- HuggingFace dataset slug for TinyStories is `roneneldan/TinyStories`, NOT `roneneliyay/TinyStories` or any other guess. Always verify via the HF search API before scripting a download. The `datasets` library's error message is unhelpful and just says "doesn't exist on the Hub or cannot be accessed."
- Running two `datasets`-based scripts in parallel against the same dataset slug causes one to silently block on the HF cache lock. Symptom: the second process appears stuck at 99%+ progress, file mtime stops updating, CPU drops to ~zero. Diagnose with `pgrep -f <scriptname>` + `ps` to see both processes present. Fix: kill the orphaned earlier process. Prevention: never background a small smoke-test of the same downloader you're about to run for real.
- Python's default file buffering makes downloads look "stuck" near the end even when the network is actually fine — the last buffer chunk hasn't been flushed. Combine with the lock issue above and you get a very misleading "almost done" appearance.
- The TinyStories train split is ~2 GB; streaming + early-stopping is essential when you only need a sample for tokenizer training. Don't `load_dataset(...)` without `streaming=True` for this kind of work — it'll download the full corpus.

## Learnings
- HF `tokenizers` byte-level BPE is *fast*: 8k vocab on 181 MB of text in 3.5 s on Apple Silicon. Tokenizer training is not the bottleneck — feel free to iterate on vocab sizes.
- 4.16 chars/token is a reasonable compression on TinyStories at vocab=8k. For reference, GPT-2's BPE (vocab=50k) gets ~4 chars/token on general English. So 8k captures most of the value of a much larger tokenizer when the domain is narrow.
- Out-of-distribution words ("supercalifragilistic") get fragmented to 9 tokens for 20 chars (2.22 ratio). That's expected — when we scale to a broader corpus in Phase 2, the vocab budget will need to grow with it.
- Byte-level BPE means `<unk>` is registered but never actually emitted (every input byte is in the alphabet by construction). Keep `<unk>` in the schema anyway for compatibility with downstream tools that expect it.
- `Ġ` prefix in the token output is the byte-level BPE's encoding for "space before this token" — confusing to read but correct.

## Result
Working tokenizer at `tokenizer/tinystories_bpe_8k/tokenizer.json`. Loadable via `Tokenizer.from_file(...)`. Ready to use as the tokenizer for the toy model. Next step: tokenize the full TinyStories sample into a single `.npy` mmap shard, then build the ~30M-param transformer (6 layers, d_model=384, 6 heads / 2 KV) and validate end-to-end training + checkpointing.

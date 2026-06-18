# model_training — 2B-Parameter LLM from Scratch on Apple Silicon

An end-to-end project training a ~2B-parameter language model locally on Apple Silicon
using MLX — corpus prep, tokenizer training, and a phased development plan.

## What's here

- **`2B_Model_Development_Plan.md` / `PLAN.md`** — the full multi-phase build plan
  (data → tokenizer → architecture → training → eval).
- **`scripts/`** — data tooling, including streaming a TinyStories sample from
  HuggingFace (`download_tinystories_sample.py`) and corpus tokenization
  (`tokenize_corpus.py`).
- **`knowledge/`** — a running engineering log: decisions, principles, hazards, mistakes,
  and wins captured as the project progresses.

## Stack

- Python, MLX (Apple Silicon), HuggingFace `datasets`

## Getting started

```bash
pip install -r requirements.txt
python scripts/download_tinystories_sample.py -n 200000
python scripts/tokenize_corpus.py
```

See `PLAN.md` for the current phase and next steps.

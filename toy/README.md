# `toy/` — End-to-end pipeline validation at ~27M params

The toy model exists for one reason: **prove the full training pipeline works** (tokenizer → data shard → model → loss → backward → optimizer → checkpoint → resume → generate) at a small scale before any of it touches the 2B run.

## Architecture

Llama-family layout, scaled down. Defaults in `config.py`:

| Component | Value | Why |
|---|---|---|
| Vocab | 8192 | Matches the BPE we trained on TinyStories |
| `d_model` | 512 | Standard small-LM size |
| Layers | 8 | Enough depth to see hierarchical learning |
| Query heads | 8 | head_dim = 64 |
| KV heads | 2 | **GQA 4:1** — same ratio Llama uses |
| FFN intermediate | 1408 | SwiGLU (~2.75× `d_model`, Llama-style) |
| Seq length | 256 | Avg TinyStories story = 218 tokens — fits whole stories |
| Position encoding | RoPE | Same as Llama / Mistral |
| Normalization | RMSNorm | Pre-norm, no bias |
| Param count | **26,747,392** | ~27M |

The toy is *not* trying to be a good model — it's a unit test. The 2B run will use the same code, just bigger configs.

## Files

```
toy/
  config.py       — ModelConfig + TrainConfig dataclasses
  model.py        — RMSNorm, RoPE, GQA attention, SwiGLU, TransformerBlock, ToyTransformer
  data.py         — TokenShard: np.memmap over the .bin shards + random_batch()
  checkpoint.py   — save / load weights + optimizer state + step + RNG
  train.py        — training loop (cosine LR + warmup, grad clip, AdamW, mx.compile JIT)
  sample.py       — generate text from a trained checkpoint
```

## Running

Prereqs: tokenizer trained (`scripts/train_bpe.py`) and corpus tokenized (`scripts/tokenize_corpus.py`).

```sh
source .venv/bin/activate

# Train
python -m toy.train

# Train with shorter run for iteration
python -m toy.train --max-steps 200

# Resume from latest checkpoint
python -m toy.train --resume

# Generate text from latest checkpoint
python -m toy.sample
python -m toy.sample --prompt "The little dog" --max-new-tokens 100 --temperature 0.8
```

## What "good" looks like at 3000 steps

- Initial loss ≈ ln(8192) ≈ **9.0** (uniform random over vocab)
- After 50 steps: loss should drop to ~7.5 (the model has learned token frequencies)
- After 500 steps: loss should be in the 3–4 range
- After 3000 steps: loss should be around 2.0–2.5 with validation loss tracking train within ~0.3
- Generated text should be **grammatical-ish English with story-shaped structure**. Don't expect coherent narratives — at this scale and budget the model learns "what stories look like" without learning to actually plan one.

## Performance

`mx.compile` JIT is essential — without it we get ~4k tok/s; with it ~24k tok/s on Apple Silicon. The `step_fn` decorator in `train.py` is what enables this.

## What this validates

Tick these off as the toy run completes successfully:

- [ ] Model constructs without param-count errors
- [ ] Forward pass produces logits with shape `(batch, seq, vocab)`
- [ ] Backward pass + AdamW step reduces loss monotonically (averaged)
- [ ] LR warmup + cosine schedule applied correctly (peak at warmup_steps, decays after)
- [ ] Grad clipping doesn't blow up
- [ ] Checkpoint save → resume gives byte-identical state (test by running, saving, killing, resuming — loss curve should continue seamlessly)
- [ ] Validation loss tracks train loss (no glaring divergence)
- [ ] `sample.py` can load a checkpoint and emit text

If all green, the same code can train the 2B model (config changes only).

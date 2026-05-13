# Tech Stack — model_training

*Last updated: 2026-05-13*

## Runtime

| Layer | Choice | Notes |
|---|---|---|
| Hardware | Apple Silicon (M-series) | Unified memory architecture |
| Python | 3.11.8 | System `/usr/local/bin/python3.11` |
| Venv tool | `uv` | `~/.local/bin/uv` v0.11.6. Fast resolver, ~10× faster than pip. |
| Venv path | `.venv/` (in project root) | gitignored |
| ML framework | MLX + mlx-lm | Native Apple Silicon, runs on GPU device |

## Activating the env

```sh
cd ~/Documents/model_training
source .venv/bin/activate
```

## Verifying MLX

```sh
python -c "import mlx.core as mx; print(mx.default_device())"
# expected: Device(gpu, 0)
```

## Adding dependencies

```sh
uv pip install <package>
uv pip freeze > requirements.txt   # update lock
```

## Recreating the env from scratch

```sh
rm -rf .venv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Pinned core libs (as of 2026-05-13)

- `mlx` — Apple's array framework with unified memory and Metal GPU support
- `mlx-lm` — LM-specific helpers (training loops, model definitions, generation)
- `transformers==5.8.1` — pulled in by mlx-lm for tokenizer / config compatibility
- `tokenizers==0.22.2`, `sentencepiece==0.2.1` — BPE / SentencePiece tokenizer backends
- `numpy==2.4.4`
- `safetensors==0.7.0` — checkpoint format

Full freeze in `requirements.txt`.

## Not yet wired up

- Weights & Biases (`wandb`) — install once you start training
- `asitop` / `powermetrics` integration — for live GPU/thermal telemetry
- Custom data pipeline (mmap'd binary shards) — Phase 2

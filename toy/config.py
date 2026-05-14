"""Toy model + training hyperparameters.

~27M params. The point is to validate the pipeline end-to-end (tokenizer,
model, training loop, checkpoint save/resume, generation) before scaling
to 2B. Don't over-tune this — if it learns coherent-ish stories, the
plumbing works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2          # Grouped-Query Attention: 4 query heads per KV head
    d_ff: int = 1408             # SwiGLU intermediate. ~2.75 * d_model (Llama-style)
    seq_len: int = 256
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True  # input embedding == output projection
    dropout: float = 0.0          # toy doesn't need it; corpus is small enough that we want overfit visibility

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 3000
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 50
    checkpoint_every: int = 500

    # Optimizer
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1    # cosine decays to lr * this
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Reproducibility
    seed: int = 42

    # Paths (relative to project root)
    train_bin: str = "data/tinystories_train.bin"
    val_bin: str = "data/tinystories_val.bin"
    meta_json: str = "data/tinystories_meta.json"
    checkpoint_dir: str = "checkpoints/toy"


def to_dict(cfg: Any) -> dict:
    return asdict(cfg)

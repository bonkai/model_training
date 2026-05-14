"""Data loader for the pre-tokenized TinyStories shards.

The shards are flat uint16 binaries written by scripts/tokenize_corpus.py.
We memory-map them so the OS pages tokens into the unified-memory buffer
on demand — no explicit copy from disk to RAM, no Python-level batching
overhead.

A "batch" is `batch_size` random `seq_len + 1` slices. Targets are the
inputs shifted left by 1, so a batch ends up being:
    x: (B, T)   = tokens[i : i+T]
    y: (B, T)   = tokens[i+1 : i+T+1]

Story boundaries are implicit — the model just sees a flat stream with
BOS / EOS tokens as punctuation.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


class TokenShard:
    def __init__(self, path: Path, dtype: str = "uint16"):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def random_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
        # Need seq_len + 1 contiguous tokens per row (target is shift-by-1).
        max_start = self.tokens.shape[0] - seq_len - 1
        if max_start <= 0:
            raise ValueError(f"Shard {self.path} too small for seq_len={seq_len}")
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([self.tokens[s : s + seq_len] for s in starts]).astype(np.int32)
        y = np.stack([self.tokens[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def load_meta(meta_path: Path) -> dict:
    return json.loads(Path(meta_path).read_text())

"""Save / load checkpoints for the toy model.

A checkpoint is a directory containing:
    weights.safetensors   — model parameters (MLX-native serialization)
    optimizer.safetensors — optimizer state (Adam moments)
    state.json            — step, RNG state, config snapshot

Resume semantics: loading restores model weights, optimizer state, and
the data-loader RNG so training can pick up byte-identically. The 2B run
will use the same shape so the toy run is the unit test.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from toy.config import ModelConfig, TrainConfig


def _flatten(prefix: str, obj, out: dict) -> None:
    """Flatten nested params (dicts/lists of mx.array) into a flat dict
    with dotted keys, matching what mlx.utils.tree_flatten would do."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}" if prefix else str(i), v, out)
    elif isinstance(obj, mx.array):
        out[prefix] = obj


def _unflatten(flat: dict) -> dict:
    """Inverse of _flatten — turn dotted keys back into nested dict/list."""
    root: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        cur = root
        for i, p in enumerate(parts[:-1]):
            nxt = parts[i + 1]
            child_is_list = nxt.isdigit()
            if p.isdigit():
                p = int(p)
                while not isinstance(cur, list):
                    raise RuntimeError("flat-key shape mismatch")
                while len(cur) <= p:
                    cur.append([] if child_is_list else {})
                cur = cur[p]
            else:
                if p not in cur:
                    cur[p] = [] if child_is_list else {}
                cur = cur[p]
        last = parts[-1]
        if last.isdigit():
            last = int(last)
            while len(cur) <= last:
                cur.append(None)
            cur[last] = v
        else:
            cur[last] = v
    return root


def save(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer,
    step: int,
    rng_state: dict,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    val_loss: float | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    out = checkpoint_dir / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)

    weights: dict = {}
    _flatten("", model.parameters(), weights)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)

    opt_state: dict = {}
    _flatten("", optimizer.state, opt_state)
    # optimizer.state may contain ints (step counter) — only save arrays.
    opt_state = {k: v for k, v in opt_state.items() if isinstance(v, mx.array)}
    if opt_state:
        mx.save_safetensors(str(out / "optimizer.safetensors"), opt_state)

    state = {
        "step": step,
        "val_loss": val_loss,
        "rng_bit_generator_state": rng_state,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    (out / "state.json").write_text(json.dumps(state, indent=2, default=str))
    return out


def load(checkpoint_dir: Path, model: nn.Module, optimizer=None) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    weights_flat = mx.load(str(checkpoint_dir / "weights.safetensors"))
    weights = _unflatten(dict(weights_flat))
    model.update(weights)

    if optimizer is not None:
        opt_path = checkpoint_dir / "optimizer.safetensors"
        if opt_path.exists():
            opt_flat = mx.load(str(opt_path))
            opt_state = _unflatten(dict(opt_flat))
            for k, v in opt_state.items():
                optimizer.state[k] = v

    state = json.loads((checkpoint_dir / "state.json").read_text())
    return state


def latest(checkpoint_dir: Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*"))
    return candidates[-1] if candidates else None

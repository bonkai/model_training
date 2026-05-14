"""Toy transformer with modern components: RMSNorm, RoPE, GQA, SwiGLU.

Same architectural family as Llama / Mistral, scaled down. Built for MLX.

The model is intentionally small (~27M params with default config). Its
job is to prove the training pipeline works end-to-end before we scale up.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from toy.config import ModelConfig


class RMSNorm(nn.Module):
    """Root mean square normalization (Llama-style, no centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., dim)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x * rms)


def _apply_rope(x: mx.array, base: float = 10000.0) -> mx.array:
    """Apply rotary positional embeddings to the last dim of `x`.

    x shape: (batch, n_heads, seq_len, head_dim) — head_dim must be even.

    We compute frequencies on the fly each forward. For toy-scale runs this
    is negligible; in the big model we'll cache.
    """
    *_, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2

    # Frequencies: 1 / base^(2i/head_dim) for i in [0, half)
    inv_freq = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))
    # Positions: (seq_len,)
    pos = mx.arange(seq_len, dtype=mx.float32)
    # Outer product → (seq_len, half)
    freqs = mx.outer(pos, inv_freq)
    cos = mx.cos(freqs).astype(x.dtype)
    sin = mx.sin(freqs).astype(x.dtype)
    # Broadcast to (1, 1, seq_len, half)
    cos = cos.reshape(1, 1, seq_len, half)
    sin = sin.reshape(1, 1, seq_len, half)

    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads queries share n_kv_heads K/V projections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim()
        self.scale = self.head_dim ** -0.5
        self.rope_base = cfg.rope_base

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _apply_rope(q, self.rope_base)
        k = _apply_rope(k, self.rope_base)

        # GQA: expand K, V to match Q's head count by repetition.
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeats=repeats, axis=1)
            v = mx.repeat(v, repeats=repeats, axis=1)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN: y = (silu(x W_gate) * x W_up) W_down.

    The gating gives the network an in-place attention mechanism over its
    own features; outperforms GELU/ReLU at this scale.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Lower-triangular mask of shape (seq_len, seq_len) with 0 on allowed
    positions and -inf on masked. Compatible with mx.fast.SDPA."""
    mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf, dtype=dtype), k=1)
    return mask


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tied_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None  # weight tied via the embedding

    def __call__(self, idx: mx.array) -> mx.array:
        """idx: (batch, seq_len) int → logits (batch, seq_len, vocab)."""
        _, T = idx.shape
        x = self.tok_embed(idx)
        mask = causal_mask(T, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.final_norm(x)
        if self.lm_head is not None:
            return self.lm_head(x)
        # Tied: project back through the embedding table
        return x @ self.tok_embed.weight.T

    def num_params(self) -> int:
        def _count(node) -> int:
            n = 0
            if isinstance(node, dict):
                for v in node.values():
                    n += _count(v)
            elif isinstance(node, list):
                for v in node:
                    n += _count(v)
            elif isinstance(node, mx.array):
                n += node.size
            return n
        return _count(self.parameters())

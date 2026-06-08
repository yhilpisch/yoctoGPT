"""Advanced GPT variant focused on accuracy improvements.

Implements a GPT-style Transformer with the following changes relative to the
baseline model:
- Rotary positional embeddings (RoPE) applied to Q/K (replaces learned pos_emb)
- RMSNorm instead of LayerNorm
- Biasless Linear projections (as in LLaMA)
- SwiGLU MLP
- Residual branch scaling and zero-init on final projections for stability

The public API mirrors the baseline `GPT` to remain drop-in compatible for
training and generation scripts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import GPTBase
from .config import AdvancedModelConfig


class AdvancedGPTConfig(AdvancedModelConfig):
    pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k (pairwise interleaved dims).

    Expects q,k shape (B, H, T, D). We treat the last dimension as pairs of
    size 2: D = 2 * (D/2). For each pair (x1, x2) and angle θ_t we apply:
      x1' = x1 * cos θ_t - x2 * sin θ_t
      x2' = x1 * sin θ_t + x2 * cos θ_t
    """
    B, H, T, D = q.shape
    half = D // 2
    cos = cos_cached.index_select(0, position_ids).to(dtype=q.dtype).view(1, 1, T, half)
    sin = sin_cached.index_select(0, position_ids).to(dtype=q.dtype).view(1, 1, T, half)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x = x.view(B, H, T, half, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).view(B, H, T, D)

    return _apply(q), _apply(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AdvancedGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        if config.use_rope:
            assert (self.head_dim % 2) == 0, "head_dim must be even when using RoPE"
        self.use_rope = config.use_rope
        self.rope_theta = float(config.rope_theta)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask.bool())
        if self.use_rope:
            half = self.head_dim // 2
            inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            pos = torch.arange(config.block_size, dtype=torch.float32)
            freqs = torch.einsum("t,d->td", pos, inv_freq)  # (T, half)
            self.register_buffer("rope_cos_cached", torch.cos(freqs), persistent=False)
            self.register_buffer("rope_sin_cached", torch.sin(freqs), persistent=False)
            assert self.rope_cos_cached.shape == (config.block_size, half)
            assert self.rope_sin_cached.shape == (config.block_size, half)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            pos_ids = torch.arange(pos_offset, pos_offset + T, device=x.device)
            q, k = apply_rope(q, k, pos_ids, self.rope_cos_cached, self.rope_sin_cached)

        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.size(-2)
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)
            if k.size(-2) > self.mask.size(-1):
                k = k[:, :, -self.mask.size(-1) :, :]
                v = v[:, :, -self.mask.size(-1) :, :]
                past_len = max(0, k.size(-2) - T)

        att = (q @ k.transpose(-2, -1)) * self.scale
        if past_len == 0:
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        else:
            S = k.size(-2)
            key_pos = torch.arange(S, device=x.device)
            qry_pos = past_len + torch.arange(T, device=x.device)
            allow = key_pos.unsqueeze(0) <= qry_pos.unsqueeze(1)  # (T, S)
            att = att.masked_fill(~allow.view(1, 1, T, S), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        if use_cache:
            return y, (k, v)
        return y


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # Standard formula: inner = 2/3 * mult * dim, rounded for hardware efficiency.
        # With mult=4 this yields int(8*dim/3) ≈ 2.67*dim, keeping FLOPs comparable
        # to a standard 4*dim MLP while adding the gated activation benefit.
        inner = int(2 * mult * dim / 3)
        # Ensure inner is divisible by common tensor core tile sizes.
        inner = max(inner, 1)
        inner = ((inner + 255) // 256) * 256 if inner > 256 else inner
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, config: AdvancedGPTConfig, resid_scale: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, mult=4, dropout=config.dropout)
        self.resid_scale = resid_scale

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache, pos_offset=pos_offset)
        if use_cache:
            assert isinstance(attn_out, tuple)
            y, present = attn_out
        else:
            y = attn_out
            present = None
        x = x + self.resid_scale * y
        x = x + self.resid_scale * self.mlp(self.ln2(x))
        if use_cache:
            assert present is not None
            return x, present
        return x


class AdvancedGPT(GPTBase):
    def __init__(self, config: AdvancedGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        scale = 1.0 / math.sqrt(2 * config.n_layer)
        self.blocks = nn.ModuleList([Block(config, resid_scale=scale) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for b in self.blocks:
            if isinstance(b, Block):
                nn.init.zeros_(b.attn.c_proj.weight)
                nn.init.zeros_(b.mlp.w3.weight)

        self._tie_weights()

        # Track position offset across generate steps for RoPE.
        self._pos_offset: int = 0

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(
        self,
        idx: torch.Tensor,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
    ) -> tuple[torch.Tensor, int, int]:
        B, T = idx.shape
        # AdvancedGPT uses RoPE and has no learned positional embeddings,
        # so pos_start is effectively the pos_offset for RoPE.
        pos_start = self._pos_offset
        past_len = 0
        if past_kv is not None and len(past_kv) > 0:
            past_len = int(past_kv[0][0].size(-2))
        x = self.tok_emb(idx)
        x = self.drop(x)
        return x, pos_start, past_len

    def _block_kwargs(
        self,
        i: int,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        pos_start: int,
    ) -> dict:
        return {"past_kv": layer_past, "use_cache": use_cache, "pos_offset": pos_start}

    def _on_cache_reset(self) -> None:
        self._pos_offset = 0

    def forward(
        self,
        idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # Stash pos_offset so the base-class forward loop can pass it to blocks.
        self._pos_offset = pos_offset
        out = super().forward(idx, labels=labels, past_kv=past_kv, use_cache=use_cache)
        # Update pos_offset after a successful forward (used by generate).
        if use_cache and isinstance(out, tuple):
            _, presents = out
            if presents:
                seq_len = presents[0][0].size(-2)
                self._pos_offset = pos_offset + idx.size(1)
        return out

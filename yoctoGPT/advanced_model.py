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

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _top_k_top_p_mask


@dataclass
class AdvancedGPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    tie_weights: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0


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
    seq_len: int,
    head_dim: int,
    base: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to last-dim pairs of q and k.

    Expects q,k shape (B, H, T, D). Builds cos/sin of shape (1,1,T,D) by
    computing half-dim frequencies and repeating each along the last axis.
    """
    device = q.device
    dtype = q.dtype
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", pos, theta)  # (T, half)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    # Expand to full D by repeating each along the last axis
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # (1,1,T,D)
    sin = torch.repeat_interleave(sin, 2, dim=-1)  # (1,1,T,D)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, k, T, self.head_dim, self.rope_theta)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = 2 * dim
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_scale * self.attn(self.ln1(x))
        x = x + self.resid_scale * self.mlp(self.ln2(x))
        return x


class AdvancedGPT(nn.Module):
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

        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(_top_k_top_p_mask(logits, top_k=top_k, top_p=top_p), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token is not None and (next_id == eos_token).all():
                break
        return idx

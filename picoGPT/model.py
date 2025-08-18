"""Minimal GPT model in PyTorch.

This file implements a compact, readable GPT-style Transformer suitable for
character- or token-level language modeling. It is intentionally light on
abstractions, with careful in-line documentation to make the core algorithmic
ideas easy to follow and modify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Model hyperparameters defining the network architecture."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with a fixed causal mask.

    The causal mask prevents attending to future positions (strictly upper
    triangular of the attention matrix) so that predictions at position `t`
    depend only on positions `< t`.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5  # for scaled dot-product

        # Projections for query, key, value in a single linear layer for speed
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask (buffer) of shape (1, 1, T, T) to allow broadcasting across batch/heads
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project once and split into query, key, value
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)
        # Apply causal mask: allow only the lower-triangular including main diagonal
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network applied after attention in each block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → residual → LayerNorm → MLP → residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """A tiny GPT-style language model suitable for small datasets.

    - Input is token IDs of shape (B, T).
    - Outputs logits over the vocabulary of shape (B, T, vocab_size).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Final language modeling head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init parameters following GPT-2 style (approx.)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass: produce next-token logits for each position.

        idx: LongTensor with shape (B, T) containing token IDs.
        returns: FloatTensor logits with shape (B, T, vocab_size)
        """

        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        # Gather token and position embeddings and combine them
        tok = self.tok_emb(idx)  # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.drop(tok + pos)

        # Transformer stack
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Language modeling head to vocab logits
        logits = self.head(x)
        return logits

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
        """Autoregressively sample tokens from the model.

        - idx: starting tokens, shape (B, T0)
        - Will append up to `max_new_tokens` tokens, truncating context to
          `block_size` as needed.
        """

        self.eval()
        for _ in range(max_new_tokens):
            # If the sequence grows beyond block_size, crop the left side
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(_top_k_top_p_mask(logits, top_k=top_k, top_p=top_p), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token is not None and (next_id == eos_token).all():
                break
        return idx


def _top_k_top_p_mask(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    """Apply top-k and/or nucleus (top-p) filtering to logits.

    Returns masked logits where filtered entries are set to -inf so that their
    probability after softmax becomes zero.
    """

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        kth = v[..., -1, None]
        mask = logits < kth
        logits = logits.masked_fill(mask, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cdf > top_p
        # Keep at least one token
        mask[..., 0] = False
        # Map back to original order
        indices_to_remove = torch.zeros_like(mask)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=mask)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


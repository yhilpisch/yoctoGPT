"""Performance-optimized GPT variant.

Key changes for training speed (quality-neutral):
- Uses PyTorch scaled_dot_product_attention (Flash/SDPA when available)
  with is_causal=True instead of explicit masking + softmax.
- Biasless Linear layers to enable better kernel fusions.
- Keeps learned positional embeddings (same as baseline) for compatibility.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tckpt

from .model import _top_k_top_p_mask
from .config import PerformanceModelConfig


class PerformanceGPTConfig(PerformanceModelConfig):
    pass


class CausalSelfAttention(nn.Module):
    def __init__(self, config: PerformanceGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # Fused qkv projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.size(-2)
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)
        # SDPA will select the best backend (Flash, MemEfficient, Math)
        # dropout is applied only during training
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )  # (B,H,T,D)
        else:
            S = k.size(-2)
            key_pos = torch.arange(S, device=x.device)
            qry_pos = past_len + torch.arange(T, device=x.device)
            allow = key_pos.unsqueeze(0) <= qry_pos.unsqueeze(1)  # (T, S)
            attn_mask = torch.where(
                allow,
                torch.zeros((T, S), device=x.device, dtype=q.dtype),
                torch.full((T, S), float("-inf"), device=x.device, dtype=q.dtype),
            )
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )  # (B,H,T,D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        if use_cache:
            return y, (k, v)
        return y


class MLP(nn.Module):
    def __init__(self, config: PerformanceGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Block(nn.Module):
    def __init__(self, config: PerformanceGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            assert isinstance(attn_out, tuple)
            y, present = attn_out
        else:
            y = attn_out
            present = None
        x = x + y
        x = x + self.mlp(self.ln2(x))
        if use_cache:
            assert present is not None
            return x, present
        return x


class PerformanceGPT(nn.Module):
    def __init__(self, config: PerformanceGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]
    ):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")
        past_len = 0
        if past_kv is not None and len(past_kv) > 0:
            past_len = int(past_kv[0][0].size(-2))
        pos_start = past_len
        if pos_start + T > self.config.block_size:
            pos_start = max(0, self.config.block_size - T)
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(pos_start, pos_start + T, device=idx.device))
        x = self.drop(tok + pos)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        use_act_ckpt = bool(
            getattr(self, "activation_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
            and not use_cache
        )
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            if use_act_ckpt:
                x = tckpt.checkpoint(block, x, use_reentrant=False)
                continue
            out = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                assert isinstance(out, tuple)
                x, present = out
                presents.append(present)
            else:
                x = out
        x = self.ln_f(x)
        logits = self.head(x)
        if use_cache:
            return logits, presents
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
        finished: Optional[torch.Tensor] = None
        if eos_token is not None:
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        for _ in range(max_new_tokens):
            if past_kv is not None and len(past_kv) > 0 and past_kv[0][0].size(-2) >= self.config.block_size:
                past_kv = None
            if past_kv is None:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx[:, -1:]
            out = self(idx_cond, past_kv=past_kv, use_cache=True)
            assert isinstance(out, tuple)
            logits, past_kv = out
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(_top_k_top_p_mask(logits, top_k=top_k, top_p=top_p), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if finished is not None and eos_token is not None:
                eos_fill = torch.full_like(next_id, int(eos_token))
                next_id = torch.where(finished.unsqueeze(1), eos_fill, next_id)
                finished = finished | next_id.squeeze(1).eq(int(eos_token))
            idx = torch.cat([idx, next_id], dim=1)
            if finished is not None and bool(finished.all()):
                break
        return idx

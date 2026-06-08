"""Base class for GPT-family models in yoctoGPT.

Extracts the shared forward-pass loop, weight initialization, autoregressive
generation, and top-k/top-p sampling that were previously duplicated across
the baseline, advanced, and performance model variants.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tckpt

from .config import ModelConfigBase


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


class GPTBase(nn.Module):
    """Shared infrastructure for GPT-family language models.

    Subclasses must set:
    - self.config: a ModelConfigBase-derived config
    - self.blocks: nn.ModuleList of transformer blocks
    - self.ln_f: final layer norm / RMS norm
    - self.head: LM head (Linear)
    - self.tok_emb: token embedding

    And implement:
    - _embed(idx, past_kv, use_cache) -> (x, pos_start, past_len)
      to build the input embedding vector and return positional metadata.
    - _block_args(i, layer_past, use_cache, pos_start) -> dict
      to supply variant-specific keyword arguments to each block.
    - _on_cache_reset() -> None
      called when the KV cache is reset during generation (e.g., to reset
      pos_offset for RoPE-based models).
    """

    config: ModelConfigBase
    blocks: nn.ModuleList
    ln_f: nn.Module
    head: nn.Linear
    tok_emb: nn.Embedding

    # ---- Subclass hooks ----

    def _embed(
        self,
        idx: torch.Tensor,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
    ) -> tuple[torch.Tensor, int, int]:
        """Build input embeddings and return (x, pos_start, past_len)."""
        raise NotImplementedError

    def _block_kwargs(
        self,
        i: int,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        pos_start: int,
    ) -> dict:
        """Return keyword arguments for block forward call."""
        raise NotImplementedError

    def _on_cache_reset(self) -> None:
        """Called when the KV cache is reset during generation."""

    # ---- Shared implementations ----

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tie_weights(self) -> None:
        """Tie token embedding and LM head weights if configured."""
        if getattr(self.config, "tie_weights", False):
            self.head.weight = self.tok_emb.weight

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
        """Forward pass: produce next-token logits for each position.

        idx: LongTensor with shape (B, T) containing token IDs.
        returns: FloatTensor logits with shape (B, T, vocab_size)
        """

        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x, pos_start, past_len = self._embed(idx, past_kv, use_cache)

        # Transformer stack
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
                # Checkpointing only runs during training (use_cache is False).
                # KV cache is not used here.
                x = tckpt.checkpoint(block, x, use_reentrant=False)
                continue
            out = block(x, **self._block_kwargs(i, layer_past, use_cache, pos_start))
            if use_cache:
                assert isinstance(out, tuple)
                x, present = out
                presents.append(present)
            else:
                x = out
        x = self.ln_f(x)

        # Language modeling head to vocab logits
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
        """Autoregressively sample tokens from the model.

        - idx: starting tokens, shape (B, T0)
        - Will append up to `max_new_tokens` tokens, truncating context to
          `block_size` as needed.
        """

        self.eval()
        finished: Optional[torch.Tensor] = None
        if eos_token is not None:
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._on_cache_reset()
        for _ in range(max_new_tokens):
            if past_kv is not None and len(past_kv) > 0 and past_kv[0][0].size(-2) >= self.config.block_size:
                past_kv = None
                self._on_cache_reset()
            if past_kv is None:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx[:, -1:]
            out = self(idx_cond, past_kv=past_kv, use_cache=True)
            assert isinstance(out, tuple)
            logits, past_kv = out
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(_top_k_top_p_mask(logits, top_k=top_k, top_p=top_p), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            if finished is not None and eos_token is not None:
                eos_fill = torch.full_like(next_id, int(eos_token))
                next_id = torch.where(finished.unsqueeze(1), eos_fill, next_id)
                finished = finished | next_id.squeeze(1).eq(int(eos_token))
            idx = torch.cat([idx, next_id], dim=1)
            if finished is not None and bool(finished.all()):
                break
        return idx

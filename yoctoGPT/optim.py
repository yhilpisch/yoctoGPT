"""Optimizer-related helpers for yoctoGPT."""

from __future__ import annotations

import torch.nn as nn


def build_weight_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Create AdamW parameter groups with selective weight decay.

    Applies no decay to:
    - bias terms
    - normalization layers
    - embedding parameters
    - 1D parameters
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_l = name.lower()
        no_decay = (
            name.endswith(".bias")
            or param.ndim == 1
            or "norm" in name_l
            or ".ln" in name_l
            or "tok_emb" in name_l
            or "pos_emb" in name_l
            or "embedding" in name_l
        )
        if no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

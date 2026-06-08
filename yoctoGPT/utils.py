"""Shared utilities for yoctoGPT.

Centralizes device detection and checkpoint loading that were previously
duplicated across multiple modules and scripts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .model import GPT, GPTConfig
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig


def detect_device(explicit: Optional[str] = None) -> str:
    """Select the compute device with a preference for Apple Silicon (MPS).

    On Macs with Apple Silicon, PyTorch's MPS backend provides good defaults.
    This function prefers MPS when available, then falls back to CUDA, then CPU.
    Users can override via the --device flag.  Pass "auto" to auto-detect.
    """
    if explicit and explicit != "auto":
        return explicit
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_from_checkpoint(
    path: str,
    device: str = "cpu",
    weights_only: bool = True,
) -> tuple[nn.Module, dict]:
    """Load a model from a yoctoGPT checkpoint.

    Returns (model, checkpoint_dict) where the model is already moved to
    `device` and set to eval mode. The full checkpoint dict is returned so
    callers can extract additional metadata (arch, mode, etc.).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=weights_only)
    arch = ckpt.get("arch", "gpt")
    config_dict = ckpt["model_config"]

    if arch == "gpt_plus":
        model = AdvancedGPT(AdvancedGPTConfig(**config_dict))
    elif arch == "gpt_fast":
        model = PerformanceGPT(PerformanceGPTConfig(**config_dict))
    else:
        model = GPT(GPTConfig(**config_dict))

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt

"""picoGPT package

Minimal GPT implementation and tooling.

This package exposes the core `GPT` model and its configuration for importers,
while CLI entry points live in sibling modules (train, sampler, chat).
"""

from .model import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
__version__ = "0.1.0"


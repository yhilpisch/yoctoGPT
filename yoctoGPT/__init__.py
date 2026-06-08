"""yoctoGPT package — Minimal GPT from Scratch.

(c) Dr. Yves J. Hilpisch
AI-Powered by Different LLMs.

Minimal GPT implementation and tooling.

This package exposes all model variants and their configurations for importers,
while CLI entry points live in sibling modules (train, sampler, chat).
"""

from .model import GPT, GPTConfig
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig

__all__ = [
    "GPT",
    "GPTConfig",
    "AdvancedGPT",
    "AdvancedGPTConfig",
    "PerformanceGPT",
    "PerformanceGPTConfig",
]
__version__ = "0.1.0"

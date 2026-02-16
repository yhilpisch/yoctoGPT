"""Centralized configuration dataclasses and CLI parsing helpers.

These small dataclasses make it easy to pass configuration between modules
without introducing a heavy dependency on external configuration libraries.
Training scripts can override any of these via CLI flags and construct the
final config objects in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture.

    - vocab_size: Number of discrete input/output tokens.
    - block_size: Maximum context length (sequence length) the model can see.
    - n_layer: Number of Transformer blocks.
    - n_head: Number of attention heads per block.
    - n_embd: Embedding (hidden) dimension size.
    - dropout: Dropout probability used in attention and MLP blocks.
    """

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Configuration for a training run.

    Paths use a single `data_dir` containing `train.bin` and `val.bin` in both
    char and token modes. In token mode we also require a `tokenizer_path`.
    """

    # Data
    mode: str = "char"  # "char" or "token"
    data_dir: str = "data/char"
    tokenizer_path: Optional[str] = None  # required for token mode
    vocab_json: Optional[str] = None  # path to char vocab.json (auto-detected)

    # Training
    batch_size: int = 64
    max_iters: int = 5000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    label_smoothing: float = 0.0
    cosine_lr: bool = False
    warmup_iters: int = 100

    # Model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    tie_weights: bool = False
    auto_tie_weights: bool = False
    model_type: str = "gpt"  # "gpt" or "gpt_plus"
    # EMA options
    ema: bool = False
    ema_decay: float = 0.999
    ema_eval: bool = True

    # Misc
    ckpt_dir: str = "checkpoints/run"
    resume: Optional[str] = None  # path to checkpoint (model+optimizer)
    init_from: Optional[str] = None  # path to checkpoint (model only)
    strict_init: bool = True  # if False, allow partial warm starts (strict=False)
    seed: int = 1337
    device: Optional[str] = None  # auto-detect if None
    amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" or "fp16" when amp is enabled
    compile: bool = False
    grad_accum_steps: int = 1
    activation_checkpointing: bool = False
    auto_microbatch: bool = False

    def model_config(self, vocab_size: int) -> ModelConfig:
        """Produce a ModelConfig coupled to the provided `vocab_size`."""

        return ModelConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
        )

    def to_dict(self):  # convenience for checkpoint serialization
        return asdict(self)

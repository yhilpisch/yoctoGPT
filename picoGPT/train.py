"""Training entry point for picoGPT.

Supports both character-level and token-level training by sharing a common
training loop and only switching out the vocabulary/encoding and dataset
loading. Designed for clarity and minimal dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from .config import TrainConfig
from .data import CharVocab, load_bin, make_windows
from .model import GPT, GPTConfig
from .tokenizer import load_tokenizer


def detect_device(explicit: str | None = None) -> str:
    """Select the compute device with a preference for Apple Silicon (MPS).

    On Macs with Apple Silicon, PyTorch's MPS backend provides good defaults.
    This function prefers MPS when available, then falls back to CUDA, then CPU.
    Users can override via --device.
    """
    if explicit:
        return explicit
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a minimal GPT (char or token mode)")
    # Modes and paths
    p.add_argument("--mode", choices=["char", "token"], default="char")
    p.add_argument("--data_dir", type=str, default="data/char")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/run")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to fully resume (model+optimizer)")
    p.add_argument("--init_from", type=str, default=None, help="Path to checkpoint to warm start (model weights only)")
    p.add_argument("--strict_init", dest="strict_init", action="store_true", help="Strict weight loading for --init_from (default)")
    p.add_argument("--no_strict_init", dest="strict_init", action="store_false", help="Allow partial warm start (ignore missing/unexpected)")
    p.set_defaults(strict_init=True)

    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)

    # Model hyperparameters
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)

    args = p.parse_args()
    tc = TrainConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
        device=args.device,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        resume=args.resume,
        init_from=args.init_from,
        strict_init=args.strict_init,
    )
    return tc


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_batch(
    data_ids: torch.LongTensor, block_size: int, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    ixs = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
    x, y = make_windows(data_ids, block_size, ixs)
    return x.to(device), y.to(device)


def evaluate(model: GPT, ids: torch.LongTensor, cfg: TrainConfig, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_iters):
            xb, yb = get_batch(ids, cfg.block_size, cfg.batch_size, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _diff_model_configs(a: dict, b: GPTConfig) -> dict:
    fields = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    diffs = {}
    for k in fields:
        av = a.get(k)
        bv = getattr(b, k)
        if av != bv:
            diffs[k] = (av, bv)
    return diffs


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = detect_device(cfg.device)

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    assert train_path.exists() and val_path.exists(), (
        "Expected train.bin and val.bin in data_dir. Use scripts/prepare_*.py first."
    )

    # Load encoded datasets
    train_ids = load_bin(train_path)
    val_ids = load_bin(val_path)

    # Resolve vocab/tokenizer and determine vocab size
    vocab_size = None
    char_vocab = None
    tokenizer = None
    if cfg.mode == "char":
        # Load vocab.json from data_dir
        vocab_json = data_dir / "vocab.json"
        assert vocab_json.exists(), "Missing vocab.json for char mode"
        char_vocab = CharVocab.load(vocab_json)
        vocab_size = char_vocab.vocab_size
    else:  # token mode
        assert cfg.tokenizer_path is not None, "--tokenizer_path is required for token mode"
        tokenizer = load_tokenizer(cfg.tokenizer_path)
        vocab_size = tokenizer.vocab_size

    # Create model; support warm-start or resume
    start_iter = 0
    resume_path = cfg.resume
    init_from_path = cfg.init_from

    # Desired architecture from CLI/current dataset
    desired_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )

    model = GPT(desired_config).to(device)

    # Optionally load weights (warm start) or full state (resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = ckpt.get("model_config")
        assert ckpt_cfg, "Checkpoint missing model_config"
        diffs = _diff_model_configs(ckpt_cfg, desired_config)
        if diffs:
            raise ValueError(f"Cannot resume: model config mismatch {diffs}")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "opt_state" in ckpt:
            optimizer.load_state_dict(ckpt["opt_state"])
        start_iter = int(ckpt.get("iters_completed", 0))
        print(f"Resumed from {resume_path} at iter {start_iter}")
    elif init_from_path:
        ckpt = torch.load(init_from_path, map_location="cpu")
        ckpt_cfg = ckpt.get("model_config")
        assert ckpt_cfg, "Checkpoint missing model_config"
        diffs = _diff_model_configs(ckpt_cfg, desired_config)
        if diffs and cfg.strict_init:
            raise ValueError(
                "Warm start strict mode: model config mismatch. "
                f"Pass --no_strict_init to allow partial load. Diffs: {diffs}"
            )
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=cfg.strict_init)
        print(
            f"Warm-started from {init_from_path}; strict={cfg.strict_init} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    # Training loop
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Interpret --max_iters as additional steps from the current state.
    # If resuming, we will run `cfg.max_iters` more steps on top of `start_iter`.
    end_iter = start_iter + cfg.max_iters
    pbar = tqdm(
        range(start_iter, end_iter),
        initial=start_iter,
        total=end_iter,
        desc="training",
    )
    best_val = float("inf")
    last_val_loss = None  # remember last validation loss for display between evals
    for it in pbar:
        xb, yb = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Lightweight progress update: refresh train loss every 10 steps to keep tqdm current
        if (it + 1) % 10 == 0 or it == start_iter:
            postfix = {"train_loss": loss.item()}
            if last_val_loss is not None:
                postfix["val_loss"] = last_val_loss
            pbar.set_postfix(postfix)

        if (it + 1) % cfg.eval_interval == 0 or it == start_iter:
            val_loss = evaluate(model, val_ids, cfg, device)
            last_val_loss = val_loss
            pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
            # Save a checkpoint if improved
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model_state": model.state_dict(),
                    "model_config": desired_config.__dict__,
                    "mode": cfg.mode,
                    "tokenizer_path": cfg.tokenizer_path,
                    "char_vocab_path": str(data_dir / "vocab.json") if cfg.mode == "char" else None,
                    "opt_state": optimizer.state_dict(),
                    "iters_completed": it + 1,
                    "train_config": cfg.to_dict(),
                }
                torch.save(ckpt, ckpt_dir / "best.pt")
        # Optional occasional save of the latest state
        if (it + 1) % max(1, cfg.eval_interval // 2) == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "model_config": desired_config.__dict__,
                "mode": cfg.mode,
                "tokenizer_path": cfg.tokenizer_path,
                "char_vocab_path": str(data_dir / "vocab.json") if cfg.mode == "char" else None,
                "opt_state": optimizer.state_dict(),
                "iters_completed": it + 1,
                "train_config": cfg.to_dict(),
            }
            torch.save(ckpt, ckpt_dir / "latest.pt")

    # Ensure a final evaluation at the end regardless of eval_interval
    final_val = evaluate(model, val_ids, cfg, device)
    print({"final_val_loss": final_val})


if __name__ == "__main__":
    main()

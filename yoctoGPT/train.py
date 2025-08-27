"""Training entry point for yoctoGPT.

Supports both character-level and token-level training by sharing a common
training loop and only switching out the vocabulary/encoding and dataset
loading. Designed for clarity and minimal dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import copy
import csv
import time
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
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig
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
    p.add_argument("--min_lr", type=float, default=1e-5, help="Min LR for cosine schedule")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--cosine_lr", action="store_true", help="Use cosine learning rate schedule with warmup")
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)

    # Model hyperparameters
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie_weights", action="store_true", help="Tie token embedding and LM head weights")
    p.add_argument("--auto_tie_weights", action="store_true", help="Enable weight tying automatically for small datasets")
    p.add_argument("--model_type", choices=["gpt", "gpt_plus", "gpt_fast"], default="gpt", help="Choose baseline, advanced (accuracy), or fast (SDPA) variant")
    # EMA options
    p.add_argument("--ema", action="store_true", help="Track an exponential moving average of weights")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (closer to 1.0 = slower updates)")
    p.add_argument("--ema_eval", action="store_true", help="Use EMA weights for evaluation and best checkpoint")
    p.add_argument("--no_ema_eval", dest="ema_eval", action="store_false")
    p.set_defaults(ema_eval=True)

    args = p.parse_args()
    tc = TrainConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        label_smoothing=args.label_smoothing,
        cosine_lr=args.cosine_lr,
        warmup_iters=args.warmup_iters,
        seed=args.seed,
        device=args.device,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        auto_tie_weights=args.auto_tie_weights,
        model_type=args.model_type,
        ema=args.ema,
        ema_decay=args.ema_decay,
        ema_eval=args.ema_eval,
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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                label_smoothing=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.0,
            )
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
    # Decide weight tying policy
    auto_tie = False
    if cfg.auto_tie_weights and not cfg.tie_weights:
        # Heuristic: enable tying for smaller datasets (< 1M tokens)
        try:
            auto_tie = (len(train_ids) < 1_000_000)
        except Exception:
            auto_tie = False

    arch = cfg.model_type if hasattr(cfg, "model_type") else "gpt"
    if arch == "gpt_plus":
        desired_config = AdvancedGPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            tie_weights=(cfg.tie_weights or auto_tie),
        )
        model = AdvancedGPT(desired_config).to(device)
    elif arch == "gpt_fast":
        desired_config = PerformanceGPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            tie_weights=(cfg.tie_weights or auto_tie),
        )
        model = PerformanceGPT(desired_config).to(device)
    else:
        desired_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            tie_weights=(cfg.tie_weights or auto_tie),
        )
        model = GPT(desired_config).to(device)

    # Optionally load weights (warm start) or full state (resume)
    # Build parameter-wise weight decay groups: no decay for biases/norms/embeddings
    def build_param_groups(m: nn.Module, weight_decay: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            nd = (
                n.endswith(".bias")
                or ".ln" in n
                or "ln1" in n
                or "ln2" in n
                or "ln_f" in n
                or "norm" in n
                or "tok_emb" in n
                or "pos_emb" in n
            )
            (no_decay if nd else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay), lr=cfg.lr)

    # EMA model (optional)
    ema_model = None
    if cfg.ema:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = ckpt.get("model_config")
        assert ckpt_cfg, "Checkpoint missing model_config"
        ckpt_arch = ckpt.get("arch", "gpt")
        if ckpt_arch != arch:
            raise ValueError(f"Cannot resume: arch mismatch ckpt={ckpt_arch} vs requested={arch}")
        if ckpt_cfg != desired_config.__dict__:
            raise ValueError("Cannot resume: model config mismatch with checkpoint")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "opt_state" in ckpt:
            optimizer.load_state_dict(ckpt["opt_state"])
        if ema_model is not None and "ema_state" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state"], strict=False)
        start_iter = int(ckpt.get("iters_completed", 0))
        print(f"Resumed from {resume_path} at iter {start_iter}")
    elif init_from_path:
        ckpt = torch.load(init_from_path, map_location="cpu")
        ckpt_cfg = ckpt.get("model_config")
        assert ckpt_cfg, "Checkpoint missing model_config"
        ckpt_arch = ckpt.get("arch", "gpt")
        if cfg.strict_init:
            if ckpt_arch != arch or ckpt_cfg != desired_config.__dict__:
                raise ValueError(
                    "Warm start strict mode: model/arch mismatch. "
                    "Pass --no_strict_init to allow partial load."
                )
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
        else:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if ema_model is not None:
            if "ema_state" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state"], strict=False)
            else:
                ema_model.load_state_dict(model.state_dict(), strict=False)
        print(
            f"Warm-started from {init_from_path}; strict={cfg.strict_init} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    # Training loop
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Metrics CSV in checkpoint directory
    metrics_path = ckpt_dir / "metrics.csv"
    metrics_fields = [
        "iter",
        "train_loss",
        "val_loss",
        "best_val_loss",
        "lr",
        "time_sec",
        "tokens_seen",
        "throughput_tps",
        "grad_norm",
    ]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            writer.writeheader()
    # Persist a one-time run metadata JSON for later analysis
    meta_path = ckpt_dir / "run_meta.json"
    if not meta_path.exists():
        try:
            param_count = int(sum(p.numel() for p in model.parameters()))
        except Exception:
            param_count = None
        meta = {
            "arch": arch,
            "device": device,
            "train_config": cfg.to_dict(),
            "model_config": desired_config.__dict__,
            "params": param_count,
            "tokens_per_step": cfg.batch_size * cfg.block_size,
            "created_at": time.time(),
        }
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

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

    # LR scheduling helpers
    base_lr = cfg.lr
    min_lr = cfg.min_lr
    warmup_iters = cfg.warmup_iters
    total_iters = end_iter

    def get_lr(step: int) -> float:
        if not cfg.cosine_lr:
            return base_lr
        if step < warmup_iters:
            return base_lr * step / max(1, warmup_iters)
        # Cosine decay from base_lr to min_lr
        progress = (step - warmup_iters) / max(1, (total_iters - warmup_iters))
        progress = min(max(progress, 0.0), 1.0)
        import math as _math
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + _math.cos(_math.pi * progress))

    start_wall = time.time()
    tokens_per_step = cfg.batch_size * cfg.block_size
    for it in pbar:
        xb, yb = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            label_smoothing=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.0,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                d = float(cfg.ema_decay)
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
        # Update learning rate
        new_lr = get_lr(it + 1)
        for g in optimizer.param_groups:
            g["lr"] = new_lr

        # Lightweight progress update: refresh train loss every 10 steps to keep tqdm current
        if (it + 1) % 10 == 0 or it == start_iter:
            postfix = {"train_loss": loss.item()}
            if last_val_loss is not None:
                postfix["val_loss"] = last_val_loss
            pbar.set_postfix(postfix)

        if (it + 1) % cfg.eval_interval == 0 or it == start_iter:
            eval_model = ema_model if (cfg.ema and cfg.ema_eval and ema_model is not None) else model
            val_loss = evaluate(eval_model, val_ids, cfg, device)
            last_val_loss = val_loss
            pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
            # Save a checkpoint if improved
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model_state": model.state_dict(),
                    "model_config": desired_config.__dict__,
                    "arch": arch,
                    "mode": cfg.mode,
                    "tokenizer_path": cfg.tokenizer_path,
                    "char_vocab_path": str(data_dir / "vocab.json") if cfg.mode == "char" else None,
                    "opt_state": optimizer.state_dict(),
                    "iters_completed": it + 1,
                    "train_config": cfg.to_dict(),
                }
                if ema_model is not None:
                    ckpt["ema_state"] = ema_model.state_dict()
                torch.save(ckpt, ckpt_dir / "best.pt")
        # Optional occasional save of the latest state
        if (it + 1) % max(1, cfg.eval_interval // 2) == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "model_config": desired_config.__dict__,
                "arch": arch,
                "mode": cfg.mode,
                "tokenizer_path": cfg.tokenizer_path,
                "char_vocab_path": str(data_dir / "vocab.json") if cfg.mode == "char" else None,
                "opt_state": optimizer.state_dict(),
                "iters_completed": it + 1,
                "train_config": cfg.to_dict(),
            }
            if ema_model is not None:
                ckpt["ema_state"] = ema_model.state_dict()
            torch.save(ckpt, ckpt_dir / "latest.pt")

        # Append metrics row
        elapsed = time.time() - start_wall
        tokens_seen = (it + 1 - start_iter) * tokens_per_step
        tps = tokens_seen / max(elapsed, 1e-9)
        # Round times/throughput to integers for readability
        time_sec_out = int(round(elapsed))
        tps_out = int(round(tps))
        best_val_out = "" if best_val == float("inf") else round(float(best_val), 5)
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            writer.writerow(
                {
                    "iter": it + 1,
                    "train_loss": round(float(loss.item()), 5),
                    "val_loss": round(float(last_val_loss), 5) if last_val_loss is not None else "",
                    "best_val_loss": best_val_out,
                    "lr": round(float(new_lr), 5),
                    "time_sec": time_sec_out,
                    "tokens_seen": int(tokens_seen),
                    "throughput_tps": tps_out,
                    "grad_norm": round(float(grad_total_norm), 5),
                }
            )

    # Ensure a final evaluation at the end regardless of eval_interval
    final_eval_model = ema_model if (cfg.ema and cfg.ema_eval and ema_model is not None) else model
    final_val = evaluate(final_eval_model, val_ids, cfg, device)
    print({"final_val_loss": final_val})
    # Append a final metrics row capturing final validation
    try:
        elapsed = time.time() - start_wall
        tokens_seen = (end_iter - start_iter) * tokens_per_step
        tps = tokens_seen / max(elapsed, 1e-9)
        time_sec_out = int(round(elapsed))
        tps_out = int(round(tps))
        best_val_out = "" if best_val == float("inf") else round(float(best_val), 5)
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            writer.writerow(
                {
                    "iter": end_iter,
                    "train_loss": "",
                    "val_loss": round(float(final_val), 5),
                    "best_val_loss": best_val_out,
                    "lr": round(float(get_lr(end_iter)), 5),
                    "time_sec": time_sec_out,
                    "tokens_seen": int(tokens_seen),
                    "throughput_tps": tps_out,
                    "grad_norm": "",
                }
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()

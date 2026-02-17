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
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .config import TrainConfig
from .data import CharVocab, TokenIdArray, load_ids_adaptive, make_windows
from .model import GPT, GPTConfig
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig
from .tokenizer import load_tokenizer
from .optim import build_weight_decay_param_groups


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
    p.add_argument("--memmap_threshold_mb", type=int, default=128, help="Use memmap loading above this dataset size (MB)")
    p.add_argument("--always_memmap", action="store_true", help="Always load train/val .bin via numpy memmap")
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
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16", help="Autocast dtype when --amp is enabled")
    p.add_argument("--compile", action="store_true", help="Compile model.forward with torch.compile when available")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Number of micro-steps to accumulate before optimizer step")
    p.add_argument("--activation_checkpointing", action="store_true", help="Enable activation checkpointing through transformer blocks")
    p.add_argument("--auto_microbatch", action="store_true", help="Auto-reduce micro-batch size on CUDA OOM")
    p.add_argument("--save_strategy", choices=["both", "best", "latest", "none"], default="both", help="Checkpoint save policy")
    p.add_argument("--early_stopping_patience", type=int, default=0, help="Stop after N evals without improvement (0 disables)")
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum val-loss improvement to reset early stopping")
    p.add_argument("--ddp", action="store_true", help="Enable DDP (single-node multi-process via torchrun)")
    p.add_argument("--ddp_backend", choices=["nccl", "gloo"], default=None, help="DDP backend (auto if omitted)")

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
        memmap_threshold_mb=max(0, int(args.memmap_threshold_mb)),
        always_memmap=args.always_memmap,
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
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        compile=args.compile,
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        activation_checkpointing=args.activation_checkpointing,
        auto_microbatch=args.auto_microbatch,
        save_strategy=args.save_strategy,
        early_stopping_patience=max(0, int(args.early_stopping_patience)),
        early_stopping_min_delta=max(0.0, float(args.early_stopping_min_delta)),
        ddp=args.ddp,
        ddp_backend=args.ddp_backend,
    )
    return tc


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_batch(
    data_ids: TokenIdArray,
    block_size: int,
    batch_size: int,
    device: str,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = int(len(data_ids)) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset too small for requested block_size")
    if ddp_world_size > 1:
        global_batch = int(batch_size) * int(ddp_world_size)
        ixs_all = torch.randint(0, max_start, (global_batch,))
        ixs = ixs_all[int(ddp_rank) :: int(ddp_world_size)]
        if ixs.numel() > batch_size:
            ixs = ixs[:batch_size]
    else:
        ixs = torch.randint(0, max_start, (batch_size,))
    x, y = make_windows(data_ids, block_size, ixs)
    return x.to(device), y.to(device)


def evaluate(
    model: GPT,
    ids: TokenIdArray,
    cfg: TrainConfig,
    device: str,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    batch_size_override: int | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_iters):
            bs = batch_size_override if batch_size_override is not None else cfg.batch_size
            xb, yb = get_batch(
                ids,
                cfg.block_size,
                bs,
                device,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
            )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with amp_ctx:
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


def _reduce_mean_scalar(value: float, device: str, enabled: bool, world_size: int) -> float:
    if not enabled or world_size <= 1:
        return float(value)
    reduce_device = device if str(device).startswith("cuda") else "cpu"
    t = torch.tensor([float(value)], device=reduce_device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(world_size)
    return float(t.item())


def main() -> None:
    cfg = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(cfg.ddp or world_size > 1)
    if cfg.ddp and world_size <= 1:
        raise RuntimeError("DDP requested but WORLD_SIZE<=1. Launch with torchrun (e.g. --nproc_per_node=2).")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    ddp_inited = False
    if use_ddp:
        if not dist.is_available():
            raise RuntimeError("DDP requested but torch.distributed is unavailable")
        backend = cfg.ddp_backend
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        ddp_inited = True
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    is_master = (rank == 0)
    set_seed(cfg.seed + rank)
    if use_ddp and torch.cuda.is_available():
        device = cfg.device or f"cuda:{local_rank}"
    else:
        device = detect_device(cfg.device)

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    assert train_path.exists() and val_path.exists(), (
        "Expected train.bin and val.bin in data_dir. Use scripts/prepare_*.py first."
    )

    # Load encoded datasets
    train_ids = load_ids_adaptive(
        train_path,
        memmap_threshold_mb=cfg.memmap_threshold_mb,
        prefer_memmap=cfg.always_memmap,
    )
    val_ids = load_ids_adaptive(
        val_path,
        memmap_threshold_mb=cfg.memmap_threshold_mb,
        prefer_memmap=cfg.always_memmap,
    )
    if is_master:
        train_backend = "memmap" if isinstance(train_ids, np.memmap) else "in-memory"
        val_backend = "memmap" if isinstance(val_ids, np.memmap) else "in-memory"
        print(f"Loaded train.bin with {train_backend} backend ({len(train_ids)} tokens)")
        print(f"Loaded val.bin with {val_backend} backend ({len(val_ids)} tokens)")

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
    optimizer = torch.optim.AdamW(
        build_weight_decay_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
    )

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
        if is_master:
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
        if is_master:
            print(
                f"Warm-started from {init_from_path}; strict={cfg.strict_init} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    # Optional DDP wrap (single-node multi-process; launch with torchrun).
    model_train: nn.Module = model
    if use_ddp:
        ddp_device_ids = [local_rank] if str(device).startswith("cuda") else None
        model_train = DDP(model, device_ids=ddp_device_ids)

    # Optional compile (PyTorch 2.x): compile forward only to keep checkpoint keys stable.
    if cfg.compile:
        if use_ddp:
            if is_master:
                print("Skipping torch.compile in DDP mode for stability")
        elif hasattr(torch, "compile"):
            try:
                model_train.forward = torch.compile(model_train.forward)  # type: ignore[method-assign]
                if is_master:
                    print("Compiled model.forward with torch.compile")
            except Exception as e:
                if is_master:
                    print(f"torch.compile unavailable for this run; continuing uncompiled ({e})")
        else:
            if is_master:
                print("torch.compile not available in this PyTorch build; continuing uncompiled")

    # AMP setup (CUDA only)
    amp_enabled = bool(cfg.amp and str(device).startswith("cuda"))
    if cfg.amp and not amp_enabled:
        if is_master:
            print("AMP requested but only CUDA AMP is enabled in this script; continuing in full precision")
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler_enabled = bool(amp_enabled and cfg.amp_dtype == "fp16")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    # Colab-focused memory knobs
    setattr(model_train, "activation_checkpointing", bool(cfg.activation_checkpointing))
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    current_batch_size = int(cfg.batch_size)
    if cfg.auto_microbatch and not str(device).startswith("cuda"):
        if is_master:
            print("auto_microbatch is currently CUDA-only; continuing with fixed batch_size")
    auto_microbatch = bool(cfg.auto_microbatch and str(device).startswith("cuda"))
    if grad_accum_steps > 1:
        if is_master:
            print(f"Using gradient accumulation: {grad_accum_steps} micro-steps per optimizer step")
    if bool(cfg.activation_checkpointing):
        if is_master:
            print("Activation checkpointing enabled")

    # Training loop
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Metrics CSV in checkpoint directory
    metrics_path = ckpt_dir / "metrics.csv"
    metrics_fields = [
        "iter",
        "train_loss",
        "val_loss",
        "val_ppl",
        "val_loss_raw",
        "val_ppl_raw",
        "val_loss_ema",
        "val_ppl_ema",
        "best_val_loss",
        "best_val_ppl",
        "best_val_loss_raw",
        "best_val_ppl_raw",
        "best_val_loss_ema",
        "best_val_ppl_ema",
        "lr",
        "time_sec",
        "tokens_seen",
        "throughput_tps",
        "grad_norm",
        "micro_batch_size",
        "grad_accum_steps",
    ]
    if is_master and not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            writer.writeheader()
    # Persist a one-time run metadata JSON for later analysis
    meta_path = ckpt_dir / "run_meta.json"
    if is_master and not meta_path.exists():
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
            "tokens_per_step": current_batch_size * cfg.block_size * grad_accum_steps,
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
        disable=not is_master,
    )
    best_val = float("inf")
    best_val_raw = float("inf")
    best_val_ema = float("inf")
    no_improve_evals = 0
    last_val_loss = None  # metric used for display (raw or ema)
    last_val_loss_raw = None
    last_val_loss_ema = None
    last_iter = start_iter

    def to_ppl(loss_value: float | None) -> float | None:
        if loss_value is None:
            return None
        return float(math.exp(min(float(loss_value), 20.0)))

    def make_checkpoint(iters_completed: int) -> dict:
        ckpt = {
            "model_state": model.state_dict(),
            "model_config": desired_config.__dict__,
            "arch": arch,
            "mode": cfg.mode,
            "tokenizer_path": cfg.tokenizer_path,
            "char_vocab_path": str(data_dir / "vocab.json") if cfg.mode == "char" else None,
            "opt_state": optimizer.state_dict(),
            "iters_completed": iters_completed,
            "train_config": cfg.to_dict(),
        }
        if ema_model is not None:
            ckpt["ema_state"] = ema_model.state_dict()
        return ckpt

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
    tokens_per_step = current_batch_size * cfg.block_size * grad_accum_steps
    for it in pbar:
        last_iter = it + 1
        while True:
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            try:
                for _ in range(grad_accum_steps):
                    xb, yb = get_batch(
                        train_ids,
                        cfg.block_size,
                        current_batch_size,
                        device,
                        ddp_rank=rank if use_ddp else 0,
                        ddp_world_size=world_size if use_ddp else 1,
                    )
                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if amp_enabled
                        else nullcontext()
                    )
                    with amp_ctx:
                        logits = model_train(xb)
                        micro_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            yb.view(-1),
                            label_smoothing=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.0,
                        )
                        loss = micro_loss / grad_accum_steps
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    micro_losses.append(float(micro_loss.item()))
                break
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                if not (auto_microbatch and oom and current_batch_size > 1):
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                tokens_per_step = current_batch_size * cfg.block_size * grad_accum_steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_master:
                    print(f"OOM detected; reducing micro-batch size to {current_batch_size}")
                continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()
        train_loss_out = float(sum(micro_losses) / max(1, len(micro_losses)))
        train_loss_out = _reduce_mean_scalar(train_loss_out, device, use_ddp, world_size)
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
            postfix = {"train_loss": train_loss_out}
            if last_val_loss is not None:
                postfix["val_loss"] = last_val_loss
            postfix["mbs"] = current_batch_size
            if is_master:
                pbar.set_postfix(postfix)

        if (it + 1) % cfg.eval_interval == 0 or it == start_iter:
            # Compute both raw and EMA validation losses when possible
            val_raw = evaluate(
                model_train,
                val_ids,
                cfg,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                batch_size_override=current_batch_size,
                ddp_rank=rank if use_ddp else 0,
                ddp_world_size=world_size if use_ddp else 1,
            )
            val_ema = (
                evaluate(
                    ema_model,
                    val_ids,
                    cfg,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    batch_size_override=current_batch_size,
                    ddp_rank=rank if use_ddp else 0,
                    ddp_world_size=world_size if use_ddp else 1,
                )
                if ema_model is not None
                else None
            )
            val_raw = _reduce_mean_scalar(val_raw, device, use_ddp, world_size)
            if val_ema is not None:
                val_ema = _reduce_mean_scalar(val_ema, device, use_ddp, world_size)
            use_ema = bool(cfg.ema and cfg.ema_eval and (val_ema is not None))
            val_loss = val_ema if use_ema else val_raw
            # Track last values
            last_val_loss_raw = float(val_raw)
            last_val_loss_ema = float(val_ema) if val_ema is not None else None
            last_val_loss = float(val_loss)
            if is_master:
                pbar.set_postfix({"train_loss": train_loss_out, "val_loss": val_loss, "mbs": current_batch_size})
            # Update best trackers
            if val_raw < best_val_raw:
                best_val_raw = float(val_raw)
            if val_ema is not None and val_ema < best_val_ema:
                best_val_ema = float(val_ema)
            # Save a checkpoint if primary val improved
            improved = bool(val_loss < (best_val - float(cfg.early_stopping_min_delta)))
            if improved:
                best_val = float(val_loss)
                no_improve_evals = 0
                if is_master and cfg.save_strategy in ("both", "best"):
                    torch.save(make_checkpoint(it + 1), ckpt_dir / "best.pt")
            else:
                no_improve_evals += 1
            if cfg.early_stopping_patience > 0 and no_improve_evals >= cfg.early_stopping_patience:
                if is_master:
                    print(
                        f"Early stopping triggered at iter {it + 1}: "
                        f"no improvement for {no_improve_evals} evals"
                    )
                if is_master and cfg.save_strategy in ("both", "latest"):
                    torch.save(make_checkpoint(it + 1), ckpt_dir / "latest.pt")
                break
        # Optional occasional save of the latest state
        if is_master and cfg.save_strategy in ("both", "latest") and ((it + 1) % max(1, cfg.eval_interval // 2) == 0):
            torch.save(make_checkpoint(it + 1), ckpt_dir / "latest.pt")

        # Append metrics row
        elapsed = time.time() - start_wall
        tokens_seen = (it + 1 - start_iter) * tokens_per_step
        tps = tokens_seen / max(elapsed, 1e-9)
        # Round times/throughput to integers for readability
        time_sec_out = int(round(elapsed))
        tps_out = int(round(tps))
        best_val_out = "" if best_val == float("inf") else round(float(best_val), 5)
        best_val_ppl_out = "" if best_val == float("inf") else round(float(to_ppl(best_val)), 5)
        val_ppl = to_ppl(last_val_loss)
        val_ppl_raw = to_ppl(last_val_loss_raw)
        val_ppl_ema = to_ppl(last_val_loss_ema)
        best_val_ppl_raw = "" if best_val_raw == float("inf") else round(float(to_ppl(best_val_raw)), 5)
        best_val_ppl_ema = "" if best_val_ema == float("inf") else round(float(to_ppl(best_val_ema)), 5)
        if is_master:
            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics_fields)
                writer.writerow(
                    {
                        "iter": it + 1,
                        "train_loss": round(train_loss_out, 5),
                        "val_loss": round(float(last_val_loss), 5) if last_val_loss is not None else "",
                        "val_ppl": round(float(val_ppl), 5) if val_ppl is not None else "",
                        "val_loss_raw": round(float(last_val_loss_raw), 5) if last_val_loss_raw is not None else "",
                        "val_ppl_raw": round(float(val_ppl_raw), 5) if val_ppl_raw is not None else "",
                        "val_loss_ema": round(float(last_val_loss_ema), 5) if last_val_loss_ema is not None else "",
                        "val_ppl_ema": round(float(val_ppl_ema), 5) if val_ppl_ema is not None else "",
                        "best_val_loss": best_val_out,
                        "best_val_ppl": best_val_ppl_out,
                        "best_val_loss_raw": (round(float(best_val_raw), 5) if best_val_raw != float("inf") else ""),
                        "best_val_ppl_raw": best_val_ppl_raw,
                        "best_val_loss_ema": (round(float(best_val_ema), 5) if best_val_ema != float("inf") else ""),
                        "best_val_ppl_ema": best_val_ppl_ema,
                        "lr": round(float(new_lr), 5),
                        "time_sec": time_sec_out,
                        "tokens_seen": int(tokens_seen),
                        "throughput_tps": tps_out,
                        "grad_norm": round(float(grad_total_norm), 5),
                        "micro_batch_size": int(current_batch_size),
                        "grad_accum_steps": int(grad_accum_steps),
                    }
                )
        if cfg.early_stopping_patience > 0 and no_improve_evals >= cfg.early_stopping_patience:
            break

    # Ensure a final evaluation at the end regardless of eval_interval
    # Final evaluation on raw and EMA
    final_val_raw = evaluate(
        model_train,
        val_ids,
        cfg,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        batch_size_override=current_batch_size,
        ddp_rank=rank if use_ddp else 0,
        ddp_world_size=world_size if use_ddp else 1,
    )
    final_val_ema = (
        evaluate(
            ema_model,
            val_ids,
            cfg,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            batch_size_override=current_batch_size,
            ddp_rank=rank if use_ddp else 0,
            ddp_world_size=world_size if use_ddp else 1,
        )
        if ema_model is not None
        else None
    )
    final_val_raw = _reduce_mean_scalar(final_val_raw, device, use_ddp, world_size)
    if final_val_ema is not None:
        final_val_ema = _reduce_mean_scalar(final_val_ema, device, use_ddp, world_size)
    use_ema_final = bool(cfg.ema and cfg.ema_eval and (final_val_ema is not None))
    final_val = final_val_ema if use_ema_final else final_val_raw
    if is_master:
        print({"final_val_loss": final_val})
    # Append a final metrics row capturing final validation
    try:
        elapsed = time.time() - start_wall
        tokens_seen = (last_iter - start_iter) * tokens_per_step
        tps = tokens_seen / max(elapsed, 1e-9)
        time_sec_out = int(round(elapsed))
        tps_out = int(round(tps))
        best_val_out = "" if best_val == float("inf") else round(float(best_val), 5)
        best_val_ppl_out = "" if best_val == float("inf") else round(float(to_ppl(best_val)), 5)
        best_val_ppl_raw = "" if best_val_raw == float("inf") else round(float(to_ppl(best_val_raw)), 5)
        best_val_ppl_ema = "" if best_val_ema == float("inf") else round(float(to_ppl(best_val_ema)), 5)
        final_val_ppl = to_ppl(final_val)
        final_val_ppl_raw = to_ppl(final_val_raw)
        final_val_ppl_ema = to_ppl(final_val_ema)
        if is_master:
            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics_fields)
                writer.writerow(
                    {
                        "iter": last_iter,
                        "train_loss": "",
                        "val_loss": round(float(final_val), 5),
                        "val_ppl": round(float(final_val_ppl), 5) if final_val_ppl is not None else "",
                        "val_loss_raw": round(float(final_val_raw), 5) if final_val_raw is not None else "",
                        "val_ppl_raw": round(float(final_val_ppl_raw), 5) if final_val_ppl_raw is not None else "",
                        "val_loss_ema": round(float(final_val_ema), 5) if final_val_ema is not None else "",
                        "val_ppl_ema": round(float(final_val_ppl_ema), 5) if final_val_ppl_ema is not None else "",
                        "best_val_loss": best_val_out,
                        "best_val_ppl": best_val_ppl_out,
                        "best_val_loss_raw": (round(float(best_val_raw), 5) if best_val_raw != float("inf") else ""),
                        "best_val_ppl_raw": best_val_ppl_raw,
                        "best_val_loss_ema": (round(float(best_val_ema), 5) if best_val_ema != float("inf") else ""),
                        "best_val_ppl_ema": best_val_ppl_ema,
                        "lr": round(float(get_lr(last_iter)), 5),
                        "time_sec": time_sec_out,
                        "tokens_seen": int(tokens_seen),
                        "throughput_tps": tps_out,
                        "grad_norm": "",
                        "micro_batch_size": int(current_batch_size),
                        "grad_accum_steps": int(grad_accum_steps),
                    }
                )
    except Exception:
        pass

    if ddp_inited:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

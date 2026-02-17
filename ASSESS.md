# Assessment and Open Improvements for yoctoGPT

This file tracks open technical points after a fresh code review.

### 1. Training Scalability
- **Distributed training (DDP/FSDP) is still missing**: `yoctoGPT/train.py` currently runs single-process only. Add basic `torch.distributed` initialization, rank-aware logging/checkpointing, and distributed samplers.
- **Streaming token dataset is still missing**: `yoctoGPT/data.py` loads full `.bin` into RAM via `np.fromfile`. Add a memory-mapped dataset (`np.memmap`) path for large corpora.

### 2. API Consistency
- **Forward API mismatch remains**: `AdvancedGPT.forward` includes `labels` and can return `(logits, loss)`, while `GPT` and `PerformanceGPT` return logits-only (or cache tuples). This complicates generic training/inference wrappers across model types.
- **Config class divergence remains**: `GPTConfig`, `AdvancedGPTConfig`, and `PerformanceGPTConfig` are separate and drift-prone. Consolidate shared fields via a common base/config adapter used by CLI entry points.

### 3. Additional Review Findings
- **Sampling/chat are CPU-only by construction**: `yoctoGPT/sampler.py` and `yoctoGPT/chat.py` load checkpoints with `map_location="cpu"` and never expose a `--device` argument or move models/tensors to CUDA/MPS. This is a practical performance limitation for inference.
- **Batch generation EOS behavior is coarse**: all `generate` implementations stop only when all batch rows hit EOS (`(next_id == eos_token).all()`). Add per-sequence stopping/masking to avoid over-generating completed samples in batched decoding.

### 4. Verification Snapshot
- Local tests pass in the provided environment: `12 passed, 6 skipped, 6 subtests passed` (pytest).

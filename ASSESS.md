# Assessment and Open Improvements for yoctoGPT

This file tracks open technical points after a fresh code review.

### 1. Training Scalability
- **Distributed training (DDP/FSDP) is still missing**: `yoctoGPT/train.py` currently runs single-process only. Add basic `torch.distributed` initialization, rank-aware logging/checkpointing, and distributed samplers.
- **Streaming token dataset is still missing**: `yoctoGPT/data.py` loads full `.bin` into RAM via `np.fromfile`. Add a memory-mapped dataset (`np.memmap`) path for large corpora.

### 2. API Consistency
- **Config class divergence remains**: `GPTConfig`, `AdvancedGPTConfig`, and `PerformanceGPTConfig` are separate and drift-prone. Consolidate shared fields via a common base/config adapter used by CLI entry points.

### 3. Additional Review Findings
- (No additional open findings at this time.)

### 4. Verification Snapshot
- Local tests pass in the provided environment: `13 passed, 6 skipped, 9 subtests passed` (pytest).

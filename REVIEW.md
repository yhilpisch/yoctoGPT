# yoctoGPT — Open Issues

**Updated**: 2025-06-08

Issues resolved in the quality sweep have been removed. Only open items remain.

---

## HIGH — Bugs & Correctness Risks

### H1. `evaluate()` uses `model_train` (DDP-wrapped) but `make_checkpoint()` uses raw `model`

When DDP is active, `evaluate()` receives `model_train` (the DDP wrapper) while `make_checkpoint()` saves `model.state_dict()` using the unwrapped original. This happens to work, but the asymmetry is fragile — if someone refactors to use `model_train` consistently, they'd save the DDP `.module` prefix and break checkpoint loading.

---

## LOW — Style, Nits, Minor Improvements

### L1. `top_k=0` sentinel means "disabled" in CLI but `None` internally

In `sampler.py` and `chat.py`:
```python
top_k=args.top_k if args.top_k > 0 else None,
```
Using `0` as a sentinel for "no top-k" works but is error-prone. A future caller passing `top_k=0` directly to `_top_k_top_p_mask` would get unexpected behavior (it's guarded with `top_k > 0`), but the API contract is unclear.

### L2. `README.md` is ~500 lines — could benefit from splitting

The README combines quickstart, architecture explanation, M1 configs, performance tips, overfitting guide, and GPU setup. A `docs/` folder with separate pages would improve navigability.

### L3. No `py.typed` marker

The package uses inline type hints (Python 3.10+ style `X | Y`) but lacks `py.typed`, so type checkers won't recognize it as a typed package.

### L4. Metrics CSV rows written one-by-one with `open("a")` inside the training loop

Each iteration opens the file, appends a row, and closes it. This is safe but slow. A buffered writer or periodic flush would be more efficient for long runs.

---

## TEST COVERAGE GAPS

| Area | Status |
|---|---|
| **Sampler CLI end-to-end** | **Not covered** |
| **Chat CLI end-to-end** | **Not covered** |
| **PerformanceGPT generate with KV cache** | **Not covered** (only baseline GPT tested for MPS/CUDA generation) |
| **AdvancedGPT with RoPE + cache generation** | **Not covered** |
| **DDP multi-process** | **Not covered** (only single-process paths tested) |
| **EMA evaluation path** | **Not covered** (no test exercises `--ema`) |
| **`gpt_fast` checkpoint save/resume** | **Not covered** (integration test only uses baseline) |

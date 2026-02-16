# Test Suite Plan

This directory uses fast local tests first, then optional hardware-specific
tests.

## Layers

1. CPU baseline (always run)
- `test_models_cpu.py`: model forward/backward/generate sanity for all variants
- `test_kv_cache_cpu.py`: cache-vs-full-forward equivalence for all variants
- `test_tokenizer.py`: BOS/EOS encode/decode behavior (word + BPE if available)
- `test_data_cpu.py`: char vocab + windowing utility behavior
- `test_train_cli_cpu.py`: parser coverage for training defaults and new flags
- `test_optim.py`: weight-decay parameter-group assignment
- `test_train_integration_cpu.py`: short `train.py` integration coverage for
  save strategy, early stopping, resume/warm start, and metrics schema

2. Apple Silicon (MPS) optional
- `test_mps_smoke.py`: tiny forward/backward on MPS, skipped if unavailable

3. CUDA optional
- `test_cuda_smoke.py`: tiny forward/backward + AMP path, skipped if unavailable
- `test_train_cuda_flags.py`: short train smoke with CUDA-only flags (`--amp`,
  `--auto_microbatch`)

## Running

Run all local fast tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Run only CPU baseline:

```bash
python3 -m unittest tests.test_models_cpu tests.test_tokenizer tests.test_optim -v
```

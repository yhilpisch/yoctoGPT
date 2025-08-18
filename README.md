# picoGPT

Minimal GPT from scratch in PyTorch. Supports:

- Character-level training and sampling
- Token-level training with a BPE tokenizer (falls back to simple word-level if unavailable)
- A tiny REPL-style chat interface on top of the sampler

The default example corpus is `data/philosophy.txt`. Replace it with your own
text to experiment.

![TPQ Logo](https://hilpisch.com/tpq_logo.png)

Authors: The Python Quants with Codex and GPT-5

## Purpose and Approach

The aim of picoGPT is to provide a compact, readable, end-to-end GPT implementation for learning and lightweight experiments.

- Minimal PyTorch model: no external model frameworks; clear, well-commented code.
- Two training modes: character-level and token-level; BPE tokenizer by default (Hugging Face `tokenizers`) with a simple word-level fallback.
- Small-first design: friendly to CPU and Apple Silicon (MPS) with modest defaults for quick iteration.
- Simple data pipeline: raw `.txt` → `{train,val}.bin`; supports multiple texts via `--all_txt_in_dir`.
- CLIs included: data preparation, training, sampling, and a tiny chat REPL.
- Checkpointing: warm start (`--init_from`) and full resume (`--resume`); when resuming, `--max_iters` means additional steps and the progress bar reflects total steps.
- Educational, not production: defaults favor clarity over speed; easy to extend.

## Quickstart

1) Prepare data (char-level):

```
python -m scripts.prepare_char_data --text_path data/philosophy.txt --out_dir data/char
```

2) Train (char-level):

```
python -m picoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 64 --max_iters 2000
```

3) Sample (char-level):

```
python -m picoGPT.sampler --mode char --ckpt checkpoints/char/latest.pt --vocab_path data/char/vocab.json --prompt "What is wisdom?\n" --max_new_tokens 200
```

---

Token-level path (BPE by default):

1) Prepare data:

```
python -m scripts.prepare_tokenizer --text_path data/philosophy.txt --out_dir data/token --vocab_size 8000
```

If `tokenizers` (Hugging Face) is not installed, the script falls back to a simple word-level tokenizer. Force the backend if needed:

```
python -m scripts.prepare_tokenizer --text_path data/philosophy.txt --out_dir data/token --vocab_size 8000 --backend word
```

Use multiple texts by including all `.txt` files from a directory (non-recursive):

```
python -m scripts.prepare_tokenizer --all_txt_in_dir --text_dir data --out_dir data/token --vocab_size 8000
```

2) Train:

```
python -m picoGPT.train --mode token --data_dir data/token --tokenizer_path data/token/tokenizer.json --ckpt_dir checkpoints/token --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 --max_iters 5000
```

3) Sample:

```
python -m picoGPT.sampler --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --prompt "Q: What is knowledge?\nA:" --max_new_tokens 200
```

4) Chat:

```
python -m picoGPT.chat --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --system_prompt "You are picoGPT, a helpful assistant."

Resume or warm-start full training:

```
# Resume training from a saved checkpoint (restores optimizer).
# Note: --max_iters means additional steps to run. The progress bar shows
# total steps completed across all runs.
python -m picoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --resume checkpoints/char/latest.pt --max_iters 1000

# Warm start from weights only
python -m picoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --init_from checkpoints/char/best.pt

# If the vocab/head changed and you want to ignore mismatches
python -m picoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --init_from checkpoints/char/best.pt --no_strict_init
```
```

## Notes

- The implementation prioritizes readability and minimalism over speed.
- The tokenizer is intentionally simple; feel free to swap in a more advanced
  implementation later.
- Checkpoints store model state and enough metadata to reload the encoder.
- Prompts in smoke tests are normalized to characters present in the corpus to
  avoid unknown-character errors in char-level mode.

## CPU Smoke Test

Run a quick end-to-end generation on CPU with random weights to validate the stack:

```
python -m scripts.cpu_smoke_test --text_path data/philosophy.txt --prompt "What is virtue?\n"
```

This does not require training and simply verifies that model construction,
encoding, and autoregressive sampling work on CPU.

Run as a file (fixes sys.path automatically):

```
python scripts/cpu_smoke_test.py --text_path data/philosophy.txt
```

## Training Smoke Test

Run a short training loop on a tiny model to verify backprop and device setup (prefers Apple Silicon `mps`):

```
python scripts/train_smoke_test.py --text_path data/philosophy.txt --iters 200
```

Warm start or resume the smoke test:

```
# After first run saves to checkpoints/smoke/latest.pt
python scripts/train_smoke_test.py --resume checkpoints/smoke/latest.pt --iters 200

# Or warm start weights only
python scripts/train_smoke_test.py --init_from checkpoints/smoke/latest.pt --iters 200
```

## Apple Silicon (MPS)

Training defaults to Apple Silicon `mps` device when available. To override:

```
python -m picoGPT.train --device cpu ...
```

If you prefer CUDA on a multi-backend system, pass `--device cuda`.

## Disclaimer

This repository and its contents are provided solely for illustration and educational purposes. No guarantees or representations of any kind are given, express or implied, including but not limited to fitness for a particular purpose or non-infringement, to the extent permitted by law. Use at your own risk.

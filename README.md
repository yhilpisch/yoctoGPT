# yoctoGPT

Minimal GPT from scratch in PyTorch. Supports:

- Character-level training and sampling
- Token-level training with a BPE tokenizer (falls back to simple word-level if unavailable)
- A tiny REPL-style chat interface on top of the sampler
- Two model variants: baseline GPT and an accuracy-focused AdvancedGPT

The default example corpus is `data/philosophy.txt`. Replace it with your own
text to experiment.

<img src="https://hilpisch.com/tpq_logo.png" alt="TPQ Logo" width="350" />

Authors: The Python Quants with Codex and GPT-5

## Installation

Assuming you already created and activated a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`), install dependencies and verify the setup:

1) Upgrade core packaging tools

```
python -m pip install -U pip setuptools wheel
```

2) Install project requirements

```
pip install -r requirements.txt
```

Notes:
- BPE tokenizer: `tokenizers` is included and provides the default BPE backend.
- PyTorch: If your platform requires a specific wheel (CUDA vs. CPU, Apple Silicon/MPS), follow the official selector to install the right build, then re-run step (2) if needed:
  - https://pytorch.org/get-started/locally/

3) Optional: Apple Silicon verification (MPS)

```
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('mps available:', hasattr(torch.backends,'mps') and torch.backends.mps.is_available())
PY
```

4) Package sanity check

```
python - <<'PY'
import yoctoGPT
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig  # optional
print('yoctoGPT version:', getattr(yoctoGPT, '__version__', 'unknown'))
cfg = GPTConfig(vocab_size=100)
model = GPT(cfg)
print('model params:', sum(p.numel() for p in model.parameters()))
PY
```

## Purpose and Approach

The aim of yoctoGPT is to provide a compact, readable, end-to-end GPT implementation for learning and lightweight experiments.

- Minimal PyTorch model: no external model frameworks; clear, well-commented code.
- Two training modes: character-level and token-level; BPE tokenizer by default (Hugging Face `tokenizers`) with a simple word-level fallback.
- Small-first design: friendly to CPU and Apple Silicon (MPS) with modest defaults for quick iteration.
- Simple data pipeline: raw `.txt` → `{train,val}.bin`; supports multiple texts via `--all_txt_in_dir`.
- CLIs included: data preparation, training, sampling, and a tiny chat REPL.
- Checkpointing: warm start (`--init_from`) and full resume (`--resume`); when resuming, `--max_iters` means additional steps and the progress bar reflects total steps.
- Educational, not production: defaults favor clarity over speed; easy to extend.

### Model Variants

yoctoGPT now offers two selectable architectures:

- `gpt` (default): the original minimal GPT with learned positional embeddings and GELU MLP.
- `gpt_plus`: an accuracy‑focused variant with Rotary Positional Embeddings (RoPE), RMSNorm, biasless Linear layers, SwiGLU MLP, and stabilized initialization/residual scaling. Intended to improve validation loss on small/mid‑size setups while remaining compact and readable.

Select the variant at train time via `--model_type` and it will be recorded in the checkpoint. The chat utility auto‑detects it on load.

### Why AdvancedGPT

- RoPE: rotary positions improve inductive bias and longer‑context generalization; typically yields small but consistent validation‑loss/perplexity gains, especially when eval context approaches or exceeds train context.
- RMSNorm + biasless linears: stabilizes optimization and reduces gradient noise; often enables slightly higher learning rates and modest val‑loss improvements versus LayerNorm + biased linears.
- SwiGLU MLP: gated feed‑forward increases expressivity at similar FLOPs; commonly observed 1–3% relative perplexity improvements on small/mid models.

### Choosing a Variant

- Prefer `gpt` when: you want the smallest, most readable reference; are teaching/learning the basics; need maximum simplicity and compatibility with older checkpoints/scripts; or are running on very limited hardware and value minimalism over small quality gains.
- Prefer `gpt_plus` when: you care about lower validation loss at similar model sizes; want better behavior at longer contexts; or expect slightly more stable training. It uses the same CLI and data pipeline, but you should train new checkpoints for this architecture.
- Warm starting across variants: technically possible with `--no_strict_init` as long as dims/vocab match, but quality depends on overlap and is not guaranteed. For best results, train from scratch for the chosen variant.

## Quickstart

1) Prepare data (char-level):

```
python -m scripts.prepare_char_data --text_path data/philosophy.txt --out_dir data/char
```

2) Train (char-level):

```
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 64 --max_iters 2000
```

Use the accuracy‑focused variant:

```
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char_plus --model_type gpt_plus --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 64 --max_iters 2000
```

3) Sample (char-level):

```
python -m yoctoGPT.sampler --mode char --ckpt checkpoints/char/latest.pt --vocab_path data/char/vocab.json --prompt "What is wisdom?\n" --max_new_tokens 200
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

Randomize the train/val split (helps reduce distribution shift):

```
python -m scripts.prepare_tokenizer --all_txt_in_dir --text_dir data --out_dir data/token --vocab_size 8000 --random_split --split_seed 1337
```

2) Train:

```
python -m yoctoGPT.train --mode token --data_dir data/token --tokenizer_path data/token/tokenizer.json --ckpt_dir checkpoints/token --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 --max_iters 5000
```

Accuracy‑focused variant (token mode):

```
python -m yoctoGPT.train --mode token --data_dir data/token --tokenizer_path data/token/tokenizer.json --ckpt_dir checkpoints/token_plus --model_type gpt_plus --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 --max_iters 5000
```

3) Sample:

```
python -m yoctoGPT.sampler --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --prompt "Q: What is knowledge?\nA:" --max_new_tokens 200
```

4) Chat:

```
python -m yoctoGPT.chat --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --system_prompt "You are yoctoGPT, a helpful assistant."
```

Notes:
- Chat auto‑detects the architecture from the checkpoint (`arch` field). No extra flags are required whether the checkpoint is `gpt` or `gpt_plus`.

Resume or warm-start full training:

```
# Resume training from a saved checkpoint (restores optimizer).
# Note: --max_iters means additional steps to run. The progress bar shows
# total steps completed across all runs.
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --resume checkpoints/char/latest.pt --max_iters 1000

# Warm start from weights only
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --init_from checkpoints/char/best.pt

# If the vocab/head changed and you want to ignore mismatches
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char --init_from checkpoints/char/best.pt --no_strict_init
```

Architecture compatibility:

```
# When resuming, the checkpoint's architecture must match the requested one.
# For AdvancedGPT checkpoints, add --model_type gpt_plus when training.
python -m yoctoGPT.train --mode char --data_dir data/char --ckpt_dir checkpoints/char_plus --resume checkpoints/char_plus/latest.pt --model_type gpt_plus --max_iters 1000
```

## Notes

- The implementation prioritizes readability and minimalism over speed.
- Default tokenization uses a BPE tokenizer (via `tokenizers`); a simple word-level fallback is available.
- Checkpoints store model state and enough metadata to reload the encoder.
- Checkpoints also store the architecture under `arch` ("gpt" or "gpt_plus").
- Prompts in smoke tests are normalized to characters present in the corpus to
  avoid unknown-character errors in char-level mode.

## Avoiding Overfitting

Overfitting appears when training loss continues to fall while validation loss rises. Common causes: model too large for the dataset, too little regularization, aggressive learning rate, or distribution shift between train and validation. Practical strategies:

- Regularize more:
  - Dropout: increase `--dropout` from 0.0 up to 0.1–0.3. Typical starting point: `--dropout 0.1`.
  - Weight decay: set `--weight_decay 0.01`–`0.1`. A balanced default is `--weight_decay 0.05`.
  - Label smoothing: set `--label_smoothing 0.05`–`0.1` to soften targets and improve generalization.
  - Weight tying: add `--tie_weights` to share the token embedding and LM head weights (fewer params; better generalization on small data).
  - Auto weight tying: add `--auto_tie_weights` to enable tying automatically for small datasets (< 1M tokens).

- Reduce capacity:
  - Use smaller dims/layers: e.g., `--n_layer 4 --n_head 4 --n_embd 128`.
  - Keep `block_size` to 128–256 for small corpora to limit compute and stabilize training.

- Train gentler and monitor:
  - Lower LR (e.g., `--lr 1e-4` or `2e-4`).
  - Use cosine LR with warmup: `--cosine_lr --warmup_iters 100 --min_lr 1e-5`.
  - Evaluate more frequently and with enough samples: `--eval_interval 100 --eval_iters 100–200`.
  - Early stop manually: rely on `best.pt` and stop when val no longer improves.

- Improve validation signal & data:
  - Use more data (`--all_txt_in_dir`) to broaden coverage and reduce variance.
  - Use `--random_split` in tokenization prep to reduce distribution drift between train and val.
  - Consider a randomized validation split (not yet built in) if your corpus is ordered by topic; contiguous splits can exaggerate distribution shift.

- Tokenization choices:
  - Prefer BPE (default) with a reasonable `--vocab_size` (e.g., 4k–16k for small–medium corpora).
  - If overfitting persists, you can reduce vocab to shrink embedding/head params (trade-off: more `<unk>` or longer subword sequences).

Notes:
- Dropout and label smoothing apply only during training; sampling runs with `model.eval()`.
- Changing architecture or vocab requires a fresh run (new `--ckpt_dir`). You may warm start with `--init_from ... --no_strict_init` but results depend on compatibility.

## M1 (8 GB) Best‑Practice Configs

Suggested configurations for Apple Silicon M1 with 8 GB RAM (using MPS). Adjust `batch_size` downwards if you hit OOM; prefer reducing batch size before shrinking `block_size`.

- Tiny (fastest):
  - `--n_layer 4 --n_head 4 --n_embd 128 --block_size 128 --batch_size 32 --dropout 0.1 --weight_decay 0.05 --lr 2e-4 --eval_interval 100 --eval_iters 100`
- Small (balanced):
  - `--n_layer 6 --n_head 8 --n_embd 256 --block_size 256 --batch_size 32 --dropout 0.1 --weight_decay 0.05 --lr 2e-4 --eval_interval 100 --eval_iters 200`
- Regularized small (overfitting observed):
  - add `--dropout 0.2 --weight_decay 0.1 --label_smoothing 0.1 --tie_weights` and consider `--cosine_lr --warmup_iters 100 --min_lr 1e-5`

Tokens/step ≈ `batch_size × block_size`. For ~0.4M tokens, 20× coverage is ≈ 500 steps at 8k tokens/step. Use `--max_iters` accordingly and rely on `best.pt` for early stopping by hand.

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

## Metrics Logging

- Training (`yoctoGPT.train`) writes a CSV log to the checkpoint directory as `metrics.csv` with columns:
  - `iter, train_loss, val_loss, lr, time_sec, tokens_seen, throughput_tps, grad_norm`.
  - A `run_meta.json` is written once with training/model configs, device, parameter count, and tokens per step.
- The training smoke test also writes `metrics.csv` (without validation columns) in the same folder as its `--ckpt` path.

## Apple Silicon (MPS)

Training defaults to Apple Silicon `mps` device when available. To override:

```
python -m yoctoGPT.train --device cpu ...
```

If you prefer CUDA on a multi-backend system, pass `--device cuda`.

## Disclaimer

This repository and its contents are provided solely for illustration and educational purposes. No guarantees or representations of any kind are given, express or implied, including but not limited to fitness for a particular purpose or non-infringement, to the extent permitted by law. Use at your own risk.

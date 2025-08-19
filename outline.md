# yoctoGPT — Minimal GPT from scratch (PyTorch)

This document outlines a minimal, end-to-end project structure to:
- Implement a small GPT model from scratch in PyTorch
- Train it on either character-level or token-level data
- Sample/generate text from trained checkpoints
- Offer a simple terminal chat interface built on the sampler

The emphasis is on minimal, readable code with clear seams to extend later.


## Directory Layout

```
yoctogpt/
├─ yoctoGPT/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ tokenizer.py
│  ├─ model.py
│  ├─ train.py
│  ├─ sampler.py
│  └─ chat.py
├─ scripts/
│  ├─ prepare_char_data.py
│  └─ prepare_tokenizer.py
├─ data/
│  └─ philosophy.txt
├─ README.md
├─ outline.md  ← this file
├─ requirements.txt (or pyproject.toml)
└─ .gitignore
```


## High-Level Flow

- Char-level path:
  1) `scripts/prepare_char_data.py` builds char vocab from raw text and writes encoded `train.bin` and `val.bin`.
  2) `yoctoGPT/train.py --mode char` trains GPT using `CharDataset` from `data.py`.
  3) `yoctoGPT/sampler.py --mode char` loads checkpoint and generates text.

- Token-level path:
  1) `scripts/prepare_tokenizer.py` trains/loads a tokenizer and encodes raw text into token IDs.
  2) `yoctoGPT/train.py --mode token` trains GPT using `TokenDataset` from `data.py`.
  3) `yoctoGPT/sampler.py --mode token` loads checkpoint, tokenizer, and generates text.

- Chat interface:
  - `yoctoGPT/chat.py` wraps the sampler with a simple REPL, preserving context and applying generation settings.


## File-by-File Details

### `yoctoGPT/__init__.py`
- Purpose: Mark the package; optionally expose convenience imports.
- Minimal content: set `__version__`, re-export `GPT`, `GPTConfig`.


### `yoctoGPT/config.py`
- Purpose: Centralize configuration via small dataclasses to keep scripts clean.
- Contents:
  - `ModelConfig`: `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`.
  - `TrainConfig`: dataset paths, `batch_size`, `max_iters`, `lr`, `weight_decay`, `grad_clip`, `eval_interval`, `eval_iters`, `ckpt_dir`, `seed`, `device`.
  - `DataConfig`: for char or token mode; file paths and tokenizer paths.
  - Simple `from_cli()` helper to parse `argparse` flags into configs.


### `yoctoGPT/data.py`
- Purpose: Datasets and simple utilities for producing model inputs.
- Contents:
  - `CharVocab`: build char-to-id and id-to-char from raw text; `encode`, `decode`.
  - `CharDataset`: memory-mapped or tensor-backed dataset over `train.bin` / `val.bin` returning `(x, y)` where `y` is `x` shifted by 1.
  - `TokenDataset`: same as `CharDataset` but uses token IDs produced by a tokenizer; expects `train.bin` / `val.bin` of token IDs.
  - `load_bin(path)`: returns a `torch.LongTensor` of IDs.
  - Splitting & batching assumptions: fixed `block_size`, random index sampling to produce subsequences.


### `yoctoGPT/tokenizer.py`
- Purpose: Provide a minimal tokenization interface for the token-level path.
- Contents:
  - `BaseTokenizer` protocol: `encode(str) -> List[int]`, `decode(List[int]) -> str`, `vocab_size`.
  - `SimpleBPETokenizer` (minimal BPE or WordLevel fallback) with:
    - `train(text, vocab_size)` and `save(path)` / `load(path)`.
    - For true minimalism, start with a whitespace/punctuation-splitting `WordLevelTokenizer` and leave BPE as a later enhancement.
  - Optional adapter to Hugging Face `tokenizers` or `tiktoken` if available; otherwise use the simple built-in.


### `yoctoGPT/model.py`
- Purpose: Implement a small GPT in PyTorch, no external model libs.
- Contents:
  - `GPTConfig` (could alias `ModelConfig`).
  - Modules: `CausalSelfAttention`, `MLP`, `Block`, `GPT`.
  - `GPT.forward(idx)`:
    - `idx`: shape `(B, T)` with token IDs.
    - Embeddings: token + positional.
    - Stack `n_layer` Transformer blocks; each block is LayerNorm → Attention → residual, then LayerNorm → MLP → residual.
    - Final LayerNorm and linear head to logits of shape `(B, T, vocab_size)`.
  - `GPT.generate(idx, max_new_tokens, temperature, top_k, top_p, eos_token=None)` for autoregressive sampling.
  - Masking: use a fixed causal mask or upper-triangular attention mask.


### `yoctoGPT/train.py`
- Purpose: Single training entry point for both char-level and token-level modes.
- Key CLI flags:
  - `--mode {char,token}`
  - `--data_dir`, `--tokenizer_path` (token mode), `--text_path` (char mode setup), `--ckpt_dir`
  - Model/training hyperparams override flags (e.g., `--n_layer`, `--n_head`, `--n_embd`, `--block_size`, `--batch_size`, `--max_iters`, `--lr`)
- Contents:
  - Seeding and device setup (CPU/CUDA/MPS autodetect).
  - Data loading:
    - Char mode: load `CharVocab`, read encoded `.bin` files.
    - Token mode: load tokenizer, read encoded `.bin` files.
  - Model init:
    - Infer `vocab_size` from `CharVocab` or tokenizer.
    - Create `GPT(config)` and move to device.
  - Optimizer: `torch.optim.AdamW`.
  - Optional cosine LR schedule; gradient clipping.
  - Training loop:
    - Sample batches by random start indices from the 1D ID tensor with `block_size` windows.
    - Compute logits, cross-entropy loss, backward, step.
    - Periodic eval on validation split.
    - Save checkpoints: model weights, optimizer state, config, vocab/tokenizer artifacts.


### `yoctoGPT/sampler.py`
- Purpose: Load a checkpoint + tokenizer/vocab and generate text.
- Key CLI flags:
  - `--mode {char,token}`, `--ckpt`, `--tokenizer_path` (token), `--vocab_path` (char)
  - `--prompt`, `--max_new_tokens`, `--temperature`, `--top_k`, `--top_p`, `--seed`
- Contents:
  - Load config and weights onto CPU/GPU.
  - Encode prompt with the correct path (char or token), call `model.generate`, decode and print.


### `yoctoGPT/chat.py`
- Purpose: Minimal terminal chat on top of the sampler.
- Contents:
  - REPL loop that maintains a running prompt/context window truncated to `block_size`.
  - Input prefix formatting, e.g., `User: ...` / `Assistant: ...` to guide style.
  - Stream tokens or print full assistant turn; exit on `/exit`.
  - CLI flags mirror `sampler.py` plus `--system_prompt` and `--max_ctx_tokens` for context management.


### `scripts/prepare_char_data.py`
- Purpose: Turn raw text into char-level training binaries.
- Contents:
  - Read a text file (default: `data/philosophy.txt`).
  - Build `CharVocab` from corpus chars.
  - Encode entire text to IDs; split into train/val; save to `{train,val}.bin` via `numpy.memmap` or `torch.save`.
  - Save `vocab.json` with `stoi`/`itos`.
- CLI flags: `--text_path`, `--out_dir`, `--val_ratio`.


### `scripts/prepare_tokenizer.py`
- Purpose: Train or load a tokenizer and encode raw text into IDs for token-level training.
- Contents:
  - Train a simple tokenizer (initially word-level). Save `tokenizer.json`.
  - Encode corpus, split to `{train,val}.bin`.
  - Print resulting `vocab_size`.
- CLI flags: `--text_path`, `--out_dir`, `--vocab_size`, `--val_ratio`.


### `data/philosophy.txt`
- Purpose: Default corpus for quick experiments (replace with your text as needed).


### `requirements.txt` (or `pyproject.toml`)
- Minimal dependencies:
  - `torch`, `tqdm`, `numpy`
  - Optional for better tokenization: `tokenizers` or `tiktoken`


### `README.md`
- Quickstart for both char and token paths.
- Examples of commands, checkpoints, and sampling.


## Minimal Code Sketches (for alignment)

### `model.py` (sketch)
```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init); self.mask = None

    def forward(self, idx):
        B, T = idx.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError("Sequence length > block_size")
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-8)
            probs = top_k_top_p_filtering(logits, top_k, top_p).softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token is not None and next_id.item() == eos_token:
                break
        return idx
```


### Batch sampling in `train.py` (sketch)
```python
def get_batch(data_ids: torch.LongTensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data_ids) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in ix])
    y = torch.stack([data_ids[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)
```


## Example Commands

- Char-level preparation:
  - `python -m scripts.prepare_char_data --text_path data/philosophy.txt --out_dir data/char`

- Char-level training:
  - `python -m yoctoGPT.train --mode char --data_dir data/char --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 64 --max_iters 5000`

- Char-level sampling:
  - `python -m yoctoGPT.sampler --mode char --ckpt checkpoints/char/latest.pt --vocab_path data/char/vocab.json --prompt "To be, or not to be" --max_new_tokens 200`

- Token-level preparation:
  - `python -m scripts.prepare_tokenizer --text_path data/philosophy.txt --out_dir data/token --vocab_size 8000`

- Token-level training:
  - `python -m yoctoGPT.train --mode token --data_dir data/token --tokenizer_path data/token/tokenizer.json --n_layer 8 --n_head 8 --n_embd 512 --block_size 512 --batch_size 32 --max_iters 20000`

- Token-level sampling:
  - `python -m yoctoGPT.sampler --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --prompt "Q: What is love?\nA:" --max_new_tokens 200`

- Chat interface:
  - `python -m yoctoGPT.chat --mode token --ckpt checkpoints/token/latest.pt --tokenizer_path data/token/tokenizer.json --system_prompt "You are yoctoGPT, a helpful assistant."`


## Checkpoints and Artifacts

- `checkpoints/<run_name>/model.pt`: `state_dict` and `ModelConfig`.
- `checkpoints/<run_name>/opt.pt`: optimizer state (optional).
- Char mode extras: `vocab.json` (`stoi`, `itos`).
- Token mode extras: `tokenizer.json` and optionally merges/vocab files.


## Design Notes

- Keep the model intentionally small to enable CPU or single-GPU runs.
- Prefer simple, explicit code over heavy abstractions.
- Data is stored as contiguous 1D tensors of IDs for fast random slicing.
- Both modes share the same model; only vocab/encoding differs.
- The chat REPL simply appends user/assistant turns and re-generates with truncation to fit `block_size`.


## Possible Extensions (later)

- FP16/AMP training; gradient checkpointing for deeper models.
- Better tokenization (real BPE or `tiktoken`), special tokens, BOS/EOS handling.
- Save/load full training runs; TensorBoard or WandB logging.
- Multi-GPU or FSDP; dataset streaming; instruction-tuning format.
- Unit tests for tokenizer, dataset slicing, and generation.
```

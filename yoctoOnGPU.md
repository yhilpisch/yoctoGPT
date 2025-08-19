# yoctoGPT on GPU (Ubuntu + NVIDIA H100)

This guide shows how to install dependencies and train yoctoGPT on a DigitalOcean GPU droplet with an NVIDIA H100 (80 GB VRAM). The droplet is assumed to already have the correct NVIDIA drivers and CUDA runtime.

- GPU: NVIDIA H100, 80 GB VRAM
- CPU: 20 vCPU
- RAM: 240 GB
- Disks: 720 GB NVMe (boot), 5 TB NVMe (scratch)

## 1) Setup

Assuming you cloned this repo and are in its root directory:

```
# Update packages, create a venv, install CUDA-enabled PyTorch and project deps
bash scripts/setup_gpu_ubuntu.sh

# Activate the environment for this session
source .venv/bin/activate
```

Verify CUDA is visible to PyTorch:

```
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```

## 2) Data Placement (use the scratch disk)

For large corpora, place data and checkpoints on the scratch NVMe for speed and space, e.g. `/mnt/scratch`:

```
# Example layout (adjust paths for your mount point)
mkdir -p /mnt/scratch/yocto/data /mnt/scratch/yocto/checkpoints
```

## 3) Tokenization (BPE by default)

Prepare tokens from a single file or all `.txt` files in a directory.

- Single file:
```
python -m scripts.prepare_tokenizer \
  --text_path /mnt/scratch/yocto/data/corpus.txt \
  --out_dir   /mnt/scratch/yocto/data/token \
  --vocab_size 32000 \
  --random_split --split_seed 1337
```

- Multiple files (non-recursive) with randomized split:
```
python -m scripts.prepare_tokenizer \
  --all_txt_in_dir --text_dir /mnt/scratch/yocto/data/texts \
  --out_dir   /mnt/scratch/yocto/data/token \
  --vocab_size 32000 \
  --random_split --split_seed 1337
```

Notes:
- BPE (`tokenizers`) is default; falls back to word-level if unavailable.
- `--random_split` reduces distribution drift between train/val.
- Choose `--vocab_size` based on corpus size (e.g., 16k–50k for larger corpora).

## 4) Recommended Training Profiles (H100 80 GB)

Below are conservative starting points without AMP. Increase `batch_size` as memory allows; if you hit OOM, reduce it first, then consider reducing `block_size`.

- Base (balanced):
```
python -m yoctoGPT.train \
  --mode token \
  --data_dir /mnt/scratch/yocto/data/token \
  --tokenizer_path /mnt/scratch/yocto/data/token/tokenizer.json \
  --ckpt_dir /mnt/scratch/yocto/checkpoints/base \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --block_size 1024 --batch_size 32 \
  --dropout 0.1 --weight_decay 0.05 \
  --label_smoothing 0.1 --tie_weights \
  --cosine_lr --warmup_iters 200 --min_lr 1e-5 --lr 2e-4 \
  --eval_interval 200 --eval_iters 200 --max_iters 20000
```

- Large:
```
python -m yoctoGPT.train \
  --mode token \
  --data_dir /mnt/scratch/yocto/data/token \
  --tokenizer_path /mnt/scratch/yocto/data/token/tokenizer.json \
  --ckpt_dir /mnt/scratch/yocto/checkpoints/large \
  --n_layer 24 --n_head 16 --n_embd 1024 \
  --block_size 1024 --batch_size 16 \
  --dropout 0.1 --weight_decay 0.1 \
  --label_smoothing 0.1 --tie_weights \
  --cosine_lr --warmup_iters 1000 --min_lr 1e-5 --lr 2e-4 \
  --eval_interval 200 --eval_iters 200 --max_iters 50000
```

- XL (push further if memory allows):
```
python -m yoctoGPT.train \
  --mode token \
  --data_dir /mnt/scratch/yocto/data/token \
  --tokenizer_path /mnt/scratch/yocto/data/token/tokenizer.json \
  --ckpt_dir /mnt/scratch/yocto/checkpoints/xl \
  --n_layer 24 --n_head 20 --n_embd 1280 \
  --block_size 1024 --batch_size 8 \
  --dropout 0.1 --weight_decay 0.1 \
  --label_smoothing 0.1 --tie_weights \
  --cosine_lr --warmup_iters 2000 --min_lr 1e-5 --lr 2e-4 \
  --eval_interval 200 --eval_iters 200 --max_iters 100000
```

Notes:
- If you require longer context (e.g., `--block_size 2048`), expect higher memory usage; reduce `batch_size` accordingly.
- Mixed precision (AMP/bfloat16) is not yet wired into yoctoGPT. Adding AMP would roughly halve memory and speed up training on H100; consider it as a next enhancement.
- Use `--resume` to add more steps later; `--max_iters` counts additional steps when resuming.

## 5) Monitoring

- GPU: `watch -n 1 nvidia-smi`
- Disk IO: `iostat -x 1`, `iotop`
- CPU/mem: `htop`

## 6) Sampling and Chat

- Sampling:
```
python -m yoctoGPT.sampler \
  --mode token \
  --ckpt /mnt/scratch/yocto/checkpoints/base/latest.pt \
  --tokenizer_path /mnt/scratch/yocto/data/token/tokenizer.json \
  --prompt "The philosophy of science begins with" \
  --max_new_tokens 200 --temperature 0.9 --top_p 0.95
```

- Chat (toy REPL):
```
python -m yoctoGPT.chat \
  --mode token \
  --ckpt /mnt/scratch/yocto/checkpoints/base/latest.pt \
  --tokenizer_path /mnt/scratch/yocto/data/token/tokenizer.json \
  --system_prompt "You are yoctoGPT, a concise helpful assistant."
```

## 7) Storage Tips

- Keep checkpoints on the scratch disk to avoid filling the boot disk.
- Periodically snapshot or rsync checkpoints to persistent storage.
- Use distinct `--ckpt_dir` names per run to avoid overwriting.

## 8) Troubleshooting

- CUDA not visible: ensure the droplet image includes NVIDIA drivers and reboot if needed. `nvidia-smi` should work.
- OOM: lower `--batch_size` first; then consider reducing `--block_size`, `--n_embd`, or `--n_layer`.
- Overfitting: increase `--dropout`, `--weight_decay`, enable `--label_smoothing`, and consider `--tie_weights`. See README “Avoiding Overfitting”.


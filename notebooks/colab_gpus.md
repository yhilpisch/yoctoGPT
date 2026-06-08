# Google Colab GPU Reference

GPU options available on Google Colab as of June 2026.

## Overview

| GPU | Architecture | VRAM | bf16 | FP8 | Colab Tier | Approx. \$/hr |
|-----|-------------|------|------|-----|-----------|---------------|
| **T4** | Turing | 15 GB | No | No | Free | \$0.12 |
| **L4** | Ada Lovelace | 22.5 GB | Yes | No | Pro | \$0.17 |
| **A100** | Ampere | 40 / 80 GB | Yes | No | Pro+ | \$0.54–0.75 |
| **G4** (RTX PRO 6000) | Blackwell | 96 GB | Yes | Yes | Pro+ | \$0.87 |
| **H100** | Hopper | 80 GB | Yes | Yes | Pro+ | ~\$1.50+ |

## Key Differences for Training

### bf16 vs fp16

The T4 lacks hardware bf16 support and must use fp16 for mixed-precision
training. bf16 has a wider dynamic range (same exponent bits as fp32) which
makes training more stable, especially for deeper models. All other Colab GPUs
support bf16 natively.

### VRAM and Model Scaling

Larger VRAM allows bigger batch sizes, wider models (n_embd), deeper models
(n_layer), and longer context windows (block_size) — all of which improve
training quality and speed. The `gpu_configs.yml` file alongside this document
maps each GPU to a set of training parameters that comfortably fit its VRAM.

### FlashAttention

PyTorch's `scaled_dot_product_attention` (used by `gpt_fast`) automatically
selects the best available backend — FlashAttention 2 on Ampere+ and SDPA
on Turing. The H100 additionally supports FlashAttention 3.

### G4 — NVIDIA RTX PRO 6000 (Blackwell)

The "G4" option in Colab's GPU menu maps to the NVIDIA RTX PRO 6000 on the
Blackwell architecture. It provides 96 GB of GDDR7 ECC VRAM with ~24,000
CUDA cores and native bf16 + fp8 support. At the time of writing it is the
most powerful GPU available on Colab.

## Disclaimer

**Pricing, availability, and GPU options change frequently.** The figures above
are approximate snapshots and may not reflect current rates or your region.
Always verify directly in your Colab session via the "View resources" panel or
by running `nvidia-smi`. The authors assume no responsibility for incurred
costs.

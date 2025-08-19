#!/usr/bin/env bash
set -euo pipefail

# Setup script for training yoctoGPT on an Ubuntu GPU server (e.g., DO droplet)
# Assumptions:
# - NVIDIA drivers and CUDA runtime are already installed by the provider
# - You want a local Python venv under the project directory
# - You want a CUDA-enabled PyTorch wheel (cu121 at the time of writing)

echo "[yoctoGPT] Updating apt packages and installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git build-essential

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[yoctoGPT] Warning: nvidia-smi not found. Ensure NVIDIA drivers are installed."
else
  echo "[yoctoGPT] NVIDIA GPU detected:" && nvidia-smi || true
fi

echo "[yoctoGPT] Creating Python venv (.venv) ..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

echo "[yoctoGPT] Installing CUDA-enabled PyTorch (cu121) ..."
# Adjust the URL for newer CUDA versions if needed: https://pytorch.org/get-started/locally/
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "[yoctoGPT] Installing project dependencies (excluding torch to keep the CUDA wheel) ..."
TMP_REQ=$(mktemp)
grep -v '^torch\b' requirements.txt > "$TMP_REQ" || true
pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"

echo "[yoctoGPT] Verifying CUDA availability in PyTorch ..."
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY

echo "[yoctoGPT] Setup complete. Activate the venv with: source .venv/bin/activate"
echo "[yoctoGPT] Next: prepare data and start training (see yoctoOnGPU.md)"


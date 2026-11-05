#!/usr/bin/env bash
# Create/refresh the conda env, install the CUDA-matched PyTorch wheel,
# install this package in editable mode and run the environment check.
#
# Usage:
#   bash setup_env.sh                 # auto-detect CUDA tag
#   TORCH_CUDA=cu128 bash setup_env.sh  # force a wheel tag (cu130, cu129, cu128, cpu)
#   ENV_NAME=myenv bash setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-attackdiar}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else
  echo "[setup] conda/mamba not found. Install Miniforge first:"
  echo "  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  echo "  bash Miniforge3-Linux-x86_64.sh -b -p \$HOME/miniforge3 && source \$HOME/miniforge3/bin/activate"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] updating env $ENV_NAME"
  $CONDA env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "[setup] creating env $ENV_NAME"
  $CONDA env create -n "$ENV_NAME" -f environment.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- choose the PyTorch wheel ------------------------------------------------
if [[ -z "${TORCH_CUDA:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1 || true)"
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true)"
    echo "[setup] driver=${DRIVER_VER:-?} cuda=${CUDA_VER:-?}"
    MAJOR="${CUDA_VER%%.*}"; MINOR="${CUDA_VER#*.}"
    if [[ -n "$CUDA_VER" && "$MAJOR" -ge 13 ]]; then TORCH_CUDA=cu130
    elif [[ -n "$CUDA_VER" && "$MAJOR" -eq 12 && "$MINOR" -ge 8 ]]; then TORCH_CUDA=cu128
    else
      echo "[setup] WARNING: CUDA ${CUDA_VER:-unknown} < 12.8. Blackwell (B300) needs >= 12.8."
      echo "[setup] Falling back to cu128; check_env.py will tell you if the driver is too old."
      TORCH_CUDA=cu128
    fi
  else
    echo "[setup] nvidia-smi not found -> CPU wheel"
    TORCH_CUDA=cpu
  fi
fi
echo "[setup] installing torch wheel: $TORCH_CUDA"
pip install --upgrade "torch>=2.8" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
  || { echo "[setup] torch $TORCH_CUDA failed, trying cu128"; pip install --upgrade "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128; }

pip install -e .
python scripts/check_env.py
echo "[setup] done. Activate with: conda activate $ENV_NAME"

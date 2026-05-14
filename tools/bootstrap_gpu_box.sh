#!/usr/bin/env bash
# bootstrap_gpu_box.sh — fresh Ubuntu 22.04 GPU box → ready-to-bench in ~10 min.
#
# Assumes:
#   - NVIDIA driver already installed (true on every cloud GPU image).
#   - You can sudo without password OR are root.
#
# Usage (on the GPU box):
#   curl -fsSL https://raw.githubusercontent.com/USERNAME/gpu-kernels-lab/main/tools/bootstrap_gpu_box.sh | bash
#
# Or copy this file to the box and:
#   bash bootstrap_gpu_box.sh

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-12-8}"             # use 12-6 for H200-only, 12-4 for A100-only
CONDA_ENV="${CONDA_ENV:-kernels-gpu}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

bold() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

bold "1. apt packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential ninja-build cmake git tmux htop curl wget \
    pkg-config zlib1g-dev libssl-dev ca-certificates gnupg

bold "2. CUDA toolkit ${CUDA_VERSION//-/.}"
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q "release ${CUDA_VERSION//-/.}"; then
    UBU=$(. /etc/os-release && echo "ubuntu${VERSION_ID//./}")
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_VERSION}"
    echo "export PATH=/usr/local/cuda-${CUDA_VERSION//-/.}/bin:\$PATH"          >>~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION//-/.}/lib64:\$LD_LIBRARY_PATH" >>~/.bashrc
    export PATH=/usr/local/cuda-${CUDA_VERSION//-/.}/bin:$PATH
fi

bold "3. Nsight Compute + Nsight Systems"
sudo apt-get install -y --no-install-recommends \
    nsight-compute-2024.3.2 nsight-systems-2024.4.1 || \
    echo "  (skip if already installed via cuda-toolkit metapackage)"

bold "4. Miniforge (conda)"
if [ ! -d "$HOME/miniforge3" ]; then
    ARCH=$(uname -m)
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    "$HOME/miniforge3/bin/conda" init bash
    rm /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"

bold "5. Python env: ${CONDA_ENV}"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    conda create -n "$CONDA_ENV" python=3.11 -y
fi
conda activate "$CONDA_ENV"

bold "6. PyTorch (matched to CUDA ${CUDA_VERSION//-/.})"
pip install --upgrade pip
pip install --index-url "$PYTORCH_INDEX" torch

bold "7. Python deps from requirements-gpu.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/requirements-gpu.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements-gpu.txt"
elif [ -f "./requirements.txt" ]; then
    pip install -r ./requirements.txt
else
    echo "  (no requirements file found; install ad-hoc later)"
fi

bold "8. Reference repos in ~/refs"
mkdir -p "$HOME/refs" && cd "$HOME/refs"
for repo in \
    NVIDIA/cutlass \
    flashinfer-ai/flashinfer \
    linkedin/Liger-Kernel \
    deepseek-ai/DeepGEMM \
    karpathy/llm.c \
    vllm-project/vllm \
    NVIDIA/cccl \
    srush/GPU-Puzzles \
    srush/Triton-Puzzles \
    siboehm/SGEMM_CUDA \
; do
    name=${repo##*/}
    [ -d "$name" ] || git clone --depth=1 "https://github.com/${repo}.git" "$name"
done
cd - >/dev/null

bold "9. Sanity check"
if [ -x "$SCRIPT_DIR/sanity_check.sh" ]; then
    "$SCRIPT_DIR/sanity_check.sh"
else
    echo "(sanity_check.sh not found at $SCRIPT_DIR; run it manually after copying)"
fi

bold "Done. Activate with:  conda activate ${CONDA_ENV}"

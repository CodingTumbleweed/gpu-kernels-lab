#!/usr/bin/env bash
# sanity_check.sh — verify the GPU box toolchain after install or driver upgrade.
# Paste output into the "My verified setups" table in Exercises.md.
#
# Usage:  ./sanity_check.sh [SM_ARCH]
#   SM_ARCH defaults to detecting from nvidia-smi; override with sm_80 / sm_90a / sm_103a.

set -uo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

bold "==> nvidia-smi (driver + GPU)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  fail "nvidia-smi not found — install the NVIDIA driver"
  exit 1
fi

bold "==> CUDA driver runtime"
nvidia-smi | grep -i "CUDA Version" || fail "Could not parse driver CUDA version"

bold "==> nvcc (CUDA toolkit)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n1
  NVCC_MAJOR=$(nvcc --version | grep -oE 'release [0-9]+' | awk '{print $2}')
  NVCC_MINOR=$(nvcc --version | grep -oE 'V[0-9]+\.[0-9]+' | head -1 | awk -F. '{print $2}')
  ok "nvcc ${NVCC_MAJOR}.${NVCC_MINOR}"
else
  fail "nvcc not found — install CUDA toolkit (apt install cuda-toolkit-12-8)"
fi

bold "==> PyTorch + CUDA binding"
python - <<'PY'
import sys
try:
    import torch
    print(f"  torch {torch.__version__}  built-with-cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    else:
        print("  CUDA not visible to torch — wrong wheel? (try: pip install torch --index-url https://download.pytorch.org/whl/cu128)")
        sys.exit(2)
except ImportError:
    print("  torch not installed — pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(2)
PY

bold "==> Triton"
python - <<'PY'
try:
    import triton
    print(f"  triton {triton.__version__}")
except ImportError:
    print("  triton not installed (ships with PyTorch 2.x; reinstall PyTorch)")
PY

bold "==> Nsight Compute / Systems"
command -v ncu  >/dev/null && ncu  --version | head -n1 || fail "ncu  not found (apt install nsight-compute-2024.3.2)"
command -v nsys >/dev/null && nsys --version | head -n1 || fail "nsys not found (apt install nsight-systems-2024.4.1)"

bold "==> Smoke compile (sm-arch detection)"
SM_ARCH="${1:-}"
if [ -z "$SM_ARCH" ]; then
  CC=$(python -c "import torch; m,n = torch.cuda.get_device_capability(0); print(f'{m}{n}')" 2>/dev/null)
  if [ -n "$CC" ]; then
    case "$CC" in
      80) SM_ARCH=sm_80 ;;
      89) SM_ARCH=sm_89 ;;
      90) SM_ARCH=sm_90a ;;          # Hopper — note the 'a'
      100|103) SM_ARCH=sm_103a ;;    # Blackwell DC
      *)  SM_ARCH=sm_${CC} ;;
    esac
  else
    SM_ARCH=sm_80
  fi
fi
echo "  Compiling smoke kernel for $SM_ARCH"
TMP=$(mktemp -d)
cat >"$TMP/smoke.cu" <<'CU'
#include <cstdio>
__global__ void k() { if (threadIdx.x==0 && blockIdx.x==0) printf("smoke ok\n"); }
int main() {
    k<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
CU
if nvcc -O2 -arch="$SM_ARCH" "$TMP/smoke.cu" -o "$TMP/smoke" 2>"$TMP/err"; then
  "$TMP/smoke" && ok "compiled and ran for $SM_ARCH"
else
  fail "nvcc -arch=$SM_ARCH failed — see $TMP/err"
  cat "$TMP/err"
fi
rm -rf "$TMP"

bold "==> Done"
echo "Paste the table rows above into the 'My verified setups' table in Exercises.md."

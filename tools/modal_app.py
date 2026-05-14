"""Modal template — dispatch a CUDA / Triton job from the Mac to a remote GPU.

Local prereqs::

    pip install modal
    modal setup                 # one-time browser auth

Run::

    modal run modal_template.py                       # default: H200 sgemm smoke
    modal run modal_template.py::run --gpu A100        # change GPU
    modal run modal_template.py::triton_softmax        # different entrypoint

Switch GPU types by editing the ``gpu=`` argument on each ``@app.function``.
Valid as of 2026: "A100", "A100-40GB", "H100", "H200", "B200", "L40S", "T4".
B300 is not yet on Modal — use OEM cloud (Lambda / TensorWave / Vultr).
"""

from __future__ import annotations

import modal

# Pin CUDA 12.8 — works for sm_80 (A100), sm_90a (H200), sm_103a (B300 not on Modal).
CUDA_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "cmake")
    .pip_install(
        # PyTorch + Triton (Triton ships with PyTorch 2.x)
        "torch",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy", "pandas", "matplotlib", "tabulate", "rich", "pytest",
    )
    # Mount the lab repo at /root/lab — change path to your local checkout.
    # .add_local_dir("./gpu-kernels-lab", remote_path="/root/lab")
)

app = modal.App("kernel-bench", image=CUDA_IMAGE)


# ---------------------------------------------------------------------------
# Entrypoint 1: compile and run a CUDA file on the remote GPU.
# ---------------------------------------------------------------------------
@app.function(gpu="H200", timeout=600)
def run_cuda(arch: str = "sm_90a", source: str = "smoke") -> str:
    """Compile and execute the named CUDA source from /root/lab.

    Drop the source file into ``gpu-kernels-lab/<exercise>/`` and uncomment
    the ``add_local_dir`` line above so it's mounted.
    """
    import subprocess
    import textwrap
    from pathlib import Path

    # Inline smoke kernel so the template runs out-of-the-box.
    if source == "smoke":
        src = textwrap.dedent(
            """
            #include <cstdio>
            __global__ void k() { if (threadIdx.x==0) printf("hello from %s\\n", "GPU"); }
            int main() { k<<<1,32>>>(); cudaDeviceSynchronize(); return 0; }
            """
        )
        Path("/tmp/smoke.cu").write_text(src)
        path = "/tmp/smoke.cu"
    else:
        path = f"/root/lab/{source}"
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} — did you uncomment add_local_dir?")

    out = "/tmp/run"
    subprocess.run(
        ["nvcc", "-O3", f"-arch={arch}", path, "-o", out, "-lcublas"],
        check=True,
    )
    res = subprocess.run([out], capture_output=True, text=True, check=True)
    return res.stdout


# ---------------------------------------------------------------------------
# Entrypoint 2: run a Triton kernel and benchmark it.
# ---------------------------------------------------------------------------
@app.function(gpu="H200", timeout=600)
def triton_softmax(B: int = 4096, N: int = 8192) -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * n_cols + cols, num / den, mask=mask)

    x = torch.randn(B, N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)

    def fn():
        softmax_kernel[(B,)](out, x, N, BLOCK=BLOCK, num_warps=8)

    # Warmup + time
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    for i in range(50):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))

    # Validate vs torch
    ref = torch.softmax(x, dim=-1)
    max_err = (out - ref).abs().max().item()

    return {
        "p50_us": times_us[25],
        "p95_us": times_us[47],
        "max_err": max_err,
        "shape": (B, N),
    }


@app.local_entrypoint()
def main():
    print(run_cuda.remote())
    stats = triton_softmax.remote()
    print(f"triton softmax: p50={stats['p50_us']:.1f}us  p95={stats['p95_us']:.1f}us  err={stats['max_err']:.2e}")

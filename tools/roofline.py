"""roofline.py — plot a single kernel against the GPU roofline.

Usage::

    from tools.roofline import roofline_plot, GPU_PEAKS

    # AI = arithmetic intensity (FLOPs / byte moved through DRAM)
    ai = 2 * M * N * K / (M * K * 2 + K * N * 2 + M * N * 2)
    achieved = gflops(2 * M * N * K, p50_us) * 1e9      # FLOPs/sec
    roofline_plot("sgemm_stage5", ai, achieved, gpu="A100_80GB", dtype="fp16_tc")

The plot is saved as PNG next to the caller (use chdir before calling, or
pass an absolute outpath). Commit the PNG into the per-exercise folder for
the README, per the plan's submission checklist.
"""

from __future__ import annotations

import numpy as np

# (peak_FLOPs_per_sec, peak_HBM_bytes_per_sec) per dtype.
# Sources: NVIDIA / AMD spec sheets. Use sparse=False (dense) numbers.
GPU_PEAKS: dict[str, dict[str, tuple[float, float]]] = {
    "A100_80GB": {
        # FP32 vector / FP16 Tensor Core / BF16 Tensor Core / INT8 TC
        "fp32":     (19.5e12, 2.039e12),
        "fp16_tc":  (312e12,  2.039e12),
        "bf16_tc":  (312e12,  2.039e12),
        "int8_tc":  (624e12,  2.039e12),
    },
    "H100_SXM": {
        "fp32":     (67e12,   3.35e12),
        "fp16_tc":  (989.4e12, 3.35e12),
        "bf16_tc":  (989.4e12, 3.35e12),
        "fp8_tc":   (1979e12, 3.35e12),
    },
    "H200": {
        "fp32":     (67e12,   4.8e12),
        "fp16_tc":  (989.4e12, 4.8e12),
        "bf16_tc":  (989.4e12, 4.8e12),
        "fp8_tc":   (1979e12, 4.8e12),
    },
    "B200": {
        "fp32":     (75e12,   8.0e12),
        "fp16_tc":  (2250e12, 8.0e12),
        "bf16_tc":  (2250e12, 8.0e12),
        "fp8_tc":   (4500e12, 8.0e12),
        "fp4_tc":   (9000e12, 8.0e12),
    },
    "B300": {
        # Public numbers will firm up; treat as approximate.
        "fp32":     (80e12,   8.0e12),
        "fp16_tc":  (2500e12, 8.0e12),
        "bf16_tc":  (2500e12, 8.0e12),
        "fp8_tc":   (5000e12, 8.0e12),
        "fp4_tc":   (10000e12, 8.0e12),
    },
    "MI300X": {
        "fp32":     (163.4e12, 5.3e12),
        "fp16_tc":  (1300e12,  5.3e12),
        "bf16_tc":  (1300e12,  5.3e12),
        "fp8_tc":   (2600e12,  5.3e12),
    },
}


def roofline_plot(
    name: str,
    ai: float,
    achieved_flops_per_sec: float,
    *,
    gpu: str = "A100_80GB",
    dtype: str = "fp16_tc",
    extra_points: list[tuple[str, float, float]] | None = None,
    outpath: str | None = None,
    show_ridge: bool = True,
):
    """Plot one or more kernels against the GPU's roofline.

    Parameters
    ----------
    name :
        Label for the primary point.
    ai :
        Arithmetic intensity (FLOPs / byte moved through DRAM).
    achieved_flops_per_sec :
        Measured throughput in FLOPs/sec (NOT GFLOP/s — pass raw FLOPs).
    gpu :
        Key into ``GPU_PEAKS``.
    dtype :
        Sub-key into ``GPU_PEAKS[gpu]`` ("fp32", "fp16_tc", "bf16_tc", ...).
    extra_points :
        Optional list of ``(label, ai, flops_per_sec)`` for stage comparisons.
    outpath :
        Output PNG path. Default: ``<name>_<gpu>_<dtype>_roofline.png``.
    """
    import matplotlib.pyplot as plt

    if gpu not in GPU_PEAKS:
        raise KeyError(f"unknown gpu={gpu}; pick from {list(GPU_PEAKS)}")
    if dtype not in GPU_PEAKS[gpu]:
        raise KeyError(f"unknown dtype={dtype} for {gpu}; pick from {list(GPU_PEAKS[gpu])}")

    peak_flops, peak_bw = GPU_PEAKS[gpu][dtype]
    ridge = peak_flops / peak_bw

    xs = np.logspace(-2, 4, 400)
    roof = np.minimum(peak_flops, xs * peak_bw)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(xs, roof, "-", label=f"{gpu} {dtype} peak", linewidth=2)
    ax.loglog(xs, xs * peak_bw, "--", alpha=0.4, label=f"HBM {peak_bw / 1e12:.2f} TB/s")
    ax.axhline(peak_flops, linestyle=":", alpha=0.4, label=f"compute {peak_flops / 1e12:.0f} TFLOP/s")
    if show_ridge:
        ax.axvline(ridge, linestyle="--", alpha=0.3, color="grey")
        ax.text(ridge, peak_flops * 1.1, f"AI*={ridge:.0f}", fontsize=9, alpha=0.6)

    ax.scatter([ai], [achieved_flops_per_sec], s=80, color="red", zorder=5, label=name)
    if extra_points:
        for lbl, x, y in extra_points:
            ax.scatter([x], [y], s=60, zorder=5, label=lbl)

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (FLOPs / sec)")
    ax.set_title(f"Roofline: {name} on {gpu} ({dtype})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1e10, peak_flops * 5)

    out = outpath or f"{name}_{gpu}_{dtype}_roofline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Demo: SGEMM stage 1 vs stage 5 on A100.
    M = N = K = 4096
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 4  # fp32, no reuse
    ai_naive = flops / bytes_moved              # ~1365 — but stage 1 doesn't realize this
    out = roofline_plot(
        "sgemm_stage5",
        ai=ai_naive,
        achieved_flops_per_sec=200e12,
        gpu="A100_80GB",
        dtype="fp32",
        extra_points=[("sgemm_stage1", 0.5, 50e9)],
    )
    print(f"wrote {out}")

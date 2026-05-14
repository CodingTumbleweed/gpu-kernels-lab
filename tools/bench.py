"""bench.py — standard timing harness for CUDA / Triton kernels.

Drop into ``gpu-kernels-lab/tools/bench.py``. Use everywhere; do NOT use
``time.perf_counter`` for GPU work (it ignores kernel launch overhead).

Usage::

    from tools.bench import bench, gflops, fmt

    def fn():
        out = my_kernel(a, b)
        # do not return — the work just needs to be issued

    stats = bench(fn, warmup=20, iters=200)
    print(fmt(stats))
    print(f"{gflops(2*M*N*K, stats['p50']):.1f} GFLOP/s @ p50")
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

import torch


def bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 20,
    iters: int = 200,
    return_all: bool = False,
) -> dict | tuple[dict, list[float]]:
    """Time a no-arg callable that issues GPU work.

    Returns a dict with mean, p50, p95, p99, stdev, min, max in **microseconds**.
    Always synchronizes between warmup and measurement so cold-cache effects
    don't contaminate the early iters.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("bench() requires a CUDA device")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    out = {
        "mean": sum(times_us) / iters,
        "p50": times_us[iters // 2],
        "p95": times_us[int(iters * 0.95)],
        "p99": times_us[int(iters * 0.99)],
        "stdev": statistics.pstdev(times_us),
        "min": times_us[0],
        "max": times_us[-1],
        "iters": iters,
    }
    return (out, times_us) if return_all else out


def gflops(flops: float, time_us: float) -> float:
    """Convert (FLOPs done in time_us microseconds) to GFLOP/s."""
    return flops / (time_us * 1e-6) / 1e9


def gbps(bytes_moved: float, time_us: float) -> float:
    """Convert (bytes moved in time_us microseconds) to GB/s."""
    return bytes_moved / (time_us * 1e-6) / 1e9


def fmt(stats: dict, unit: str = "us") -> str:
    """One-line summary of a bench() result."""
    scale = {"us": 1, "ms": 1e-3, "s": 1e-6}[unit]
    return (
        f"mean={stats['mean'] * scale:.2f}{unit} "
        f"p50={stats['p50'] * scale:.2f}{unit} "
        f"p95={stats['p95'] * scale:.2f}{unit} "
        f"p99={stats['p99'] * scale:.2f}{unit} "
        f"std={stats['stdev'] * scale:.2f}{unit}"
    )


def report_table(rows: list[dict], headers: list[str]) -> str:
    """Render a markdown table for README pasting."""
    try:
        from tabulate import tabulate

        return tabulate(rows, headers="keys", tablefmt="github", floatfmt=".2f")
    except ImportError:
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for r in rows:
            out.append("| " + " | ".join(f"{r.get(h, ''):.2f}" if isinstance(r.get(h), float) else str(r.get(h, "")) for h in headers) + " |")
        return "\n".join(out)


if __name__ == "__main__":
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    def fn():
        torch.matmul(a, b)

    stats = bench(fn)
    flops = 2 * 4096**3
    print(f"torch.matmul(4096^3 fp32):  {fmt(stats)}  ->  {gflops(flops, stats['p50']):.0f} GFLOP/s")

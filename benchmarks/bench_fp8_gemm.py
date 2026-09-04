"""Benchmark FP8 GEMM vs fp16 matmul (cuBLAS via torch.matmul).

Unlike the other kernels here, GEMM is compute-bound, so the metric is TFLOP/s
(2·M·N·K / time). FP8 tensor cores on Ada have ~2x the fp16 throughput, but
cuBLAS fp16 is heavily tuned, so the point is to show the FP8 path works and
measure the TFLOP/s our simple tiled kernel reaches -- not to beat cuBLAS.

Usage:
    python benchmarks/bench_fp8_gemm.py

Requires a CUDA GPU with FP8 tensor cores (Ada / RTX 4090+).
"""

import argparse

import torch

from triton_llm_kernels import fp8_gemm

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def tflops(ms, M, N, K):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def run(sizes):
    do_bench = triton.testing.do_bench
    rows = []
    for S in sizes:
        M = N = K = S
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        t_fp16 = do_bench(lambda: torch.matmul(a, b), warmup=25, rep=50)
        t_fp8 = do_bench(lambda: fp8_gemm(a, b), warmup=25, rep=50)
        rows.append((S, t_fp16, t_fp8))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="assets/fp8_gemm_bench.png")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    if not HAS_TRITON:
        raise SystemExit("Triton is required.")

    sizes = [512, 1024, 2048, 4096, 8192]
    rows = run(sizes)

    gpu = torch.cuda.get_device_name()
    print(f"\nFP8 GEMM (square M=N=K)  |  GPU={gpu}\n")
    header = f"{'size':>6} | {'fp16 ms':>9} {'fp8 ms':>9} | {'fp16 TFLOP/s':>13} {'fp8 TFLOP/s':>12}"
    print(header); print("-" * len(header))
    for S, t16, t8 in rows:
        print(f"{S:>6} | {t16:>9.4f} {t8:>9.4f} | {tflops(t16,S,S,S):>13.1f} {tflops(t8,S,S,S):>12.1f}")

    _plot(rows, args)


def _plot(rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
        return
    import os
    sizes = [r[0] for r in rows]
    fp16_tf = [tflops(r[1], r[0], r[0], r[0]) for r in rows]
    fp8_tf = [tflops(r[2], r[0], r[0], r[0]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, fp16_tf, marker="o", label="torch.matmul fp16 (cuBLAS)")
    ax.plot(sizes, fp8_tf, marker="o", label="FP8 GEMM (ours)")
    ax.set_xlabel("matrix size (M=N=K)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title("FP8 GEMM vs fp16 matmul (RTX 4090)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()

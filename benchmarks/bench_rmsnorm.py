"""Benchmark RMSNorm: eager PyTorch vs. torch.compile vs. fused Triton.

RMSNorm is memory-bound, so the headline metric is effective memory bandwidth
(GB/s): how close we get to the GPU's peak. We also report latency and speedup
vs. eager. Results are printed as a table and saved as a PNG for the README.

Usage:
    python benchmarks/bench_rmsnorm.py --M 8192 --dtype fp16
    python benchmarks/bench_rmsnorm.py --out assets/rmsnorm_bench.png

Requires a CUDA GPU.
"""

import argparse

import torch

from triton_llm_kernels import rmsnorm, rmsnorm_reference

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def bytes_moved(M, N, dtype):
    """Read x (M*N) + weight (N) + write y (M*N), in bytes. rstd is negligible."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return (2 * M * N + N) * elem


def gbps(ms, M, N, dtype):
    return bytes_moved(M, N, dtype) / (ms * 1e-3) / 1e9


def run(M, hidden_sizes, dtype, provider):
    """Return list of (N, latency_ms) for a given implementation."""
    do_bench = triton.testing.do_bench
    results = []
    for N in hidden_sizes:
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype)

        if provider == "eager":
            fn = lambda: rmsnorm_reference(x, w)
        elif provider == "compile":
            compiled = torch.compile(rmsnorm_reference)
            compiled(x, w)  # warm up / trigger compilation outside timing
            fn = lambda: compiled(x, w)
        elif provider == "triton":
            fn = lambda: rmsnorm(x, w)
        else:
            raise ValueError(provider)

        ms = do_bench(fn, warmup=25, rep=100)
        results.append((N, ms))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192, help="number of rows (tokens)")
    p.add_argument("--dtype", choices=DTYPE_MAP, default="fp16")
    p.add_argument("--out", type=str, default="assets/rmsnorm_bench.png")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    if not HAS_TRITON:
        raise SystemExit("Triton is required for benchmarking.")

    dtype = DTYPE_MAP[args.dtype]
    hidden_sizes = [1024, 2048, 3072, 4096, 5120, 6144, 8192]
    providers = ["eager", "compile", "triton"]

    data = {prov: run(args.M, hidden_sizes, dtype, prov) for prov in providers}

    gpu = torch.cuda.get_device_name()
    print(f"\nRMSNorm  |  M={args.M}  dtype={args.dtype}  GPU={gpu}\n")
    header = f"{'N':>6} | {'eager ms':>9} {'compile ms':>11} {'triton ms':>10} " \
             f"| {'triton GB/s':>11} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for i, N in enumerate(hidden_sizes):
        e = data["eager"][i][1]
        c = data["compile"][i][1]
        t = data["triton"][i][1]
        bw = gbps(t, args.M, N, dtype)
        print(f"{N:>6} | {e:>9.4f} {c:>11.4f} {t:>10.4f} | {bw:>11.1f} {e / t:>7.2f}x")

    _plot(data, hidden_sizes, args)


def _plot(data, hidden_sizes, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
        return

    import os

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {"eager": "PyTorch eager", "compile": "torch.compile", "triton": "Triton (ours)"}
    for prov in ["eager", "compile", "triton"]:
        xs = [n for n, _ in data[prov]]
        ys = [ms for _, ms in data[prov]]
        ax1.plot(xs, ys, marker="o", label=labels[prov])
    ax1.set_xlabel("hidden size N")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title(f"RMSNorm latency (M={args.M}, {args.dtype})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    dtype = DTYPE_MAP[args.dtype]
    for prov in ["eager", "compile", "triton"]:
        xs = [n for n, _ in data[prov]]
        ys = [gbps(ms, args.M, n, dtype) for n, ms in data[prov]]
        ax2.plot(xs, ys, marker="o", label=labels[prov])
    ax2.set_xlabel("hidden size N")
    ax2.set_ylabel("effective bandwidth (GB/s)")
    ax2.set_title("Memory throughput (higher = better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()

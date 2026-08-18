"""Benchmark the fused SwiGLU activation: eager vs. torch.compile vs. Triton.

SwiGLU's activation (SiLU(gate)*up) is memory-bound: read gate + up, write out.
Headline metric is effective bandwidth (GB/s). Matmuls are excluded on purpose
-- this measures the fusable activation, not the GEMMs (which belong to cuBLAS).

Usage:
    python benchmarks/bench_swiglu.py --M 8192 --dtype fp16

Requires a CUDA GPU.
"""

import argparse

import torch

from triton_llm_kernels import swiglu, swiglu_reference

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def bytes_moved(M, N, dtype):
    elem = torch.tensor([], dtype=dtype).element_size()
    # read gate + up, write out
    return 3 * M * N * elem


def gbps(ms, M, N, dtype):
    return bytes_moved(M, N, dtype) / (ms * 1e-3) / 1e9


def run(M, ffn_sizes, dtype, provider):
    do_bench = triton.testing.do_bench
    results = []
    for N in ffn_sizes:
        gate = torch.randn(M, N, device="cuda", dtype=dtype)
        up = torch.randn(M, N, device="cuda", dtype=dtype)

        if provider == "eager":
            fn = lambda: swiglu_reference(gate, up)
        elif provider == "compile":
            compiled = torch.compile(swiglu_reference)
            compiled(gate, up)  # warm up outside timing
            fn = lambda: compiled(gate, up)
        elif provider == "triton":
            fn = lambda: swiglu(gate, up)
        else:
            raise ValueError(provider)

        ms = do_bench(fn, warmup=25, rep=100)
        results.append((N, ms))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192, help="rows (tokens)")
    p.add_argument("--dtype", choices=DTYPE_MAP, default="fp16")
    p.add_argument("--out", type=str, default="assets/swiglu_bench.png")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    if not HAS_TRITON:
        raise SystemExit("Triton is required for benchmarking.")

    dtype = DTYPE_MAP[args.dtype]
    # intermediate (d_ff) sizes seen in real LLMs
    ffn_sizes = [2048, 4096, 5504, 8192, 11008, 14336]
    providers = ["eager", "compile", "triton"]
    data = {prov: run(args.M, ffn_sizes, dtype, prov) for prov in providers}

    gpu = torch.cuda.get_device_name()
    print(f"\nSwiGLU activation  |  M={args.M}  dtype={args.dtype}  GPU={gpu}\n")
    header = f"{'d_ff':>6} | {'eager ms':>9} {'compile ms':>11} {'triton ms':>10} " \
             f"| {'triton GB/s':>11} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for i, N in enumerate(ffn_sizes):
        e = data["eager"][i][1]
        c = data["compile"][i][1]
        t = data["triton"][i][1]
        bw = gbps(t, args.M, N, dtype)
        print(f"{N:>6} | {e:>9.4f} {c:>11.4f} {t:>10.4f} | {bw:>11.1f} {e / t:>7.2f}x")

    _plot(data, ffn_sizes, args, dtype)


def _plot(data, ffn_sizes, args, dtype):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
        return
    import os

    labels = {"eager": "PyTorch eager", "compile": "torch.compile", "triton": "Triton (ours)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for prov in ["eager", "compile", "triton"]:
        xs = [n for n, _ in data[prov]]
        ax1.plot(xs, [ms for _, ms in data[prov]], marker="o", label=labels[prov])
    ax1.set_xlabel("intermediate size d_ff")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title(f"SwiGLU activation latency (M={args.M}, {args.dtype})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for prov in ["eager", "compile", "triton"]:
        xs = [n for n, _ in data[prov]]
        ys = [gbps(ms, args.M, n, dtype) for n, ms in data[prov]]
        ax2.plot(xs, ys, marker="o", label=labels[prov])
    ax2.set_xlabel("intermediate size d_ff")
    ax2.set_ylabel("effective bandwidth (GB/s)")
    ax2.set_title("Memory throughput (higher = better)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()

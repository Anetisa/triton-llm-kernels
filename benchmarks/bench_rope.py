"""Benchmark RoPE: eager PyTorch vs. torch.compile vs. fused Triton.

Like RMSNorm, RoPE is memory-bound (a few multiply-adds per element), so the
headline metric is effective memory bandwidth (GB/s). Results print as a table
and save as a PNG for the README.

Usage:
    python benchmarks/bench_rope.py --B 8 --H 32 --D 128 --dtype fp16

Requires a CUDA GPU.
"""

import argparse

import torch

from triton_llm_kernels import build_rope_cache, rope, rope_reference

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def bytes_moved(B, S, H, D, dtype):
    elem = torch.tensor([], dtype=dtype).element_size()
    # read x + write out (M*D each); cos/sin (2*S*D) are negligible
    M = B * S * H
    return (2 * M * D + 2 * S * D) * elem


def gbps(ms, B, S, H, D, dtype):
    return bytes_moved(B, S, H, D, dtype) / (ms * 1e-3) / 1e9


def run(B, H, D, seq_lens, dtype, provider):
    do_bench = triton.testing.do_bench
    results = []
    for S in seq_lens:
        x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        cos, sin = build_rope_cache(S, D, device="cuda", dtype=dtype)

        if provider == "eager":
            fn = lambda: rope_reference(x, cos, sin)
        elif provider == "compile":
            compiled = torch.compile(rope_reference)
            compiled(x, cos, sin)  # warm up outside timing
            fn = lambda: compiled(x, cos, sin)
        elif provider == "triton":
            fn = lambda: rope(x, cos, sin)
        else:
            raise ValueError(provider)

        ms = do_bench(fn, warmup=25, rep=100)
        results.append((S, ms))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8, help="batch size")
    p.add_argument("--H", type=int, default=32, help="num heads")
    p.add_argument("--D", type=int, default=128, help="head dim (even)")
    p.add_argument("--dtype", choices=DTYPE_MAP, default="fp16")
    p.add_argument("--out", type=str, default="assets/rope_bench.png")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    if not HAS_TRITON:
        raise SystemExit("Triton is required for benchmarking.")

    dtype = DTYPE_MAP[args.dtype]
    seq_lens = [512, 1024, 2048, 4096, 8192]
    providers = ["eager", "compile", "triton"]
    data = {prov: run(args.B, args.H, args.D, seq_lens, dtype, prov) for prov in providers}

    gpu = torch.cuda.get_device_name()
    print(f"\nRoPE  |  B={args.B} H={args.H} D={args.D}  dtype={args.dtype}  GPU={gpu}\n")
    header = f"{'S':>6} | {'eager ms':>9} {'compile ms':>11} {'triton ms':>10} " \
             f"| {'triton GB/s':>11} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for i, S in enumerate(seq_lens):
        e = data["eager"][i][1]
        c = data["compile"][i][1]
        t = data["triton"][i][1]
        bw = gbps(t, args.B, S, args.H, args.D, dtype)
        print(f"{S:>6} | {e:>9.4f} {c:>11.4f} {t:>10.4f} | {bw:>11.1f} {e / t:>7.2f}x")

    _plot(data, seq_lens, args, dtype)


def _plot(data, seq_lens, args, dtype):
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
        xs = [s for s, _ in data[prov]]
        ax1.plot(xs, [ms for _, ms in data[prov]], marker="o", label=labels[prov])
    ax1.set_xlabel("sequence length S")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title(f"RoPE latency (B={args.B} H={args.H} D={args.D}, {args.dtype})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for prov in ["eager", "compile", "triton"]:
        xs = [s for s, _ in data[prov]]
        ys = [gbps(ms, args.B, s, args.H, args.D, dtype) for s, ms in data[prov]]
        ax2.plot(xs, ys, marker="o", label=labels[prov])
    ax2.set_xlabel("sequence length S")
    ax2.set_ylabel("effective bandwidth (GB/s)")
    ax2.set_title("Memory throughput (higher = better)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()

"""Benchmark FlashAttention forward vs. PyTorch SDPA (and a naive materialized attn).

Attention is compute-bound (two matmuls per block), so the headline here is
latency and, for the naive baseline, the memory blow-up it avoids. We compare:
  * naive  : explicit S = QK^T, softmax, P@V  (materializes the S×S matrix)
  * sdpa   : torch.nn.functional.scaled_dot_product_attention (Flash backend)
  * triton : this kernel

Usage:
    python benchmarks/bench_flash_attention.py --B 4 --H 32 --D 64 --dtype fp16

Requires a CUDA GPU.
"""

import argparse
import math

import torch
import torch.nn.functional as F

from triton_llm_kernels import flash_attention, flash_attention_forward

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def naive_attention(q, k, v, causal=True):
    D = q.shape[-1]
    s = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    if causal:
        S = q.shape[-2]
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


def run(B, H, D, seq_lens, dtype, provider, causal=True, backward=False):
    do_bench = triton.testing.do_bench
    results = []
    for S in seq_lens:
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=backward)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=backward)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=backward)
        do = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        if provider == "naive":
            fwd = lambda: naive_attention(q, k, v, causal)
        elif provider == "sdpa":
            fwd = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        elif provider == "triton":
            fwd = (lambda: flash_attention(q, k, v, causal=causal)) if backward \
                else (lambda: flash_attention_forward(q, k, v, causal=causal))
        else:
            raise ValueError(provider)

        if backward:
            def step():
                o = fwd()
                o.backward(do, retain_graph=True)
            fn = step
        else:
            fn = fwd

        try:
            ms = do_bench(fn, warmup=25, rep=50)
        except RuntimeError:  # naive can OOM at long S
            ms = float("nan")
        results.append((S, ms))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--dtype", choices=DTYPE_MAP, default="fp16")
    p.add_argument("--out", type=str, default="assets/flash_bench.png")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    if not HAS_TRITON:
        raise SystemExit("Triton is required for benchmarking.")

    dtype = DTYPE_MAP[args.dtype]
    seq_lens = [512, 1024, 2048, 4096, 8192]
    providers = ["naive", "sdpa", "triton"]
    data = {prov: run(args.B, args.H, args.D, seq_lens, dtype, prov) for prov in providers}

    gpu = torch.cuda.get_device_name()
    print(f"\nFlashAttention fwd (causal)  |  B={args.B} H={args.H} D={args.D}  "
          f"dtype={args.dtype}  GPU={gpu}\n")
    header = f"{'S':>6} | {'naive ms':>10} {'sdpa ms':>9} {'triton ms':>10} | {'vs naive':>9}"
    print(header)
    print("-" * len(header))
    for i, S in enumerate(seq_lens):
        n = data["naive"][i][1]
        s = data["sdpa"][i][1]
        t = data["triton"][i][1]
        speed = f"{n / t:.2f}x" if n == n else "OOM"  # n!=n checks NaN
        print(f"{S:>6} | {n:>10.4f} {s:>9.4f} {t:>10.4f} | {speed:>9}")

    _plot(data, seq_lens, args)


def _plot(data, seq_lens, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
        return
    import os

    labels = {"naive": "naive (materialized)", "sdpa": "PyTorch SDPA", "triton": "Triton (ours)"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prov in ["naive", "sdpa", "triton"]:
        xs = [s for s, _ in data[prov]]
        ys = [ms for _, ms in data[prov]]
        ax.plot(xs, ys, marker="o", label=labels[prov])
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"FlashAttention fwd, causal (B={args.B} H={args.H} D={args.D}, {args.dtype})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()

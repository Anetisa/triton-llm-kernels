# SwiGLU kernel: the math and why it's fast

## What SwiGLU computes

SwiGLU (Shazeer, 2020, "GLU Variants Improve Transformer") is the gated MLP
activation used by LLaMA, Mistral, Qwen and PaLM. The full feed-forward block is

    FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )

where ⊙ is elementwise multiply and SiLU (a.k.a. swish) is

    SiLU(z) = z · sigmoid(z)

Two projections produce `gate = W_gate·x` and `up = W_up·x`; the activation gates
one by a smooth function of the other, then a third projection maps back. The
gating lets the network modulate information flow per channel, which empirically
beats a plain ReLU/GELU MLP at the same parameter count (hence `d_ff` is scaled
by ~2/3 to keep the FLOPs matched).

## What this kernel fuses (and what it doesn't)

The three matmuls (`W_gate`, `W_up`, `W_down`) are compute-bound GEMMs — cuBLAS
and tensor cores already handle them near-optimally, and a hand-written Triton
GEMM rarely wins. The *fusable* part is the elementwise middle:

    swiglu(gate, up) = SiLU(gate) ⊙ up

given `gate, up` of shape `[.., d_ff]`. This is **memory-bound**: read `gate`,
read `up`, write `out` — a few flops per element. Eager PyTorch
(`F.silu(gate) * up`) materializes the `SiLU(gate)` intermediate and launches
multiple kernels; the fused kernel reads each element once, computes in
registers, and writes once. The metric is **GB/s**.

## Backward: the SiLU derivative

With `s = sigmoid(gate)` and `silu = gate·s`:

    ∂out/∂up   = silu               →  d(up)   = dout · silu
    ∂out/∂gate = up · silu'(gate)   →  d(gate) = dout · up · silu'(gate)

The SiLU derivative is the one non-obvious piece:

    silu'(z) = d/dz [ z·σ(z) ] = σ(z) + z·σ'(z)
             = σ(z) + z·σ(z)(1 − σ(z))
             = σ(z) · ( 1 + z·(1 − σ(z)) )

Both gradients are pure elementwise — no reductions, no atomics — so the backward
kernel has the same shape as the forward. The formula is checked against autograd
to fp64 precision in development.

## Implementation notes

- **1-D elementwise grid.** Unlike RMSNorm (per-row reduction) or RoPE (paired
  rotation), SwiGLU is fully elementwise, so the kernel flattens the tensors and
  processes `BLOCK` elements per program — the simplest possible layout.
- **fp32 compute.** `gate`/`up` may be fp16/bf16; sigmoid and the products are
  done in fp32 and cast back on store, mirrored by the reference.
- **Sigmoid numerics.** `tl.sigmoid` is numerically stable across the input
  range, so no manual clamping is needed.
- **Full MLP.** `SwiGLUMLP` wires the fused activation between the three
  `nn.Linear` projections, showing how the kernel drops into a real model.
- **Roadmap.** Fusing the activation with the `W_down` matmul epilogue (so the
  intermediate never hits DRAM) is a natural next optimization.

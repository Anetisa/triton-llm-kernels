# RMSNorm kernel: the math and why it's fast

## What RMSNorm computes

For a row `x ∈ R^N` (the hidden dimension), with weight `w ∈ R^N`:

```
rstd = 1 / sqrt( (1/N) · Σ_i x_i²  +  eps )
y_i  = x_i · rstd · w_i
```

Unlike LayerNorm there is no mean-subtraction and no bias — just a
root-mean-square rescale followed by a per-channel gain. That's what makes it
cheap and why every recent LLM (LLaMA, Mistral, Qwen, Gemma) uses it.

## Why it's memory-bound

Count the memory traffic for an `[M, N]` activation tensor:

- read `x`: `M·N` elements
- read `w`: `N` elements (negligible)
- write `y`: `M·N` elements

So ~`2·M·N` elements move, and the arithmetic is a handful of flops per element.
The ratio of flops to bytes is tiny → the kernel is **bandwidth-bound**. The
best you can do is touch each element the minimum number of times. A fused
kernel reads each row once into registers, does the reduction and the rescale
without going back to DRAM, and writes once. Eager PyTorch instead materializes
intermediates (`x²`, its mean, `rsqrt`, the product) and launches several
kernels, each paying launch overhead and re-reading memory.

The right yardstick is therefore **effective bandwidth (GB/s)**, compared to the
GPU's spec sheet peak — not FLOP/s.

## Backward pass

Let `x_hat_i = x_i · rstd` and `dy_hat_i = dy_i · w_i`.

**Weight gradient** (reduction over the batch/sequence rows):

```
dw_i = Σ_rows dy_i · x_hat_i
```

**Input gradient.** Starting from `y_i = w_i · x_i · rstd` and
`rstd = (s + eps)^(-1/2)` with `s = (1/N)·Σ_k x_k²`:

```
∂rstd/∂x_j = -(1/N) · x_j · rstd³
```

Propagating through both the direct term and the shared `rstd`:

```
dx_j = rstd · dy_hat_j  −  (1/N) · x_j · rstd³ · Σ_i (dy_hat_i · x_i)
```

which simplifies, using `x_hat`, to the form the kernel implements:

```
c    = (1/N) · Σ_i (dy_hat_i · x_hat_i)     # one scalar per row
dx_j = rstd · ( dy_hat_j − x_hat_j · c )
```

Both `dx` and the per-row `c` need only a single reduction over the hidden dim,
so the backward kernel keeps the same one-row-per-program structure as the
forward. This derivation is checked against PyTorch autograd to fp64 precision
in the tests.

## Implementation notes

- **Accumulate in fp32.** Inputs may be fp16/bf16, but the sum-of-squares and
  the normalization are done in fp32 to avoid precision loss, then cast back on
  store. The reference mirrors this so the two match tightly.
- **One program per row.** Each Triton program handles one row with
  `BLOCK_SIZE = next_power_of_2(N)`, so the whole hidden dim lives in registers.
  For very large `N` this could be split into column blocks with a two-pass
  reduction — a future optimization.
- **Weight-grad reduction.** `dw` sums over all rows. The current kernel uses
  `tl.atomic_add` into `GROUP_M` partial buffers, then sums them in PyTorch.
  It's simple and correct; the classic optimization (grouped accumulation with
  locks, as in the Triton LayerNorm tutorial) is on the roadmap.

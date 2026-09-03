# LayerNorm kernel: the math and the grown-up weight-grad reduction

## What LayerNorm computes

LayerNorm (Ba et al., 2016) standardizes each row to zero mean and unit
variance, then applies a per-channel gain and bias:

    mean = (1/N) Σ x_i
    var  = (1/N) Σ (x_i − mean)²
    y_i  = (x_i − mean) / sqrt(var + eps) · w_i + b_i

Compared to RMSNorm it does two extra things: it **subtracts the mean** (so the
backward has an extra term) and it has a **bias** `b` (a third gradient). Like
the others it's memory-bound — read `x`, write `y` — so the metric is GB/s.

## Forward

One program per row. It reduces the row to `mean` and `var` in fp32, saves
`mean` and `rstd = 1/sqrt(var+eps)` for the backward, and writes
`(x − mean)·rstd·w + b`. A masking detail matters: when computing the variance,
out-of-range lanes must be re-zeroed *after* subtracting the mean
(`where(mask, x − mean, 0)`), otherwise they'd contribute `mean²` to the sum.

## Backward for dx: two reductions

With `x_hat = (x − mean)·rstd`, `dy_hat = dy·w`, and per-row means over `N`:

    c1 = mean( dy_hat · x_hat )
    c2 = mean( dy_hat )
    dx = ( dy_hat − (x_hat·c1 + c2) ) · rstd

The `c2` term is exactly what mean-subtraction adds relative to RMSNorm (whose
`dx` has only the `c1`-style term). Both are single row reductions, so the dx
kernel keeps one-program-per-row.

## The interesting part: dw and db over all rows

`dw` and `db` are reductions over the *batch/sequence* dimension — every row
contributes to the same `[N]` vectors. RMSNorm in this repo uses the simplest
correct thing, `tl.atomic_add` into `GROUP_M` partial buffers. LayerNorm uses
the **grown-up pattern from the Triton tutorial**, in two stages:

1. **Lock-guarded partial accumulation.** Each row maps to one of `GROUP_M`
   partial buffers via `row % GROUP_M`. To add its contribution it takes a
   spin-lock on that slot with `atomic_cas(Lock, 0, 1)`, uses a per-slot `Count`
   flag to decide *initialize vs. accumulate*, writes back, and releases with
   `atomic_xchg(Lock, 0)`. Contention is bounded to `GROUP_M` slots no matter
   how many rows there are, and each slot's running sum is updated atomically.

2. **Final reduction kernel.** A second kernel tiles the `[GROUP_M, N]` partials
   in blocks and sums down to the final `[N]` `dw` and `db`.

Why bother versus a plain `atomic_add`? Atomics on every element of every row
contend heavily and are non-deterministic in float accumulation order. The
grouped-lock scheme caps contention at `GROUP_M` and makes each slot's
accumulation ordered, which is the standard production approach. It's the main
new technique this kernel adds to the repo.

## Correctness

The forward (mean≈0, var≈1) and the affine behaviour are checked on CPU. The
kernel is validated against `torch.nn.functional.layer_norm` for forward and all
three gradients, including a shape with `M > GROUP_M` so the lock actually sees
contention (multiple rows sharing a slot). The dx/dw/db formulas are checked
against autograd to fp64 precision in development.

## Notes and roadmap

- **fp32 accumulation** throughout; cast back on store, mirrored by the reference.
- **`GROUP_M = min(M, 128)`** balances the partial-buffer size against contention.
- A natural follow-up is autotuning `GROUP_M`/`BLOCK` and fusing the residual add
  that usually precedes LayerNorm in a transformer block.

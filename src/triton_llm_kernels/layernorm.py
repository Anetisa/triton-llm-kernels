"""Fused LayerNorm in Triton (forward + backward, with weight and bias).

LayerNorm (Ba et al., 2016) normalizes each row to zero mean and unit variance,
then applies a per-channel gain and bias:

    mean = (1/N) Σ x_i
    var  = (1/N) Σ (x_i − mean)²
    y_i  = (x_i − mean) / sqrt(var + eps) · w_i + b_i

It differs from RMSNorm in two ways: it subtracts the mean (so the backward has
an extra reduction term), and it has a learnable bias.

Weight/bias gradients: the grown-up reduction
----------------------------------------------
dw and db are sums over *all rows*. RMSNorm here uses a simple `atomic_add`;
LayerNorm instead uses the classic Triton-tutorial pattern: each row accumulates
its partial dw/db into one of GROUP_M buffers guarded by a spin-lock
(`atomic_cas`), and a second kernel reduces the GROUP_M partials to the final
[N] gradients. This bounds contention to GROUP_M lock slots regardless of how
many rows there are, and keeps the accumulation deterministic per slot.

Math (backward)
---------------
With x_hat = (x − mean)·rstd, dy_hat = dy·w, and per-row means over N:

    c1   = mean(dy_hat · x_hat)
    c2   = mean(dy_hat)
    dx   = ( dy_hat − (x_hat·c1 + c2) ) · rstd
    dw_i = Σ_rows dy_i · x_hat_i
    db_i = Σ_rows dy_i
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# Reference (pure PyTorch, autograd-native, runs on CPU)                      #
# --------------------------------------------------------------------------- #
def layernorm_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """LayerNorm ground truth over the last dim. Computes in fp32."""
    input_dtype = x.dtype
    out = torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
    )
    return out.to(input_dtype)


if HAS_TRITON:

    @triton.jit
    def _ln_fwd_kernel(
        X, Y, W, B, Mean, Rstd, stride_row, N, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        X += row * stride_row
        Y += row * stride_row
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)          # re-mask before squaring
        var = tl.sum(xc * xc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = xc * rstd * w + b
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _ln_bwd_dx_kernel(
        DX, DY, X, W, Mean, Rstd, DW, DB, Lock,
        stride_row, N, GROUP_M, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride_row
        DY += row * stride_row
        DX += row * stride_row
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        xhat = tl.where(mask, (x - mean) * rstd, 0.0)
        dyhat = tl.where(mask, dy * w, 0.0)
        c1 = tl.sum(dyhat * xhat, axis=0) / N
        c2 = tl.sum(dyhat, axis=0) / N
        dx = (dyhat - (xhat * c1 + c2)) * rstd
        tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)

        # --- grouped, lock-guarded partial accumulation of dw/db ---
        partial_dw = (dy * xhat)
        partial_db = dy
        lock_id = row % GROUP_M
        Lock += lock_id
        Count = Lock + GROUP_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols

        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask, other=0.0)
            partial_db += tl.load(DB, mask=mask, other=0.0)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)
        tl.atomic_xchg(Lock, 0)

    @triton.jit
    def _ln_bwd_dwdb_kernel(
        DW, DB, FINAL_DW, FINAL_DB, GROUP_M, N,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        dw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(0, GROUP_M, BLOCK_M):
            rows = i + tl.arange(0, BLOCK_M)
            mask = (rows[:, None] < GROUP_M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.0)
            db += tl.load(DB + offs, mask=mask, other=0.0)
        tl.store(FINAL_DW + cols, tl.sum(dw, axis=0), mask=cols < N)
        tl.store(FINAL_DB + cols, tl.sum(db, axis=0), mask=cols < N)

    class _LayerNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            orig_shape = x.shape
            x2d = x.contiguous().view(-1, orig_shape[-1])
            weight = weight.contiguous()
            bias = bias.contiguous()
            M, N = x2d.shape

            y = torch.empty_like(x2d)
            mean = torch.empty(M, dtype=torch.float32, device=x2d.device)
            rstd = torch.empty(M, dtype=torch.float32, device=x2d.device)
            BLOCK = triton.next_power_of_2(N)
            num_warps = max(1, min(32, BLOCK // 256))

            _ln_fwd_kernel[(M,)](
                x2d, y, weight, bias, mean, rstd, x2d.stride(0), N, eps,
                BLOCK=BLOCK, num_warps=num_warps,
            )
            ctx.save_for_backward(x2d, weight, mean, rstd)
            ctx.BLOCK = BLOCK
            ctx.num_warps = num_warps
            ctx.orig_shape = orig_shape
            return y.view(orig_shape)

        @staticmethod
        def backward(ctx, dy):
            x2d, weight, mean, rstd = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            dy2d = dy.contiguous().view(-1, orig_shape[-1])
            M, N = x2d.shape

            dx = torch.empty_like(x2d)
            GROUP_M = min(M, 128)
            dw_partial = torch.zeros((GROUP_M, N), dtype=torch.float32, device=x2d.device)
            db_partial = torch.zeros((GROUP_M, N), dtype=torch.float32, device=x2d.device)
            # Lock buffer: GROUP_M locks + GROUP_M init-flags, laid out contiguously.
            locks = torch.zeros(2 * GROUP_M, dtype=torch.int32, device=x2d.device)

            _ln_bwd_dx_kernel[(M,)](
                dx, dy2d, x2d, weight, mean, rstd, dw_partial, db_partial, locks,
                x2d.stride(0), N, GROUP_M,
                BLOCK=ctx.BLOCK, num_warps=ctx.num_warps,
            )

            dw = torch.empty(N, dtype=weight.dtype, device=x2d.device)
            db = torch.empty(N, dtype=weight.dtype, device=x2d.device)
            BLOCK_N = 64
            grid = (triton.cdiv(N, BLOCK_N),)
            _ln_bwd_dwdb_kernel[grid](
                dw_partial, db_partial, dw, db, GROUP_M, N,
                BLOCK_M=32, BLOCK_N=BLOCK_N,
            )
            return dx.view(orig_shape), dw, db, None


def layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Fused LayerNorm. Requires a CUDA tensor, or TRITON_INTERPRET=1 for CPU."""
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use layernorm_reference() on CPU, or install "
            "triton (bundled with PyTorch on Linux) and run on a CUDA GPU."
        )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not x.is_cuda and not interpret:
        raise RuntimeError(
            "layernorm() runs a Triton GPU kernel and needs a CUDA tensor. "
            "For CPU debugging use layernorm_reference(), or set TRITON_INTERPRET=1."
        )
    return _LayerNormFn.apply(x, weight, bias, eps)


class TritonLayerNorm(torch.nn.Module):
    """Drop-in LayerNorm layer backed by the fused Triton kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm(x, self.weight, self.bias, self.eps)

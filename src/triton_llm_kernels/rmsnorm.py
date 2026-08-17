"""Fused RMSNorm in Triton (forward + backward).

RMSNorm (Zhang & Sennrich, 2019) is the normalization used by LLaMA, Mistral,
Qwen and most modern LLMs. It is *memory-bound*: the arithmetic per element is
trivial, so runtime is dominated by reading `x` and writing `y`. A fused Triton
kernel loads each row once, does the whole normalization in registers, and
writes once -- avoiding the extra passes and kernel-launch overhead of a naive
`x * rsqrt(mean(x**2) + eps) * w` written in eager PyTorch.

Math
----
Forward, per row (reduction over the hidden dim of size N):

    rstd = 1 / sqrt(mean(x_i^2) + eps)
    y_i  = x_i * rstd * w_i

Backward. Let x_hat_i = x_i * rstd and dy_hat_i = dy_i * w_i. Then

    c    = (1/N) * sum_j (dy_hat_j * x_hat_j)          # a per-row scalar
    dx_i = rstd * (dy_hat_i - x_hat_i * c)
    dw_i = sum_over_rows (dy_i * x_hat_i)              # reduction over rows

The full derivation is in docs/rmsnorm.md.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # allows `import`/reference use on a machine without Triton
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# Reference implementation (pure PyTorch, runs anywhere -- CPU included)       #
# --------------------------------------------------------------------------- #
def rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Numerically-standard RMSNorm. Compute in fp32, cast back at the end.

    This mirrors exactly what the Triton kernel does, so the two can be compared
    with tight tolerances. It is fully autograd-compatible, which the tests use
    as the ground truth for the backward pass.
    """
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = xf * weight.float()
    return out.to(input_dtype)


if HAS_TRITON:

    # ----------------------------------------------------------------------- #
    # Kernels                                                                  #
    # ----------------------------------------------------------------------- #
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X,          # [M, N] input
        Y,          # [M, N] output
        W,          # [N]    weight
        Rstd,       # [M]    saved 1/rms per row (fp32), reused in backward
        stride_row,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride_row
        Y += row * stride_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Everything is accumulated in fp32 regardless of input dtype.
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        rstd = 1.0 / tl.sqrt(sum_sq / N + eps)
        tl.store(Rstd + row, rstd)

        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DY,          # [M, N] grad of output
        DX,          # [M, N] grad of input (written)
        X,           # [M, N] input
        W,           # [N]    weight
        Rstd,        # [M]    saved 1/rms
        DW_partial,  # [GROUP_M, N] partial weight grads (fp32, atomically summed)
        stride_row,
        N,
        GROUP_M,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride_row
        DY += row * stride_row
        DX += row * stride_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        x_hat = x * rstd
        dy_hat = dy * w
        # per-row scalar c = mean_j(dy_hat_j * x_hat_j)
        c = tl.sum(dy_hat * x_hat, axis=0) / N
        dx = rstd * (dy_hat - x_hat * c)
        tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)

        # Weight grad is a reduction over rows. Each row atomically adds its
        # contribution into one of GROUP_M partial buffers; PyTorch sums them.
        # Simple and correct; see roadmap for the locked/grouped optimization.
        dw = dy * x_hat
        lock_id = row % GROUP_M
        DW_partial += lock_id * N
        tl.atomic_add(DW_partial + cols, dw, mask=mask)

    # ----------------------------------------------------------------------- #
    # Autograd wiring                                                          #
    # ----------------------------------------------------------------------- #
    class _RMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            orig_shape = x.shape
            x2d = x.contiguous().reshape(-1, orig_shape[-1])
            weight = weight.contiguous()
            M, N = x2d.shape

            y = torch.empty_like(x2d)
            rstd = torch.empty(M, dtype=torch.float32, device=x2d.device)

            BLOCK_SIZE = triton.next_power_of_2(N)
            num_warps = max(1, min(32, BLOCK_SIZE // 256))

            _rmsnorm_fwd_kernel[(M,)](
                x2d, y, weight, rstd,
                x2d.stride(0), N, eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )

            ctx.save_for_backward(x2d, weight, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.orig_shape = orig_shape
            return y.reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy):
            x2d, weight, rstd = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            dy2d = dy.contiguous().reshape(-1, orig_shape[-1])
            M, N = x2d.shape

            dx = torch.empty_like(x2d)
            GROUP_M = min(M, 128)
            dw_partial = torch.zeros((GROUP_M, N), dtype=torch.float32, device=x2d.device)

            _rmsnorm_bwd_kernel[(M,)](
                dy2d, dx, x2d, weight, rstd, dw_partial,
                x2d.stride(0), N, GROUP_M,
                BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
            )

            dw = dw_partial.sum(dim=0).to(weight.dtype)
            return dx.reshape(orig_shape), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm. Requires a CUDA tensor (Triton kernels run on GPU).

    Falls back with a clear message if Triton is unavailable so the reference
    path stays usable on CPU-only machines.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use rmsnorm_reference() on CPU, or install "
            "triton (bundled with PyTorch on Linux) and run on a CUDA GPU."
        )
    # In interpreter mode (TRITON_INTERPRET=1) kernels run on CPU tensors, so we
    # only require CUDA when running real kernels.
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not x.is_cuda and not interpret:
        raise RuntimeError(
            "rmsnorm() runs a Triton GPU kernel and needs a CUDA tensor. "
            "For CPU debugging use rmsnorm_reference(), or set TRITON_INTERPRET=1."
        )
    return _RMSNormFn.apply(x, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in RMSNorm layer backed by the fused Triton kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

"""FP8 GEMM in Triton — matmul on Ada-native FP8 tensor cores (RTX 4090+).

FP8 (8-bit float) doubles tensor-core throughput vs fp16 on Ada/Hopper, at the
cost of precision: the e4m3 format has 4 exponent + 3 mantissa bits and a max
magnitude of 448. To use it for a matmul you must **scale** the inputs into fp8
range, accumulate in fp32, then dequantize:

    scale_a = amax(A) / 448 ;  A8 = (A / scale_a) as fp8      (per-tensor)
    scale_b = amax(B) / 448 ;  B8 = (B / scale_b) as fp8
    C ≈ (A8 @ B8) · (scale_a · scale_b)        # dot in fp32, then dequant

This is a lossy op by design (~a few % Frobenius error), so correctness is
checked by a norm-based relative error, not element-wise. The kernel itself is a
standard tiled GEMM whose only special ingredient is fp8 `tl.dot`.

Scope: per-tensor scaling, e4m3. Per-row/col scaling and e5m2 are natural
follow-ups. This is compute-bound (unlike the other kernels here) — the goal is
to exercise the FP8 tensor cores correctly, not to beat cuBLAS.
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

FP8_E4M3_MAX = 448.0


# --------------------------------------------------------------------------- #
# Reference + quantization helpers (pure PyTorch, run on CPU)                  #
# --------------------------------------------------------------------------- #
def gemm_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Full-precision matmul ground truth (fp32)."""
    return (a.float() @ b.float())


def quantize_fp8(x: torch.Tensor):
    """Per-tensor scale into e4m3 range. Returns (x_fp8, scale) with x ≈ x_fp8·scale."""
    scale = x.abs().amax().float() / FP8_E4M3_MAX + 1e-12
    x8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x8, scale


def fp8_relative_error(c: torch.Tensor, ref: torch.Tensor) -> float:
    """Frobenius relative error — the right metric for a lossy fp8 matmul."""
    return ((c - ref).norm() / ref.norm()).item()


if HAS_TRITON:

    @triton.jit
    def _fp8_gemm_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_mask = offs_k[None, :] < K - k
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)                      # fp8 x fp8 -> fp32 accumulate
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        acc = acc * scale                            # dequantize
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """FP8 matmul: A [M, K] @ B [K, N]. Inputs fp16/bf16/fp32; per-tensor scaled.

    Requires a CUDA tensor (Ada FP8 tensor cores), or TRITON_INTERPRET=1 for CPU.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed; use gemm_reference() on CPU.")
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not a.is_cuda and not interpret:
        raise RuntimeError(
            "fp8_gemm() runs a Triton GPU kernel and needs CUDA tensors. "
            "For CPU debugging set TRITON_INTERPRET=1, or use gemm_reference()."
        )
    assert a.shape[1] == b.shape[0], "inner dimensions must match"
    M, K = a.shape
    K2, N = b.shape

    a8, scale_a = quantize_fp8(a.contiguous())
    b8, scale_b = quantize_fp8(b.contiguous())
    scale = (scale_a * scale_b).item()

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fp8_gemm_kernel[grid](
        a8, b8, c, M, N, K,
        a8.stride(0), a8.stride(1),
        b8.stride(0), b8.stride(1),
        c.stride(0), c.stride(1),
        scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c

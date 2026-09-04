"""Correctness tests for the FP8 GEMM kernel.

FP8 is lossy by design, so correctness is a **norm-based relative error**, not
element-wise closeness (near-zero output elements have huge relative error even
when the matmul is correct). Two groups:
  * Property tests (CPU): the quantize/dequantize round-trip stays in fp8 range
    and reconstructs within fp8 precision.
  * Kernel tests (GPU or interpreter): fp8 GEMM matches the fp32 matmul within a
    few percent Frobenius error, across shapes that aren't multiples of the block.
"""

import os

import pytest
import torch

from triton_llm_kernels import fp8_gemm, gemm_reference, quantize_fp8
from triton_llm_kernels.fp8_gemm import FP8_E4M3_MAX, fp8_relative_error

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

SHAPES = [(128, 256, 128), (64, 64, 64), (96, 128, 80), (200, 64, 128), (129, 257, 65)]
# fp8 e4m3 Frobenius error is ~3-4% for well-scaled random inputs
FP8_REL_TOL = 0.08


# --------------------------------------------------------------------------- #
# CPU property tests                                                          #
# --------------------------------------------------------------------------- #
def test_quantize_stays_in_fp8_range():
    """After scaling, the largest magnitude maps to ~the fp8 max, not beyond."""
    x = torch.randn(256, 256) * 17.0
    x8, scale = quantize_fp8(x)
    # dequantized max should be close to the original max, and fp8 codes finite
    assert torch.isfinite(x8.float()).all()
    assert (x8.float().abs().max() <= FP8_E4M3_MAX + 1e-3)


def test_quantize_roundtrip_reconstructs():
    """x ≈ x8 · scale within fp8 precision (norm-based)."""
    x = torch.randn(128, 128)
    x8, scale = quantize_fp8(x)
    recon = x8.float() * scale
    rel = (recon - x).norm() / x.norm()
    assert rel < 0.10, f"roundtrip rel err {rel:.4f} too high"


# --------------------------------------------------------------------------- #
# GPU / interpreter: fp8 kernel vs fp32 matmul                                 #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("M,K,N", SHAPES)
def test_fp8_gemm_matches_fp32_matmul(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c = fp8_gemm(a, b)
    ref = gemm_reference(a, b)
    rel = fp8_relative_error(c.float(), ref)
    assert rel < FP8_REL_TOL, f"fp8 GEMM rel err {rel:.4f} exceeds {FP8_REL_TOL}"


@requires_cuda
def test_fp8_gemm_scale_invariance():
    """Scaling an input by a constant scales the output by the same constant."""
    torch.manual_seed(0)
    a = torch.randn(128, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(128, 128, device=DEVICE, dtype=torch.float16)
    c1 = fp8_gemm(a, b).float()
    c2 = fp8_gemm(a * 10.0, b).float()
    # per-tensor scaling makes fp8 GEMM invariant to input scale up to fp8 noise
    rel = ((c2 / 10.0) - c1).norm() / c1.norm()
    assert rel < 0.05, f"scale-invariance rel err {rel:.4f}"

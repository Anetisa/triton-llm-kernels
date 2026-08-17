"""Correctness tests for the RMSNorm kernel.

Two groups:
  * Reference-property tests run anywhere (CPU included) so `pytest` is never
    a no-op on a machine without a GPU.
  * Kernel-vs-reference tests are skipped unless a CUDA GPU is present.

The reference (`rmsnorm_reference`) is autograd-native pure PyTorch and serves
as ground truth for both forward and backward.
"""

import os

import pytest
import torch

from triton_llm_kernels import rmsnorm, rmsnorm_reference

# Kernel tests run on GPU normally, but can also run on CPU under Triton's
# interpreter (TRITON_INTERPRET=1) -- handy for validating kernel logic locally
# without a GPU. Device is chosen accordingly.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

SHAPES = [(4, 128), (32, 1024), (8, 4096), (2, 8192), (16, 5120)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype):
    return {
        torch.float32: dict(atol=1e-5, rtol=1e-5),
        torch.float16: dict(atol=1e-2, rtol=1e-2),
        torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    }[dtype]


# --------------------------------------------------------------------------- #
# CPU-runnable sanity checks on the reference                                 #
# --------------------------------------------------------------------------- #
def test_reference_unit_rms():
    """With weight=1, each normalized row should have RMS ~= 1."""
    x = torch.randn(64, 512)
    w = torch.ones(512)
    y = rmsnorm_reference(x, w, eps=1e-6)
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)


def test_reference_scale_invariance():
    """RMSNorm is invariant to positive scaling of the input."""
    x = torch.randn(8, 256)
    w = torch.randn(256)
    y1 = rmsnorm_reference(x, w)
    y2 = rmsnorm_reference(7.5 * x, w)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# GPU: fused kernel vs. reference                                             #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    w = torch.randn(N, device=DEVICE, dtype=dtype)

    y_tri = rmsnorm(x, w)
    y_ref = rmsnorm_reference(x, w)
    torch.testing.assert_close(y_tri, y_ref, **_tol(dtype))


@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype, requires_grad=True)
    w = torch.randn(N, device=DEVICE, dtype=dtype, requires_grad=True)
    dy = torch.randn(M, N, device=DEVICE, dtype=dtype)

    # Triton path
    rmsnorm(x, w).backward(dy)
    dx_tri, dw_tri = x.grad.detach().clone(), w.grad.detach().clone()
    x.grad = None
    w.grad = None

    # Reference path
    rmsnorm_reference(x, w).backward(dy)
    dx_ref, dw_ref = x.grad.detach(), w.grad.detach()

    tol = _tol(dtype)
    torch.testing.assert_close(dx_tri, dx_ref, **tol)
    # dw accumulates over rows, so give it a little more slack in low precision.
    dw_tol = tol if dtype == torch.float32 else dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dw_tri, dw_ref, **dw_tol)


@requires_cuda
def test_non_contiguous_input():
    """A transposed (non-contiguous) view should still work."""
    x = torch.randn(1024, 64, device=DEVICE).t()  # -> [64, 1024], non-contiguous
    w = torch.randn(1024, device=DEVICE)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-5, rtol=1e-5)


@requires_cuda
def test_3d_input():
    """Batched [B, T, H] input, as used in a real transformer forward pass."""
    x = torch.randn(4, 128, 2048, device=DEVICE, dtype=torch.float16)
    w = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w), **_tol(torch.float16))

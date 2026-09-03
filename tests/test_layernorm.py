"""Correctness tests for the LayerNorm kernel.

Two groups, same pattern as the other kernels:
  * Property tests run anywhere (CPU): normalized rows have ~zero mean and
    ~unit variance; the affine params recover the identity.
  * Kernel-vs-reference tests run on GPU, or on CPU under TRITON_INTERPRET=1,
    and cover forward and all three gradients (dx, dw, db). Includes a shape
    with M > GROUP_M so the lock-guarded reduction is exercised under contention.
"""

import os

import pytest
import torch

from triton_llm_kernels import layernorm, layernorm_reference

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

# 200 > GROUP_M(=128) forces lock contention in the dw/db reduction.
SHAPES = [(8, 64), (32, 1024), (5, 4096), (16, 5120), (200, 128)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype):
    return {
        torch.float32: dict(atol=1e-5, rtol=1e-5),
        torch.float16: dict(atol=1e-2, rtol=1e-2),
        torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    }[dtype]


# --------------------------------------------------------------------------- #
# CPU property tests (reference only)                                         #
# --------------------------------------------------------------------------- #
def test_normalized_stats():
    """With weight=1, bias=0, each row has ~zero mean and ~unit variance."""
    x = torch.randn(64, 512, dtype=torch.float64) * 3 + 1
    w = torch.ones(512, dtype=torch.float64)
    b = torch.zeros(512, dtype=torch.float64)
    y = layernorm_reference(x, w, b, eps=1e-5)
    torch.testing.assert_close(y.mean(-1), torch.zeros(64, dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        y.var(-1, unbiased=False), torch.ones(64, dtype=torch.float64), atol=1e-3, rtol=1e-3
    )


def test_affine_recovers_shift_and_scale():
    """weight and bias apply an affine map on the normalized values."""
    x = torch.randn(8, 256, dtype=torch.float64)
    w = torch.full((256,), 2.0, dtype=torch.float64)
    b = torch.full((256,), 5.0, dtype=torch.float64)
    y = layernorm_reference(x, w, b)
    y0 = layernorm_reference(x, torch.ones_like(w), torch.zeros_like(b))
    torch.testing.assert_close(y, y0 * 2.0 + 5.0, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# GPU / interpreter: fused kernel vs. reference                               #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    w = torch.randn(N, device=DEVICE, dtype=dtype)
    b = torch.randn(N, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(layernorm(x, w, b), layernorm_reference(x, w, b), **_tol(dtype))


@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype, requires_grad=True)
    w = torch.randn(N, device=DEVICE, dtype=dtype, requires_grad=True)
    b = torch.randn(N, device=DEVICE, dtype=dtype, requires_grad=True)
    dy = torch.randn(M, N, device=DEVICE, dtype=dtype)

    layernorm(x, w, b).backward(dy)
    dx_t, dw_t, db_t = x.grad.detach().clone(), w.grad.detach().clone(), b.grad.detach().clone()
    x.grad = w.grad = b.grad = None

    layernorm_reference(x, w, b).backward(dy)
    dx_r, dw_r, db_r = x.grad.detach(), w.grad.detach(), b.grad.detach()

    tol = _tol(dtype)
    torch.testing.assert_close(dx_t, dx_r, **tol)
    dwb_tol = tol if dtype == torch.float32 else dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dw_t, dw_r, **dwb_tol)
    torch.testing.assert_close(db_t, db_r, **dwb_tol)


@requires_cuda
def test_module_matches_torch_layernorm():
    """TritonLayerNorm matches torch.nn.LayerNorm on a 3D input."""
    from triton_llm_kernels import TritonLayerNorm

    x = torch.randn(2, 128, 1024, device=DEVICE, dtype=torch.float32)
    ln = TritonLayerNorm(1024).to(DEVICE)
    ref = torch.nn.LayerNorm(1024).to(DEVICE)
    with torch.no_grad():
        ref.weight.copy_(ln.weight)
        ref.bias.copy_(ln.bias)
    torch.testing.assert_close(ln(x), ref(x), atol=1e-4, rtol=1e-4)

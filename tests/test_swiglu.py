"""Correctness tests for the SwiGLU activation kernel.

Two groups, same pattern as the RMSNorm/RoPE tests:
  * Property tests run anywhere (CPU): SiLU-specific values and the gating
    behaviour, checked against the reference.
  * Kernel-vs-reference tests run on GPU, or on CPU under TRITON_INTERPRET=1.
"""

import os

import pytest
import torch

from triton_llm_kernels import swiglu, swiglu_reference

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

SHAPES = [(8, 64), (32, 1024), (4, 11008), (2, 128, 512), (16, 4096)]
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
def test_silu_zero_gate_is_zero():
    """SiLU(0) = 0, so gate=0 zeroes the output regardless of up."""
    gate = torch.zeros(4, 32, dtype=torch.float64)
    up = torch.randn(4, 32, dtype=torch.float64)
    out = swiglu_reference(gate, up)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-12, rtol=0)


def test_gate_acts_as_up_for_large_positive():
    """For large positive gate, SiLU(gate) -> gate, so out -> gate * up."""
    gate = torch.full((4, 32), 30.0, dtype=torch.float64)
    up = torch.randn(4, 32, dtype=torch.float64)
    out = swiglu_reference(gate, up)
    torch.testing.assert_close(out, gate * up, atol=1e-6, rtol=1e-6)


def test_gate_suppresses_for_large_negative():
    """For large negative gate, SiLU(gate) -> 0, so the output is suppressed."""
    gate = torch.full((4, 32), -30.0, dtype=torch.float64)
    up = torch.randn(4, 32, dtype=torch.float64)
    out = swiglu_reference(gate, up)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-10, rtol=0)


def test_reference_matches_functional_silu():
    """Reference equals the textbook F.silu(gate) * up (fp32-level precision:
    the reference computes in fp32 by design, to mirror the kernel)."""
    gate = torch.randn(8, 128, dtype=torch.float64)
    up = torch.randn(8, 128, dtype=torch.float64)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(swiglu_reference(gate, up), expected, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# GPU / interpreter: fused kernel vs. reference                               #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_reference(shape, dtype):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device=DEVICE, dtype=dtype)
    up = torch.randn(*shape, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(swiglu(gate, up), swiglu_reference(gate, up), **_tol(dtype))


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_reference(shape, dtype):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    up = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    dout = torch.randn(*shape, device=DEVICE, dtype=dtype)

    swiglu(gate, up).backward(dout)
    dg_tri, du_tri = gate.grad.detach().clone(), up.grad.detach().clone()
    gate.grad = up.grad = None

    swiglu_reference(gate, up).backward(dout)
    dg_ref, du_ref = gate.grad.detach(), up.grad.detach()

    tol = _tol(dtype)
    torch.testing.assert_close(dg_tri, dg_ref, **tol)
    torch.testing.assert_close(du_tri, du_ref, **tol)


@requires_cuda
def test_full_mlp_runs():
    """The full SwiGLUMLP (fused activation between matmuls) produces finite output."""
    from triton_llm_kernels import SwiGLUMLP

    mlp = SwiGLUMLP(d_model=512, d_ff=1376).to(DEVICE)
    x = torch.randn(2, 16, 512, device=DEVICE)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

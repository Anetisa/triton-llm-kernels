"""Correctness tests for FlashAttention forward.

Two groups, same pattern as the other kernels:
  * Property tests run anywhere (CPU): non-causal uniform attention averages V;
    the causal mask zeroes out future positions.
  * Kernel tests run on GPU, or on CPU under TRITON_INTERPRET=1, and check the
    kernel against both a hand reference and PyTorch's scaled_dot_product_attention,
    across causal/non-causal and sequence lengths that are not multiples of the
    block size.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

from triton_llm_kernels import attention_reference, flash_attention, flash_attention_forward

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

# small blocks so multi-block paths run even for modest S; include non-multiples
BLOCK = 32
# (B, H, S, D)
SHAPES = [(1, 2, 64, 64), (2, 4, 128, 64), (1, 1, 100, 64), (1, 2, 96, 32), (2, 2, 200, 64)]
DTYPES = [torch.float32, torch.float16]


def _tol(dtype):
    return {
        torch.float32: dict(atol=2e-4, rtol=2e-4),
        torch.float16: dict(atol=2e-2, rtol=2e-2),
    }[dtype]


# --------------------------------------------------------------------------- #
# CPU property tests (reference only)                                         #
# --------------------------------------------------------------------------- #
def test_noncausal_uniform_scores_average_v():
    """If all scores are equal, softmax is uniform and output is mean(V)."""
    B, H, S, D = 1, 1, 8, 4
    q = torch.zeros(B, H, S, D, dtype=torch.float64)  # zero q -> all scores 0
    k = torch.randn(B, H, S, D, dtype=torch.float64)
    v = torch.randn(B, H, S, D, dtype=torch.float64)
    out = attention_reference(q, k, v, causal=False)
    torch.testing.assert_close(
        out, v.mean(dim=-2, keepdim=True).expand_as(out), atol=1e-5, rtol=1e-5
    )


def test_causal_first_row_attends_only_to_itself():
    """Under causal masking, query 0 sees only key 0, so out[0] = v[0]."""
    B, H, S, D = 1, 1, 6, 4
    q = torch.zeros(B, H, S, D, dtype=torch.float64)
    k = torch.randn(B, H, S, D, dtype=torch.float64)
    v = torch.randn(B, H, S, D, dtype=torch.float64)
    out = attention_reference(q, k, v, causal=True)
    torch.testing.assert_close(out[:, :, 0], v[:, :, 0], atol=1e-5, rtol=1e-5)


def test_reference_matches_sdpa():
    """Hand reference agrees with PyTorch's scaled_dot_product_attention."""
    B, H, S, D = 2, 2, 32, 16
    q = torch.randn(B, H, S, D, dtype=torch.float64)
    k = torch.randn(B, H, S, D, dtype=torch.float64)
    v = torch.randn(B, H, S, D, dtype=torch.float64)
    ref = attention_reference(q, k, v, causal=True)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(ref, sdpa, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# GPU / interpreter: kernel vs reference + SDPA                                #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("B,H,S,D", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_reference(B, H, S, D, causal, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    out = flash_attention_forward(q, k, v, causal=causal, block_m=BLOCK, block_n=BLOCK)
    ref = attention_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, **_tol(dtype))


@requires_cuda
@pytest.mark.parametrize("causal", [True, False])
def test_forward_matches_sdpa(causal):
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
    out = flash_attention_forward(q, k, v, causal=causal, block_m=BLOCK, block_n=BLOCK)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, sdpa, atol=2e-4, rtol=2e-4)


@requires_cuda
@pytest.mark.parametrize("B,H,S,D", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_backward_matches_reference(B, H, S, D, causal):
    """dQ, dK, dV from the Triton backward match autograd through the reference."""
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    do = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)

    flash_attention(q, k, v, causal=causal, block_m=BLOCK, block_n=BLOCK).backward(do)
    dq_t, dk_t, dv_t = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()
    q.grad = k.grad = v.grad = None

    attention_reference(q, k, v, causal=causal).backward(do)
    dq_r, dk_r, dv_r = q.grad.detach(), k.grad.detach(), v.grad.detach()

    torch.testing.assert_close(dq_t, dq_r, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dk_t, dk_r, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dv_t, dv_r, atol=2e-4, rtol=2e-4)


@requires_cuda
def test_backward_matches_sdpa():
    """Full grads match differentiating PyTorch's scaled_dot_product_attention."""
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    do = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)

    flash_attention(q, k, v, causal=True, block_m=BLOCK, block_n=BLOCK).backward(do)
    dq_t, dk_t, dv_t = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()
    q.grad = k.grad = v.grad = None

    F.scaled_dot_product_attention(q, k, v, is_causal=True).backward(do)
    dq_s, dk_s, dv_s = q.grad.detach(), k.grad.detach(), v.grad.detach()

    torch.testing.assert_close(dq_t, dq_s, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dk_t, dk_s, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dv_t, dv_s, atol=2e-4, rtol=2e-4)

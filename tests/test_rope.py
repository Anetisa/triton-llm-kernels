"""Correctness tests for the RoPE kernel.

Two groups, same pattern as the RMSNorm tests:
  * Property tests run anywhere (CPU): position-0 identity, norm preservation,
    and the relative-position (translation-invariance) property that defines
    RoPE. These pin down correctness without an external reference.
  * Kernel-vs-reference tests run on GPU, or on CPU under TRITON_INTERPRET=1.
"""

import os

import pytest
import torch

from triton_llm_kernels import build_rope_cache, rope, rope_reference
from triton_llm_kernels.rope import _rotate_half

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() or INTERPRET),
    reason="needs a CUDA GPU, or set TRITON_INTERPRET=1 to run on CPU",
)

SHAPES = [(2, 16, 4, 64), (1, 32, 8, 128), (3, 8, 2, 32), (2, 128, 4, 128)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype):
    return {
        torch.float32: dict(atol=1e-5, rtol=1e-5),
        torch.float16: dict(atol=1e-2, rtol=1e-2),
        torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    }[dtype]


# --------------------------------------------------------------------------- #
# CPU property tests -- fp64 cache to isolate the math from float32 precision  #
# --------------------------------------------------------------------------- #
def _cache_fp64(S, D, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    freqs = torch.outer(torch.arange(S, dtype=torch.float64), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate(x, cos, sin):
    return x * cos[None, :, None, :] + _rotate_half(x) * sin[None, :, None, :]


def test_position_zero_is_identity():
    """At position 0, cos=1 and sin=0, so RoPE leaves the vector unchanged."""
    B, S, H, D = 2, 8, 4, 64
    x = torch.randn(B, S, H, D, dtype=torch.float64)
    cos, sin = _cache_fp64(S, D)
    y = _rotate(x, cos, sin)
    torch.testing.assert_close(y[:, 0], x[:, 0], atol=1e-12, rtol=0)


def test_norm_preservation():
    """RoPE is an orthogonal rotation, so every vector keeps its L2 norm."""
    B, S, H, D = 2, 16, 4, 64
    x = torch.randn(B, S, H, D, dtype=torch.float64)
    cos, sin = _cache_fp64(S, D)
    y = _rotate(x, cos, sin)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), atol=1e-12, rtol=0)


def test_relative_position_property():
    """<RoPE(q,m), RoPE(k,n)> depends only on (m-n): shifting both by d is invariant."""
    S, D = 32, 64
    cos, sin = _cache_fp64(S, D)
    q0 = torch.randn(D, dtype=torch.float64)
    k0 = torch.randn(D, dtype=torch.float64)
    q = q0.view(1, 1, 1, D).expand(1, S, 1, D).contiguous()
    k = k0.view(1, 1, 1, D).expand(1, S, 1, D).contiguous()
    Q, K = _rotate(q, cos, sin)[0, :, 0], _rotate(k, cos, sin)[0, :, 0]
    m, n, d = 5, 2, 7
    lhs = (Q[m] * K[n]).sum()
    rhs = (Q[m + d] * K[n + d]).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-11, rtol=0)


# --------------------------------------------------------------------------- #
# GPU / interpreter: fused kernel vs. reference                               #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("B,S,H,D", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_reference(B, S, H, D, dtype):
    torch.manual_seed(0)
    x = torch.randn(B, S, H, D, device=DEVICE, dtype=dtype)
    cos, sin = build_rope_cache(S, D, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(rope(x, cos, sin), rope_reference(x, cos, sin), **_tol(dtype))


@requires_cuda
@pytest.mark.parametrize("B,S,H,D", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_reference(B, S, H, D, dtype):
    torch.manual_seed(0)
    x = torch.randn(B, S, H, D, device=DEVICE, dtype=dtype, requires_grad=True)
    cos, sin = build_rope_cache(S, D, device=DEVICE, dtype=dtype)
    dout = torch.randn(B, S, H, D, device=DEVICE, dtype=dtype)

    rope(x, cos, sin).backward(dout)
    dx_tri = x.grad.detach().clone()
    x.grad = None

    rope_reference(x, cos, sin).backward(dout)
    dx_ref = x.grad.detach()

    torch.testing.assert_close(dx_tri, dx_ref, **_tol(dtype))


@requires_cuda
def test_module_forward():
    """The TritonRoPE nn.Module builds its own cache and matches the reference."""
    from triton_llm_kernels import TritonRoPE

    B, S, H, D = 2, 64, 8, 128
    x = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16)
    layer = TritonRoPE(head_dim=D, max_seq_len=128).to(DEVICE)
    cos, sin = build_rope_cache(128, D, device=DEVICE, dtype=torch.float16)
    torch.testing.assert_close(layer(x), rope_reference(x, cos[:S], sin[:S]), **_tol(torch.float16))

"""Fused Rotary Position Embedding (RoPE) in Triton (forward + backward).

RoPE (Su et al., 2021, "RoFormer") injects position information by *rotating*
pairs of features by a position-dependent angle, instead of adding a learned
positional vector. It is used by LLaMA, Mistral, Qwen, and most modern LLMs, and
is applied to the query and key projections inside attention.

Convention
----------
This implements the "rotate_half" (GPT-J / HuggingFace-LLaMA) layout: the
head_dim `D` is split into two halves and feature `i` is paired with feature
`i + D/2`. For a token at position `m` and pair `i` with angle
`m·θ_i` (θ_i = base^(-2i/D)):

    out[i]        = x[i]·cos_i − x[i+D/2]·sin_i
    out[i+D/2]    = x[i+D/2]·cos_i + x[i]·sin_i

Equivalently, with rotate_half(x) = cat(−x[D/2:], x[:D/2]):

    out = x·cos + rotate_half(x)·sin        (this is exactly HF's formula)

Backward
--------
Each pair is multiplied by an orthogonal 2x2 rotation matrix R = [[c,−s],[s,c]].
Its transpose R^T = [[c,s],[−s,c]] is the rotation by the *negative* angle, so
the gradient is just the inverse rotation -- no reduction, no atomics:

    dx[i]      = dout[i]·cos_i    + dout[i+D/2]·sin_i
    dx[i+D/2]  = dout[i+D/2]·cos_i − dout[i]·sin_i

cos/sin depend only on position (not on learnable params), so they get no grad.
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
# cos/sin cache                                                               #
# --------------------------------------------------------------------------- #
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device="cpu",
    dtype: torch.dtype = torch.float32,
):
    """Precompute cos/sin tables of shape [seq_len, head_dim].

    Frequencies are duplicated across the two halves (HF layout), so the tables
    can be used directly by `rope`/`rope_reference`.
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )                                              # [D/2]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)               # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [S, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


# --------------------------------------------------------------------------- #
# Reference (pure PyTorch, autograd-native, runs on CPU)                      #
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE ground truth. x: [B, S, H, D]; cos/sin: [S, D]. Matches HF LLaMA."""
    input_dtype = x.dtype
    xf = x.float()
    cosf = cos.float()[None, :, None, :]           # [1, S, 1, D]
    sinf = sin.float()[None, :, None, :]
    out = xf * cosf + _rotate_half(xf) * sinf
    return out.to(input_dtype)


if HAS_TRITON:

    @triton.jit
    def _rope_fwd_kernel(
        X, OUT, COS, SIN,
        stride_row,
        S, H, HALF,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)          # one program per (b, s, h) row
        seq = (pid // H) % S            # position of this row, for [B,S,H,D] layout
        X += pid * stride_row
        OUT += pid * stride_row
        COS += seq * (2 * HALF)
        SIN += seq * (2 * HALF)

        cols = tl.arange(0, BLOCK)
        mask = cols < HALF

        x1 = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X + HALF + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(COS + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SIN + cols, mask=mask, other=0.0).to(tl.float32)

        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        tl.store(OUT + cols, o1.to(OUT.dtype.element_ty), mask=mask)
        tl.store(OUT + HALF + cols, o2.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _rope_bwd_kernel(
        DOUT, DX, COS, SIN,
        stride_row,
        S, H, HALF,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        seq = (pid // H) % S
        DOUT += pid * stride_row
        DX += pid * stride_row
        COS += seq * (2 * HALF)
        SIN += seq * (2 * HALF)

        cols = tl.arange(0, BLOCK)
        mask = cols < HALF

        d1 = tl.load(DOUT + cols, mask=mask, other=0.0).to(tl.float32)
        d2 = tl.load(DOUT + HALF + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(COS + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SIN + cols, mask=mask, other=0.0).to(tl.float32)

        # inverse rotation (transpose of the forward rotation matrix)
        dx1 = d1 * c + d2 * s
        dx2 = d2 * c - d1 * s
        tl.store(DX + cols, dx1.to(DX.dtype.element_ty), mask=mask)
        tl.store(DX + HALF + cols, dx2.to(DX.dtype.element_ty), mask=mask)

    class _RoPEFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin):
            B, S, H, D = x.shape
            assert D % 2 == 0, "head_dim must be even"
            x2d = x.contiguous().view(-1, D)
            cos_c, sin_c = cos.contiguous(), sin.contiguous()

            out = torch.empty_like(x2d)
            HALF = D // 2
            BLOCK = triton.next_power_of_2(HALF)
            num_warps = max(1, min(16, BLOCK // 128))

            _rope_fwd_kernel[(x2d.shape[0],)](
                x2d, out, cos_c, sin_c, x2d.stride(0), S, H, HALF,
                BLOCK=BLOCK, num_warps=num_warps,
            )
            ctx.save_for_backward(cos_c, sin_c)
            ctx.shape = (B, S, H, D)
            ctx.BLOCK = BLOCK
            ctx.num_warps = num_warps
            return out.view(B, S, H, D)

        @staticmethod
        def backward(ctx, dout):
            cos_c, sin_c = ctx.saved_tensors
            B, S, H, D = ctx.shape
            HALF = D // 2
            dout2d = dout.contiguous().view(-1, D)
            dx = torch.empty_like(dout2d)

            _rope_bwd_kernel[(dout2d.shape[0],)](
                dout2d, dx, cos_c, sin_c, dout2d.stride(0), S, H, HALF,
                BLOCK=ctx.BLOCK, num_warps=ctx.num_warps,
            )
            # no grad for cos / sin (position-derived, not parameters)
            return dx.view(B, S, H, D), None, None


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused RoPE. x: [B, S, H, D]; cos/sin: [S, D] from build_rope_cache().

    Requires a CUDA tensor, or TRITON_INTERPRET=1 for CPU debugging.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use rope_reference() on CPU, or install "
            "triton (bundled with PyTorch on Linux) and run on a CUDA GPU."
        )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not x.is_cuda and not interpret:
        raise RuntimeError(
            "rope() runs a Triton GPU kernel and needs a CUDA tensor. "
            "For CPU debugging use rope_reference(), or set TRITON_INTERPRET=1."
        )
    return _RoPEFn.apply(x, cos, sin)


class TritonRoPE(torch.nn.Module):
    """Builds and caches cos/sin, then applies fused RoPE to [B, S, H, D] input."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        cos, sin = build_rope_cache(max_seq_len, head_dim, base=base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        return rope(x, self.cos[:S], self.sin[:S])

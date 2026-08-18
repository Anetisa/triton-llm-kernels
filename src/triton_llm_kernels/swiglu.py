"""Fused SwiGLU activation in Triton (forward + backward).

SwiGLU (Shazeer, 2020) is the gated MLP activation used by LLaMA, Mistral, Qwen,
PaLM, and most modern LLMs. The full feed-forward block is

    FFN(x) = W_down @ ( SiLU(W_gate @ x) * (W_up @ x) )

The three matmuls are best left to cuBLAS/tensor cores; the *fusable*,
memory-bound piece is the middle activation, given the two projections
`gate = W_gate @ x` and `up = W_up @ x` (both [.., d_ff]):

    swiglu(gate, up) = SiLU(gate) * up,      SiLU(z) = z · sigmoid(z)

Eager PyTorch (`F.silu(gate) * up`) materializes the SiLU intermediate and
launches several kernels; the fused kernel reads gate/up once, computes in
registers, and writes once.

Backward
--------
With s = sigmoid(gate) and silu = gate·s:

    d(up)   = dout · silu
    d(gate) = dout · up · silu'(gate),   silu'(z) = s·(1 + z·(1 − s))

Both are pure elementwise -- no reductions, no atomics.
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
def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation ground truth: SiLU(gate) * up. Computes in fp32."""
    input_dtype = gate.dtype
    out = torch.nn.functional.silu(gate.float()) * up.float()
    return out.to(input_dtype)


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(GATE, UP, OUT, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(g)
        out = (g * s) * u                      # SiLU(g) * u
        tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        GATE, UP, DOUT, DGATE, DUP, n_elements, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DOUT + offs, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(g)
        silu = g * s
        # silu'(z) = s * (1 + z*(1 - s))
        dsilu = s * (1.0 + g * (1.0 - s))

        dgate = do * u * dsilu
        dup = do * silu
        tl.store(DGATE + offs, dgate.to(DGATE.dtype.element_ty), mask=mask)
        tl.store(DUP + offs, dup.to(DUP.dtype.element_ty), mask=mask)

    class _SwiGLUFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            assert gate.shape == up.shape, "gate and up must have the same shape"
            gate_c = gate.contiguous()
            up_c = up.contiguous()
            out = torch.empty_like(gate_c)
            n = gate_c.numel()
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)

            _swiglu_fwd_kernel[grid](gate_c, up_c, out, n, BLOCK=BLOCK)
            ctx.save_for_backward(gate_c, up_c)
            ctx.BLOCK = BLOCK
            return out

        @staticmethod
        def backward(ctx, dout):
            gate_c, up_c = ctx.saved_tensors
            dout_c = dout.contiguous()
            dgate = torch.empty_like(gate_c)
            dup = torch.empty_like(up_c)
            n = gate_c.numel()
            grid = (triton.cdiv(n, ctx.BLOCK),)

            _swiglu_bwd_kernel[grid](
                gate_c, up_c, dout_c, dgate, dup, n, BLOCK=ctx.BLOCK
            )
            return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation: SiLU(gate) * up.

    Requires a CUDA tensor, or TRITON_INTERPRET=1 for CPU debugging.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use swiglu_reference() on CPU, or install "
            "triton (bundled with PyTorch on Linux) and run on a CUDA GPU."
        )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not gate.is_cuda and not interpret:
        raise RuntimeError(
            "swiglu() runs a Triton GPU kernel and needs a CUDA tensor. "
            "For CPU debugging use swiglu_reference(), or set TRITON_INTERPRET=1."
        )
    return _SwiGLUFn.apply(gate, up)


class TritonSwiGLU(torch.nn.Module):
    """Fused SwiGLU activation as a Module. Takes (gate, up), returns SiLU(gate)*up."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return swiglu(gate, up)


class SwiGLUMLP(torch.nn.Module):
    """A full LLaMA-style SwiGLU MLP using the fused activation.

    FFN(x) = W_down @ ( SiLU(W_gate @ x) * (W_up @ x) )
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = torch.nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = torch.nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

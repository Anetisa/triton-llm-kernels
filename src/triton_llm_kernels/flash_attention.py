"""FlashAttention forward in Triton (online-softmax, causal + non-causal).

Standard attention forms the full S×S score matrix:

    S = Q·Kᵀ · scale ;  P = softmax(S) ;  O = P·V

which costs O(S²) memory. FlashAttention (Dao et al., 2022) never materializes S
or P: it tiles over key/value blocks and keeps a running softmax per query row —
a running max `m`, running normalizer `l`, and running output `acc` — rescaling
`acc` and `l` by `exp(m_old − m_new)` as the max grows. Memory drops from O(S²)
to O(S), and the whole thing stays in registers/SRAM.

This module implements the **forward** pass (inference). Backward is a separate,
larger follow-up. Layout is [B, H, S, D]; scale defaults to 1/√D; causal masking
is supported.

Online-softmax recurrence (per query row, per key block s_j):
    m_new = max(m, rowmax(s_j))
    p     = exp(s_j − m_new)
    alpha = exp(m − m_new)                 # correction for the running state
    l     = l·alpha + rowsum(p)
    acc   = acc·alpha + p·V_block
    m     = m_new
Final: O = acc / l.
"""

from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# Reference (pure PyTorch, runs on CPU)                                        #
# --------------------------------------------------------------------------- #
def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, scale=None
) -> torch.Tensor:
    """Plain attention ground truth. q,k,v: [B, H, S, D]. Computes in fp32."""
    input_dtype = q.dtype
    qf, kf, vf = q.float(), k.float(), v.float()
    D = qf.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    s = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        Sq, Sk = qf.shape[-2], kf.shape[-2]
        mask = torch.arange(Sk, device=q.device)[None, :] > torch.arange(Sq, device=q.device)[:, None]
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.matmul(p, vf)
    return out.to(input_dtype)


if HAS_TRITON:

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O,
        stride_qbh, stride_qs, stride_qd,
        stride_kbh, stride_ks, stride_kd,
        stride_vbh, stride_vs, stride_vd,
        stride_obh, stride_os, stride_od,
        S, scale,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)          # query block index
        bh = tl.program_id(1)             # flattened (batch, head) index

        # base pointers for this (batch, head) slice of the [B*H, S, D] view
        q_base = Q + bh * stride_qbh
        k_base = K + bh * stride_kbh
        v_base = V + bh * stride_vbh
        o_base = O + bh * stride_obh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        # load Q block [BLOCK_M, D]
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < S
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)

        # causal: only iterate key blocks up to the diagonal of this query block
        hi = (pid_m + 1) * BLOCK_M if CAUSAL else S

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            n_mask = offs_n[:, None] < S
            k = tl.load(k_ptrs, mask=n_mask, other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * scale        # [BLOCK_M, BLOCK_N]

            # validity mask: in-range keys, and (causal) key <= query
            valid = offs_n[None, :] < S
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(valid, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            vblk = tl.load(v_ptrs, mask=n_mask, other=0.0).to(tl.float32)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vblk.dtype), vblk)
            m_i = m_new

        out = acc / l_i[:, None]
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=q_mask)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """FlashAttention forward. q,k,v: [B, H, S, D]. Returns [B, H, S, D].

    Forward/inference only for now (no autograd). Requires a CUDA tensor, or
    TRITON_INTERPRET=1 for CPU debugging.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use attention_reference() on CPU, or install "
            "triton and run on a CUDA GPU."
        )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not q.is_cuda and not interpret:
        raise RuntimeError(
            "flash_attention_forward() runs a Triton GPU kernel and needs a CUDA "
            "tensor. For CPU debugging use attention_reference(), or set "
            "TRITON_INTERPRET=1."
        )

    B, H, S, D = q.shape
    assert k.shape == v.shape == (B, H, S, D), "q, k, v must share shape [B,H,S,D]"
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    # flatten (B, H) -> BH so the kernel indexes heads with a single stride
    qf = q.view(B * H, S, D)
    kf = k.view(B * H, S, D)
    vf = v.view(B * H, S, D)
    of = o.view(B * H, S, D)

    grid = (triton.cdiv(S, block_m), B * H)
    _flash_fwd_kernel[grid](
        qf, kf, vf, of,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        vf.stride(0), vf.stride(1), vf.stride(2),
        of.stride(0), of.stride(1), of.stride(2),
        S, scale,
        CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
    )
    return o

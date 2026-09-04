"""FlashAttention forward in Triton (online-softmax, causal + non-causal).

Standard attention forms the full S×S score matrix:

    S = Q·Kᵀ · scale ;  P = softmax(S) ;  O = P·V

which costs O(S²) memory. FlashAttention (Dao et al., 2022) never materializes S
or P: it tiles over key/value blocks and keeps a running softmax per query row —
a running max `m`, running normalizer `l`, and running output `acc` — rescaling
`acc` and `l` by `exp(m_old − m_new)` as the max grows. Memory drops from O(S²)
to O(S), and the whole thing stays in registers/SRAM.

This module implements **forward and backward** (training-ready) and supports
**GQA/MQA**: Q has H heads while K/V have H_kv heads (H_kv divides H), so each KV
head is shared by H/H_kv query heads. MQA is H_kv=1; MHA is H_kv=H. Layout is
[B, H, S, D] for Q and [B, H_kv, S, D] for K/V; scale defaults to 1/√D; causal
masking is supported.

The backward reuses the saved log-sum-exp `L` from the forward to recompute `P`
on the fly, and uses the identity `delta_i = Σ_d O_id·dO_id` to collapse the
softmax-gradient term. It runs as three passes with no atomics: a delta
preprocess, a dQ kernel parallelized over query blocks, and a dK/dV kernel
parallelized over key/value blocks.

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
    """Plain attention ground truth. q: [B, H, S, D]; k, v: [B, H_kv, S, D].

    Supports GQA/MQA: if H_kv < H, each KV head is shared by H // H_kv query
    heads (query head h uses KV head h // (H // H_kv)). Computes in fp32.
    """
    input_dtype = q.dtype
    qf, kf, vf = q.float(), k.float(), v.float()
    H, H_kv = qf.shape[1], kf.shape[1]
    if H_kv != H:
        assert H % H_kv == 0, "H must be divisible by H_kv"
        group = H // H_kv
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
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
        Q, K, V, O, L,
        stride_qbh, stride_qs, stride_qd,
        stride_kbh, stride_ks, stride_kd,
        stride_vbh, stride_vs, stride_vd,
        stride_obh, stride_os, stride_od,
        stride_lbh, stride_ls,
        S, scale,
        H: tl.constexpr, H_KV: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)          # query block index
        bh = tl.program_id(1)             # flattened (batch, query-head) index

        # GQA/MQA: map this query head to its shared KV head.
        b = bh // H
        h = bh % H
        bh_kv = b * H_KV + h // (H // H_KV)

        q_base = Q + bh * stride_qbh
        k_base = K + bh_kv * stride_kbh
        v_base = V + bh_kv * stride_vbh
        o_base = O + bh * stride_obh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        # load Q block [BLOCK_M, D]
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < S
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)

        # causal: only iterate key blocks up to the diagonal of this query block
        hi = (pid_m + 1) * BLOCK_M if CAUSAL else S

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            n_mask = offs_n[:, None] < S
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale        # [BLOCK_M, BLOCK_N]

            # validity mask: in-range keys, and (causal) key <= query
            valid = offs_n[None, :] < S
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(valid, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            vblk = tl.load(v_ptrs, mask=n_mask, other=0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vblk.dtype), vblk, input_precision="ieee")
            m_i = m_new

        out = acc / l_i[:, None]
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=q_mask)

        # save log-sum-exp per query row (L = m + log l), needed by the backward
        l_ptrs = L + bh * stride_lbh + offs_m * stride_ls
        L_row = m_i + tl.log(l_i)
        tl.store(l_ptrs, L_row, mask=offs_m < S)


    # ----------------------------------------------------------------------- #
    # Backward kernels                                                        #
    # ----------------------------------------------------------------------- #
    @triton.jit
    def _bwd_preprocess_kernel(
        O, DO, Delta,
        stride_obh, stride_os, stride_od,
        stride_dbh, stride_ds,
        S, BLOCK_M: tl.constexpr, D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask = offs_m[:, None] < S

        o = tl.load(O + bh * stride_obh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                    mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DO + bh * stride_obh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                     mask=mask, other=0.0).to(tl.float32)
        delta = tl.sum(o * do, axis=1)                      # per-row scalar
        tl.store(Delta + bh * stride_dbh + offs_m * stride_ds, delta, mask=offs_m < S)

    @triton.jit
    def _bwd_dq_kernel(
        Q, K, V, DO, DQ, L, Delta,
        stride_qbh, stride_qs, stride_qd,
        stride_lbh, stride_ls,
        S, scale,
        H: tl.constexpr, H_KV: tl.constexpr, CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)          # query block -> owns dQ rows (no atomics)
        bh = tl.program_id(1)
        # GQA/MQA: this query head reads its shared KV head.
        b = bh // H
        bh_kv = b * H_KV + (bh % H) // (H // H_KV)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_base = Q + bh * stride_qbh
        k_base = K + bh_kv * stride_qbh
        v_base = V + bh_kv * stride_qbh
        do_base = DO + bh * stride_qbh

        m_mask = offs_m[:, None] < S
        q = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                    mask=m_mask, other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                     mask=m_mask, other=0.0)
        L_i = tl.load(L + bh * stride_lbh + offs_m * stride_ls, mask=offs_m < S, other=0.0)
        delta_i = tl.load(Delta + bh * stride_lbh + offs_m * stride_ls, mask=offs_m < S, other=0.0)

        dq = tl.zeros((BLOCK_M, D), tl.float32)
        hi = (pid_m + 1) * BLOCK_M if CAUSAL else S
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n[:, None] < S
            k = tl.load(k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                        mask=n_mask, other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                        mask=n_mask, other=0.0)
            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = offs_n[None, :] < S
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(valid, tl.exp(qk - L_i[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")                    # dO_i · V_j
            ds = p * (dp - delta_i[:, None])
            dq += scale * tl.dot(ds.to(k.dtype), k, input_precision="ieee")
        tl.store(DQ + bh * stride_qbh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                 dq.to(DQ.dtype.element_ty), mask=m_mask)

    @triton.jit
    def _bwd_dkv_kernel(
        Q, K, V, DO, DK, DV, L, Delta,
        stride_qbh, stride_qs, stride_qd,
        stride_lbh, stride_ls,
        S, scale,
        H: tl.constexpr, H_KV: tl.constexpr, GROUP: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)          # kv block -> owns dK,dV rows (no atomics)
        bh_kv = tl.program_id(1)          # flattened (batch, kv-head) index
        b = bh_kv // H_KV
        kv_head = bh_kv % H_KV
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        # K, V for this kv head/block (loaded once, reused across the whole group)
        k_base = K + bh_kv * stride_qbh
        v_base = V + bh_kv * stride_qbh
        n_mask = offs_n[:, None] < S
        k = tl.load(k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                    mask=n_mask, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                    mask=n_mask, other=0.0)

        dk = tl.zeros((BLOCK_N, D), tl.float32)
        dv = tl.zeros((BLOCK_N, D), tl.float32)

        lo = pid_n * BLOCK_N if CAUSAL else 0
        lo = (lo // BLOCK_M) * BLOCK_M

        # GQA/MQA: this kv head receives gradient from GROUP query heads.
        for g in range(GROUP):
            bh_q = b * H + (kv_head * GROUP + g)
            q_base = Q + bh_q * stride_qbh
            do_base = DO + bh_q * stride_qbh
            for start_m in range(lo, S, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                m_mask = offs_m[:, None] < S
                q = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                            mask=m_mask, other=0.0)
                do = tl.load(do_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                             mask=m_mask, other=0.0)
                L_i = tl.load(L + bh_q * stride_lbh + offs_m * stride_ls, mask=offs_m < S, other=0.0)
                delta_i = tl.load(Delta + bh_q * stride_lbh + offs_m * stride_ls, mask=offs_m < S, other=0.0)

                qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale             # [BLOCK_M, BLOCK_N]
                valid = (offs_n[None, :] < S) & (offs_m[:, None] < S)
                if CAUSAL:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])
                p = tl.where(valid, tl.exp(qk - L_i[:, None]), 0.0)

                dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")      # P^T @ dO
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta_i[:, None])
                dk += scale * tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")

        tl.store(DK + bh_kv * stride_qbh + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                 dk.to(DK.dtype.element_ty), mask=n_mask)
        tl.store(DV + bh_kv * stride_qbh + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                 dv.to(DV.dtype.element_ty), mask=n_mask)


def _flash_forward_impl(q, k, v, causal, scale, block_m, block_n):
    """Run the forward kernel, returning (o, L). Internal; assumes contiguous.

    q: [B, H, S, D]; k, v: [B, H_kv, S, D] with H % H_kv == 0 (GQA/MQA).
    """
    B, H, S, D = q.shape
    H_kv = k.shape[1]
    o = torch.empty_like(q)
    L = torch.empty((B * H, S), dtype=torch.float32, device=q.device)
    qf = q.view(B * H, S, D); kf = k.view(B * H_kv, S, D)
    vf = v.view(B * H_kv, S, D); of = o.view(B * H, S, D)
    grid = (triton.cdiv(S, block_m), B * H)
    _flash_fwd_kernel[grid](
        qf, kf, vf, of, L,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        vf.stride(0), vf.stride(1), vf.stride(2),
        of.stride(0), of.stride(1), of.stride(2),
        L.stride(0), L.stride(1),
        S, scale, H=H, H_KV=H_kv, CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
    )
    return o, L


if HAS_TRITON:

    class _FlashAttnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, causal, scale, block_m, block_n):
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            o, L = _flash_forward_impl(q, k, v, causal, scale, block_m, block_n)
            ctx.save_for_backward(q, k, v, o, L)
            ctx.causal = causal; ctx.scale = scale
            ctx.block_m = block_m; ctx.block_n = block_n
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, L = ctx.saved_tensors
            B, H, S, D = q.shape
            H_kv = k.shape[1]
            group = H // H_kv
            causal, scale = ctx.causal, ctx.scale
            block_m, block_n = ctx.block_m, ctx.block_n
            do = do.contiguous()

            dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
            delta = torch.empty((B * H, S), dtype=torch.float32, device=q.device)

            qf = q.view(B * H, S, D); kf = k.view(B * H_kv, S, D); vf = v.view(B * H_kv, S, D)
            of = o.view(B * H, S, D); dof = do.view(B * H, S, D)
            dqf = dq.view(B * H, S, D); dkf = dk.view(B * H_kv, S, D); dvf = dv.view(B * H_kv, S, D)

            # 1) delta_i = rowsum(dO_i * O_i)  -- per query head
            _bwd_preprocess_kernel[(triton.cdiv(S, block_m), B * H)](
                of, dof, delta,
                of.stride(0), of.stride(1), of.stride(2),
                delta.stride(0), delta.stride(1),
                S, BLOCK_M=block_m, D=D,
            )
            # 2) dQ (parallel over query blocks x query heads)
            _bwd_dq_kernel[(triton.cdiv(S, block_m), B * H)](
                qf, kf, vf, dof, dqf, L, delta,
                qf.stride(0), qf.stride(1), qf.stride(2),
                L.stride(0), L.stride(1),
                S, scale, H=H, H_KV=H_kv, CAUSAL=causal,
                BLOCK_M=block_m, BLOCK_N=block_n, D=D,
            )
            # 3) dK, dV (parallel over kv blocks x kv heads; inner loop over the group)
            _bwd_dkv_kernel[(triton.cdiv(S, block_n), B * H_kv)](
                qf, kf, vf, dof, dkf, dvf, L, delta,
                qf.stride(0), qf.stride(1), qf.stride(2),
                L.stride(0), L.stride(1),
                S, scale, H=H, H_KV=H_kv, GROUP=group, CAUSAL=causal,
                BLOCK_M=block_m, BLOCK_N=block_n, D=D,
            )
            return dq, dk, dv, None, None, None, None


def _prepare(q, k, v, causal, scale, block_m, block_n):
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Use attention_reference() on CPU, or install "
            "triton and run on a CUDA GPU."
        )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not q.is_cuda and not interpret:
        raise RuntimeError(
            "flash_attention runs a Triton GPU kernel and needs a CUDA tensor. "
            "For CPU debugging use attention_reference(), or set TRITON_INTERPRET=1."
        )
    B, H, S, D = q.shape
    H_kv = k.shape[1]
    assert k.shape == v.shape == (B, H_kv, S, D), "k, v must share shape [B, H_kv, S, D]"
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv}) for GQA/MQA"
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    return scale


def flash_attention(q, k, v, causal=True, scale=None, block_m=64, block_n=64):
    """Differentiable FlashAttention (forward + backward). q,k,v: [B, H, S, D].

    Supports autograd for dQ, dK, dV. Requires a CUDA tensor, or
    TRITON_INTERPRET=1 for CPU debugging.
    """
    scale = _prepare(q, k, v, causal, scale, block_m, block_n)
    return _FlashAttnFn.apply(q, k, v, causal, scale, block_m, block_n)


def flash_attention_forward(q, k, v, causal=True, scale=None, block_m=64, block_n=64):
    """Forward-only FlashAttention (no autograd graph). q,k,v: [B, H, S, D]."""
    scale = _prepare(q, k, v, causal, scale, block_m, block_n)
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o, _ = _flash_forward_impl(q, k, v, causal, scale, block_m, block_n)
    return o

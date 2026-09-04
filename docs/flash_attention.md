# FlashAttention (forward): online softmax without the S×S matrix

## The problem with standard attention

Standard attention computes

    S = Q·Kᵀ · scale        # [S, S]
    P = softmax(S)          # [S, S]
    O = P·V                 # [S, D]

The score matrix `S` is `S×S`. At 8k context that's 64M entries **per head** —
materializing it dominates memory and bandwidth, and it's the reason naive
attention scales as O(S²) memory. FlashAttention (Dao et al., 2022) computes the
exact same `O` **without ever forming `S` or `P`**.

## Online softmax

Softmax needs the row max (for numerical stability) and the row sum. Normally you
need the whole row at once. The trick is to compute both *incrementally* as you
stream over blocks of keys, keeping three running quantities per query row:

- `m` — running max of the scores seen so far
- `l` — running sum of `exp(score − m)`
- `acc` — running weighted sum of `V`

For each new key block with scores `s_j`:

    m_new = max(m, rowmax(s_j))
    p     = exp(s_j − m_new)
    alpha = exp(m − m_new)          # rescale the old state to the new max
    l     = l·alpha + rowsum(p)
    acc   = acc·alpha + p·V_block
    m     = m_new

The `alpha` factor retro-corrects everything accumulated under the old max when a
larger score appears. After the last block, `O = acc / l`. This is algebraically
identical to a full softmax — verified against plain attention to fp64 precision
in development — but uses O(S) memory instead of O(S²).

## Kernel structure

One Triton program handles one `(batch, head, query-block)`. It loads its
`[BLOCK_M, D]` block of Q once, then loops over key/value blocks:

1. `qk = Q · Kᵀ · scale`  (a `tl.dot`, `[BLOCK_M, BLOCK_N]`)
2. apply masks (see below), then the online-softmax update above
3. `acc += P · V_block`   (a second `tl.dot`)

Q, `m`, `l`, `acc` stay in registers/SRAM across the whole key loop; only K and V
blocks stream in. That data-reuse is why the kernel is fast and memory-light.

## Causal masking, done efficiently

With causal masking, query `i` may attend only to keys `j ≤ i`. Two consequences:

- **Skip future blocks entirely.** For query block starting at `m₀`, key indices
  above `m₀ + BLOCK_M − 1` are all masked, so the loop stops there
  (`hi = (pid_m + 1)·BLOCK_M`) instead of scanning all of `S`. This roughly halves
  the work versus non-causal.
- **Mask the diagonal block element-wise.** Where `j > i` inside a straddling
  block, set the score to `−∞` so `exp` sends it to 0.

Boundary handling (sequence length not a multiple of the block size) uses the
same `−∞` trick for out-of-range keys, and a load mask for the ragged Q/K/V
edges. Both are covered by tests with `S = 100, 96, 200`.

## Correctness

The kernel is checked against two independent oracles — a hand-written reference
and PyTorch's `scaled_dot_product_attention` — for causal and non-causal, fp32
and fp16, and sequence lengths that are not multiples of the block size. The
online-softmax recurrence itself is verified against plain attention to fp64.

## Backward

Given `dO`, we need `dQ`, `dK`, `dV`. The backward reuses two tricks so it never
stores `P` or the `S×S` matrix either:

**Recompute P from L.** The forward saves `L_i = m_i + log(l_i)` (the log-sum-exp
per query row). Then `P_ij = exp(scale·q_i·k_j − L_i)` is recomputed block-by-block
— no need to have kept `P`.

**The delta identity.** The softmax gradient needs
`Σ_j P_ij·dP_ij`. Substituting `dP_ij = dO_i·V_j` collapses this to a cheap
per-row scalar:

    delta_i = Σ_j P_ij (dO_i·V_j) = dO_i · O_i = Σ_d O_id·dO_id

So a small preprocess pass computes `delta = rowsum(O ⊙ dO)`. Then, per block:

    P    = exp(scale·Q·Kᵀ − L)          # recomputed
    dV  += Pᵀ · dO
    dP   = dO · Vᵀ
    dS   = P ⊙ (dP − delta)
    dQ  += scale · dS · K
    dK  += scale · dSᵀ · Q

**No atomics.** Rather than one pass with atomic accumulation, the backward runs
as two independent kernels that each *own* their output rows:

- **dQ kernel** — one program per query block, loops over key blocks (up to the
  causal diagonal). Each program writes its own `dQ` rows.
- **dK/dV kernel** — one program per key/value block, loops over query blocks
  (from the causal diagonal onward). Each program writes its own `dK`, `dV` rows.

This trades a little recompute (P is formed twice) for fully deterministic,
atomic-free accumulation — clean and easy to verify.

## Correctness

Forward is checked against a hand reference and PyTorch's
`scaled_dot_product_attention`. Backward is validated three ways: the full
formula set matches autograd to fp64 precision in the derivation; the kernel's
`dQ/dK/dV` match autograd through both the reference and SDPA to fp32 precision
across causal/non-causal and ragged sequence lengths. (`torch.autograd.gradcheck`
is not used as a gate because the kernel accumulates in fp32, which makes
fp64 finite-difference checking unreliable — expected, not a bug.)

## Scope and roadmap

Forward + backward are done (training-ready). Next:

- **GQA / MQA** — fewer KV heads than Q heads (broadcast KV across query groups).
- **One-pass backward** with atomic dQ accumulation (less recompute), and
  autotuned `BLOCK_M`/`BLOCK_N` per head dim.
- Non-power-of-two `D` and a dropout path.

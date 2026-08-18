# RoPE kernel: the math and why it's fast

## What RoPE computes

Rotary Position Embedding encodes position by **rotating** pairs of features of
each query/key head vector by a position-dependent angle, rather than adding a
learned positional vector. For a token at position `m`, feature pair `i` is
rotated by angle `m·θ_i`, where `θ_i = base^(-2i/D)` (base is usually 10000).
Low `i` rotate slowly (long wavelength), high `i` rotate fast — a spectrum of
frequencies, like a continuous analogue of binary position encoding.

## The rotate_half convention

There are two equivalent layouts for which features form a rotating pair. This
kernel uses the **rotate_half** layout that HuggingFace-LLaMA uses: the head
dim `D` is split in half and feature `i` pairs with feature `i + D/2`. With

    rotate_half(x) = cat(−x[D/2:], x[:D/2])

the forward pass is exactly HF's one-liner:

    out = x·cos + rotate_half(x)·sin

Per pair `(i, i+D/2)` this expands to a 2-D rotation:

    out[i]      = x[i]·cos_i − x[i+D/2]·sin_i
    out[i+D/2]  = x[i+D/2]·cos_i + x[i]·sin_i

where `cos_i`, `sin_i` come from a precomputed `[S, D]` table (frequencies
duplicated across the two halves, so the table drops straight into the formula).

## Why it's memory-bound

Like RMSNorm, RoPE does only a handful of multiply-adds per element, so runtime
is dominated by reading `x` and writing the output — it's bandwidth-bound. The
fused kernel reads each head vector once, rotates it in registers, and writes
once, avoiding the extra passes and kernel launches of an eager
`x*cos + rotate_half(x)*sin`. The right metric is **GB/s**, not FLOP/s.

The cos/sin tables are tiny (`[S, D]`) and reused across batch and heads, so
they stay in cache and don't dominate traffic.

## Backward is an inverse rotation

Each pair is multiplied by an orthogonal 2×2 matrix `R = [[c,−s],[s,c]]`. Its
transpose `Rᵀ = [[c,s],[−s,c]]` is the rotation by the *negative* angle, so the
gradient is just the inverse rotation — no reduction, no atomics, cheaper than
the RMSNorm backward:

    dx[i]      = dout[i]·cos_i    + dout[i+D/2]·sin_i
    dx[i+D/2]  = dout[i+D/2]·cos_i − dout[i]·sin_i

`cos`/`sin` are functions of position, not learnable parameters, so they receive
no gradient.

## Correctness without an external reference

Three intrinsic properties pin down RoPE, and the tests check all three in fp64:

1. **Position 0 is identity** — `cos=1, sin=0`, so the vector is unchanged.
2. **Norm preservation** — a rotation is orthogonal, so `‖out‖ = ‖x‖` per vector.
3. **Relative-position property** — `⟨RoPE(q,m), RoPE(k,n)⟩` depends only on
   `m−n`. Concretely `⟨R(m)q, R(n)k⟩ = qᵀR(n−m)k`, so shifting both positions by
   the same amount leaves the score unchanged. This is *the* reason RoPE works:
   attention scores see relative, not absolute, position.

## Implementation notes

- **Layout.** Input is `[B, S, H, D]`. Each Triton program handles one
  `(b, s, h)` row and recovers its position as `seq = (pid // H) % S`.
- **fp32 accumulation.** Inputs may be fp16/bf16; the rotation is done in fp32
  and cast back on store, mirrored by the reference so they match tightly.
- **Cache precision.** cos/sin are built in fp32 (as in HF). That's the source
  of the ~1e-7 residual vs. a fully-fp64 cache — expected, not a bug.
- **Roadmap.** The interleaved (GPT-NeoX) pair layout and a `position_ids`
  variant for packed/padded sequences are natural follow-ups.

# FP8 GEMM: matmul on Ada's FP8 tensor cores

## Why FP8

Ada (RTX 4090) and Hopper have tensor cores that do **FP8** matmuls at roughly
double the fp16 throughput. FP8 comes in two formats; this kernel uses **e4m3**
(4 exponent, 3 mantissa bits, max magnitude 448) — the one used for weights and
activations in FP8 inference/training. The catch: 3 mantissa bits is very little
precision, so you can't just cast and multiply — you must **scale** the inputs
into fp8 range first.

## Scaling (quantize → matmul → dequantize)

With per-tensor scaling:

    scale_a = amax(A) / 448 ;  A8 = (A / scale_a) as fp8_e4m3
    scale_b = amax(B) / 448 ;  B8 = (B / scale_b) as fp8_e4m3
    C ≈ (A8 @ B8) · (scale_a · scale_b)

The division maps each tensor's largest magnitude onto the fp8 max (448), using
the format's whole dynamic range. The matmul runs in fp8 with **fp32
accumulation** (the tensor core accumulates in fp32, so summing over K doesn't
lose precision), and a final multiply by `scale_a·scale_b` dequantizes back.

Per-tensor scaling is the simplest choice. Per-row (A) / per-column (B) scaling
is more accurate for inputs with uneven magnitudes and is a natural follow-up.

## The kernel

It's a standard tiled GEMM — one program per `[BLOCK_M, BLOCK_N]` output tile,
looping over K in `BLOCK_K` chunks — whose only special ingredient is that
`tl.dot` receives fp8 operands and accumulates into an fp32 tile. Boundary masks
handle M/N/K that aren't multiples of the block. The scale is folded into a
single scalar applied to the fp32 accumulator before the store.

## Why correctness is measured by norm, not element-wise

FP8 is **lossy by construction**. The Frobenius relative error
`‖C − C_ref‖ / ‖C_ref‖` for well-scaled random inputs is ~3–4% — that's the
expected precision of e4m3, not a bug. Element-wise `assert_close` is the *wrong*
test here: wherever the true product is near zero, the relative error explodes to
huge values even when the matmul is exactly right. So the tests assert a
norm-based relative error below a few percent, and separately check the
quantize/dequantize round-trip and scale-invariance.

## Scope, and honesty about performance

This is the only **compute-bound** kernel in the repo. It competes with cuBLAS,
which is extremely tuned — the goal here is **not** to beat it, but to exercise
the FP8 tensor cores correctly and measure the TFLOP/s a straightforward tiled
kernel reaches. Real speedups from FP8 come with autotuning (block sizes, num
stages/warps, pipelining) and better scaling granularity, which are follow-ups.

## Roadmap

- Per-row / per-column scaling for higher accuracy on skewed inputs.
- Autotuned block sizes and software pipelining (`num_stages`).
- e5m2 format and a mixed e4m3/e5m2 path (weights vs gradients).
- Fused dequant + epilogue (bias, activation).

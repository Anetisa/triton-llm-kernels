"""triton-llm-kernels: fused Triton kernels for LLM building blocks."""

from .rmsnorm import (
    TritonRMSNorm,
    rmsnorm,
    rmsnorm_reference,
)
from .rope import (
    TritonRoPE,
    build_rope_cache,
    rope,
    rope_reference,
)
from .swiglu import (
    SwiGLUMLP,
    TritonSwiGLU,
    swiglu,
    swiglu_reference,
)
from .layernorm import (
    TritonLayerNorm,
    layernorm,
    layernorm_reference,
)
from .flash_attention import (
    attention_reference,
    flash_attention,
    flash_attention_forward,
)
from .fp8_gemm import (
    fp8_gemm,
    gemm_reference,
    quantize_fp8,
)

__all__ = [
    "rmsnorm",
    "rmsnorm_reference",
    "TritonRMSNorm",
    "rope",
    "rope_reference",
    "build_rope_cache",
    "TritonRoPE",
    "swiglu",
    "swiglu_reference",
    "TritonSwiGLU",
    "SwiGLUMLP",
    "layernorm",
    "layernorm_reference",
    "TritonLayerNorm",
    "flash_attention_forward",
    "flash_attention",
    "attention_reference",
    "fp8_gemm",
    "gemm_reference",
    "quantize_fp8",
]

__version__ = "0.8.0"

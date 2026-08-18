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
]

__version__ = "0.3.0"

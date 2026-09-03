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
    flash_attention_forward,
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
    "attention_reference",
]

__version__ = "0.5.0"

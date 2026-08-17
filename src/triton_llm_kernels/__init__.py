"""triton-llm-kernels: fused Triton kernels for LLM building blocks."""

from .rmsnorm import (
    TritonRMSNorm,
    rmsnorm,
    rmsnorm_reference,
)

__all__ = [
    "rmsnorm",
    "rmsnorm_reference",
    "TritonRMSNorm",
]

__version__ = "0.1.0"

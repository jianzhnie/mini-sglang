"""Attention backend dispatcher.

Static router that selects the attention implementation from the configured
backend name:

- "fa": FlashAttention (if installed), else PyTorch SDPA fallback.
- "pt": always PyTorch SDPA (works on CUDA, NPU, CPU).
"""

__all__ = ["AttentionBackend"]
import torch

from minisgl.models.attention.fa_backend import (
    _FLASH_ATTN_AVAILABLE,
    FlashAttentionBackend,
)
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.models.attention.pt_backend import PyTorchBackend


class AttentionBackend:
    """Static dispatcher for attention computation.

    Routes to the appropriate backend based on configuration.
    """

    _backend_name: str = "fa"

    @classmethod
    def configure(cls, backend: str) -> None:
        """Set the attention backend.

        Args:
            backend: "fa" (FlashAttention) or "pt" (PyTorch SDPA).
        """
        cls._backend_name = backend

    @classmethod
    def forward(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Compute attention output. Routes to the configured backend.

        Backend selection:
        - "fa": FlashAttention (if available), else PyTorch fallback
        - "pt": Always use PyTorch SDPA (works on CUDA, NPU, CPU)

        On NPU devices, PyTorch backend is used automatically since
        flash_attn is not available. torch_npu provides SDPA acceleration.
        """
        if cls._backend_name == "pt":
            return PyTorchBackend.forward(q, k, v, k_cache, v_cache, attn_meta)
        if _FLASH_ATTN_AVAILABLE:
            return FlashAttentionBackend.forward(q, k, v, k_cache, v_cache, attn_meta)
        return PyTorchBackend.forward(q, k, v, k_cache, v_cache, attn_meta)

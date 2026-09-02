"""Attention backend dispatcher.

Static router that selects the attention implementation from the configured
backend name:

- "fa": FlashAttention (if installed), else PyTorch SDPA fallback.
- "pt": always PyTorch SDPA (works on CUDA, NPU, CPU).
- "fi": FlashInfer is NOT implemented; the backend class delegates every
  call to FlashAttention (see FlashInferBackend in fa_backend.py).
- "fa,fi": hybrid — equivalent to "fa" today, everything runs on
  FlashAttention.
"""

__all__ = ["AttentionBackend"]
import torch

from minisgl.models.attention.fa_backend import (
    _FLASH_ATTN_AVAILABLE,
    _FLASHINFER_AVAILABLE,
    FlashAttentionBackend,
    FlashInferBackend,
)
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
            backend: "fa" (FlashAttention), "fi" (FlashInfer), or "fa,fi" (hybrid).
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
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention output. Routes to the configured backend.

        Backend selection:
        - "fi": FlashInfer (if available)
        - "fa": FlashAttention (if available), else PyTorch fallback
        - "pt": Always use PyTorch SDPA (works on CUDA, NPU, CPU)

        On NPU devices, PyTorch backend is used automatically since
        flash_attn is not available. torch_npu provides SDPA acceleration.
        """
        backend = cls._backend_name

        if backend == "pt":
            return PyTorchBackend.forward(
                q, k, v, k_cache, v_cache, write_loc, **kwargs
            )

        if "fi" in backend and _FLASHINFER_AVAILABLE:
            return FlashInferBackend.forward(
                q, k, v, k_cache, v_cache, write_loc, **kwargs
            )
        elif _FLASH_ATTN_AVAILABLE:
            return FlashAttentionBackend.forward(
                q, k, v, k_cache, v_cache, write_loc, **kwargs
            )
        else:
            return PyTorchBackend.forward(
                q, k, v, k_cache, v_cache, write_loc, **kwargs
            )

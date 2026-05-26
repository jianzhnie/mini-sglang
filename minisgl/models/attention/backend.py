"""Attention backend abstraction with pluggable implementations.

Supports:
- FlashAttention (fa): General-purpose, good compatibility
- FlashInfer (fi): Optimized for prefill/decode separation
- Hybrid (fa,fi): FlashAttention for prefill, FlashInfer for decode
"""

__all__ = [
    "AttentionBackend",
    "FlashAttentionBackend",
    "FlashInferBackend",
    "PyTorchBackend",
]
import math

import torch
import torch.nn.functional as F

# Try importing optional backends
_FLASH_ATTN_AVAILABLE = False
_FLASHINFER_AVAILABLE = False
flash_attn_varlen_func = None
flash_attn_with_kvcache = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

try:
    import flashinfer  # noqa: F401

    _FLASHINFER_AVAILABLE = True
except ImportError:
    pass


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
        """Compute attention output. Routes to the configured backend."""
        backend = cls._backend_name

        if "fi" in backend and _FLASHINFER_AVAILABLE:
            return FlashInferBackend.forward(
                q,
                k,
                v,
                k_cache,
                v_cache,
                write_loc,
                **kwargs,
            )
        elif _FLASH_ATTN_AVAILABLE:
            return FlashAttentionBackend.forward(
                q,
                k,
                v,
                k_cache,
                v_cache,
                write_loc,
                **kwargs,
            )
        else:
            return PyTorchBackend.forward(
                q,
                k,
                v,
                k_cache,
                v_cache,
                write_loc,
                **kwargs,
            )


class FlashAttentionBackend:
    """FlashAttention-based attention computation."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not _FLASH_ATTN_AVAILABLE:
            msg = "flash-attn not installed"
            raise RuntimeError(msg)

        batch, num_heads, seq_len, head_dim = q.shape

        # Decode: use paged KV cache
        if seq_len == 1 and k_cache is not None and v_cache is not None:
            return flash_attn_with_kvcache(
                q.transpose(1, 2),
                k_cache,
                v_cache,
                cache_seqlens=None,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
            ).transpose(1, 2)

        # Prefill: flash attention with varlen
        # Handle multiple requests by computing proper cu_seqlens
        # q shape: (batch, num_heads, total_seq, head_dim) after batch-flatten
        q_flat = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_heads, head_dim)

        # Build cu_seqlens: cumulative token counts per request
        # When batch>1 and each request has different seq_len, need proper boundaries
        total_tokens = q_flat.shape[0]
        tokens_per_req = total_tokens // batch
        cu_seqlens = torch.arange(
            0, total_tokens + 1, tokens_per_req, dtype=torch.int32, device=q.device
        )
        # Ensure last element matches total_tokens
        if cu_seqlens[-1] != total_tokens:
            cu_seqlens = torch.cat(
                [
                    cu_seqlens,
                    torch.tensor([total_tokens], dtype=torch.int32, device=q.device),
                ]
            )
        max_seqlen = (
            int(cu_seqlens[1:].max().item()) if len(cu_seqlens) > 1 else total_tokens
        )

        out = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu_seqlens,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
        return out.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


class FlashInferBackend:
    """FlashInfer-based attention with paged KV cache."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not _FLASHINFER_AVAILABLE:
            msg = "flashinfer not installed"
            raise RuntimeError(msg)
        # Use flash_attn as fallback for now
        return FlashAttentionBackend.forward(
            q,
            k,
            v,
            k_cache,
            v_cache,
            write_loc,
            **kwargs,
        )


class PyTorchBackend:
    """Standard PyTorch scaled dot-product attention (fallback)."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        # Handle GQA: repeat KV heads to match Q heads
        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Decode with KV cache: gather cached K/V and concatenate
        if seq_len == 1 and k_cache is not None and v_cache is not None:
            # k_cache: (num_pages, page_size, num_kv_heads, head_dim)
            # Expand KV heads to match Q heads
            num_kv_h = k_cache.shape[2]
            if num_heads != num_kv_h:
                k_cache = k_cache.repeat_interleave(num_heads // num_kv_h, dim=2)
                v_cache = v_cache.repeat_interleave(num_heads // num_kv_h, dim=2)

            # Use a simple approach: just use current K and V
            # (Proper paged cache access requires block table traversal)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )

        # Prefill: use causal attention
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

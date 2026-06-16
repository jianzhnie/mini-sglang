"""Attention backend abstraction with pluggable implementations.

Supports:
- FlashAttention (fa): General-purpose, good compatibility
- FlashInfer (fi): Optimized for prefill/decode separation (not yet implemented)
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
            cache_seqlens = kwargs.get("cache_seqlens")
            block_table = kwargs.get("block_table")
            return flash_attn_with_kvcache(
                q.transpose(1, 2),
                k_cache,
                v_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
            ).transpose(1, 2)

        # Prefill: flash attention with varlen
        q_flat = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_heads, head_dim)

        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        if cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=q.device)
        else:
            total_tokens = q_flat.shape[0]
            cu_seqlens_q = torch.tensor(
                [0, total_tokens], dtype=torch.int32, device=q.device
            )

        max_seqlen = (
            int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
            if len(cu_seqlens_q) > 1
            else q_flat.shape[0]
        )

        out = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu_seqlens_q,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
        return out.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


class FlashInferBackend:
    """FlashInfer-based attention with paged KV cache.

    TODO: Implement FlashInfer decode path with proper page table support.
    Currently falls back to FlashAttention.
    """

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
    """Standard PyTorch scaled dot-product attention (fallback).

    During decode, gathers cached K,V from the paged KV cache to provide
    full context for each request.
    """

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

        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        if seq_len == 1 and k_cache is not None and v_cache is not None:
            return PyTorchBackend._decode_with_cache(
                q, num_heads, head_dim, k_cache, v_cache, scale, **kwargs
            )

        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        if cu_seqlens_q is not None and len(cu_seqlens_q) > 2:
            return PyTorchBackend._prefill_varlen(
                q, k, v, cu_seqlens_q, scale
            )

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

    @staticmethod
    def _prefill_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Handle multi-sequence prefill by processing each sequence separately."""
        batch, num_heads, total_len, head_dim = q.shape
        outputs = torch.zeros_like(q)
        num_seqs = len(cu_seqlens) - 1
        for i in range(num_seqs):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            qi = q[:, :, start:end, :]
            ki = k[:, :, start:end, :]
            vi = v[:, :, start:end, :]
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
            outputs[:, :, start:end, :] = out_i
        return outputs

    @staticmethod
    def _decode_with_cache(
        q: torch.Tensor,
        num_heads: int,
        head_dim: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        **kwargs,
    ) -> torch.Tensor:
        """Gather cached K,V from page table and run full-context attention."""
        req_to_token = kwargs.get("req_to_token")
        cache_seqlens = kwargs.get("cache_seqlens")

        if req_to_token is None or cache_seqlens is None:
            return F.scaled_dot_product_attention(
                q,
                q,
                q,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )

        num_kv_heads = k_cache.shape[2]

        if num_heads != num_kv_heads:
            k_cache = k_cache.repeat_interleave(num_heads // num_kv_heads, dim=2)
            v_cache = v_cache.repeat_interleave(num_heads // num_kv_heads, dim=2)

        flat_k = k_cache.reshape(-1, num_heads, head_dim)
        flat_v = v_cache.reshape(-1, num_heads, head_dim)

        max_len = int(cache_seqlens.max().item()) + 1
        idxs = req_to_token[:, :max_len]
        valid_mask = idxs >= 0
        idxs_safe = idxs.clamp(min=0)

        gathered_k = flat_k[idxs_safe]
        gathered_v = flat_v[idxs_safe]
        gathered_k = gathered_k * valid_mask[:, :, None, None]
        gathered_v = gathered_v * valid_mask[:, :, None, None]

        gathered_k = gathered_k.transpose(1, 2)
        gathered_v = gathered_v.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            gathered_k,
            gathered_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        return output

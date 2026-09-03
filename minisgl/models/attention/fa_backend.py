"""FlashAttention and FlashInfer attention backends.

Holds the flash-attn / flashinfer availability detection and exports it
(``_FLASH_ATTN_AVAILABLE``, ``_FLASHINFER_AVAILABLE``) for the dispatcher.
``FlashInferBackend`` is a stub: flashinfer is not implemented, so every
call is delegated to FlashAttention.
"""

from __future__ import annotations

__all__ = [
    "FlashAttentionBackend",
    "FlashInferBackend",
]
import math

import torch

from minisgl.models.attention.metadata import AttentionMetadata

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


class FlashAttentionBackend:
    """FlashAttention-based attention computation."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        attn_meta: AttentionMetadata | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        if not _FLASH_ATTN_AVAILABLE:
            msg = "flash-attn not installed"
            raise RuntimeError(msg)

        batch, num_heads, seq_len, head_dim = q.shape
        # GQA: k/v carry their own (smaller) head count — never reshape
        # them with q's num_heads.
        num_kv_heads = k.shape[1]

        # Sliding window (Mistral): FA expects (left, right) offsets.
        window_size = (sliding_window - 1, 0) if sliding_window else (-1, -1)

        # Decode: use paged KV cache
        if attn_meta is not None and attn_meta.forward_mode == "decode":
            # cache_seqlens semantics: total length including the current
            # token, which is exactly what flash_attn_with_kvcache expects.
            return flash_attn_with_kvcache(
                q.transpose(1, 2),
                k_cache,
                v_cache,
                block_table=attn_meta.block_table,
                cache_seqlens=attn_meta.cache_seqlens,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
                window_size=window_size,
            ).transpose(1, 2)

        # Prefill (attn_meta=None is a single cache-less causal sequence).
        q_flat = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        cu_seqlens_q = attn_meta.cu_seqlens_q if attn_meta is not None else None
        if cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=q.device)
        else:
            total_tokens = q_flat.shape[0]
            cu_seqlens_q = torch.tensor(
                [0, total_tokens], dtype=torch.int32, device=q.device
            )

        # Extend attention: requests with a cached prefix read the prefix KV
        # from the paged cache instead of the (suffix-only) k/v tensors.
        prefix_lens = attn_meta.prefix_lens if attn_meta is not None else None
        block_table = attn_meta.block_table if attn_meta is not None else None
        if (
            prefix_lens is not None
            and block_table is not None
            and k_cache is not None
            and int(prefix_lens.max()) > 0
        ):
            return (
                FlashAttentionBackend._prefill_extend(
                    q_flat,
                    k_cache,
                    v_cache,
                    cu_seqlens_q,
                    prefix_lens,
                    block_table,
                    window_size=window_size,
                )
                .view(batch, seq_len, num_heads, head_dim)
                .transpose(1, 2)
            )

        # Prefer the scheduler-computed max sequence length (a plain Python
        # int, no host sync); fall back to deriving it from cu_seqlens for
        # cache-less callers (attn_meta=None — tests and teaching demos).
        max_seqlen = attn_meta.max_seqlen if attn_meta is not None else None
        if max_seqlen is None:
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
            window_size=window_size,
        )
        return out.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _prefill_extend(
        q_flat: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_table: torch.Tensor,
        window_size: tuple[int, int] = (-1, -1),
    ) -> torch.Tensor:
        """Per-request extend attention against the paged KV cache.

        flash_attn_with_kvcache aligns the causal mask to the bottom-right,
        so a suffix query at relative position p attends to the first
        cached_len + p + 1 cached KV entries — exactly extend semantics.
        Note: cache_seqlens here already counts the suffix tokens, matching
        the "total length including current token" convention.
        """
        head_dim = q_flat.shape[2]
        out_flat = torch.empty_like(q_flat)
        # One host sync for the whole loop instead of 3 .item() per request;
        # cache_seqlens slices are views of one pre-built tensor, not a fresh
        # tiny tensor allocation per request.
        cu_list = cu_seqlens.tolist()
        prefix_list = prefix_lens.tolist()
        totals = torch.tensor(
            [p + cu_list[i + 1] - cu_list[i] for i, p in enumerate(prefix_list)],
            dtype=torch.int32,
            device=q_flat.device,
        )
        for i in range(len(prefix_list)):
            start = cu_list[i]
            end = cu_list[i + 1]
            out_i = flash_attn_with_kvcache(
                q_flat[start:end].unsqueeze(0),
                k_cache,
                v_cache,
                block_table=block_table[i : i + 1],
                cache_seqlens=totals[i : i + 1],
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
                window_size=window_size,
            )
            out_flat[start:end] = out_i[0]
        return out_flat


class FlashInferBackend:
    """FlashInfer is NOT implemented — this is a stub that delegates to FA.

    The class exists so that ``--attention-backend fi`` (or ``fa,fi``) parses
    and runs, but every forward call goes straight to FlashAttentionBackend.
    A real implementation would use flashinfer's prefill/decode runners with
    the paged KV cache.
    """

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        attn_meta: AttentionMetadata | None = None,
        sliding_window: int | None = None,
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
            attn_meta,
            sliding_window,
        )

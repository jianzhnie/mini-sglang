"""Attention backend abstraction with pluggable implementations.

Supports:
- FlashAttention (fa): General-purpose, good compatibility
- FlashInfer (fi): NOT implemented; the name is accepted but every call is
  delegated to FlashAttention (see FlashInferBackend).
- Hybrid (fa,fi): Same as "fi" today — everything runs on FlashAttention.
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
        # GQA: k/v carry their own (smaller) head count — never reshape
        # them with q's num_heads.
        num_kv_heads = k.shape[1]

        # Sliding window (Mistral): FA expects (left, right) offsets.
        sliding_window = kwargs.get("sliding_window")
        window_size = (sliding_window - 1, 0) if sliding_window else (-1, -1)

        forward_mode = kwargs.get("forward_mode")
        if forward_mode is None:
            # Legacy callers (e.g. CUDA graph capture) don't pass a mode.
            forward_mode = (
                "decode"
                if (seq_len == 1 and k_cache is not None and v_cache is not None)
                else "prefill"
            )

        # Decode: use paged KV cache
        if forward_mode == "decode":
            cache_seqlens = kwargs.get("cache_seqlens")
            block_table = kwargs.get("block_table")
            # cache_seqlens semantics: total length including the current
            # token, which is exactly what flash_attn_with_kvcache expects.
            return flash_attn_with_kvcache(
                q.transpose(1, 2),
                k_cache,
                v_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
                window_size=window_size,
            ).transpose(1, 2)

        # Prefill
        q_flat = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        if cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=q.device)
        else:
            total_tokens = q_flat.shape[0]
            cu_seqlens_q = torch.tensor(
                [0, total_tokens], dtype=torch.int32, device=q.device
            )

        # Extend attention: requests with a cached prefix read the prefix KV
        # from the paged cache instead of the (suffix-only) k/v tensors.
        prefix_lens = kwargs.get("prefix_lens")
        block_table = kwargs.get("block_table")
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
        # legacy callers.
        max_seqlen = kwargs.get("max_seqlen")
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
        for i in range(len(prefix_lens)):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            total = int(prefix_lens[i].item()) + (end - start)
            out_i = flash_attn_with_kvcache(
                q_flat[start:end].unsqueeze(0),
                k_cache,
                v_cache,
                block_table=block_table[i : i + 1],
                cache_seqlens=torch.tensor(
                    [total], dtype=torch.int32, device=q_flat.device
                ),
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

    During decode and extend-prefill, gathers cached K,V from the paged KV
    cache to provide full context for each request.
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

        forward_mode = kwargs.get("forward_mode")
        if forward_mode is None:
            # Legacy callers (e.g. CUDA graph capture) don't pass a mode.
            forward_mode = (
                "decode"
                if (seq_len == 1 and k_cache is not None and v_cache is not None)
                else "prefill"
            )

        if forward_mode == "decode":
            return PyTorchBackend._decode_with_cache(
                q, num_heads, head_dim, k_cache, v_cache, scale, **kwargs
            )

        # Prefill. Requests with a cached prefix use extend attention: the
        # prefix KV is read back from the paged cache (the suffix KV was
        # already written there by _write_kv_cache before this call).
        prefix_lens = kwargs.get("prefix_lens")
        req_to_token = kwargs.get("req_to_token")
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        sliding_window = kwargs.get("sliding_window")
        if (
            prefix_lens is not None
            and req_to_token is not None
            and k_cache is not None
            and len(prefix_lens) > 0
            and int(prefix_lens.max()) > 0
        ):
            return PyTorchBackend._prefill_extend(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                prefix_lens,
                req_to_token,
                scale,
                sliding_window,
            )

        if cu_seqlens_q is not None and len(cu_seqlens_q) > 2:
            return PyTorchBackend._prefill_varlen(
                q, k, v, cu_seqlens_q, scale, sliding_window
            )

        if sliding_window is not None:
            # Sliding window (Mistral): query i attends keys j in
            # [i - window + 1, i] — causal plus a band condition.
            pos = torch.arange(seq_len, device=q.device)
            allow = (pos.unsqueeze(0) <= pos.unsqueeze(1)) & (
                pos.unsqueeze(0) > pos.unsqueeze(1) - sliding_window
            )
            attn_bias = torch.zeros(seq_len, seq_len, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(~allow, float("-inf"))
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
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
    def _prefill_extend(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        scale: float,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        """Extend attention: suffix queries attend to cached prefix + suffix KV.

        For request i with cached prefix length c and u uncached tokens, all
        c+u KV entries are gathered via req_to_token, and the query at
        relative position p attends to keys [0, c + p] (causal with offset).
        With a sliding window, keys older than the window are masked out too.
        """
        _, num_heads, _, head_dim = q.shape
        num_kv_heads = k_cache.shape[2]
        flat_k = k_cache.reshape(-1, num_kv_heads, head_dim)
        flat_v = v_cache.reshape(-1, num_kv_heads, head_dim)

        outputs = torch.zeros_like(q)
        for i in range(len(prefix_lens)):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            cached = int(prefix_lens[i].item())
            u = end - start
            total = cached + u

            idxs = req_to_token[i, :total].long()
            k_i = flat_k[idxs].transpose(0, 1)  # (num_kv_heads, total, head_dim)
            v_i = flat_v[idxs].transpose(0, 1)
            if num_heads != num_kv_heads:
                k_i = k_i.repeat_interleave(num_heads // num_kv_heads, dim=0)
                v_i = v_i.repeat_interleave(num_heads // num_kv_heads, dim=0)

            q_i = q[0, :, start:end, :]  # (num_heads, u, head_dim)
            q_pos = torch.arange(u, device=q.device) + cached
            k_pos = torch.arange(total, device=q.device)
            allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            if sliding_window is not None:
                allow &= k_pos.unsqueeze(0) > q_pos.unsqueeze(1) - sliding_window
            attn_bias = torch.zeros(u, total, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(~allow, float("-inf"))

            out_i = F.scaled_dot_product_attention(
                q_i.unsqueeze(0),
                k_i.unsqueeze(0),
                v_i.unsqueeze(0),
                attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
            outputs[0, :, start:end, :] = out_i[0]
        return outputs

    @staticmethod
    def _prefill_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
        sliding_window: int | None = None,
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
            attn_mask = None
            is_causal = True
            if sliding_window is not None:
                # Sliding window (Mistral): query i attends keys j in
                # [i - window + 1, i].
                seq_i = end - start
                pos = torch.arange(seq_i, device=q.device)
                allow = (pos.unsqueeze(0) <= pos.unsqueeze(1)) & (
                    pos.unsqueeze(0) > pos.unsqueeze(1) - sliding_window
                )
                attn_mask = torch.zeros(seq_i, seq_i, dtype=q.dtype, device=q.device)
                attn_mask.masked_fill_(~allow, float("-inf"))
                is_causal = False
            out_i = F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
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
            msg = (
                "PyTorchBackend decode requires req_to_token and cache_seqlens "
                "(paged KV cache metadata)"
            )
            raise RuntimeError(msg)

        num_kv_heads = k_cache.shape[2]
        flat_k = k_cache.reshape(-1, num_kv_heads, head_dim)
        flat_v = v_cache.reshape(-1, num_kv_heads, head_dim)

        # cache_seqlens are total lengths INCLUDING the current token.
        # Prefer the scheduler-computed max (a plain Python int): it avoids a
        # per-layer .item() host sync and is legal during CUDA graph capture.
        max_len = kwargs.get("max_seqlen")
        if max_len is None:
            max_len = int(cache_seqlens.max().item())
        idxs = req_to_token[:, :max_len]
        valid_mask = idxs >= 0

        sliding_window = kwargs.get("sliding_window")
        if sliding_window is not None:
            # The current query sits at absolute position cache_seqlens-1;
            # key j sits at absolute position j. Mask keys outside the window.
            q_pos = (cache_seqlens.long() - 1).unsqueeze(1)  # (num_reqs, 1)
            k_pos = torch.arange(max_len, device=q.device).unsqueeze(0)
            valid_mask = valid_mask & (k_pos > q_pos - sliding_window)

        idxs_safe = idxs.clamp(min=0).long()

        # Gather KV-head rows first, then expand GQA groups on the small
        # gathered tensors — repeat_interleave on the whole cache pool would
        # be an O(pool_size) allocation per layer per step.
        gathered_k = flat_k[idxs_safe]  # (num_reqs, max_len, num_kv_heads, head_dim)
        gathered_v = flat_v[idxs_safe]
        if num_heads != num_kv_heads:
            gathered_k = gathered_k.repeat_interleave(num_heads // num_kv_heads, dim=2)
            gathered_v = gathered_v.repeat_interleave(num_heads // num_kv_heads, dim=2)

        gathered_k = gathered_k.transpose(1, 2)
        gathered_v = gathered_v.transpose(1, 2)

        num_reqs = q.shape[0]
        attn_mask = (
            valid_mask.unsqueeze(1).unsqueeze(2).expand(num_reqs, num_heads, 1, max_len)
        )
        attn_bias = torch.zeros(
            num_reqs, num_heads, 1, max_len, dtype=q.dtype, device=q.device
        )
        attn_bias.masked_fill_(~attn_mask, float("-inf"))

        output = F.scaled_dot_product_attention(
            q,
            gathered_k,
            gathered_v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        return output

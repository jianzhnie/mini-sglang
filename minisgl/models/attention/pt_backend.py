"""Pure-PyTorch attention backend based on F.scaled_dot_product_attention.

Fallback backend that works on CUDA, NPU and CPU. During decode and
extend-prefill it gathers cached K,V from the paged KV cache to provide
full context for each request.
"""

from __future__ import annotations

__all__ = ["PyTorchBackend"]
import math

import torch
import torch.nn.functional as F

from minisgl.models.attention.metadata import AttentionMetadata


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
        attn_meta: AttentionMetadata | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        if attn_meta is not None and attn_meta.forward_mode == "decode":
            return PyTorchBackend._decode_with_cache(
                q,
                num_heads,
                head_dim,
                k_cache,
                v_cache,
                scale,
                attn_meta,
                sliding_window,
            )

        # Prefill (attn_meta=None is just a cache-less causal prefill).
        # Requests with a cached prefix use extend attention: the prefix KV
        # is read back from the paged cache (the suffix KV was already
        # written there by _write_kv_cache before this call).
        prefix_lens = attn_meta.prefix_lens if attn_meta is not None else None
        req_to_token = attn_meta.req_to_token if attn_meta is not None else None
        cu_seqlens_q = attn_meta.cu_seqlens_q if attn_meta is not None else None
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

        # One host sync for the whole loop instead of 3 .item() per request.
        cu_list = cu_seqlens.tolist()
        prefix_list = prefix_lens.tolist()
        outputs = torch.zeros_like(q)
        for i in range(len(prefix_list)):
            start = cu_list[i]
            end = cu_list[i + 1]
            cached = prefix_list[i]
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
        # One host sync for the whole loop instead of 2 .item() per request.
        cu_list = cu_seqlens.tolist()
        num_seqs = len(cu_list) - 1
        for i in range(num_seqs):
            start = cu_list[i]
            end = cu_list[i + 1]
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
        attn_meta: AttentionMetadata,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        """Gather cached K,V from page table and run full-context attention."""
        req_to_token = attn_meta.req_to_token
        cache_seqlens = attn_meta.cache_seqlens

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
        max_len = attn_meta.max_seqlen
        if max_len is None:
            max_len = int(cache_seqlens.max().item())
        idxs = req_to_token[:, :max_len]
        valid_mask = idxs >= 0

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

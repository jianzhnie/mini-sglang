"""Base attention module with shared KV cache write and forward logic.

All model-specific attention classes (Qwen2, Qwen3, Llama, Mistral, OPT)
inherit from this base and only override projection creation and optional
pre/post-processing hooks.
"""

from __future__ import annotations

__all__ = ["BaseAttention"]

import torch
import torch.nn as nn

from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.metadata import AttentionMetadata


class BaseAttention(nn.Module):
    """Shared attention forward logic for all model architectures.

    Subclasses must set: num_heads, num_kv_heads, head_dim, hidden_size.
    Subclasses must implement: _project_qkv(), _project_output().
    Optional hooks: _pre_rope_hook(q, k).
    Optional attributes: sliding_window (band width for Mistral-style
    sliding-window attention).

    KV cache ownership
    ------------------
    Each layer holds its own slice of the paged KV cache pool: the engine
    calls set_kv_cache() once at startup (with this layer's slice), and
    forward() writes new K/V into it at attn_meta.write_loc. The per-batch
    inputs (page tables, varlen boundaries, sequence lengths) travel in a
    single typed AttentionMetadata object instead of the old **kwargs
    passthrough chain.

    attn_meta=None means plain causal self-attention without a KV cache
    (used by tests and teaching demos). logits_indices never reaches this
    level — it is consumed by the CausalLM wrapper to select which hidden
    rows feed the lm_head.

    Implicit shape convention: prefill input is always a flat
    (total_tokens, hidden) tensor; forward() unsqueezes it into a fake
    batch=1 and squeezes the output back before returning.
    """

    num_local_heads: int
    num_local_kv_heads: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    sliding_window: int | None = None
    # Per-layer slices of the paged KV cache pool, bound once by the engine
    # via set_kv_cache(); None means "no KV cache" (plain causal attention).
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        """Bind this layer's slice of the paged KV cache pool (one-time setup)."""
        self.k_cache = k_cache
        self.v_cache = v_cache

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _pre_rope_hook(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional normalization before RoPE (e.g., Qwen3 QK norm)."""
        return q, k

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Apply RoPE to Q and K. Override to skip for models without RoPE (e.g., OPT)."""
        self.rotary_emb(q, k, positions)

    def _reshape_for_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim
        ).transpose(1, 2)
        return q, k, v

    def _write_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        write_loc: torch.Tensor | None,
    ) -> None:
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache is None or write_loc is None:
            return

        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
        flat_k = k_cache.view(-1, num_kv_heads, head_dim)
        flat_v = v_cache.view(-1, num_kv_heads, head_dim)
        flat_in_k = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        flat_in_v = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        idx = write_loc.long()
        # Filter out invalid slots (idx == -1); otherwise negative indices would
        # silently write into the last rows of the cache.
        valid = idx >= 0
        flat_k[idx[valid]] = flat_in_k[valid]
        flat_v[idx[valid]] = flat_in_v[valid]

    def _reshape_output(
        self,
        output: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        squeeze_out = hidden_states.dim() == 2
        if squeeze_out:
            hidden_states = hidden_states.unsqueeze(0)

        if positions.dim() > 1:
            positions = positions.squeeze(-1)

        batch_size, seq_len = hidden_states.shape[:2]

        q, k, v = self._project_qkv(hidden_states)
        q, k, v = self._reshape_for_attention(q, k, v, batch_size, seq_len)
        q, k = self._pre_rope_hook(q, k)

        self._apply_rope(q, k, positions)

        write_loc = attn_meta.write_loc if attn_meta is not None else None
        self._write_kv_cache(k, v, write_loc)

        output = AttentionBackend.forward(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            attn_meta,
            self.sliding_window,
        )
        output = self._reshape_output(output, batch_size, seq_len)
        output = self._project_output(output)

        if squeeze_out:
            output = output.squeeze(0)
        return output

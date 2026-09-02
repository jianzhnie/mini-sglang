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


class BaseAttention(nn.Module):
    """Shared attention forward logic for all model architectures.

    Subclasses must set: num_heads, num_kv_heads, head_dim, hidden_size.
    Subclasses must implement: _project_qkv(), _project_output().
    Optional hooks: _pre_rope_hook(q, k), _extra_backend_kwargs().
    """

    num_local_heads: int
    num_local_kv_heads: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int

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

    def _extra_backend_kwargs(self) -> dict:
        """Optional extra kwargs for attention backend (e.g., sliding_window)."""
        return {}

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
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        write_loc: torch.Tensor | None,
    ) -> None:
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
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
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

        self._write_kv_cache(k, v, k_cache, v_cache, write_loc)

        backend_kwargs = self._extra_backend_kwargs()
        backend_kwargs.update(kwargs)
        output = AttentionBackend.forward(
            q,
            k,
            v,
            k_cache,
            v_cache,
            write_loc,
            **backend_kwargs,
        )
        output = self._reshape_output(output, batch_size, seq_len)
        output = self._project_output(output)

        if squeeze_out:
            output = output.squeeze(0)
        return output

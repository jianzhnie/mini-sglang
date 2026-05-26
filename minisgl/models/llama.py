"""Llama model implementation.

LlamaForCausalLM architecture:
- Classic decoder-only transformer
- RMSNorm, RoPE, SwiGLU MLP
- Supports tie_word_embeddings
"""

from typing import Optional

import torch
import torch.nn as nn

from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rms_norm import RMSNorm
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.utils.device import get_tp_size


class LlamaAttention(nn.Module):
    """Multi-head attention for Llama with RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, max_position_embeddings: int, rope_theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_size = get_tp_size()

        self.num_local_heads = num_heads // self.tp_size
        self.num_local_kv_heads = max(1, num_kv_heads // self.tp_size)

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        write_loc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        if positions.dim() > 1:
            positions = positions.squeeze(-1)

        batch_size, seq_len = hidden_states.shape[:2]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        self.rotary_emb(q, k, positions)

        if k_cache is not None and write_loc is not None:
            page_size = k_cache.shape[1]
            num_kv_heads = k_cache.shape[2]
            head_dim = k_cache.shape[3]
            flat_k = k_cache.view(-1, num_kv_heads, head_dim)
            flat_v = v_cache.view(-1, num_kv_heads, head_dim)
            flat_in_k = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
            flat_in_v = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
            flat_k[write_loc] = flat_in_k
            flat_v[write_loc] = flat_in_v

        from minisgl.models.attention.backend import AttentionBackend
        output = AttentionBackend.forward(q, k, v, k_cache, v_cache, write_loc)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.o_proj(output)
        if squeeze_out:
            output = output.squeeze(0)
        return output


class LlamaMLP(nn.Module):
    """Gated MLP (SwiGLU) for Llama."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer for Llama."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int, max_position_embeddings: int,
                 rope_theta: float, rms_norm_eps: float) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            max_position_embeddings, rope_theta,
        )
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        write_loc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(hidden_states, positions, k_cache, v_cache, write_loc)
        hidden_states = attn_out + residual

        hidden_states, residual = self.post_attention_layernorm(hidden_states)
        if residual is None:
            residual = hidden_states
        mlp_out = self.mlp(hidden_states)
        return mlp_out + residual


class LlamaModel(nn.Module):
    """Llama transformer model (without lm_head)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                config.head_dim, config.intermediate_size, config.max_position_embeddings,
                config.rope_theta, config.rms_norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        write_loc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for i, layer in enumerate(self.layers):
            layer_k_cache = k_cache[i] if k_cache is not None else None
            layer_v_cache = v_cache[i] if v_cache is not None else None
            hidden_states = layer(
                hidden_states, positions, residual,
                layer_k_cache, layer_v_cache, write_loc,
            )

        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """Llama model with language modeling head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        write_loc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, k_cache, v_cache, write_loc)
        return self.lm_head(hidden_states)

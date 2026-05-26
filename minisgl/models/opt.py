"""OPT model implementation.

OPT (Open Pre-trained Transformer) architecture:
- Learned positional embeddings (not RoPE)
- LayerNorm (not RMSNorm)
- Standard FFN: fc1 -> ReLU -> fc2 (not gated SwiGLU)
- Decoder: LN -> Attention -> LN -> FFN
"""

__all__ = [
    "LayerNorm",
    "OPTAttention",
    "OPTDecoderLayer",
    "OPTModel",
    "OPTForCausalLM",
    "load_opt_weights",
]
import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from minisgl.utils.device import get_tp_size


class LayerNorm(nn.Module):
    """Standard Layer Normalization."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        mean = x_fp32.mean(-1, keepdim=True)
        var = x_fp32.var(-1, keepdim=True, unbiased=False)
        x_normed = (x_fp32 - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x_normed = self.weight.float() * x_normed + self.bias.float()
        return x_normed.to(x.dtype)


class OPTAttention(nn.Module):
    """Multi-head attention for OPT with learned positional bias and no RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.scaling = self.head_dim**-0.5
        self.tp_size = get_tp_size()
        self.num_local_heads = num_heads // self.tp_size

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * self.head_dim, bias=bias
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_heads * self.head_dim, bias=bias
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_heads * self.head_dim, bias=bias
        )
        self.out_proj = RowParallelLinear(
            num_heads * self.head_dim, hidden_size, bias=bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
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
        k = k.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if k_cache is not None and write_loc is not None:
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

        output = self.out_proj(output)
        if squeeze_out:
            output = output.squeeze(0)
        return output


class OPTDecoderLayer(nn.Module):
    """Single OPT decoder layer: LN -> Attention -> LN -> FFN(ReLU)."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.self_attn = OPTAttention(hidden_size, num_heads, bias=True)
        self.self_attn_layer_norm = LayerNorm(hidden_size)
        self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
        self.fc2 = RowParallelLinear(intermediate_size, hidden_size, bias=True)
        self.final_layer_norm = LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None = None,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention block with LN before
        if residual is None:
            residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        attn_out = self.self_attn(normed, positions, k_cache, v_cache, write_loc)
        hidden_states = attn_out + residual

        # FFN block with LN before
        residual = hidden_states
        normed = self.final_layer_norm(hidden_states)
        hidden_states = F.relu(self.fc1(normed))
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class OPTModel(nn.Module):
    """OPT transformer decoder model."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Add positional embeddings
        hidden_states = self.embed_tokens(input_ids)
        pos_emb = self.embed_positions(positions)
        hidden_states = hidden_states + pos_emb

        residual = None
        for i, layer in enumerate(self.layers):
            layer_k_cache = k_cache[i] if k_cache is not None else None
            layer_v_cache = v_cache[i] if v_cache is not None else None
            hidden_states = layer(
                hidden_states,
                positions,
                residual,
                layer_k_cache,
                layer_v_cache,
                write_loc,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTForCausalLM(nn.Module):
    """OPT model with language modeling head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = OPTModel(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, k_cache, v_cache, write_loc)
        return self.lm_head(hidden_states)


def load_opt_weights(model: OPTForCausalLM, state_dict: dict) -> None:
    """Load OPT weights with HF key remapping."""
    params = dict(model.named_parameters())

    for hf_name, weight in state_dict.items():
        # Map HF keys to our keys
        mini_name = hf_name

        if mini_name in params:
            param = params[mini_name]
            if param.shape == weight.shape:
                param.data.copy_(weight)
            else:
                param.data.copy_(weight[: param.shape[0]])  # Handle TP sharding

    # Tie lm_head with embed_tokens if needed
    if model.config.tie_word_embeddings:
        model.lm_head.weight.data.copy_(model.model.embed_tokens.weight.data)

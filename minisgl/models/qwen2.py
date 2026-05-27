"""Qwen2 model implementation.

Qwen2ForCausalLM architecture:
- Embedding + Decoder Layers + RMSNorm + LM Head
- Gated MLP (SwiGLU): gate_proj, up_proj, down_proj
- Standard attention with RoPE
"""

__all__ = [
    "Qwen2Attention",
    "Qwen2ForCausalLM",
]
import torch

from minisgl.models.decoder import (
    GatedMLP,
    RMSNormDecoderLayer,
    RMSNormForCausalLM,
    RMSNormModel,
)
from minisgl.models.layers.attention import BaseAttention
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.utils.device import get_tp_size


class Qwen2Attention(BaseAttention):
    """Multi-head attention for Qwen2 with RoPE."""

    def __init__(self, config) -> None:
        super().__init__()
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        tp_size = get_tp_size()
        self.num_local_heads = num_heads // tp_size
        self.num_local_kv_heads = num_kv_heads // tp_size

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=True
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=True
        )
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            head_dim, config.max_position_embeddings, config.rope_theta
        )

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)


Qwen2MLP = GatedMLP
Qwen2DecoderLayer = RMSNormDecoderLayer
Qwen2Model = RMSNormModel


class Qwen2ForCausalLM(RMSNormForCausalLM):
    """Qwen2 model with language modeling head."""

    def __init__(self, config) -> None:
        model = RMSNormModel(
            config,
            decoder_layer_cls=RMSNormDecoderLayer,
            attention_cls=Qwen2Attention,
            mlp_cls=GatedMLP,
        )
        super().__init__(model, config)

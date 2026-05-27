"""Qwen3 model implementation.

Differences from Qwen2:
- QK LayerNorm: separate RMSNorm for Q and K before RoPE
- GQA support (num_kv_heads != num_heads)
"""

__all__ = [
    "Qwen3Attention",
    "Qwen3ForCausalLM",
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
from minisgl.models.layers.rms_norm import RMSNorm
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.utils.device import get_tp_size


class Qwen3Attention(BaseAttention):
    """Multi-head attention for Qwen3 with QK normalization and RoPE."""

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

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
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

    def _pre_rope_hook(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        q_normed, _ = self.q_norm(q)
        k_normed, _ = self.k_norm(k)
        return q_normed, k_normed


Qwen3MLP = GatedMLP
Qwen3DecoderLayer = RMSNormDecoderLayer
Qwen3Model = RMSNormModel


class Qwen3ForCausalLM(RMSNormForCausalLM):
    """Qwen3 model with language modeling head."""

    def __init__(self, config) -> None:
        model = RMSNormModel(
            config,
            decoder_layer_cls=RMSNormDecoderLayer,
            attention_cls=Qwen3Attention,
            mlp_cls=GatedMLP,
        )
        super().__init__(model, config)

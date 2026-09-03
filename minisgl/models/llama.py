"""Llama model implementation.

LlamaForCausalLM architecture:
- Classic decoder-only transformer
- RMSNorm, RoPE, SwiGLU MLP
- Supports tie_word_embeddings via RMSNormForCausalLM
"""

__all__ = [
    "LlamaAttention",
    "LlamaForCausalLM",
]
import torch

from minisgl.config import ModelArgs
from minisgl.models.attention.layer import BaseAttention
from minisgl.models.base import RMSNormForCausalLM, RMSNormModel
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.utils.device import get_tp_size


class LlamaAttention(BaseAttention):
    """Multi-head attention for Llama with RoPE."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        tp_size = get_tp_size()
        self.num_local_heads = self.num_heads // tp_size
        self.num_local_kv_heads = max(1, self.num_kv_heads // tp_size)

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)


class LlamaForCausalLM(RMSNormForCausalLM):
    """Llama model with language modeling head."""

    def __init__(self, config: ModelArgs) -> None:
        model = RMSNormModel(config, attention_cls=LlamaAttention)
        super().__init__(model, config)

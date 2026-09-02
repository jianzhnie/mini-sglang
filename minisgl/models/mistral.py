"""Mistral model implementation.

Differences from Llama:
- Sliding Window Attention support
- GQA (Grouped Query Attention)
"""

__all__ = [
    "MistralAttention",
    "MistralForCausalLM",
]
import torch

from minisgl.models.decoder import RMSNormForCausalLM, RMSNormModel
from minisgl.models.layers.attention import BaseAttention
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.utils.device import get_tp_size


class MistralAttention(BaseAttention):
    """Multi-head attention for Mistral with Sliding Window and RoPE."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self._sliding_window = config.sliding_window
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

    def _extra_backend_kwargs(self) -> dict:
        return {"sliding_window": self._sliding_window}


class MistralForCausalLM(RMSNormForCausalLM):
    """Mistral model with language modeling head."""

    def __init__(self, config) -> None:
        model = RMSNormModel(config, attention_cls=MistralAttention)
        super().__init__(model, config)

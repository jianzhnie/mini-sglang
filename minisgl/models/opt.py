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
    "OPTForCausalLM",
    "OPTModel",
]
import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.models.layers.attention import BaseAttention
from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.utils.device import get_tp_rank, get_tp_size


class LayerNorm(nn.Module):
    """Standard Layer Normalization."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
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


class OPTAttention(BaseAttention):
    """Multi-head attention for OPT with learned positional bias and no RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads  # OPT: no GQA
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        tp_size = get_tp_size()
        self.num_local_heads = num_heads // tp_size
        self.num_local_kv_heads = max(1, num_heads // tp_size)

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

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.out_proj(attn_output)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """OPT uses learned positional embeddings at the model level, not RoPE."""
        pass


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
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        attn_out = self.self_attn(
            normed, positions, k_cache, v_cache, write_loc, **kwargs
        )
        hidden_states = attn_out + residual

        residual = hidden_states
        normed = self.final_layer_norm(hidden_states)
        hidden_states = F.relu(self.fc1(normed))
        hidden_states = self.fc2(hidden_states)
        return hidden_states + residual


class OPTModel(nn.Module):
    """OPT transformer decoder model.

    Known numerical difference vs HF: HF's OPTLearnedPositionalEmbedding has
    a +2 index offset (padding_idx) — position p reads row p + 2. This
    implementation indexes the table directly with `positions` (teaching
    simplification), so logits differ from HF OPT by that offset.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
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
        **kwargs,
    ) -> torch.Tensor:
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
                **kwargs,
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
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, k_cache, v_cache, write_loc, **kwargs
        )
        return self.lm_head(hidden_states)

    def tie_weights(self, state_dict: dict) -> None:
        """Tie lm_head weight to embed_tokens when config says so.

        HF OPT checkpoints with tie_word_embeddings=True omit lm_head.weight;
        the engine remaps model keys with model. → model.decoder., so the
        embedding lives at "model.decoder.embed_tokens.weight" in state_dict.
        """
        if not self.config.tie_word_embeddings:
            return
        if "lm_head.weight" in state_dict:
            return  # Checkpoint carries an untied lm_head; already loaded.
        embed_key = "model.decoder.embed_tokens.weight"
        if embed_key not in state_dict:
            return
        if get_tp_size() > 1:
            tp_rank = get_tp_rank()
            vocab_per_rank = self.config.vocab_size // get_tp_size()
            start = tp_rank * vocab_per_rank
            end = (tp_rank + 1) * vocab_per_rank
            self.lm_head.weight.data.copy_(state_dict[embed_key][start:end])
        else:
            self.lm_head.weight.data.copy_(state_dict[embed_key])

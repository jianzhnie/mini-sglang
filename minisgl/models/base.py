"""Shared decoder-only transformer base classes.

Eliminates duplication across Qwen2, Qwen3, Llama, and Mistral models.
All share the same RMSNorm + SwiGLU decoder architecture.
"""

__all__ = [
    "GatedMLP",
    "RMSNormDecoderLayer",
    "RMSNormModel",
    "RMSNormForCausalLM",
    "gather_last_logits",
]
import torch
import torch.nn as nn

from minisgl.config import ModelArgs
from minisgl.models.attention.layer import BaseAttention
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rms_norm import RMSNorm
from minisgl.utils.device import get_tp_rank, get_tp_size


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU) — shared by all RMSNorm-based decoder models."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def gather_last_logits(
    hidden: torch.Tensor, logits_indices: torch.Tensor
) -> torch.Tensor:
    """Gather the hidden rows that feed the lm_head.

    Prefill fast path: sampling only consumes the last uncached token of
    each request, so gather those rows and run the (very expensive)
    full-vocab lm_head projection on them alone.
    """
    if hidden.dim() == 3:
        hidden = hidden.view(-1, hidden.shape[-1])
    return hidden[logits_indices]


class RMSNormDecoderLayer(nn.Module):
    """Single transformer decoder layer: RMSNorm -> Attention -> RMSNorm -> MLP.

    Shared by the Qwen3 family. Accepts a pre-built MLP (e.g. an MoE block) or
    builds a standard GatedMLP from ``intermediate_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        attention: nn.Module,
        mlp: nn.Module | None = None,
        intermediate_size: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp if mlp is not None else GatedMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None = None,
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(hidden_states, positions, attn_meta)
        # Fused add: normalizes attn_out + residual and returns the new residual.
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        return self.mlp(hidden_states) + residual


class RMSNormModel(nn.Module):
    """Shared transformer backbone for the Qwen3 family.

    Embeds tokens, applies RoPE (via the attention layers), runs the decoder
    stack with residual flow.
    """

    def __init__(
        self,
        config: ModelArgs,
        attention_cls: type[BaseAttention],
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                RMSNormDecoderLayer(
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    attention=attention_cls(config),
                    intermediate_size=config.intermediate_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, residual, attn_meta)
        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


class RMSNormForCausalLM(nn.Module):
    """RMSNorm-based CausalLM with tie_weights support."""

    def __init__(self, model: nn.Module, config: ModelArgs) -> None:
        super().__init__()
        self.model = model
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, gather_output=True
        )
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, attn_meta)
        if logits_indices is not None:
            hidden_states = gather_last_logits(hidden_states, logits_indices)
        return self.lm_head(hidden_states)

    def tie_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Tie lm_head weight with embed_tokens if config says so."""
        if not self.config.tie_word_embeddings:
            return
        embed_key = "model.embed_tokens.weight"
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

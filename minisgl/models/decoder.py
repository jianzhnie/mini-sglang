"""Shared decoder-only transformer base classes.

Eliminates duplication across Qwen2, Qwen3, Llama, and Mistral models.
All share the same RMSNorm + SwiGLU decoder architecture.
"""

__all__ = [
    "GatedMLP",
    "RMSNormDecoderLayer",
    "RMSNormModel",
    "RMSNormForCausalLM",
]
import torch
import torch.nn as nn

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


class RMSNormDecoderLayer(nn.Module):
    """Single transformer decoder layer: RMSNorm -> Attention -> RMSNorm -> MLP.

    Shared by Qwen2, Qwen3, Llama, and Mistral.
    Accepts pre-built attention and MLP modules for flexibility.
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
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(
            hidden_states, positions, k_cache, v_cache, write_loc, **kwargs
        )
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        return mlp_out + residual


class RMSNormModel(nn.Module):
    """Shared transformer backbone for RMSNorm-based decoder models.

    Embeds tokens, applies positional encoding (via layers), runs decoder stack.
    """

    def __init__(
        self,
        config,
        attention_cls: type,
        mlp_cls: type | None = None,
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
                    mlp=(
                        mlp_cls(config.hidden_size, config.intermediate_size)
                        if mlp_cls is not None
                        else None
                    ),
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
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
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
        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


class RMSNormForCausalLM(nn.Module):
    """RMSNorm-based CausalLM with tie_weights support."""

    def __init__(self, model: nn.Module, config) -> None:
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
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        write_loc: torch.Tensor | None = None,
        logits_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, k_cache, v_cache, write_loc, **kwargs
        )
        if logits_indices is not None:
            # Prefill fast path: sampling only consumes the last uncached
            # token of each request, so gather those rows and run the (very
            # expensive) full-vocab lm_head projection on them alone.
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = hidden_states[logits_indices]
        return self.lm_head(hidden_states)

    def tie_weights(self, state_dict: dict) -> None:
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

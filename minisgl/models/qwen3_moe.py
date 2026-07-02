"""Qwen3-MoE model implementation.

Qwen3MoEForCausalLM:
- Same as Qwen3 but with MoE FFN blocks replacing standard MLP at specific layers
- MoE Router: linear layer selecting top-k experts
"""

__all__ = [
    "Qwen3MoEForCausalLM",
    "Qwen3MoEMLP",
]
import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.models.decoder import (
    GatedMLP,
    RMSNormDecoderLayer,
    RMSNormForCausalLM,
)
from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.linear import ColumnParallelLinear, RowParallelLinear
from minisgl.models.layers.rms_norm import RMSNorm
from minisgl.models.qwen3 import Qwen3Attention
from minisgl.utils.device import get_tp_size


class Qwen3MoEMLP(nn.Module):
    """Mixture of Experts MLP for Qwen3-MoE.

    Architecture:
    - Router: linear layer predicting expert probabilities
    - Shared experts: always-active experts
    - Routed experts: top-k selected experts
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        tp_size = get_tp_size()

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        self.shared_gate = ColumnParallelLinear(
            hidden_size, moe_intermediate_size, bias=False
        )
        self.shared_up = ColumnParallelLinear(
            hidden_size, moe_intermediate_size, bias=False
        )
        self.shared_down = RowParallelLinear(
            moe_intermediate_size, hidden_size, bias=False
        )

        self.num_local_experts = num_experts // tp_size
        experts_per_rank = moe_intermediate_size // tp_size

        self.expert_gate = nn.Parameter(
            torch.empty(self.num_local_experts, experts_per_rank, hidden_size),
        )
        self.expert_up = nn.Parameter(
            torch.empty(self.num_local_experts, experts_per_rank, hidden_size),
        )
        self.expert_down = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, experts_per_rank),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)

        shared_out = self.shared_down(
            F.silu(self.shared_gate(flat)) * self.shared_up(flat),
        )

        router_logits = self.router(flat)
        router_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        router_weights = F.softmax(router_weights, dim=-1)

        routed_out = self._fused_moe(flat, router_weights, selected_experts)
        return (shared_out + routed_out).view(batch_size, seq_len, hidden_dim)

    def _fused_moe(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens, hidden_dim = x.shape
        output = torch.zeros(total_tokens, hidden_dim, dtype=x.dtype, device=x.device)

        for expert_idx in range(self.num_local_experts):
            mask = (selected_experts == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            tokens_for_expert = x[mask]
            weight_idx = (selected_experts[mask] == expert_idx).nonzero(as_tuple=True)[
                1
            ]
            weights = router_weights[mask][
                torch.arange(len(weight_idx)),
                weight_idx,
            ].unsqueeze(-1)

            gate = F.linear(tokens_for_expert, self.expert_gate[expert_idx])
            up = F.linear(tokens_for_expert, self.expert_up[expert_idx])
            expert_out = F.silu(gate) * up
            expert_out = F.linear(expert_out, self.expert_down[expert_idx])
            output[mask] += expert_out * weights

        return output


class Qwen3MoEModel(nn.Module):
    """Qwen3-MoE transformer model with sparse MoE layers."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        decoder_sparse_step = config.decoder_sparse_step or 1
        self.layers = nn.ModuleList(
            [
                self._make_layer(config, i, decoder_sparse_step)
                for i in range(config.num_layers)
            ],
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _make_layer(config, layer_idx: int, sparse_step: int) -> RMSNormDecoderLayer:
        is_moe = layer_idx % sparse_step == sparse_step - 1
        mlp: nn.Module
        if is_moe:
            mlp = Qwen3MoEMLP(
                config.hidden_size,
                config.intermediate_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
            )
        else:
            mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        return RMSNormDecoderLayer(
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            attention=Qwen3Attention(config),
            mlp=mlp,
        )

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
                hidden_states, positions, residual,
                layer_k_cache, layer_v_cache, write_loc, **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


class Qwen3MoEForCausalLM(RMSNormForCausalLM):
    """Qwen3-MoE with language modeling head."""

    def __init__(self, config) -> None:
        model = Qwen3MoEModel(config)
        super().__init__(model, config)

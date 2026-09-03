"""Qwen3-MoE model implementation.

Qwen3MoEForCausalLM:
- Same as Qwen3 but with MoE FFN blocks replacing standard MLP at specific layers
- MoE Router: linear layer selecting top-k experts

Aligned with HF Qwen3MoESparseMoeBlock:
- mlp.gate is the router; mlp.experts.{i}.{gate_proj,up_proj,down_proj} are
  the per-expert weights (stored here as fused 3-D tensors).
- No shared expert.
- Routing: softmax over ALL experts, then top-k, then renormalize the
  selected weights to sum to 1 (HF norm_topk_prob=True).
"""

__all__ = [
    "Qwen3MoEForCausalLM",
    "Qwen3MoEMLP",
]
import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.config import ModelArgs
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.models.decoder import (
    GatedMLP,
    RMSNormDecoderLayer,
    RMSNormForCausalLM,
)
from minisgl.models.layers.embedding import VocabParallelEmbedding
from minisgl.models.layers.rms_norm import RMSNorm
from minisgl.models.qwen3 import Qwen3Attention
from minisgl.utils.device import get_tp_size


class Qwen3MoEMLP(nn.Module):
    """Sparse MoE MLP matching HF Qwen3MoESparseMoeBlock (no shared expert).

    Expert weights are stored fused: expert_gate/up are
    (num_experts, moe_intermediate_size, hidden_size) and expert_down is
    (num_experts, hidden_size, moe_intermediate_size). HF's per-expert keys
    are aggregated into these tensors by load_hf_experts().

    Teaching simplification: expert weights are not sharded across TP ranks
    and there is no all-reduce on expert outputs, so tensor parallelism is
    unsupported here.
    """

    def __init__(
        self,
        hidden_size: int,
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
        assert tp_size == 1, (
            f"Qwen3MoEMLP does not support tensor parallelism "
            f"(tp_size={tp_size}): experts are not sharded and there is no "
            "all-reduce. Run with --tp-size 1."
        )

        # Router (HF name: mlp.gate).
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.expert_gate = nn.Parameter(
            torch.empty(num_experts, moe_intermediate_size, hidden_size),
        )
        self.expert_up = nn.Parameter(
            torch.empty(num_experts, moe_intermediate_size, hidden_size),
        )
        self.expert_down = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size),
        )
        # torch.empty leaves garbage (possibly NaN) until weights load;
        # give experts a sane default init like nn.Linear does.
        for p in (self.expert_gate, self.expert_up, self.expert_down):
            nn.init.normal_(p, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Accept both prefill's 2-D (total_tokens, hidden) and 3-D input.
        flat = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(flat)
        # HF routing semantics: softmax over ALL experts first, then top-k,
        # then renormalize the selected weights to sum to 1.
        routing_weights = F.softmax(router_logits, dim=-1)
        top_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        out = self._fused_moe(flat, top_weights, selected_experts)
        return out.view(*hidden_states.shape)

    def _fused_moe(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens, hidden_dim = x.shape
        output = torch.zeros(total_tokens, hidden_dim, dtype=x.dtype, device=x.device)

        # One host sync per forward: which experts are actually selected this
        # step. Experts outside this list have no tokens and are skipped
        # without a per-expert mask.any() sync.
        for expert_idx in selected_experts.unique().tolist():
            mask = (selected_experts == expert_idx).any(dim=-1)

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

    def __init__(self, config: ModelArgs) -> None:
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
    def _make_layer(
        config: ModelArgs, layer_idx: int, sparse_step: int
    ) -> RMSNormDecoderLayer:
        is_moe = layer_idx % sparse_step == sparse_step - 1
        mlp: nn.Module
        if is_moe:
            mlp = Qwen3MoEMLP(
                config.hidden_size,
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
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, residual, attn_meta)
        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


class Qwen3MoEForCausalLM(RMSNormForCausalLM):
    """Qwen3-MoE with language modeling head."""

    def __init__(self, config: ModelArgs) -> None:
        model = Qwen3MoEModel(config)
        super().__init__(model, config)

    def load_hf_experts(self, state_dict: dict[str, torch.Tensor]) -> int:
        """Aggregate HF per-expert weights into the fused expert tensors.

        HF stores experts as mlp.experts.{i}.{gate_proj,up_proj,down_proj}.weight,
        which load_weights_parallel cannot match to the fused parameters.
        Called by the engine right after load_weights_parallel.

        Returns the number of expert weight tensors loaded.
        """
        loaded = 0
        for layer_idx, layer in enumerate(self.model.layers):
            mlp = layer.mlp
            if not isinstance(mlp, Qwen3MoEMLP):
                continue
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            for proj, attr in (
                ("gate_proj", "expert_gate"),
                ("up_proj", "expert_up"),
                ("down_proj", "expert_down"),
            ):
                fused = getattr(mlp, attr)
                for i in range(mlp.num_experts):
                    weight = state_dict.get(f"{prefix}.{i}.{proj}.weight")
                    if weight is None or weight.shape != fused.shape[1:]:
                        continue
                    fused.data[i].copy_(
                        weight.to(dtype=fused.dtype, device=fused.device)
                    )
                    loaded += 1
        return loaded

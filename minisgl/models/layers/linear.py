"""Tensor parallel linear layers."""

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.utils.device import get_tp_size, is_distributed


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across TP ranks.


    Forward computes partial output; gather combines all parts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.out_features_per_rank = out_features // tp_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.weight.is_column_parallel = True
        self._init_weights()
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
            # Bias is 1-D along the output dim: shard it like the weight.
            self.bias.is_column_parallel = True
        else:
            self.register_parameter("bias", None)

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output and is_distributed():
            from minisgl.engine.collectives import all_gather

            out = all_gather(out, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across TP ranks.

    Forward computes partial output; all-reduce sums all parts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.in_features_per_rank = in_features // tp_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.weight.is_row_parallel = True
        self._init_weights()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if is_distributed():
            from minisgl.engine.collectives import all_reduce

            out = all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out

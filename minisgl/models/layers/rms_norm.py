"""RMSNorm with fused residual add."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with fused residual add.

    When residual is provided: out, new_residual = rms_norm(x + residual)
    When residual is None: out, new_residual = rms_norm(x)
    Always returns a (normalized, residual) tuple for consistent API.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor = None,
    ) -> tuple:
        dtype = x.dtype

        if residual is not None:
            x = x + residual

        new_residual = x.detach() if residual is not None else None

        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        result = (self.weight.float() * x_normed).to(dtype)

        return result, new_residual

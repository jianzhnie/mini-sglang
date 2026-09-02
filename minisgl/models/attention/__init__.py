"""Attention backends: dispatcher plus FA / FlashInfer / PyTorch implementations."""

from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.fa_backend import (
    FlashAttentionBackend,
    FlashInferBackend,
)
from minisgl.models.attention.pt_backend import PyTorchBackend

__all__ = [
    "AttentionBackend",
    "FlashAttentionBackend",
    "FlashInferBackend",
    "PyTorchBackend",
]

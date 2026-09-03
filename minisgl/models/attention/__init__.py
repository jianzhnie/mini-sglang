"""Attention backends: metadata, dispatcher plus FlashAttention / PyTorch implementations."""

from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.fa_backend import FlashAttentionBackend
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.models.attention.pt_backend import PyTorchBackend

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "FlashAttentionBackend",
    "PyTorchBackend",
]

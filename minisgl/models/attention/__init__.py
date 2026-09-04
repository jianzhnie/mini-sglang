"""Attention backends: a typed per-batch metadata object and pluggable impls.

- ``metadata``: the typed AttentionMetadata the scheduler builds and backends read.
- ``dispatcher``: routes a forward call to the configured backend.
- ``layer``: BaseAttention — the shared per-layer attention module with paged-KV
  write logic (subclassed by Qwen3 / Qwen3-MoE).
- ``pt_backend`` / ``fa_backend``: the PyTorch-SDPA and FlashAttention kernels.

Use the concrete paths (e.g. ``from minisgl.models.attention.pt_backend import PyTorchBackend``);
this package ``__init__`` re-exports the public classes for convenience.
"""

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

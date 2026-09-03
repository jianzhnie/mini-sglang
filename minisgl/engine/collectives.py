"""Collective communication primitives with NCCL/HCCL dual backends.

Uses torch.distributed as the backend for collective operations
(all_reduce, all_gather) used in tensor parallelism, covering both NCCL
(GPU) and HCCL (NPU) backends.
"""

__all__ = ["all_gather", "all_reduce"]
import torch
import torch.distributed as dist

from minisgl.utils.device import get_tp_size, is_distributed


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce across TP ranks. Only 'sum' is supported."""
    if op != "sum":
        raise ValueError(f"Unsupported all_reduce op: {op!r} (only 'sum')")
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across TP ranks along given dimension."""
    if not is_distributed():
        return tensor
    tp_size = get_tp_size()
    chunks = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(chunks, tensor)
    return torch.cat(chunks, dim=dim)

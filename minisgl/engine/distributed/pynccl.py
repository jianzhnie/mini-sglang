"""PyNCCL: Python wrapper for NCCL collective operations.

Uses torch.distributed as the backend for collective operations
(all_reduce, all_gather, broadcast) used in tensor parallelism.
"""

__all__ = ["all_reduce", "all_gather", "broadcast", "barrier"]
import torch
import torch.distributed as dist

from minisgl.utils.device import get_tp_size, is_distributed


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce across TP ranks."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across TP ranks along given dimension."""
    if not is_distributed():
        return tensor
    tp_size = get_tp_size()
    chunks = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(chunks, tensor)
    return torch.cat(chunks, dim=dim)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from src rank to all ranks."""
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def barrier() -> None:
    """Synchronize all TP ranks."""
    if is_distributed():
        dist.barrier()

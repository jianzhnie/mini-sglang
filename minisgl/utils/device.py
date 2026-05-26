"""Device and distributed initialization utilities."""

import os
from typing import Optional

import torch
import torch.distributed as dist

_TP_RANK: int = 0
_TP_SIZE: int = 1
_DEVICE: Optional[torch.device] = None


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_size() -> int:
    return _TP_SIZE


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def set_device(device: torch.device) -> None:
    global _DEVICE
    _DEVICE = device
    if device.type == "cuda":
        torch.cuda.set_device(device)


def init_distributed(
    tp_rank: int = 0,
    tp_size: int = 1,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    global _TP_RANK, _TP_SIZE
    _TP_RANK = tp_rank
    _TP_SIZE = tp_size

    if tp_size > 1:
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
        dist.init_process_group(
            backend=backend,
            rank=tp_rank,
            world_size=tp_size,
        )
        device = torch.device(f"cuda:{tp_rank}")
        set_device(device)


def is_distributed() -> bool:
    return _TP_SIZE > 1 and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()

"""Device and distributed initialization utilities."""

__all__ = [
    "barrier",
    "DeviceState",
    "get_device",
    "get_tp_rank",
    "get_tp_size",
    "init_distributed",
    "is_distributed",
    "reset_device_state",
    "set_device",
]
import os

import torch
import torch.distributed as dist


class DeviceState:
    """Holds tensor-parallel state to avoid module-level globals.

    Use `reset_device_state()` to clear state between sessions.
    """

    def __init__(self) -> None:
        self.tp_rank: int = 0
        self.tp_size: int = 1
        self.device: torch.device | None = None


_state = DeviceState()


def reset_device_state() -> None:
    """Reset device state for clean multi-instance support."""
    global _state
    if dist.is_initialized():
        dist.destroy_process_group()
    _state = DeviceState()


def get_tp_rank() -> int:
    return _state.tp_rank


def get_tp_size() -> int:
    return _state.tp_size


def get_device() -> torch.device:
    if _state.device is None:
        _state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _state.device


def set_device(device: torch.device) -> None:
    _state.device = device
    if device.type == "cuda":
        torch.cuda.set_device(device)


def init_distributed(
    tp_rank: int = 0,
    tp_size: int = 1,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    _state.tp_rank = tp_rank
    _state.tp_size = tp_size

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
    return _state.tp_size > 1 and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()

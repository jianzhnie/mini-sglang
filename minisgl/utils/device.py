"""Device and distributed initialization utilities.

Supports: CUDA (NVIDIA), NPU (Ascend/torch_npu), and CPU fallback.
Device priority: NPU > CUDA > CPU (configurable via set_device).
"""

__all__ = [
    "barrier",
    "DeviceState",
    "get_device",
    "get_device_type",
    "get_tp_rank",
    "get_tp_size",
    "init_distributed",
    "is_accelerator_available",
    "is_distributed",
    "is_npu_available",
    "reset_device_state",
    "set_device",
    "synchronize",
]
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

_NPU_AVAILABLE: bool | None = None


def is_npu_available() -> bool:
    """Check if Ascend NPU is available via torch_npu."""
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is None:
        try:
            import torch_npu  # noqa: F401

            _NPU_AVAILABLE = torch.npu.is_available()
        except (ImportError, RuntimeError, AttributeError):
            _NPU_AVAILABLE = False
    return _NPU_AVAILABLE


def is_accelerator_available() -> bool:
    """Check if any accelerator (NPU or CUDA) is available."""
    return is_npu_available() or torch.cuda.is_available()


def get_device_type() -> str:
    """Return the best available device type string: 'npu', 'cuda', or 'cpu'."""
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DeviceState:
    """Holds tensor-parallel state to avoid module-level globals.

    Use `reset_device_state()` to clear state between sessions.
    """

    tp_rank: int = 0
    tp_size: int = 1
    device: torch.device | None = None


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
        _state.device = torch.device(get_device_type())
    return _state.device


def set_device(device: torch.device) -> None:
    _state.device = device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def synchronize() -> None:
    """Synchronize the current accelerator device."""
    if _state.device is None:
        return
    if _state.device.type == "cuda":
        torch.cuda.synchronize()
    elif _state.device.type == "npu":
        torch.npu.synchronize()


def mem_get_info(device: torch.device | None = None) -> tuple[int, int]:
    """Get (free, total) memory in bytes for the current accelerator.

    Returns (0, 0) for CPU.
    """
    dev = device or get_device()
    if dev.type == "cuda":
        return torch.cuda.mem_get_info(dev)
    elif dev.type == "npu":
        free = torch.npu.mem_get_info(dev)[0]
        total = torch.npu.mem_get_info(dev)[1]
        return free, total
    return 0, 0


def init_distributed(
    tp_rank: int = 0,
    tp_size: int = 1,
    backend: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """Initialize distributed process group.

    Auto-selects backend: 'hccl' for NPU, 'nccl' for CUDA, 'gloo' for CPU.
    """
    _state.tp_rank = tp_rank
    _state.tp_size = tp_size

    if tp_size > 1:
        if backend is None:
            dev_type = get_device_type()
            if dev_type == "npu":
                backend = "hccl"
            elif dev_type == "cuda":
                backend = "nccl"
            else:
                backend = "gloo"

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
        dist.init_process_group(
            backend=backend,
            rank=tp_rank,
            world_size=tp_size,
        )
        dev_type = get_device_type()
        device = torch.device(f"{dev_type}:{tp_rank}")
        set_device(device)


def is_distributed() -> bool:
    return _state.tp_size > 1 and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()

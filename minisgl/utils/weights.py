"""Weight loading utilities for HuggingFace models."""

import os
from typing import Dict, Iterator, Tuple

import torch
from safetensors.torch import load_file as safe_load

from minisgl.utils.device import get_device, get_tp_rank, get_tp_size


def _iter_hf_files(model_path: str) -> Iterator[Tuple[str, str]]:
    """Yield (file_path, file_type) pairs: 'safetensors' or 'bin'."""
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            yield os.path.join(model_path, fname), "safetensors"
        elif fname.endswith(".bin") and fname.startswith("pytorch_model"):
            yield os.path.join(model_path, fname), "bin"


def load_hf_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load all HF weights from a directory into a single state dict."""
    state_dict: Dict[str, torch.Tensor] = {}
    index_file = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_file):
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
        for fname in sorted(set(index["weight_map"].values())):
            file_path = os.path.join(model_path, fname)
            state_dict.update(safe_load(file_path))
    else:
        for file_path, ftype in _iter_hf_files(model_path):
            if ftype == "safetensors":
                state_dict.update(safe_load(file_path))
            elif ftype == "bin":
                state_dict.update(torch.load(file_path, map_location="cpu", weights_only=True))

    return state_dict


def shard_tensor(tensor: torch.Tensor, dim: int, rank: int, world_size: int) -> torch.Tensor:
    """Shard a tensor along the given dimension."""
    chunk_size = tensor.shape[dim] // world_size
    start = rank * chunk_size
    end = start + chunk_size
    return tensor.narrow(dim, start, chunk_size).contiguous()


def load_weights_parallel(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Load weights into a model, handling tensor parallelism sharding.

    Handles:
    - ColumnParallel: weight is sharded along dim 0 (output dim) — each rank gets a portion
    - RowParallel: weight is sharded along dim 1 (input dim) — each rank gets a portion
    - Embedding: weight sharded along dim 0 (vocab dim)
    - Non-parallel: each rank gets full copy
    """
    device = get_device()
    tp_rank = get_tp_rank()
    tp_size = get_tp_size()

    for name, param in model.named_parameters():
        if name not in state_dict:
            continue

        weight = state_dict[name]

        # Handle ColumnParallelLinear weights (e.g., q_proj, k_proj, v_proj, gate_proj, up_proj)
        if hasattr(param, "is_column_parallel") and param.is_column_parallel:
            weight = shard_tensor(weight, dim=0, rank=tp_rank, world_size=tp_size)
        # Handle RowParallelLinear weights (e.g., o_proj, down_proj)
        elif hasattr(param, "is_row_parallel") and param.is_row_parallel:
            weight = shard_tensor(weight, dim=1, rank=tp_rank, world_size=tp_size)
        # Handle VocabParallelEmbedding
        elif hasattr(param, "is_vocab_parallel") and param.is_vocab_parallel:
            weight = shard_tensor(weight, dim=0, rank=tp_rank, world_size=tp_size)

        param.data.copy_(weight.to(device=device, dtype=param.dtype))

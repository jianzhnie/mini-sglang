"""Weight loading utilities for HuggingFace models."""

__all__ = ["load_hf_weights", "load_weights_parallel", "shard_tensor"]
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load

from minisgl.utils.device import get_device


def _iter_hf_files(model_path: str) -> Iterator[tuple[str, str]]:
    """Yield (file_path, file_type) pairs: 'safetensors' or 'bin'."""

    for fname in sorted(p.name for p in Path(model_path).iterdir()):
        if fname.endswith(".safetensors"):
            yield str(Path(model_path) / fname), "safetensors"
        elif fname.endswith(".bin") and fname.startswith("pytorch_model"):
            yield str(Path(model_path) / fname), "bin"


def load_hf_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load all HF weights from a directory into a single state dict."""
    state_dict: dict[str, torch.Tensor] = {}
    index_file = Path(model_path) / "model.safetensors.index.json"

    if index_file.exists():
        import json

        with index_file.open() as f:
            index = json.load(f)
        for fname in sorted(set(index["weight_map"].values())):
            file_path = str(Path(model_path) / fname)
            state_dict.update(safe_load(file_path))
    else:
        for file_path, ftype in _iter_hf_files(model_path):
            if ftype == "safetensors":
                state_dict.update(safe_load(file_path))
            elif ftype == "bin":
                state_dict.update(
                    torch.load(file_path, map_location="cpu", weights_only=True),
                )

    return state_dict


def shard_tensor(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Shard a tensor along the given dimension."""
    chunk_size = tensor.shape[dim] // world_size
    start = rank * chunk_size
    return tensor.narrow(dim, start, chunk_size).contiguous()


def load_weights_parallel(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    tp_rank: int = 0,
    tp_size: int = 1,
    remap_fn=None,
) -> None:
    """Load weights into a model, handling tensor parallelism sharding.

    Args:
        model: The model to load weights into.
        state_dict: HF weight dictionary.
        tp_rank: Tensor parallelism rank.
        tp_size: Tensor parallelism size.
        remap_fn: Optional function to remap HF keys → model param names.

    Handles:
    - ColumnParallel: weight is sharded along dim 0 (output dim)
    - RowParallel: weight is sharded along dim 1 (input dim)
    - Embedding: weight sharded along dim 0 (vocab dim)
    - Non-parallel: each rank gets full copy
    """
    device = get_device()

    loaded = 0
    for name, param in model.named_parameters():
        hf_name = remap_fn(name) if remap_fn else name
        if hf_name not in state_dict:
            # Try the original name as fallback
            if name in state_dict:
                hf_name = name
            else:
                continue

        weight = state_dict[hf_name]

        # Handle ColumnParallelLinear weights
        if hasattr(param, "is_column_parallel") and param.is_column_parallel:
            weight = shard_tensor(weight, dim=0, rank=tp_rank, world_size=tp_size)
        # Handle RowParallelLinear weights
        elif hasattr(param, "is_row_parallel") and param.is_row_parallel:
            weight = shard_tensor(weight, dim=1, rank=tp_rank, world_size=tp_size)
        # Handle VocabParallelEmbedding
        elif hasattr(param, "is_vocab_parallel") and param.is_vocab_parallel:
            weight = shard_tensor(weight, dim=0, rank=tp_rank, world_size=tp_size)
        # Handle shape mismatch (e.g., embed_positions truncation)
        elif param.shape != weight.shape and param.dim() >= 2:
            weight = (
                weight[: param.shape[0]] if weight.shape[0] > param.shape[0] else weight
            )

        if param.shape == weight.shape:
            param.data.copy_(weight.to(device=device, dtype=param.dtype))
            loaded += 1
        elif param.dim() >= 2 and weight.shape[0] >= param.shape[0]:
            param.data.copy_(
                weight[: param.shape[0]].to(device=device, dtype=param.dtype),
            )
            loaded += 1

    return loaded

"""Model registry for auto-detection and instantiation.

Supports the Qwen3 family:
- Qwen3ForCausalLM
- Qwen3MoEForCausalLM
"""

from __future__ import annotations

__all__ = ["create_model", "detect_model_type"]
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from minisgl.utils.logger import logger

if TYPE_CHECKING:
    import torch.nn as nn

    from minisgl.config import ModelArgs


# Registry: model_type → (lazy_import_path, class_name)
_MODEL_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "qwen3": ("minisgl.models.qwen3", "Qwen3ForCausalLM"),
    "qwen3_moe": ("minisgl.models.qwen3_moe", "Qwen3MoEForCausalLM"),
}


def detect_model_type(model_path: str) -> str:
    """Detect the model architecture from config.json.

    Args:
        model_path: Path to the HF model directory.

    Returns:
        Model type string ("qwen3" or "qwen3_moe").
    """
    config_file = Path(model_path) / "config.json"
    with config_file.open() as f:
        cfg = json.load(f)

    architectures = cfg.get("architectures", [])

    for arch in architectures:
        arch_lower = arch.lower()
        if "qwen3moe" in arch_lower or "qwen3_moe" in arch_lower:
            return "qwen3_moe"
        if "qwen3" in arch_lower:
            return "qwen3"

    # Fallback heuristics (for configs that omit `architectures`).
    if cfg.get("num_experts", 0) > 0:
        return "qwen3_moe"
    if cfg.get("qk_norm", False):
        return "qwen3"

    logger.warning(
        f"Could not detect model type from architectures: {architectures}. "
        "Only the Qwen3 family is supported; defaulting to qwen3.",
    )
    return "qwen3"


def create_model(config: ModelArgs, model_type: str) -> nn.Module:
    """Create a model instance from config.

    Args:
        config: Model configuration dataclass.
        model_type: Model architecture type string.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model_type is unknown.
    """
    entry = _MODEL_ENTRYPOINTS.get(model_type)
    if entry is None:
        msg = (
            f"Unknown model type: {model_type!r}. Available: {list(_MODEL_ENTRYPOINTS)}"
        )
        raise ValueError(msg)

    module_path, class_name = entry
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    logger.info("Creating model: %s (type=%s)", model_cls.__name__, model_type)
    return model_cls(config)

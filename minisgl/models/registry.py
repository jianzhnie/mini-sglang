"""Model registry for auto-detection and instantiation.

Supports all model architectures:
- OPTForCausalLM
- Qwen2ForCausalLM
- Qwen3ForCausalLM
- Qwen3MoEForCausalLM
- LlamaForCausalLM
- MistralForCausalLM
"""

from __future__ import annotations

__all__ = ["detect_model_type", "create_model"]
import json
import os
from typing import TYPE_CHECKING

from minisgl.utils.logger import logger

if TYPE_CHECKING:
    import torch.nn as nn

    from minisgl.config import ModelArgs


# Registry: model_type → (lazy_import_path, class_name)
_MODEL_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "qwen2": ("minisgl.models.qwen2", "Qwen2ForCausalLM"),
    "qwen3": ("minisgl.models.qwen3", "Qwen3ForCausalLM"),
    "qwen3_moe": ("minisgl.models.qwen3_moe", "Qwen3MoEForCausalLM"),
    "llama": ("minisgl.models.llama", "LlamaForCausalLM"),
    "mistral": ("minisgl.models.mistral", "MistralForCausalLM"),
    "opt": ("minisgl.models.opt", "OPTForCausalLM"),
}


def detect_model_type(model_path: str) -> str:
    """Detect the model architecture from config.json.

    Args:
        model_path: Path to the HF model directory.

    Returns:
        Model type string (e.g. "qwen2", "llama", "opt").
    """
    config_file = os.path.join(model_path, "config.json")
    with open(config_file) as f:
        cfg = json.load(f)

    architectures = cfg.get("architectures", [])

    for arch in architectures:
        arch_lower = arch.lower()
        if "qwen3moe" in arch_lower or "qwen3_moe" in arch_lower:
            return "qwen3_moe"
        if "qwen3" in arch_lower:
            return "qwen3"
        if "qwen2" in arch_lower:
            return "qwen2"
        if "llama" in arch_lower:
            return "llama"
        if "opt" in arch_lower:
            return "opt"
        if "mistral" in arch_lower:
            return "mistral"

    # Fallback heuristics
    if cfg.get("num_experts", 0) > 0:
        return "qwen3_moe"
    if cfg.get("qk_norm", False):
        return "qwen3"
    if cfg.get("use_sliding_window", False):
        return "mistral"

    logger.warning(
        f"Could not detect model type from architectures: {architectures}. Defaulting to qwen2."
    )
    return "qwen2"


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
        raise ValueError(
            f"Unknown model type: {model_type!r}. Available: {list(_MODEL_ENTRYPOINTS)}"
        )

    module_path, class_name = entry
    import importlib

    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    logger.info(f"Creating model: {model_cls.__name__} (type={model_type})")
    return model_cls(config)

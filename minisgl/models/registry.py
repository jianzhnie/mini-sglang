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

__all__ = ["create_model", "detect_model_type", "get_remap_fn"]
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
        f"Could not detect model type from architectures: {architectures}. Defaulting to qwen2.",
    )
    return "qwen2"


def get_remap_fn(model_type: str):
    """Return a key remapping function for the given model type."""
    if model_type == "opt":

        def _remap(name: str) -> str:
            return name.replace("model.", "model.decoder.")

        return _remap
    return None


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

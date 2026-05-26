"""Model registry for auto-detection and instantiation.

Supports all model architectures:
- Qwen2ForCausalLM
- Qwen3ForCausalLM
- Qwen3MoEForCausalLM
- LlamaForCausalLM
- MistralForCausalLM
"""

import json
import os
from typing import Dict, Optional, Type

import torch.nn as nn

from minisgl.config import ModelArgs
from minisgl.utils.logger import logger

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def get_model_class(name: str) -> Optional[Type[nn.Module]]:
    """Get a registered model class by name."""
    return _MODEL_REGISTRY.get(name)


def detect_model_type(model_path: str) -> str:
    """Detect the model architecture from config.json.

    Args:
        model_path: Path to the HF model directory.

    Returns:
        Model type string (e.g., "qwen2", "llama", "mistral").
    """
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, "r") as f:
        cfg = json.load(f)

    architectures = cfg.get("architectures", [])

    architecture_map = {
        "qwen2": "qwen2",
        "qwen3moe": "qwen3_moe",
        "qwen3_moe": "qwen3_moe",
        "qwen3": "qwen3",
        "llama": "llama",
        "mistral": "mistral",
        "llamaforcausallm": "llama",
        "mistralforcausallm": "mistral",
    }

    for arch in architectures:
        arch_lower = arch.lower().replace(" ", "")

        # Check for MoE variants first
        if "moe" in arch_lower:
            for prefix, model_type in architecture_map.items():
                if prefix in arch_lower:
                    return model_type

        # Check standard architectures
        for prefix, model_type in architecture_map.items():
            if prefix in arch_lower:
                return model_type

    # Heuristic detection from config features
    if cfg.get("num_experts", 0) > 0:
        return "qwen3_moe"

    if cfg.get("qk_norm", False):
        return "qwen3"

    if cfg.get("use_sliding_window", False):
        return "mistral"

    # Default to Qwen2 as the safest bet
    logger.warning(f"Could not detect model type from architectures: {architectures}. "
                   f"Defaulting to qwen2.")
    return "qwen2"


def create_model(config: ModelArgs, model_type: Optional[str] = None) -> nn.Module:
    """Create a model instance from config.

    Args:
        config: Model configuration.
        model_type: Model architecture type. Auto-detected if None.

    Returns:
        Instantiated model (nn.Module).
    """
    if model_type is None and config.model_path:
        model_type = detect_model_type(config.model_path)
    elif model_type is None:
        model_type = "qwen2"

    # Lazy imports to avoid circular dependencies
    from minisgl.models.qwen2 import Qwen2ForCausalLM
    from minisgl.models.qwen3 import Qwen3ForCausalLM
    from minisgl.models.qwen3_moe import Qwen3MoEForCausalLM
    from minisgl.models.llama import LlamaForCausalLM
    from minisgl.models.mistral import MistralForCausalLM

    model_map = {
        "qwen2": Qwen2ForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        "qwen3_moe": Qwen3MoEForCausalLM,
        "llama": LlamaForCausalLM,
        "mistral": MistralForCausalLM,
    }

    model_cls = model_map.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(model_map.keys())}"
        )

    logger.info(f"Creating model: {model_cls.__name__}")
    return model_cls(config)


# Register all model types
register_model("qwen2")(None)  # Will be replaced by actual imports
register_model("qwen3")(None)
register_model("qwen3_moe")(None)
register_model("llama")(None)
register_model("mistral")(None)

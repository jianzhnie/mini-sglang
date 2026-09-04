"""Model architectures (Qwen3 family) and shared layers.

Public entry points are the registry helpers:
``create_model(config, model_type)`` and ``detect_model_type(model_path)``.
Import concrete classes from their own modules (e.g.
``from minisgl.models.qwen3 import Qwen3ForCausalLM``).
"""

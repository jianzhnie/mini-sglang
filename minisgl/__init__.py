"""Mini-SGLang: A lightweight educational LLM inference framework.

Usage:
    python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000
    python -m minisgl --model-path Qwen/Qwen3-0.6B --shell
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import SamplingParams, ServerArgs
    from minisgl.engine.llm import LLM

__all__ = ["LLM", "SamplingParams", "ServerArgs"]

# The heavy symbols (LLM pulls in torch + the whole engine) are resolved lazily
# via module __getattr__ so that importing a lightweight submodule such as
# minisgl.utils.logger does not force a torch import.
_LAZY_EXPORTS = {
    "LLM": ("minisgl.engine.llm", "LLM"),
    "SamplingParams": ("minisgl.config", "SamplingParams"),
    "ServerArgs": ("minisgl.config", "ServerArgs"),
}


def __getattr__(name: str):
    module_name, attr_name = _LAZY_EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

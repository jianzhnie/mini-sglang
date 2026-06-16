#!/usr/bin/env python3
"""High-level LLM API demo: generate and chat.

Usage:
    python examples/llm_generate.py
    python examples/llm_generate.py --model-path /path/to/model
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_ROOT = "/home/jianzhnie/llmtuner/hfhub/models"


def _find_model(*names: str) -> str:
    env_root = os.environ.get("MINISGL_MODELS", "")
    roots = [r for r in [env_root, MODELS_ROOT, str(Path.home() / "hfhub" / "models")] if r]
    for name in names:
        for root in roots:
            p = Path(root) / name
            if p.is_dir() and (p / "config.json").exists():
                return str(p)
    return ""


def _validate(path: str) -> None:
    if not path or not (Path(path) / "config.json").exists():
        print(f"ERROR: Model not found at: {path!r}")
        print(f"  python {sys.argv[0]} --model-path /path/to/hf_model")
        sys.exit(1)


def main(model_path: str) -> None:
    import torch

    from minisgl.engine.llm import LLM

    on_cpu = not torch.cuda.is_available()
    max_tokens = 10 if on_cpu else 60

    print("=" * 60)
    print("  Mini-SGLang LLM API Demo")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}")
    print("=" * 60)

    llm = LLM(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_seq_len=256,
        memory_ratio=0.5,
    )

    print("\n── Text Completion (greedy) ──")
    prompts = [
        "The capital of France is",
        "Machine learning is",
    ]
    for prompt in prompts:
        output = llm.generate(prompt, temperature=0.0, max_tokens=max_tokens)
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {output!r}\n")

    print("── Chat ──")
    messages = [
        {"role": "user", "content": "What is the meaning of life?"},
    ]
    response = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    print(f"  User:      {messages[0]['content']}")
    print(f"  Assistant: {response}\n")

    print("=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang LLM API Demo")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path)

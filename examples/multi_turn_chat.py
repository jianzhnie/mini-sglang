#!/usr/bin/env python3
"""Multi-turn chat demo: demonstrates conversation history management.

Shows how to use the LLM.chat() API with multi-turn conversations,
including system prompts and conversation context.

Usage:
    python examples/multi_turn_chat.py
    python examples/multi_turn_chat.py --model-path /path/to/model
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
    max_tokens = 30 if on_cpu else 100

    print("=" * 60)
    print("  Mini-SGLang Multi-Turn Chat Demo")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}")
    print("=" * 60)

    llm = LLM(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_seq_len=512,
        memory_ratio=0.5,
    )

    conversations = [
        {
            "title": "Science Q&A",
            "system": "You are a helpful science assistant. Answer concisely.",
            "turns": [
                "What is photosynthesis?",
                "How does it relate to climate change?",
                "What can humans do to help?",
            ],
        },
        {
            "title": "Code Helper",
            "system": "You are a Python programming assistant.",
            "turns": [
                "How do I read a file in Python?",
                "What about handling errors?",
            ],
        },
    ]

    for conv in conversations:
        print(f"\n{'─' * 60}")
        print(f"  Conversation: {conv['title']}")
        print(f"  System: {conv['system']}")
        print(f"{'─' * 60}")

        messages = [{"role": "system", "content": conv["system"]}]

        for turn_idx, user_msg in enumerate(conv["turns"], 1):
            messages.append({"role": "user", "content": user_msg})
            print(f"\n  [Turn {turn_idx}]")
            print(f"  User: {user_msg}")

            response = llm.chat(
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            print(f"  Assistant: {response}")

            messages.append({"role": "assistant", "content": response})

    print(f"\n── Parallel Multi-Turn (batch) ──")
    batch_convs = [
        [
            {"role": "user", "content": "What is 2+2?"},
        ],
        [
            {"role": "system", "content": "Reply in one word."},
            {"role": "user", "content": "What color is the sky?"},
        ],
    ]
    responses = llm.chat(batch_convs, temperature=0.0, max_tokens=max_tokens)
    for msgs, resp in zip(batch_convs, responses):
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        print(f"  Q: {user_msg!r} → A: {resp!r}")

    print(f"\n{'=' * 60}")
    print("  MULTI-TURN CHAT DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Turn Chat Demo")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "facebook/opt-125m"
    )
    _validate(model_path)
    main(model_path)

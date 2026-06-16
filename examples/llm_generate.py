#!/usr/bin/env python3
"""High-level LLM API demo: generate and chat.

Usage:
    python examples/llm_generate.py --model-path ~/hfhub/models/Qwen/Qwen3-0.6B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minisgl.engine.llm import LLM

DEFAULT_MODEL = os.path.expanduser("~/hfhub/models/Qwen/Qwen3-0.6B")


def main(model_path: str) -> None:
    print("=" * 60)
    print("  Mini-SGLang LLM API Demo")
    print(f"  Model: {model_path}")
    print("=" * 60)

    llm = LLM(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_seq_len=512,
        memory_ratio=0.5,
    )

    print("\n── Text Completion (greedy) ──")
    prompts = [
        "The capital of France is",
        "Machine learning is",
    ]
    for prompt in prompts:
        output = llm.generate(prompt, temperature=0.0, max_tokens=40)
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {output!r}\n")

    print("── Text Completion (sampling) ──")
    output = llm.generate(
        "Once upon a time",
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        max_tokens=60,
    )
    print(f"  Output: {output!r}\n")

    print("── Chat ──")
    messages = [
        {"role": "user", "content": "What is the meaning of life?"},
    ]
    response = llm.chat(messages, temperature=0.7, max_tokens=100)
    print(f"  User:      {messages[0]['content']}")
    print(f"  Assistant: {response}\n")

    print("── Multi-turn Chat ──")
    messages = [
        {"role": "user", "content": "Tell me a short joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {"role": "user", "content": "Why?"},
    ]
    response = llm.chat(messages, temperature=0.7, max_tokens=60)
    for msg in messages:
        role = msg["role"].capitalize()
        print(f"  {role}: {msg['content']}")
    print(f"  Assistant: {response}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang LLM API Demo")
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL,
        help="Path to HuggingFace model directory",
    )
    args = parser.parse_args()
    main(args.model_path)

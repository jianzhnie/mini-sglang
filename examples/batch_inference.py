#!/usr/bin/env python3
"""Batch inference demo: process multiple prompts concurrently.

Usage:
    python examples/batch_inference.py
    python examples/batch_inference.py --model-path /path/to/model
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
    from transformers import AutoTokenizer

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    on_cpu = not torch.cuda.is_available()
    max_tokens = 10 if on_cpu else 40

    print("=" * 60)
    print("  Mini-SGLang Batch Inference Demo")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_running_req=16,
        max_seq_len=256,
        page_size=16,
        memory_ratio=0.5,
        cuda_graph_bs=0,
    )
    model_args = ModelArgs.from_pretrained(model_path)

    engine = Engine(server_args, model_args, tp_rank=0)
    scheduler = Scheduler(server_args, engine)

    prompts = [
        "Python is a programming language that",
        "The theory of relativity states that",
        "In machine learning, a neural network",
    ]

    uid_to_prompt: dict[int, str] = {}
    uid_to_tokens: dict[int, list[int]] = {}

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        uid = scheduler.add_request(input_ids, sampling)
        uid_to_prompt[uid] = prompt
        uid_to_tokens[uid] = []

    print(f"\nSubmitted {len(prompts)} prompts for batch processing...\n")

    while not scheduler.is_idle():
        results = scheduler.step()
        for uid, token_id, finished in results:
            uid_to_tokens[uid].append(token_id)

    for uid in sorted(uid_to_prompt.keys()):
        prompt = uid_to_prompt[uid]
        tokens = uid_to_tokens[uid]
        output = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"{'─' * 60}")
        print(f"Prompt:  {prompt!r}")
        print(f"Output:  {output!r}")
        print(f"Tokens:  {len(tokens)}")

    print(f"\n{'=' * 60}")
    print("  BATCH DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference Demo")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path)

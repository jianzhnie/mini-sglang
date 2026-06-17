"""Mini-SGLang end-to-end demo: direct Engine + Scheduler usage.

Usage:
    python examples.py
    python examples.py --model-path /path/to/model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS_ROOT = "/home/jianzhnie/llmtuner/hfhub/models"


def _find_model(*names: str) -> str:
    """Search for the first available model directory."""
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


def run_demo(model_path: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    on_cpu = not torch.cuda.is_available()
    max_tokens = 10 if on_cpu else 60

    print("=" * 60)
    print("  Mini-SGLang Engine Demo")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_running_req=4,
        max_seq_len=256,
        page_size=16,
        memory_ratio=0.5,
        cuda_graph_bs=0,
    )
    model_args = ModelArgs.from_pretrained(model_path)
    engine = Engine(server_args, model_args, tp_rank=0)

    prompts = [
        "The capital of France is",
        "Once upon a time in a",
        "The answer to life, the universe, and everything is",
    ]

    for prompt in prompts:
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt!r}")

        scheduler = Scheduler(server_args, engine)
        input_ids = tokenizer.encode(prompt)
        scheduler.add_request(input_ids, SamplingParams(temperature=0.0, max_tokens=max_tokens))

        generated: list[int] = []
        while not scheduler.is_idle():
            for _uid, token_id, finished in scheduler.step():
                generated.append(token_id)
                if finished:
                    break

        output = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Output: {output!r}")
        print(f"({len(generated)} tokens)")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang Engine Demo")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    run_demo(model_path)

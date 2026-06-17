#!/usr/bin/env python3
"""Streaming generation demo: token-by-token output with timing.

Shows how to use the Engine+Scheduler loop to stream tokens as they are
generated, simulating a real-time chat experience.

Usage:
    python examples/streaming_demo.py
    python examples/streaming_demo.py --model-path /path/to/model
"""

import argparse
import os
import sys
import time
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
    max_tokens = 20 if on_cpu else 100

    print("=" * 60)
    print("  Mini-SGLang Streaming Generation Demo")
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
    scheduler = Scheduler(server_args, engine)

    prompts = [
        "Once upon a time in a land far away,",
        "The secret to happiness is",
        "In the year 2050, artificial intelligence",
    ]

    for prompt in prompts:
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt}")
        print("Stream: ", end="", flush=True)

        input_ids = tokenizer.encode(prompt)
        sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        uid = scheduler.add_request(input_ids, sampling)

        token_times = []
        generated_tokens = []
        start = time.perf_counter()

        while not scheduler.is_idle():
            results = scheduler.step()
            for r_uid, token_id, finished in results:
                if r_uid == uid:
                    token_times.append(time.perf_counter() - start)
                    generated_tokens.append(token_id)
                    text = tokenizer.decode([token_id], skip_special_tokens=True)
                    print(text, end="", flush=True)

        total_time = time.perf_counter() - start
        print()

        num_tokens = len(generated_tokens)
        if num_tokens > 0:
            ttft = token_times[0] * 1000
            tps = num_tokens / total_time
            print(f"  Tokens: {num_tokens} | TTFT: {ttft:.1f}ms | "
                  f"Speed: {tps:.1f} tok/s | Total: {total_time:.2f}s")

    print(f"\n{'=' * 60}")
    print("  STREAMING DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Generation Demo")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path)

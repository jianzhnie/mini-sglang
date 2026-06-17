#!/usr/bin/env python3
"""Ascend NPU inference demo.

Shows how to run mini-sglang on Huawei Ascend NPU hardware using torch_npu.
Falls back to CPU if NPU is not available (for testing the code path).

Prerequisites:
    - CANN toolkit installed (e.g., CANN 9.0.0)
    - torch_npu matching your PyTorch version
    - Environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh

Usage:
    python examples/npu_demo.py
    python examples/npu_demo.py --model-path /path/to/model
    python examples/npu_demo.py --device npu  # force NPU
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


def main(model_path: str, force_device: str = "auto") -> None:
    import torch

    from minisgl.utils.device import get_device_type, is_npu_available, set_device

    if force_device != "auto":
        set_device(torch.device(force_device))

    device_type = get_device_type()
    npu_available = is_npu_available()

    print("=" * 60)
    print("  Mini-SGLang Ascend NPU Demo")
    print(f"  Model: {model_path}")
    print(f"  NPU available: {npu_available}")
    print(f"  Using device: {device_type}")
    if npu_available:
        print(f"  NPU device count: {torch.npu.device_count()}")
        print(f"  NPU device name: {torch.npu.get_device_name(0)}")
    print("=" * 60)

    from transformers import AutoTokenizer

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    max_tokens = 40 if device_type != "cpu" else 10

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        attention_backend="pt",
        max_running_req=8,
        max_seq_len=256,
        page_size=16,
        memory_ratio=0.5,
        cuda_graph_bs=0 if device_type == "cpu" else 8,
        device=force_device,
    )
    model_args = ModelArgs.from_pretrained(model_path)

    print(f"\nLoading model to {device_type}...")
    t0 = time.perf_counter()
    engine = Engine(server_args, model_args, tp_rank=0)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    scheduler = Scheduler(server_args, engine)

    prompts = [
        "The capital of China is",
        "Artificial intelligence is transforming",
        "In deep learning, attention mechanism",
    ]

    print(f"\n── Generation (max_tokens={max_tokens}) ──")
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        uid = scheduler.add_request(input_ids, sampling)

        generated_tokens = []
        t0 = time.perf_counter()
        while not scheduler.is_idle():
            results = scheduler.step()
            for r_uid, token_id, finished in results:
                if r_uid == uid:
                    generated_tokens.append(token_id)

        elapsed = time.perf_counter() - t0
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tps = len(generated_tokens) / elapsed if elapsed > 0 else 0

        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {output!r}")
        print(f"  Tokens: {len(generated_tokens)} | {tps:.1f} tok/s | {elapsed:.3f}s")

    print(f"\n── Batch Generation ──")
    uid_map = {}
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        uid = scheduler.add_request(input_ids, sampling)
        uid_map[uid] = {"prompt": prompt, "tokens": []}

    t0 = time.perf_counter()
    while not scheduler.is_idle():
        results = scheduler.step()
        for uid, token_id, finished in results:
            if uid in uid_map:
                uid_map[uid]["tokens"].append(token_id)
    batch_time = time.perf_counter() - t0

    total_tokens = sum(len(v["tokens"]) for v in uid_map.values())
    print(f"  Batch size: {len(prompts)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens / batch_time:.1f} tok/s")
    print(f"  Time: {batch_time:.3f}s")

    print(f"\n{'=' * 60}")
    print("  NPU DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang NPU Demo")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "npu", "cuda", "cpu"],
        help="Force device type",
    )
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path, force_device=args.device)

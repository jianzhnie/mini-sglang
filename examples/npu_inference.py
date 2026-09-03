#!/usr/bin/env python3
"""Ascend NPU inference demo: single model and multi-model testing.

Demonstrates:
  1. NPU device detection and model loading
  2. Single-prompt generation with metrics
  3. Batch generation throughput
  4. Multi-model sequential testing (when --models specified)

Prerequisites:
  - CANN 9.0.0+ with torch_npu
  - Docker: torchtitan-npu:cann9.0.0-torch2.12.0

Usage:
    python examples/npu_inference.py --model-path /path/to/model
    python examples/npu_inference.py --models Qwen3-0.6B Qwen3-1.7B
    python examples/npu_inference.py --model-path /path/to/model --device cpu  # fallback test
"""

import argparse
import os
import sys
import time
from pathlib import Path

# examples/ for _common, repo root for minisgl (see examples/_common.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root

from _common import (  # noqa: E402
    banner,
    build_engine,
    drive,
    find_model,
    load_tokenizer,
)

# Qwen checkpoints tested on Ascend live under this extra root on the author's
# setup; --models looks here in addition to the standard roots.
_QWEN_ROOT = str(Path.home() / "hfhub" / "models" / "Qwen")


def test_model(model_path: str, max_tokens: int = 20) -> dict:
    """Run full inference test on one model. Returns metrics dict."""
    import torch

    from minisgl.config import SamplingParams
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.utils.device import get_device_type

    device_type = get_device_type()
    result = {
        "model": Path(model_path).name,
        "device": device_type,
        "status": "FAIL",
        "load_time": 0.0,
        "prefill_tps": 0.0,
        "decode_tps": 0.0,
        "batch_tps": 0.0,
        "output": "",
        "error": "",
    }

    try:
        # Time the engine construction (loads weights + allocates KV cache).
        t0 = time.perf_counter()
        server_args, engine = build_engine(model_path, attention_backend="pt")
        result["load_time"] = time.perf_counter() - t0

        tokenizer = load_tokenizer(model_path, trust_remote_code=True)

        # Single generation.
        scheduler = Scheduler(server_args, engine)
        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt)
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        uid = scheduler.add_request(input_ids, sampling)

        tokens = []
        t0 = time.perf_counter()
        ttft = None
        for out in drive(scheduler):
            if out.uid == uid:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens.append(out.token_id)
        total_time = time.perf_counter() - t0

        result["output"] = tokenizer.decode(tokens, skip_special_tokens=True)[:80]
        result["prefill_tps"] = len(input_ids) / ttft if ttft else 0
        result["decode_tps"] = len(tokens) / total_time if total_time > 0 else 0

        # Batch test: 3 prompts.
        prompts = [
            "Python is a programming language",
            "The capital of France is",
            "In machine learning,",
        ]
        scheduler2 = Scheduler(server_args, engine)
        for p in prompts:
            scheduler2.add_request(tokenizer.encode(p), sampling)

        total_gen = 0
        t0 = time.perf_counter()
        for _ in drive(scheduler2):
            total_gen += 1
        batch_time = time.perf_counter() - t0
        result["batch_tps"] = total_gen / batch_time if batch_time > 0 else 0

        result["status"] = "PASS"

        del engine
        if device_type == "npu":
            torch.npu.empty_cache()

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    import torch

    from minisgl.utils.device import get_device_type, is_npu_available, set_device

    parser = argparse.ArgumentParser(description="Mini-SGLang NPU Inference")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names (e.g., Qwen3-0.6B) to test in sequence",
    )
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "npu", "cuda", "cpu"]
    )
    args = parser.parse_args()

    if args.device != "auto":
        set_device(torch.device(args.device))

    device_type = get_device_type()
    npu_avail = is_npu_available()

    npu_line = (
        f"  NPU: {torch.npu.get_device_name(0)} x{torch.npu.device_count()}"
        if npu_avail
        else ""
    )
    banner(
        "Mini-SGLang NPU Inference Demo",
        lines=[
            f"Device: {device_type} | NPU available: {npu_avail}",
            npu_line,
            f"Max tokens: {args.max_tokens}",
        ],
    )

    # Determine which models to test.
    if args.models:
        model_paths = []
        for name in args.models:
            p = find_model(name, extra_roots=(_QWEN_ROOT,))
            if p:
                model_paths.append(p)
            else:
                print(f"  WARNING: Model {name!r} not found, skipping")
    elif args.model_path:
        model_paths = [args.model_path]
    else:
        p = find_model("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B")
        if not p:
            print("ERROR: No model found. Use --model-path or --models")
            sys.exit(1)
        model_paths = [p]

    results = []
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(model_paths)}] {Path(model_path).name}")
        print(f"{'─' * 60}")

        r = test_model(model_path, max_tokens=args.max_tokens)
        results.append(r)

        if r["status"] == "PASS":
            print("  Status:  PASS")
            print(f"  Load:    {r['load_time']:.2f}s")
            print(f"  Prefill: {r['prefill_tps']:.1f} tok/s")
            print(f"  Decode:  {r['decode_tps']:.1f} tok/s")
            print(f"  Batch:   {r['batch_tps']:.1f} tok/s")
            print(f"  Output:  {r['output']!r}")
        else:
            print("  Status:  FAIL")
            print(f"  Error:   {r['error']}")

    # Summary.
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Model':<22} {'Status':<7} {'Load':<7} {'Decode':<10} {'Batch':<10}")
        print(f"  {'─' * 55}")
        for r in results:
            load = f"{r['load_time']:.1f}s" if r["load_time"] else "-"
            decode = f"{r['decode_tps']:.1f} t/s" if r["decode_tps"] else "-"
            batch = f"{r['batch_tps']:.1f} t/s" if r["batch_tps"] else "-"
            print(
                f"  {r['model']:<22} {r['status']:<7} {load:<7} {decode:<10} {batch:<10}"
            )

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  Result: {passed}/{len(results)} passed")
    print("=" * 60)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

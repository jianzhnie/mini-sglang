#!/usr/bin/env python3
"""Multi-model NPU benchmark: tests multiple Qwen models on Ascend NPU.

Usage:
    python examples/npu_multi_model_test.py
    python examples/npu_multi_model_test.py --models Qwen3-0.6B Qwen2.5-0.5B
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_ROOT = "/home/jianzhnie/llmtuner/hfhub/models/Qwen"


def get_available_models(max_size_gb: float = 10.0) -> list[str]:
    """Find available Qwen models under size limit."""
    root = Path(MODELS_ROOT)
    if not root.exists():
        return []

    candidates = [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen2.5-0.5B",
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B",
        "Qwen2.5-7B-Instruct",
    ]

    available = []
    for name in candidates:
        p = root / name
        if p.is_dir() and (p / "config.json").exists():
            available.append(name)
    return available


def test_model(model_name: str, max_tokens: int = 20) -> dict:
    """Run inference test on a single model. Returns metrics dict."""
    import torch

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.utils.device import get_device_type

    model_path = str(Path(MODELS_ROOT) / model_name)
    device_type = get_device_type()

    result = {
        "model": model_name,
        "device": device_type,
        "status": "FAIL",
        "load_time": 0,
        "prefill_tps": 0,
        "decode_tps": 0,
        "batch_tps": 0,
        "output": "",
        "error": "",
    }

    try:
        server_args = ServerArgs(
            model_path=model_path,
            tp_size=1,
            attention_backend="pt",
            max_running_req=8,
            max_seq_len=256,
            page_size=16,
            memory_ratio=0.5,
            cuda_graph_bs=0,
        )
        model_args = ModelArgs.from_pretrained(model_path)

        t0 = time.perf_counter()
        engine = Engine(server_args, model_args, tp_rank=0)
        result["load_time"] = time.perf_counter() - t0

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        scheduler = Scheduler(server_args, engine)

        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt)
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        uid = scheduler.add_request(input_ids, sampling)

        generated_tokens = []
        t0 = time.perf_counter()
        first_token_time = None
        while not scheduler.is_idle():
            results = scheduler.step()
            for r_uid, token_id, finished in results:
                if r_uid == uid:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                    generated_tokens.append(token_id)
        total_time = time.perf_counter() - t0

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        result["output"] = output_text[:80]
        result["prefill_tps"] = len(input_ids) / first_token_time if first_token_time else 0
        result["decode_tps"] = len(generated_tokens) / total_time if total_time > 0 else 0

        # Batch test: 3 prompts
        prompts = [
            "Python is a programming language",
            "The capital of France is",
            "In machine learning,",
        ]
        scheduler2 = Scheduler(server_args, engine)
        for p in prompts:
            ids = tokenizer.encode(p)
            scheduler2.add_request(ids, SamplingParams(temperature=0.0, max_tokens=max_tokens))

        total_gen = 0
        t0 = time.perf_counter()
        while not scheduler2.is_idle():
            step_results = scheduler2.step()
            total_gen += len(step_results)
        batch_time = time.perf_counter() - t0
        result["batch_tps"] = total_gen / batch_time if batch_time > 0 else 0

        result["status"] = "PASS"

        del engine
        if torch.npu.is_available():
            torch.npu.empty_cache()

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-model NPU Test")
    parser.add_argument("--models", nargs="+", default=None, help="Model names to test")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    import torch
    from minisgl.utils.device import get_device_type, is_npu_available

    device_type = get_device_type()
    npu_avail = is_npu_available()

    print("=" * 70)
    print("  Mini-SGLang Multi-Model NPU Test")
    print(f"  Device: {device_type} | NPU available: {npu_avail}")
    if npu_avail:
        print(f"  NPU count: {torch.npu.device_count()}")
        print(f"  NPU name: {torch.npu.get_device_name(0)}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 70)

    if args.models:
        models = args.models
    else:
        models = get_available_models()

    if not models:
        print("ERROR: No models found!")
        sys.exit(1)

    print(f"\n  Models to test ({len(models)}):")
    for m in models:
        print(f"    - {m}")

    results = []
    for i, model_name in enumerate(models, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(models)}] Testing: {model_name}")
        print(f"{'─' * 70}")

        r = test_model(model_name, max_tokens=args.max_tokens)
        results.append(r)

        if r["status"] == "PASS":
            print(f"  ✓ Status: PASS")
            print(f"    Load time:    {r['load_time']:.2f}s")
            print(f"    Prefill:      {r['prefill_tps']:.1f} tok/s")
            print(f"    Decode:       {r['decode_tps']:.1f} tok/s")
            print(f"    Batch (3x):   {r['batch_tps']:.1f} tok/s")
            print(f"    Output:       {r['output']!r}")
        else:
            print(f"  ✗ Status: FAIL")
            print(f"    Error: {r['error']}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25} {'Status':<8} {'Load(s)':<9} {'Decode':<12} {'Batch':<12}")
    print(f"  {'─' * 65}")
    for r in results:
        status = r["status"]
        load = f"{r['load_time']:.1f}" if r['load_time'] else "-"
        decode = f"{r['decode_tps']:.1f} t/s" if r['decode_tps'] else "-"
        batch = f"{r['batch_tps']:.1f} t/s" if r['batch_tps'] else "-"
        print(f"  {r['model']:<25} {status:<8} {load:<9} {decode:<12} {batch:<12}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  Result: {passed}/{len(results)} models passed")
    print(f"{'=' * 70}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

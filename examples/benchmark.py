#!/usr/bin/env python3
"""Performance benchmark: measures prefill and decode throughput.

Benchmarks:
  1. Prefill latency (time to first token) at various input lengths
  2. Decode throughput (tokens/second) at various batch sizes
  3. End-to-end generation throughput

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --model-path /path/to/model
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _find_model(*names: str) -> str:
    """Locate a model by name under common local roots (env var first)."""
    env_root = os.environ.get("MINISGL_MODELS", "")
    roots = [
        r
        for r in [
            env_root,
            str(Path.home() / ".cache" / "huggingface" / "hub"),
            str(Path.home() / "hfhub" / "models"),
        ]
        if r
    ]
    for name in names:
        for root in roots:
            # name may be a plain "org/model" (HF hub cache layout) or an
            # absolute path already.
            candidates = [Path(name), Path(root) / name]
            if "/" in name:
                candidates.append(Path(root) / f"models--{name.replace('/', '--')}")
            for p in candidates:
                if p.is_dir() and (p / "config.json").exists():
                    return str(p)
    return ""


def _validate(path: str) -> None:
    if not path or not (Path(path) / "config.json").exists():
        print(f"ERROR: Model not found at: {path!r}")
        print(f"  python {sys.argv[0]} --model-path /path/to/hf_model")
        sys.exit(1)


def benchmark_prefill(engine, scheduler_cls, server_args, tokenizer, input_lengths):
    """Benchmark time-to-first-token at various input lengths."""
    from minisgl.config import SamplingParams

    print("\n── Prefill Latency (Time to First Token) ──")
    print(f"  {'Input Len':<12}{'TTFT (ms)':<12}{'Prefill tok/s':<15}")
    print(f"  {'─' * 38}")

    for input_len in input_lengths:
        input_ids = list(range(1, input_len + 1))
        sampling = SamplingParams(temperature=0.0, max_tokens=1)

        # Warmup: the first scheduler/forward after weight load pays cold-start
        # (page-table setup, allocator warm-up). Drop it from the measurement.
        warmup = scheduler_cls(server_args, engine)
        warmup.add_request(list(input_ids), sampling)
        while not warmup.is_idle():
            warmup.step()

        # Repeat to damp scheduler-noise; report the best (fastest) run, which
        # best reflects steady-state prefill throughput.
        best = float("inf")
        for _ in range(3):
            scheduler = scheduler_cls(server_args, engine)
            scheduler.add_request(list(input_ids), sampling)
            start = time.perf_counter()
            while not scheduler.is_idle():
                scheduler.step()
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)

        ttft_ms = best * 1000
        prefill_tps = input_len / best
        print(f"  {input_len:<12}{ttft_ms:<12.1f}{prefill_tps:<15.0f}")


def benchmark_decode_throughput(
    engine, scheduler_cls, server_args, tokenizer, batch_sizes
):
    """Benchmark decode throughput at various batch sizes."""
    from minisgl.config import SamplingParams

    decode_tokens = 20

    print("\n── Decode Throughput (batch generation) ──")
    print(
        f"  {'Batch Size':<12}{'Total tok/s':<14}{'Per-req tok/s':<15}{'Time (s)':<10}"
    )
    print(f"  {'─' * 50}")

    for batch_size in batch_sizes:
        sampling = SamplingParams(temperature=0.0, max_tokens=decode_tokens)
        input_ids = list(range(1, 17))

        # Warmup with the same shape so cold-start does not skew the numbers.
        warmup = scheduler_cls(server_args, engine)
        for _ in range(batch_size):
            warmup.add_request(list(input_ids), sampling)
        while not warmup.is_idle():
            warmup.step()

        scheduler = scheduler_cls(server_args, engine)
        for _ in range(batch_size):
            scheduler.add_request(list(input_ids), sampling)

        total_generated = 0
        start = time.perf_counter()
        while not scheduler.is_idle():
            results = scheduler.step()
            # Count real generated tokens only (aborted requests carry no token).
            total_generated += sum(1 for r in results if r.finish_reason != "abort")
        elapsed = time.perf_counter() - start

        total_tps = total_generated / elapsed if elapsed > 0 else 0
        per_req_tps = total_tps / batch_size if batch_size > 0 else 0
        print(
            f"  {batch_size:<12}{total_tps:<14.1f}{per_req_tps:<15.1f}{elapsed:<10.3f}"
        )


def benchmark_e2e(engine, scheduler_cls, server_args, tokenizer):
    """End-to-end benchmark with realistic prompts."""
    from minisgl.config import SamplingParams

    prompts = [
        "The meaning of life is",
        "Python programming language was created by",
        "Machine learning algorithms can be used to",
        "The capital of Japan is",
    ]

    decode_tokens = 30
    sampling = SamplingParams(temperature=0.0, max_tokens=decode_tokens)

    def _run() -> tuple[int, float]:
        scheduler = scheduler_cls(server_args, engine)
        for prompt in prompts:
            scheduler.add_request(tokenizer.encode(prompt), sampling)
        total_generated = 0
        start = time.perf_counter()
        while not scheduler.is_idle():
            results = scheduler.step()
            total_generated += sum(1 for r in results if r.finish_reason != "abort")
        return total_generated, time.perf_counter() - start

    print("\n── End-to-End Generation ──")
    total_input = sum(len(tokenizer.encode(p)) for p in prompts)

    # Warmup (discard), then measure.
    _run()
    total_generated, elapsed = _run()

    print(f"  Prompts: {len(prompts)}")
    print(f"  Total input tokens: {total_input}")
    print(f"  Total output tokens: {total_generated}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Overall throughput: {total_generated / elapsed:.1f} tok/s")
    print(f"  Latency per token: {elapsed / total_generated * 1000:.2f} ms/tok")


def main(model_path: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    on_cpu = not torch.cuda.is_available()

    print("=" * 60)
    print("  Mini-SGLang Performance Benchmark")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}")
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

    input_lengths = [8, 16, 32, 64, 128] if not on_cpu else [8, 16, 32]
    batch_sizes = [1, 2, 4, 8] if not on_cpu else [1, 2, 4]

    benchmark_prefill(engine, Scheduler, server_args, tokenizer, input_lengths)
    benchmark_decode_throughput(engine, Scheduler, server_args, tokenizer, batch_sizes)
    benchmark_e2e(engine, Scheduler, server_args, tokenizer)

    print(f"\n{'=' * 60}")
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang Benchmark")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path)

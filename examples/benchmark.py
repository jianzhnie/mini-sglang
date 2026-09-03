#!/usr/bin/env python3
"""Performance benchmark: measures prefill and decode throughput.

Benchmarks:
  1. Prefill latency (time to first token) at various input lengths
  2. Decode throughput (tokens/second) at various batch sizes
  3. End-to-end generation throughput

Each scenario is warmed up first so cold-start (page-table setup, allocator
warm-up right after weight load) does not skew the numbers, and only real
generated tokens count (aborted requests carry no token).

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --model-path /path/to/model
"""

import os
import sys
import time

# examples/ for _common, repo root for minisgl (see examples/_common.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root

from _common import (  # noqa: E402
    banner,
    build_engine,
    cli_main,
    drive,
    load_tokenizer,
)

from minisgl.config import SamplingParams  # noqa: E402
from minisgl.scheduler.scheduler import Scheduler  # noqa: E402


def _scheduler_tokens(scheduler) -> int:
    """Run to idle and count generated tokens, excluding aborted requests."""
    return sum(1 for out in drive(scheduler) if out.finish_reason != "abort")


def benchmark_prefill(engine, server_args, tokenizer, input_lengths):
    """Benchmark time-to-first-token at various input lengths."""
    print("\n── Prefill Latency (Time to First Token) ──")
    print(f"  {'Input Len':<12}{'TTFT (ms)':<12}{'Prefill tok/s':<15}")
    print(f"  {'─' * 38}")

    for input_len in input_lengths:
        input_ids = list(range(1, input_len + 1))
        sampling = SamplingParams(temperature=0.0, max_tokens=1)

        # Warmup: drop the cold run before measuring.
        warmup = Scheduler(server_args, engine)
        warmup.add_request(list(input_ids), sampling)
        for _ in drive(warmup):
            pass

        # Repeat to damp scheduler-noise; report the best (fastest) run, which
        # best reflects steady-state prefill throughput.
        best = float("inf")
        for _ in range(3):
            scheduler = Scheduler(server_args, engine)
            scheduler.add_request(list(input_ids), sampling)
            start = time.perf_counter()
            for _ in drive(scheduler):
                pass
            best = min(best, time.perf_counter() - start)

        ttft_ms = best * 1000
        prefill_tps = input_len / best
        print(f"  {input_len:<12}{ttft_ms:<12.1f}{prefill_tps:<15.0f}")


def benchmark_decode_throughput(engine, server_args, tokenizer, batch_sizes):
    """Benchmark decode throughput at various batch sizes."""
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
        warmup = Scheduler(server_args, engine)
        for _ in range(batch_size):
            warmup.add_request(list(input_ids), sampling)
        for _ in drive(warmup):
            pass

        scheduler = Scheduler(server_args, engine)
        for _ in range(batch_size):
            scheduler.add_request(list(input_ids), sampling)

        start = time.perf_counter()
        total_generated = _scheduler_tokens(scheduler)
        elapsed = time.perf_counter() - start

        total_tps = total_generated / elapsed if elapsed > 0 else 0
        per_req_tps = total_tps / batch_size if batch_size > 0 else 0
        print(
            f"  {batch_size:<12}{total_tps:<14.1f}{per_req_tps:<15.1f}{elapsed:<10.3f}"
        )


def benchmark_e2e(engine, server_args, tokenizer):
    """End-to-end benchmark with realistic prompts."""
    prompts = [
        "The meaning of life is",
        "Python programming language was created by",
        "Machine learning algorithms can be used to",
        "The capital of Japan is",
    ]

    decode_tokens = 30
    sampling = SamplingParams(temperature=0.0, max_tokens=decode_tokens)

    def _run() -> tuple[int, float]:
        scheduler = Scheduler(server_args, engine)
        for prompt in prompts:
            scheduler.add_request(tokenizer.encode(prompt), sampling)
        start = time.perf_counter()
        total_generated = _scheduler_tokens(scheduler)
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

    on_cpu = not torch.cuda.is_available()

    banner(
        "Mini-SGLang Performance Benchmark",
        lines=[f"Model: {model_path}", f"Device: {'CPU' if on_cpu else 'CUDA'}"],
    )

    server_args, engine = build_engine(model_path, max_running_req=16)
    tokenizer = load_tokenizer(model_path)

    input_lengths = [8, 16, 32, 64, 128] if not on_cpu else [8, 16, 32]
    batch_sizes = [1, 2, 4, 8] if not on_cpu else [1, 2, 4]

    benchmark_prefill(engine, server_args, tokenizer, input_lengths)
    benchmark_decode_throughput(engine, server_args, tokenizer, batch_sizes)
    benchmark_e2e(engine, server_args, tokenizer)

    banner("BENCHMARK COMPLETE")


if __name__ == "__main__":
    cli_main("Mini-SGLang Benchmark", main)

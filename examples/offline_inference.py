#!/usr/bin/env python3
"""Offline inference demo: covers all key inference modes.

Demonstrates:
  1. Engine + Scheduler direct usage (single & batch)
  2. High-level LLM API (generate + chat)
  3. Streaming token-by-token generation with metrics
  4. Sampling strategy comparison (greedy vs temperature vs top-p)

Usage:
    python examples/offline_inference.py --model-path /path/to/model
    python examples/offline_inference.py  # auto-detect model
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
    section,
)

from minisgl.config import SamplingParams  # noqa: E402
from minisgl.scheduler.scheduler import Scheduler  # noqa: E402


def demo_engine_scheduler(engine, server_args, tokenizer, max_tokens):
    """Part 1: Direct Engine + Scheduler usage with batch generation."""
    section("Part 1: Engine + Scheduler (Batch Generation)")

    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "In deep learning, attention mechanism",
    ]
    uid_map = {}
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    scheduler = Scheduler(server_args, engine)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        uid = scheduler.add_request(input_ids, sampling)
        uid_map[uid] = {"prompt": prompt, "tokens": []}

    t0 = time.perf_counter()
    for out in drive(scheduler):
        uid_map[out.uid]["tokens"].append(out.token_id)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(v["tokens"]) for v in uid_map.values())
    for uid in sorted(uid_map):
        entry = uid_map[uid]
        output = tokenizer.decode(entry["tokens"], skip_special_tokens=True)
        print(f"\n  Prompt: {entry['prompt']!r}")
        print(f"  Output: {output!r}")

    print(
        f"\n  [{len(prompts)} prompts, {total_tokens} tokens, "
        f"{total_tokens / elapsed:.1f} tok/s, {elapsed:.2f}s]"
    )


def demo_llm_api(model_path, max_tokens):
    """Part 2: High-level LLM API (generate + chat)."""
    from minisgl.engine.llm import LLM

    section("Part 2: LLM API (generate + chat)")

    llm = LLM(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_seq_len=256,
        memory_ratio=0.5,
    )

    prompts = ["The meaning of life is", "Machine learning is"]
    print("\n  ── Text Completion ──")
    outputs = llm.generate(prompts, temperature=0.0, max_tokens=max_tokens)
    for prompt, output in zip(prompts, outputs, strict=True):
        print(f"  {prompt!r} → {output!r}")

    print("\n  ── Chat ──")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    print("  User: What is 2+2?")
    print(f"  Assistant: {response}")

    print("\n  ── Multi-turn Chat ──")
    conversation = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "What is Python?"},
    ]
    r1 = llm.chat(conversation, temperature=0.0, max_tokens=max_tokens)
    print("  User: What is Python?")
    print(f"  Assistant: {r1}")
    conversation.append({"role": "assistant", "content": r1})
    conversation.append({"role": "user", "content": "Who created it?"})
    r2 = llm.chat(conversation, temperature=0.0, max_tokens=max_tokens)
    print("  User: Who created it?")
    print(f"  Assistant: {r2}")

    llm.cleanup()


def demo_streaming(engine, server_args, tokenizer, max_tokens):
    """Part 3: Streaming generation with TTFT and throughput metrics."""
    section("Part 3: Streaming Generation")

    prompt = "Once upon a time in a land far away,"
    input_ids = tokenizer.encode(prompt)
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)

    scheduler = Scheduler(server_args, engine)
    uid = scheduler.add_request(input_ids, sampling)

    print(f"\n  Prompt: {prompt!r}")
    print("  Stream: ", end="", flush=True)

    tokens = []
    start = time.perf_counter()
    first_token_time = None
    for out in drive(scheduler):
        if out.uid == uid:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            tokens.append(out.token_id)
            print(
                tokenizer.decode([out.token_id], skip_special_tokens=True),
                end="",
                flush=True,
            )
    total_time = time.perf_counter() - start
    print()

    ttft_ms = first_token_time * 1000 if first_token_time else 0
    tps = len(tokens) / total_time if total_time > 0 else 0
    print(f"  [{len(tokens)} tokens, TTFT={ttft_ms:.0f}ms, {tps:.1f} tok/s]")


def demo_sampling(engine, server_args, tokenizer, max_tokens):
    """Part 4: Compare sampling strategies."""
    section("Part 4: Sampling Strategies")

    prompt = "The secret to happiness is"
    input_ids = tokenizer.encode(prompt)

    strategies = [
        ("Greedy", SamplingParams(temperature=0.0, max_tokens=max_tokens)),
        ("Temp=0.7", SamplingParams(temperature=0.7, max_tokens=max_tokens)),
        (
            "Temp=1.2 + Top-p=0.9",
            SamplingParams(temperature=1.2, top_p=0.9, max_tokens=max_tokens),
        ),
        (
            "Top-k=10 + Temp=0.8",
            SamplingParams(temperature=0.8, top_k=10, max_tokens=max_tokens),
        ),
    ]

    print(f"\n  Prompt: {prompt!r}\n")
    for label, sampling in strategies:
        scheduler = Scheduler(server_args, engine)
        scheduler.add_request(list(input_ids), sampling)
        tokens = [out.token_id for out in drive(scheduler)]
        output = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  [{label:>22}] {output!r}")


def main(model_path: str):
    import torch

    on_cpu = not torch.cuda.is_available()
    max_tokens = 15 if on_cpu else 40

    banner(
        "Mini-SGLang Offline Inference Demo",
        lines=[
            f"Model: {model_path}",
            f"Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}",
        ],
    )

    server_args, engine = build_engine(model_path)
    tokenizer = load_tokenizer(model_path, trust_remote_code=True)

    demo_engine_scheduler(engine, server_args, tokenizer, max_tokens)
    demo_streaming(engine, server_args, tokenizer, max_tokens)
    demo_sampling(engine, server_args, tokenizer, max_tokens)
    demo_llm_api(model_path, max_tokens)

    banner("ALL DEMOS COMPLETE")


if __name__ == "__main__":
    cli_main("Mini-SGLang Offline Inference", main)

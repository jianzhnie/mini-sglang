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

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_ROOT = "/home/jianzhnie/llmtuner/hfhub/models"


def _find_model(*names: str) -> str:
    env_root = os.environ.get("MINISGL_MODELS", "")
    roots = [
        r for r in [env_root, MODELS_ROOT, str(Path.home() / "hfhub" / "models")] if r
    ]
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


def demo_engine_scheduler(model_path: str, max_tokens: int):
    """Part 1: Direct Engine + Scheduler usage with batch generation."""
    from transformers import AutoTokenizer

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    print("\n" + "=" * 60)
    print("  Part 1: Engine + Scheduler (Batch Generation)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_running_req=8,
        max_seq_len=256,
        page_size=16,
        memory_ratio=0.5,
        cuda_graph_bs=0,
    )
    model_args = ModelArgs.from_pretrained(model_path)
    engine = Engine(server_args, model_args, tp_rank=0)
    scheduler = Scheduler(server_args, engine)

    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "In deep learning, attention mechanism",
    ]
    uid_map = {}
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        uid = scheduler.add_request(input_ids, sampling)
        uid_map[uid] = {"prompt": prompt, "tokens": []}

    t0 = time.perf_counter()
    while not scheduler.is_idle():
        for uid, token_id, _finished, _reason in scheduler.step():
            uid_map[uid]["tokens"].append(token_id)
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
    return engine, server_args, tokenizer


def demo_llm_api(model_path: str, max_tokens: int):
    """Part 2: High-level LLM API (generate + chat)."""
    from minisgl.engine.llm import LLM

    print("\n" + "=" * 60)
    print("  Part 2: LLM API (generate + chat)")
    print("=" * 60)

    llm = LLM(
        model_path=model_path,
        tp_size=1,
        attention_backend="fa",
        max_seq_len=256,
        memory_ratio=0.5,
    )

    print("\n  ── Text Completion ──")
    outputs = llm.generate(
        ["The meaning of life is", "Machine learning is"],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    for prompt, output in zip(
        ["The meaning of life is", "Machine learning is"], outputs, strict=True
    ):
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


def demo_streaming(engine, server_args, tokenizer, max_tokens: int):
    """Part 3: Streaming generation with TTFT and throughput metrics."""
    from minisgl.config import SamplingParams
    from minisgl.scheduler.scheduler import Scheduler

    print("\n" + "=" * 60)
    print("  Part 3: Streaming Generation")
    print("=" * 60)

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
    while not scheduler.is_idle():
        for r_uid, token_id, _finished, _reason in scheduler.step():
            if r_uid == uid:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                tokens.append(token_id)
                print(
                    tokenizer.decode([token_id], skip_special_tokens=True),
                    end="",
                    flush=True,
                )
    total_time = time.perf_counter() - start
    print()

    ttft_ms = first_token_time * 1000 if first_token_time else 0
    tps = len(tokens) / total_time if total_time > 0 else 0
    print(f"  [{len(tokens)} tokens, TTFT={ttft_ms:.0f}ms, {tps:.1f} tok/s]")


def demo_sampling(engine, server_args, tokenizer, max_tokens: int):
    """Part 4: Compare sampling strategies."""
    from minisgl.config import SamplingParams
    from minisgl.scheduler.scheduler import Scheduler

    print("\n" + "=" * 60)
    print("  Part 4: Sampling Strategies")
    print("=" * 60)

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
        tokens = []
        while not scheduler.is_idle():
            for _, token_id, _, _reason in scheduler.step():
                tokens.append(token_id)
        output = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  [{label:>22}] {output!r}")


def main(model_path: str):
    import torch

    on_cpu = not torch.cuda.is_available()
    max_tokens = 15 if on_cpu else 40

    print("=" * 60)
    print("  Mini-SGLang Offline Inference Demo")
    print(f"  Model: {model_path}")
    print(f"  Device: {'CPU' if on_cpu else 'CUDA'}  max_tokens={max_tokens}")
    print("=" * 60)

    engine, server_args, tokenizer = demo_engine_scheduler(model_path, max_tokens)
    demo_streaming(engine, server_args, tokenizer, max_tokens)
    demo_sampling(engine, server_args, tokenizer, max_tokens)
    demo_llm_api(model_path, max_tokens)

    print("\n" + "=" * 60)
    print("  ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-SGLang Offline Inference")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _find_model(
        "facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"
    )
    _validate(model_path)
    main(model_path)

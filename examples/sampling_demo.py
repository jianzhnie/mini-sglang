#!/usr/bin/env python3
"""Temperature and sampling comparison demo — no model download required.

Creates a tiny random-weight model and demonstrates how different sampling
parameters (greedy, temperature, top-k, top-p) affect generation diversity.

Usage:
    python examples/sampling_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    import json
    import tempfile

    import torch
    import torch.nn as nn

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    print("=" * 60)
    print("  Mini-SGLang Sampling Strategies Demo")
    print("  (no model download required)")
    print("=" * 60)

    hidden_size = 128
    num_layers = 2
    num_heads = 4
    vocab_size = 256

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "architectures": ["OPTForCausalLM"],
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "intermediate_size": 512,
            "vocab_size": vocab_size,
            "max_position_embeddings": 64,
            "ffn_dim": 512,
            "eos_token_id": 2,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        model_args = ModelArgs.from_pretrained(tmpdir)
        server_args = ServerArgs(
            model_path=tmpdir,
            tp_size=1,
            attention_backend="fa",
            max_running_req=4,
            max_seq_len=64,
            page_size=8,
            memory_ratio=0.5,
            cuda_graph_bs=0,
        )

        engine = Engine(server_args, model_args, tp_rank=0)
        gen = torch.Generator().manual_seed(42)
        for param in engine.model.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, std=0.02, generator=gen)
            elif param.dim() == 1:
                nn.init.ones_(param)

        input_ids = [1, 10, 20, 30, 40, 50]
        max_tokens = 8

        strategies = [
            ("Greedy (temperature=0)", SamplingParams(temperature=0.0, max_tokens=max_tokens)),
            ("Low temp (t=0.3)", SamplingParams(temperature=0.3, max_tokens=max_tokens)),
            ("High temp (t=1.5)", SamplingParams(temperature=1.5, max_tokens=max_tokens)),
            ("Top-k=10, t=0.8", SamplingParams(temperature=0.8, top_k=10, max_tokens=max_tokens)),
            ("Top-p=0.9, t=0.8", SamplingParams(temperature=0.8, top_p=0.9, max_tokens=max_tokens)),
        ]

        print(f"\nInput tokens: {input_ids}")
        print(f"Max output tokens: {max_tokens}")

        for label, sampling in strategies:
            print(f"\n{'─' * 60}")
            print(f"  Strategy: {label}")

            runs = []
            for trial in range(2):
                scheduler = Scheduler(server_args, engine)
                scheduler.add_request(list(input_ids), sampling)
                generated = []
                while not scheduler.is_idle():
                    for _uid, token_id, finished in scheduler.step():
                        generated.append(token_id)
                runs.append(generated)

            for i, tokens in enumerate(runs):
                print(f"    Run {i+1}: {tokens}")

            unique = len(set(tuple(r) for r in runs))
            if unique == 1:
                print(f"    → Deterministic (all runs identical)")
            else:
                print(f"    → {unique}/2 unique sequences (stochastic)")

    print(f"\n{'=' * 60}")
    print("  SAMPLING DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

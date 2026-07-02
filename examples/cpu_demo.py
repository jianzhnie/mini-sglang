#!/usr/bin/env python3
"""Self-contained CPU demo — no model download required.

Creates a tiny random-weight OPT model and runs the full
Engine → Scheduler → Generate pipeline on CPU.

Usage:
    python examples/cpu_demo.py
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
    print("  Mini-SGLang CPU Self-Contained Demo")
    print("  (no model download required)")
    print("=" * 60)

    hidden_size = 128
    num_layers = 2
    num_heads = 4
    num_kv_heads = 4
    intermediate_size = 512
    vocab_size = 256
    max_pos = 64
    head_dim = hidden_size // num_heads

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "architectures": ["OPTForCausalLM"],
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_pos,
            "ffn_dim": intermediate_size,
            "eos_token_id": 2,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        model_args = ModelArgs.from_pretrained(tmpdir)
        server_args = ServerArgs(
            model_path=tmpdir,
            tp_size=1,
            attention_backend="pt",
            max_running_req=4,
            max_seq_len=max_pos,
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

        prompts = {
            "Single": [1, 5, 10, 15, 20, 25],
            "Short": [3, 7, 11],
            "Pair": [50, 100, 150, 200],
        }

        for label, input_ids in prompts.items():
            print(f"\n{'─' * 60}")
            print(f"Prompt ({label}): token_ids={input_ids}")

            scheduler = Scheduler(server_args, engine)
            scheduler.add_request(
                input_ids, SamplingParams(temperature=0.0, max_tokens=8)
            )

            generated: list[int] = []
            while not scheduler.is_idle():
                for _uid, token_id, finished in scheduler.step():
                    generated.append(token_id)
                    if finished:
                        break

            print(f"Generated:  token_ids={generated}")
            print(f"Tokens: {len(generated)}")

    print("\n" + "=" * 60)
    print("  CPU DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

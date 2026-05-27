"""Mini-SGLang end-to-end demo with OPT-125M.

Run: python3 examples.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

from minisgl.config import ModelArgs, SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.scheduler.scheduler import Scheduler

# MODEL_PATH = os.path.expanduser("~/hfhub/models/facebook/opt-125m")
MODEL_PATH = os.path.expanduser("~/hfhub/models/Qwen/Qwen3-0.6B")



def run_demo() -> None:
    print("=" * 60)
    print("  Mini-SGLang  Demo with OPT-125M")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    server_args = ServerArgs(
        model_path=MODEL_PATH,
        tp_size=1,
        attention_backend="fa",
        max_running_req=8,
        max_seq_len=512,
        page_size=16,
        memory_ratio=0.5,
    )
    model_args = ModelArgs.from_pretrained(MODEL_PATH)
    engine = Engine(server_args, model_args, tp_rank=0)

    prompts = [
        "The capital of France is",
        "Once upon a time in a",
        "The answer to life, the universe, and everything is",
        "To be or not to be,",
        "In the beginning, God created",
    ]

    for prompt in prompts:
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt!r}")

        scheduler = Scheduler(server_args, engine)
        input_ids = tokenizer.encode(prompt)
        scheduler.add_request(input_ids, SamplingParams(temperature=0.0, max_tokens=60))

        generated: list[int] = []
        while not scheduler.is_idle():
            for _uid, token_id, finished in scheduler.step():
                generated.append(token_id)
                if finished:
                    break

        output = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Output: {output!r}")
        print(f"({len(generated)} tokens)")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

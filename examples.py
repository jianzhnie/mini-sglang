"""Mini-SGLang end-to-end demo with real OPT-125M model.

Run: python3 examples.py
"""

import os
import sys
import torch

sys.path.insert(0, ".")

from minisgl.config import ModelArgs, SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import logger

MODEL_PATH = os.path.expanduser("~/hfhub/models/facebook/opt-125m")


def init_engine():
    server_args = ServerArgs(
        model_path=MODEL_PATH, tp_size=1, attention_backend="fa",
        max_running_req=8, max_seq_len=512, page_size=16, memory_ratio=0.5,
    )
    model_args = ModelArgs.from_pretrained(MODEL_PATH)

    from minisgl.models.opt import OPTForCausalLM
    from minisgl.models.attention.backend import AttentionBackend
    from minisgl.utils.device import get_device
    from minisgl.engine.context import BatchContext
    from minisgl.sampling.sampler import Sampler
    from minisgl.engine.engine import Engine

    AttentionBackend.configure("fa")
    device = get_device()

    logger.info(f"Creating OPT-125M on {device}...")
    model = OPTForCausalLM(model_args)
    model.to(device)
    model.eval()

    logger.info("Loading OPT-125M weights...")
    state = torch.load(
        os.path.join(MODEL_PATH, "pytorch_model.bin"),
        map_location=device, weights_only=True,
    )
    params = dict(model.named_parameters())

    for name, param in params.items():
        hf_name = name.replace("model.", "model.decoder.")
        if hf_name in state:
            w = state[hf_name]
            if param.shape == w.shape:
                param.data.copy_(w.to(device, dtype=param.dtype))
            elif name == "model.embed_positions.weight":
                param.data.copy_(w[:param.shape[0]].to(device, dtype=param.dtype))

    logger.info(f"Loaded weights")

    engine = Engine.__new__(Engine)
    engine.server_args = server_args
    engine.model_args = model_args
    engine.tp_rank = 0
    engine.tp_size = 1
    engine.device = device
    engine.model = model
    engine.kv_cache_pool = engine._allocate_kv_cache()
    engine._assign_kv_cache()
    engine.batch_context = BatchContext(
        server_args.max_running_req, server_args.max_seq_len,
        server_args.page_size, engine.device,
    )
    engine.sampler = Sampler(model_args.vocab_size)
    engine.cuda_graphs = {}
    return engine, server_args


def _decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def run_demo():
    print("=" * 60)
    print("  Mini-SGLang  Demo with OPT-125M")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    engine, server_args = init_engine()

    prompts_to_test = [
        "The capital of France is",
        "Once upon a time in a",
        "The answer to life, the universe, and everything is",
        "To be or not to be,",
        "In the beginning, God created",
    ]

    for prompt in prompts_to_test:
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt!r}")

        scheduler = Scheduler(server_args, engine)
        input_ids = tokenizer.encode(prompt)
        _uid = scheduler.add_request(
            input_ids,
            SamplingParams(temperature=0.0, max_tokens=20),
        )

        generated = []
        while not scheduler.is_idle():
            results = scheduler.step()
            for _uid, token_id, finished in results:
                generated.append(token_id)
                if finished:
                    break

        output = _decode_tokens(tokenizer, generated)
        print(f"Output: {output!r}")
        print(f"({len(generated)} tokens)")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

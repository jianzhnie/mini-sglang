"""Numerical golden test: mini-sglang vs HuggingFace reference.

Loads the same real model directory through both stacks and asserts that
greedy generation matches token-for-token. This is the strongest guard
against silent numerical drift in the RoPE / QK-norm / paged-KV / sampling
path.

Requires a local model: set MINISGL_TEST_MODELS (os.pathsep-separated) to one
or more directories. Only Qwen3-family architectures are compared; anything
else is skipped. Runs slowly on CPU — keep prompts short.

Run: python3 tests/engine/test_numerical_golden.py
     MINISGL_TEST_MODELS=/path/to/Qwen3-0.6B python3 tests/engine/test_numerical_golden.py
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_GEN_TOKENS = 6
_PROMPT = "The capital of France is"


def _model_paths() -> list[str]:
    raw = os.environ.get("MINISGL_TEST_MODELS", "")
    return [p for p in raw.split(os.pathsep) if p]


def _is_qwen3(model_path: str) -> bool:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return any("qwen3" in a.lower() for a in cfg.get("architectures", []))


def _hf_generate(model_path: str, prompt: str, n_tokens: int) -> list[int]:
    """Greedy-generate ``n_tokens`` with HF; return full token ids."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok.encode(prompt)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    inp = torch.tensor([ids])
    with torch.inference_mode():
        for _ in range(n_tokens):
            out = model(input_ids=inp)
            nxt = int(out.logits[0, -1].argmax().item())
            inp = torch.cat([inp, torch.tensor([[nxt]])], dim=1)
    return inp[0].tolist()


@unittest.skipIf(not _model_paths(), "no local model; set MINISGL_TEST_MODELS")
class TestNumericalGolden(unittest.TestCase):
    def test_greedy_matches_huggingface(self):
        from minisgl.config import ModelArgs, SamplingParams, ServerArgs
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        for model_path in _model_paths():
            if not _is_qwen3(model_path):
                continue
            with self.subTest(model=model_path):
                args = ServerArgs(
                    model_path=model_path,
                    tp_size=1,
                    attention_backend="fa",
                    max_seq_len=256,
                    page_size=16,
                    memory_ratio=0.5,
                    cuda_graph_bs=0,
                )
                ma = ModelArgs.from_pretrained(model_path)
                # Build the real Engine+Scheduler (goes through model_runner,
                # the KV allocator, and the paged prefill/decode path).
                engine = Engine(args, ma, tp_rank=0)
                scheduler = Scheduler(args, engine)

                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_path)
                input_ids = tok.encode(_PROMPT)
                scheduler.add_request(
                    list(input_ids),
                    SamplingParams(temperature=0.0, max_tokens=_GEN_TOKENS),
                )
                got = []
                for _ in range(200):
                    for out in scheduler.step():
                        got.append(out.token_id)
                    if scheduler.is_idle():
                        break
                engine.cleanup()

                expected_full = _hf_generate(model_path, _PROMPT, _GEN_TOKENS)
                self.assertEqual(
                    list(input_ids) + got,
                    expected_full,
                    msg=f"mini-sglang vs HF mismatch on {model_path}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

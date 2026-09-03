"""High-level LLM API tests (logic-level, no model download).

Run: python3 tests/engine/test_llm.py   (or: python -m pytest tests/engine/test_llm.py)
"""

import sys
import unittest
from pathlib import Path

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _FakeScheduler:
    """Replays a scripted sequence of (token_id, finished, reason) outputs."""

    def __init__(self, script):
        self._script = list(script)
        self._uid = 0
        self.requests = []  # (input_ids, sampling_params)

    def add_request(self, input_ids, sampling_params):
        uid = self._uid
        self._uid += 1
        self.requests.append((uid, input_ids, sampling_params))
        return uid

    def step(self):
        from minisgl.scheduler.batch import OutputToken

        if not self._script:
            return []
        # Each script entry is a list of (token_id, finished, reason) for one step.
        batch = self._script.pop(0)
        out = []
        for i, (tok, finished, reason) in enumerate(batch):
            uid = i if self._uid == 1 else i  # align uids 0..n-1 in order
            out.append(OutputToken(uid=uid, token_id=tok, finished=finished, finish_reason=reason))
        return out

    def is_idle(self):
        return not self._script


class _FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def encode(self, text):
        # Deterministic pseudo-tokenization for the tests.
        return [ord(c) for c in text if c != " "]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) for t in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.template_calls.append(messages)
        return "".join(m["content"] for m in messages)


def _make_llm(scheduler_script):
    """Build an LLM wired to fakes (bypasses the model-loading __init__)."""
    from minisgl.engine.llm import LLM

    llm = object.__new__(LLM)
    llm.scheduler = _FakeScheduler(scheduler_script)
    llm.tokenizer = _FakeTokenizer()
    return llm


class TestLLMGenerate(unittest.TestCase):
    def test_single_prompt_returns_str(self):
        # One request, produces tokens 65,66,67 then finishes.
        llm = _make_llm([
            [(65, False, None), (66, True, "length")],
        ])
        out = llm.generate("Hello", max_tokens=2)
        self.assertIsInstance(out, str)
        self.assertEqual(out, "AB")  # chr(65)+chr(66)
        self.assertEqual(len(llm.scheduler.requests), 1)

    def test_multiple_prompts_returns_list(self):
        # Two requests interleaved in one step.
        llm = _make_llm([
            [(72, True, "length"), (73, True, "length")],  # H then I finish
        ])
        out = llm.generate(["a", "b"], max_tokens=1)
        self.assertIsInstance(out, list)
        self.assertEqual(out, ["H", "I"])

    def test_aborted_request_yields_empty(self):
        # A request aborted up front yields no tokens.
        llm = _make_llm([
            [(0, True, "abort")],
        ])
        out = llm.generate("x", max_tokens=5)
        self.assertEqual(out, "")

    def test_sampling_params_forwarded(self):
        from minisgl.config import SamplingParams

        llm = _make_llm([
            [(65, True, "length")],
        ])
        llm.generate("Hi", temperature=0.7, top_k=50, top_p=0.9, max_tokens=3)
        _uid, _ids, params = llm.scheduler.requests[0]
        self.assertIsInstance(params, SamplingParams)
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.top_k, 50)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.max_tokens, 3)


class TestLLMChat(unittest.TestCase):
    def test_single_messages_return_str(self):
        llm = _make_llm([
            [(66, True, "length")],  # "B"
        ])
        messages = [{"role": "user", "content": "hi"}]
        out = llm.chat(messages, max_tokens=1)
        self.assertIsInstance(out, str)
        self.assertEqual(out, "B")
        # The chat template received the message list once.
        self.assertEqual(llm.tokenizer.template_calls, [messages])

    def test_multi_turn_returns_list(self):
        llm = _make_llm([
            [(67, True, "length"), (68, True, "length")],  # C, D
        ])
        conv1 = [{"role": "user", "content": "a"}]
        conv2 = [{"role": "user", "content": "b"}]
        out = llm.chat([conv1, conv2], max_tokens=1)
        self.assertIsInstance(out, list)
        self.assertEqual(out, ["C", "D"])
        self.assertEqual(llm.tokenizer.template_calls, [conv1, conv2])


class TestLLMLifecycle(unittest.TestCase):
    def test_context_manager_and_cleanup(self):
        from minisgl.engine.llm import LLM

        # cleanup() must tolerate an unconstructed engine (e.g. failed init).
        llm = object.__new__(LLM)
        llm.engine = None
        llm.cleanup()  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)

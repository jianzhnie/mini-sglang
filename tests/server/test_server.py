"""Server frontend: request lifecycle, detokenization, streaming.

Run: python3 tests/server/test_server.py   (or: python -m pytest tests/server/test_server.py)
"""

import sys
import unittest
from pathlib import Path

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



# ── TestFrontendManager ──
class TestFrontendManager(unittest.TestCase):
    @staticmethod
    def _make_frontend(scheduler):
        from minisgl.config import ServerArgs
        from minisgl.server.manager import FrontendManager

        args = ServerArgs(model_path="/tmp/test")
        return FrontendManager(args, scheduler, None)

    class _MockScheduler:
        def __init__(self):
            self._uid = 0
            self.results = []
            self.aborted = []

        def add_request(self, input_ids, sampling_params):
            uid = self._uid
            self._uid += 1
            return uid

        def is_idle(self):
            return True

        def step(self):
            results, self.results = self.results, []
            return results

        def abort_request(self, uid):
            self.aborted.append(uid)
            return True

    def test_submit_and_get_queue(self):
        import queue

        from minisgl.config import SamplingParams

        fm = self._make_frontend(self._MockScheduler())
        uid = fm.submit_request([1, 2, 3], SamplingParams())
        self.assertEqual(uid, 0)
        q = fm.get_result_queue(uid)
        self.assertIsInstance(q, queue.Queue)
        fm.remove_result(uid)
        self.assertIsNone(fm.get_result_queue(uid))

    def test_process_step_distributes_results(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import OutputToken

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)
        uid = fm.submit_request([1, 2, 3], SamplingParams())

        scheduler.results.append(
            OutputToken(uid=uid, token_id=42, finished=True, finish_reason="stop")
        )
        fm.process_step()
        self.assertEqual(fm.get_result_queue(uid).get_nowait(), (42, True, "stop"))

    def test_process_step_ignores_unknown_uid(self):
        from minisgl.scheduler.batch import OutputToken

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)

        # No queue registered for uid 999: must not raise (TOCTOU safety).
        scheduler.results.append(
            OutputToken(uid=999, token_id=1, finished=True, finish_reason="stop")
        )
        fm.process_step()

    def test_stream_timeout_aborts_request(self):
        import minisgl.server.api as api
        from minisgl.config import SamplingParams

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)

        old_frontend, old_timeout = api._frontend, api.REQUEST_TIMEOUT
        api._frontend, api.REQUEST_TIMEOUT = fm, 0.01
        try:
            uid = fm.submit_request([1, 2, 3], SamplingParams())
            result_queue = fm.get_result_queue(uid)
            chunks = list(api._stream_chat_response(uid, result_queue, "m", 3))
        finally:
            api._frontend, api.REQUEST_TIMEOUT = old_frontend, old_timeout

        # Timed-out stream reports an error chunk instead of a silent [DONE],
        # aborts the scheduler-side request, and cleans up the result queue.
        self.assertTrue(any('"error"' in chunk for chunk in chunks))
        self.assertEqual(scheduler.aborted, [uid])
        self.assertIsNone(fm.get_result_queue(uid))

    def test_stream_finish_chunk_carries_reason_and_usage(self):
        """OpenAI streaming contract: content chunks carry no finish_reason;
        a separate terminal chunk carries finish_reason + usage, then [DONE]."""
        import json

        import minisgl.server.api as api
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import OutputToken

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)

        class _FakeTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids)

        fm.tokenizer = _FakeTokenizer()
        old_frontend = api._frontend
        api._frontend = fm
        try:
            uid = fm.submit_request([1, 2, 3], SamplingParams())
            q = fm.get_result_queue(uid)
            # Two content tokens, the last one finishing with reason "length".
            scheduler.results.append(
                OutputToken(uid=uid, token_id=101, finished=False, finish_reason=None)
            )
            scheduler.results.append(
                OutputToken(uid=uid, token_id=102, finished=True, finish_reason="length")
            )
            fm.process_step()

            chunks = list(api._stream_chat_response(uid, q, "model-x", prompt_tokens=3))
        finally:
            api._frontend = old_frontend

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        payloads = [json.loads(c[len("data: "):]) for c in chunks[:-1]]
        # The two content tokens land as their own delta chunks (no reason).
        content_chunks = [p for p in payloads if p["choices"][0]["delta"].get("content")]
        self.assertEqual(len(content_chunks), 2)
        for p in content_chunks:
            self.assertIsNone(p["choices"][0]["finish_reason"])
            self.assertNotIn("usage", p)
        # Terminal chunk: empty delta + finish_reason + usage.
        term = payloads[-1]
        self.assertEqual(term["choices"][0]["delta"], {})
        self.assertEqual(term["choices"][0]["finish_reason"], "length")
        self.assertEqual(term["usage"]["completion_tokens"], 2)
        self.assertEqual(term["usage"]["prompt_tokens"], 3)
        self.assertEqual(term["usage"]["total_tokens"], 5)
        self.assertIn("created", term)


# ── Test Incremental Detokenizer ──

# ── TestIncrementalDetokenizer ──
class TestIncrementalDetokenizer(unittest.TestCase):
    def test_multibyte_char_split_across_tokens(self):
        from minisgl.server.manager import IncrementalDetokenizer

        class MockTokenizer:
            # Byte-level style: tokens 1/2 are the two halves of "中"; decoding
            # a lone half yields the U+FFFD replacement character.
            TABLE = {(): "", (1,): "\ufffd", (1, 2): "中", (1, 2, 3): "中文"}

            def decode(self, ids, skip_special_tokens=True):
                return self.TABLE[tuple(ids)]

        detok = IncrementalDetokenizer(MockTokenizer())
        pieces = [detok.add_token(t) for t in (1, 2, 3)]
        self.assertEqual("".join(pieces), "中文")
        self.assertNotIn("\ufffd", "".join(pieces))

    def test_ascii_stream(self):
        from minisgl.server.manager import IncrementalDetokenizer

        class MockTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids)

        detok = IncrementalDetokenizer(MockTokenizer())
        pieces = [detok.add_token(t) for t in (ord("h"), ord("i"))]
        self.assertEqual(pieces, ["h", "i"])


# ── Test Finish Reason and Abort ──



if __name__ == '__main__':
    unittest.main(verbosity=2)

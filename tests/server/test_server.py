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

    def test_abort_request_pushes_terminal_and_removes_queue(self):
        from minisgl.config import SamplingParams

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)
        uid = fm.submit_request([1, 2, 3], SamplingParams())
        q = fm.get_result_queue(uid)

        # A consumer blocked on q.get() must wake immediately with a terminal
        # (abort) tuple instead of waiting out its timeout.
        self.assertTrue(fm.abort_request(uid))
        self.assertEqual(q.get(timeout=1.0), (0, True, "abort"))
        # The queue is removed so it cannot accumulate for a dead request.
        self.assertIsNone(fm.get_result_queue(uid))
        self.assertEqual(scheduler.aborted, [uid])
        # Idempotent at the frontend layer: the queue is already gone, so a
        # second abort (e.g. timeout racing a client disconnect) is a no-op —
        # it must not crash, duplicate, or re-add the queue.
        fm.abort_request(uid)
        self.assertIsNone(fm.get_result_queue(uid))
        self.assertEqual(q.get_nowait() if not q.empty() else None, None)

    def test_abort_request_removes_queue_even_when_scheduler_unknown(self):
        from minisgl.config import SamplingParams

        class _RejectingScheduler(self._MockScheduler):
            def abort_request(self, uid):
                self.aborted.append(uid)
                return False  # request already gone from the scheduler

        scheduler = _RejectingScheduler()
        fm = self._make_frontend(scheduler)
        uid = fm.submit_request([1, 2, 3], SamplingParams())

        # The queue is still removed even when the scheduler reports the
        # request unknown, so a stale entry cannot leak.
        self.assertFalse(fm.abort_request(uid))
        self.assertIsNone(fm.get_result_queue(uid))

    def test_stream_error_reason_yields_error_chunk(self):
        import json

        from minisgl.config import SamplingParams
        from minisgl.server import streaming

        class _CharTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids)

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)
        fm.tokenizer = _CharTokenizer()
        uid = fm.submit_request([1, 2, 3], SamplingParams())
        q = fm.get_result_queue(uid)
        # Inject an "error" terminal directly (as Scheduler.step would after a
        # failed forward): the stream must report it, not emit a finish chunk.
        q.put((0, True, "error"))

        chunks = list(
            streaming.stream_response(
                fm, uid, q, "chat", "m", 3,
                lambda c: streaming.content_chunk(uid, "m", "chat", c),
            )
        )
        self.assertTrue(any('"error"' in chunk for chunk in chunks))
        # Cleaned up and closed out with [DONE].
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
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

    def test_process_step_survives_scheduler_error(self):
        """A scheduler step raising must not kill the event loop thread."""
        import logging

        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import OutputToken

        class _BoomScheduler(self._MockScheduler):
            def step(self):
                self.step_calls += 1
                if self.step_calls == 1:
                    raise RuntimeError("boom")
                results, self.results = self.results, []
                return results

        scheduler = _BoomScheduler()
        scheduler.step_calls = 0
        fm = self._make_frontend(scheduler)
        uid = fm.submit_request([1, 2, 3], SamplingParams())
        scheduler.results.append(
            OutputToken(uid=uid, token_id=7, finished=True, finish_reason="stop")
        )

        # First step blows up (logged, swallowed), second step distributes.
        with self.assertLogs("minisgl", level=logging.ERROR):
            fm.process_step()
        fm.process_step()
        self.assertEqual(fm.get_result_queue(uid).get_nowait(), (7, True, "stop"))

    def test_run_event_loop_stops_and_distributes(self):
        """run_event_loop polls until stop(); results still reach queues."""
        import threading
        import time

        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import OutputToken

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)
        uid = fm.submit_request([1, 2, 3], SamplingParams())

        # Give the loop a couple of non-idle steps with results, then stop.
        scheduler.results.append(
            OutputToken(uid=uid, token_id=11, finished=False, finish_reason=None)
        )
        scheduler.results.append(
            OutputToken(uid=uid, token_id=12, finished=True, finish_reason="length")
        )
        scheduler.idle_calls = 0
        real_idle = scheduler.is_idle

        def flaky_idle():
            # Non-idle for the first ~3 polls so process_step runs, then idle.
            scheduler.idle_calls += 1
            return scheduler.idle_calls > 3

        scheduler.is_idle = flaky_idle

        fm.start()
        try:
            # Let the loop make progress, then stop it cleanly.
            time.sleep(0.2)
            fm.stop()
            fm._thread.join(timeout=2.0)
        finally:
            fm._running = False
            scheduler.is_idle = real_idle

        q = fm.get_result_queue(uid)
        # The two distributed tokens (11 then 12/length) reached the queue.
        self.assertEqual(q.get_nowait(), (11, False, None))
        self.assertEqual(q.get_nowait(), (12, True, "length"))
        self.assertFalse(fm._thread.is_alive())

    def test_stream_timeout_aborts_request(self):
        from minisgl.config import SamplingParams
        from minisgl.server import streaming

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)

        uid = fm.submit_request([1, 2, 3], SamplingParams())
        result_queue = fm.get_result_queue(uid)
        chunks = list(
            streaming.stream_response(
                fm,
                uid,
                result_queue,
                "chat",
                "m",
                3,
                lambda content: streaming.content_chunk(uid, "m", "chat", content),
                timeout=0.01,
            )
        )

        # Timed-out stream reports an error chunk instead of a silent [DONE],
        # aborts the scheduler-side request, and cleans up the result queue.
        self.assertTrue(any('"error"' in chunk for chunk in chunks))
        self.assertEqual(scheduler.aborted, [uid])
        self.assertIsNone(fm.get_result_queue(uid))

    def test_stream_finish_chunk_carries_reason_and_usage(self):
        """OpenAI streaming contract: content chunks carry no finish_reason;
        a separate terminal chunk carries finish_reason + usage, then [DONE]."""
        import json

        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import OutputToken
        from minisgl.server import streaming

        scheduler = self._MockScheduler()
        fm = self._make_frontend(scheduler)

        class _FakeTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids)

        fm.tokenizer = _FakeTokenizer()
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

        chunks = list(
            streaming.stream_response(
                fm,
                uid,
                q,
                "chat",
                "model-x",
                3,
                lambda content: streaming.content_chunk(uid, "model-x", "chat", content),
            )
        )

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


# ── Test SSE Frame Serialization ──
class TestStreamingFrames(unittest.TestCase):
    """Pure serialization of OpenAI-style SSE frames (streaming module)."""

    def _parse(self, frame: str) -> dict:
        import json

        self.assertTrue(frame.startswith("data: "))
        return json.loads(frame[len("data: "):])

    def test_usage_block(self):
        from minisgl.server.streaming import usage

        self.assertEqual(usage(3, 7), {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10})

    def test_error_chunk_shape(self):
        from minisgl.server.streaming import error_chunk

        payload = self._parse(error_chunk("boom"))
        self.assertEqual(payload, {"error": {"message": "boom"}})

    def test_chat_content_chunk(self):
        from minisgl.server.streaming import content_chunk

        payload = self._parse(content_chunk(1, "m", "chat", "Hello"))
        self.assertEqual(payload["id"], "chatcmpl-1")
        self.assertEqual(payload["object"], "chat.completion.chunk")
        self.assertEqual(payload["choices"][0]["delta"], {"content": "Hello"})
        self.assertIsNone(payload["choices"][0]["finish_reason"])
        self.assertNotIn("usage", payload)
        self.assertIn("created", payload)

    def test_completion_content_chunk(self):
        from minisgl.server.streaming import content_chunk

        payload = self._parse(content_chunk(1, "m", "completion", "Hi"))
        self.assertEqual(payload["id"], "cmpl-1")
        self.assertEqual(payload["object"], "text_completion")
        self.assertEqual(payload["choices"][0]["text"], "Hi")
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_finish_chunk_carries_reason_and_usage(self):
        from minisgl.server.streaming import finish_chunk

        for api in ("chat", "completion"):
            payload = self._parse(finish_chunk(1, "m", api, "length", 3, 5))
            choice = payload["choices"][0]
            self.assertEqual(choice["finish_reason"], "length")
            if api == "chat":
                self.assertEqual(choice["delta"], {})
            else:
                self.assertEqual(choice["text"], "")
            self.assertEqual(payload["usage"]["prompt_tokens"], 3)
            self.assertEqual(payload["usage"]["completion_tokens"], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

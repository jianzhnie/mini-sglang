"""HTTP-layer tests for the FastAPI endpoints (minisgl/server/api.py).

The endpoint functions read the module-global ``_frontend`` manager, so these
tests install a fake frontend (real FrontendManager shapes + a stub scheduler /
tokenizer) and drive real HTTP requests through FastAPI's TestClient. This
covers the layer examples/server_demo.py exercises only by hand: the
submit -> queue -> collect/stream plumbing, and how abort / error / timeout
surface as HTTP status codes.

Run: python tests/server/test_api.py   (or: python -m pytest tests/server/test_api.py)
"""

import queue as _queue
import sys
import time
import unittest
from pathlib import Path

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _CharTokenizer:
    """decode maps ids back to chars so text assertions are exact."""

    def __init__(self):
        self.calls = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(("encode", text))
        # Deterministic: one token per (non-space) char, monotonic ids.
        return [ord(c) for c in text if not c.isspace()]

    def apply_chat_template(self, messages, add_generation_prompt=True) -> str:
        self.calls.append(("template", messages))
        return "USER:" + "".join(m["content"] for m in messages)

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(chr(i) for i in ids)


class _StubScheduler:
    """Tracks submissions/aborts; never produces tokens on its own."""

    def __init__(self):
        self.uid = 0
        self.aborted = []

    def add_request(self, input_ids, sampling_params):
        uid = self.uid
        self.uid += 1
        return uid

    def abort_request(self, uid):
        self.aborted.append(uid)
        return True


class _FakeFrontend:
    """Duck-typed stand-in for FrontendManager (module API level only)."""

    def __init__(self):
        from minisgl.server.manager import FrontendManager

        self.scheduler = _StubScheduler()
        # Reuse the real FrontendManager for the queue bookkeeping we rely on,
        # but with a stub scheduler and tokenizer injected.
        self._fm = FrontendManager(
            type("_Args", (), {"model_path": "/tmp/test"})(),
            self.scheduler,
            _CharTokenizer(),
        )
        self.tokenizer = self._fm.tokenizer

    def submit_request(self, input_ids, sampling_params):
        return self._fm.submit_request(input_ids, sampling_params)

    def get_result_queue(self, uid):
        return self._fm.get_result_queue(uid)

    def remove_result(self, uid):
        self._fm.remove_result(uid)

    def abort_request(self, uid):
        return self._fm.abort_request(uid)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import minisgl.server.api as api

        self.api = api
        self.client = TestClient(api.app)
        self.frontend = _FakeFrontend()
        api._frontend = self.frontend  # module global the routes read

    def tearDown(self):
        self.api._frontend = None

    def _fill_after_submit(self, items: list[tuple]) -> None:
        """Feed ``items`` to uid 0's queue once the endpoint registers it.

        The endpoint calls scheduler.add_request inside the HTTP handler, so
        the result queue does not exist until the request starts. This thread
        waits for it and then pushes the planned (token_id, finished, reason)
        tuples, which is what Scheduler.step -> FrontendManager would do.
        """
        import threading
        import time

        def producer():
            deadline = time.time() + 5.0
            q = None
            while time.time() < deadline:
                q = self.frontend.get_result_queue(0)
                if q is not None:
                    break
                time.sleep(0.005)
            self.assertIsNotNone(q, "endpoint never registered uid 0's queue")
            for item in items:
                q.put(item)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        return t

    # ── guard: server not initialized ──
    def test_server_not_initialized_503(self):
        self.api._frontend = None
        resp = self.client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 5}
        )
        self.assertEqual(resp.status_code, 503)

    # ── /v1/completions ──
    def test_completions_sync_success(self):
        producer = self._fill_after_submit([(ord("h"), False, None), (ord("i"), True, "stop")])
        resp = self.client.post(
            "/v1/completions",
            json={"prompt": "hi there", "max_tokens": 5, "stream": False},
        )
        producer.join(timeout=5)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["choices"][0]["text"], "hi")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        # Fake tokenizer drops spaces: "hithere" -> 7 tokens.
        self.assertEqual(body["usage"]["prompt_tokens"], 7)
        # Queue cleaned up after consumption.
        self.assertIsNone(self.frontend.get_result_queue(0))

    def test_completions_abort_maps_400(self):
        producer = self._fill_after_submit([(0, True, "abort")])
        resp = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 5, "stream": False},
        )
        producer.join(timeout=5)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("aborted", resp.json()["detail"].lower())

    def test_completions_error_maps_500(self):
        producer = self._fill_after_submit([(0, True, "error")])
        resp = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 5, "stream": False},
        )
        producer.join(timeout=5)
        self.assertEqual(resp.status_code, 500)

    def test_completions_timeout_maps_504_and_aborts(self):
        # Never put anything on the queue: _collect_all_tokens blocks until its
        # 120s timeout. Shrink it via monkeypatch so the test returns quickly.
        orig = self.api.REQUEST_TIMEOUT
        self.api.REQUEST_TIMEOUT = 0.1
        try:
            resp = self.client.post(
                "/v1/completions",
                json={"prompt": "never returns", "max_tokens": 5, "stream": False},
            )
        finally:
            self.api.REQUEST_TIMEOUT = orig
        self.assertEqual(resp.status_code, 504)
        # The timed-out request was aborted scheduler-side and cleaned up.
        self.assertIn(0, self.frontend.scheduler.aborted)
        self.assertIsNone(self.frontend.get_result_queue(0))

    # ── /v1/chat/completions ──
    def test_chat_sync_success(self):
        producer = self._fill_after_submit([(ord("o"), False, None), (ord("k"), True, "stop")])
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
                "stream": False,
            },
        )
        producer.join(timeout=5)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # The chat template run by the fake tokenizer prepends "USER:".
        self.assertEqual(
            self.frontend.tokenizer.calls[0],
            ("template", [{"role": "user", "content": "ping"}]),
        )
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertEqual(body["object"], "chat.completion")

    # ── SSE streaming ──
    def test_completions_stream_success(self):
        producer = self._fill_after_submit([(ord("a"), False, None), (ord("b"), True, "stop")])
        with self.client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "abc", "max_tokens": 5, "stream": True},
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(
                resp.headers["content-type"].split(";")[0], "text/event-stream"
            )
            events = [line for line in resp.iter_lines() if line.startswith("data: ")]
        producer.join(timeout=5)

        self.assertEqual(events[-1], "data: [DONE]")
        import json

        payloads = [json.loads(e[len("data: "):]) for e in events[:-1]]
        # content chunks + terminal chunk
        texts = [p["choices"][0].get("text", "") for p in payloads]
        self.assertIn("a", "".join(texts))
        self.assertIn("b", "".join(texts))
        terminal = payloads[-1]
        self.assertEqual(terminal["choices"][0]["finish_reason"], "stop")
        self.assertEqual(terminal["usage"]["completion_tokens"], 2)
        self.assertIsNone(self.frontend.get_result_queue(0))

    def test_chat_stream_success(self):
        producer = self._fill_after_submit([(ord("x"), False, None), (ord("y"), True, "length")])
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "stream": True,
            },
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            events = [line for line in resp.iter_lines() if line.startswith("data: ")]
        producer.join(timeout=5)

        self.assertEqual(events[-1], "data: [DONE]")
        import json

        payloads = [json.loads(e[len("data: "):]) for e in events[:-1]]
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "length")
        self.assertIsNone(self.frontend.get_result_queue(0))

    def test_stream_error_reason_yields_error_chunk(self):
        producer = self._fill_after_submit([(0, True, "error")])
        with self.client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "boom", "max_tokens": 5, "stream": True},
        ) as resp:
            self.assertEqual(resp.status_code, 200)  # SSE: status is 200
            events = [line for line in resp.iter_lines() if line.startswith("data: ")]
        producer.join(timeout=5)

        # An error event is emitted (not a normal finish), then [DONE].
        self.assertTrue(any('"error"' in e for e in events), msg=f"events: {events}")
        self.assertEqual(events[-1], "data: [DONE]")
        self.assertIsNone(self.frontend.get_result_queue(0))

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

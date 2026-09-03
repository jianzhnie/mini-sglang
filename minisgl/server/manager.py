"""Frontend manager: request lifecycle, scheduler event loop, detokenization."""

__all__ = ["FrontendManager", "IncrementalDetokenizer"]
import queue
import threading
import time
from typing import TYPE_CHECKING

from minisgl.config import SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import logger

if TYPE_CHECKING:
    from minisgl.tokenizer import TokenizerWorker


class FrontendManager:
    """Manages the lifecycle of inference requests and SSE streaming."""

    def __init__(
        self,
        server_args: ServerArgs,
        scheduler: Scheduler,
        tokenizer_worker: "TokenizerWorker",
    ) -> None:
        self.args = server_args
        self.scheduler = scheduler
        self.tokenizer = tokenizer_worker
        self._results: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = True

    def submit_request(
        self, input_ids: list[int], sampling_params: SamplingParams
    ) -> int:
        """Submit a tokenized request and return its UID.

        The result queue is registered under the same lock that process_step
        holds while distributing tokens, so the first token can never be
        produced before the queue exists.
        """
        with self._lock:
            uid = self.scheduler.add_request(input_ids, sampling_params)
            self._results[uid] = queue.Queue()
            return uid

    def get_result_queue(self, uid: int) -> queue.Queue:
        with self._lock:
            return self._results.get(uid)

    def remove_result(self, uid: int) -> None:
        with self._lock:
            self._results.pop(uid, None)

    def process_step(self) -> None:
        """Run one scheduler step and distribute results."""
        try:
            step_results = self.scheduler.step()
        except (RuntimeError, ValueError) as e:
            logger.error("Scheduler step error: %s", e)
            return
        # Hold the lock while distributing: remove_result() may run
        # concurrently from a streaming generator's finally block.
        with self._lock:
            for out in step_results:
                q = self._results.get(out.uid)
                if q is not None:
                    q.put((out.token_id, out.finished, out.finish_reason))

    def run_event_loop(self) -> None:
        """Background thread running the scheduler loop."""
        logger.info("Scheduler event loop started")
        while self._running:
            try:
                if self.scheduler.is_idle():
                    time.sleep(0.01)  # avoid a busy spin while idle
                    continue
                self.process_step()
            except Exception as exc:
                logger.error("Scheduler event loop error: %s", exc)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False


class IncrementalDetokenizer:
    """Incrementally decode a token stream into text deltas.

    Re-decodes the full token list on every token so multi-byte UTF-8
    characters split across tokens are never emitted as U+FFFD replacement
    characters.
    """

    def __init__(self, tokenizer: "TokenizerWorker") -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.text = ""

    def add_token(self, token_id: int) -> str:
        """Append a token and return the newly produced text."""
        self.token_ids.append(token_id)
        new_text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        # A trailing U+FFFD may be an incomplete multi-byte character still
        # waiting for its remaining bytes; hold it back for the next token.
        stable = new_text.removesuffix("\ufffd")
        delta = stable[len(self.text) :]
        self.text = stable
        return delta

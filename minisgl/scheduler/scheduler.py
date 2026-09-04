"""Main scheduler: coordinates prefill/decode cycles and tokenizer communication."""

__all__ = ["Scheduler"]
import json
from collections import deque
from pathlib import Path
from threading import Lock

from minisgl.config import SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.engine.kvcache.naive import NaiveCacheManager
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Batch, OutputToken, Req, SequenceStatus
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.utils.logger import logger

# Exceptions a single forward / sampling step can raise that must not take
# down the whole scheduler. Anything here kills only the requests in the
# offending batch (they get an "error" terminal token); the rest keep going.
STEP_ERRORS = (RuntimeError, ValueError, IndexError)


class Scheduler:
    """Coordinates the prefill/decode lifecycle for inference requests.

    Data flow (one request)::

        add_request()  ──► PrefillManager.pending
                                  │  schedule_prefill()  (match radix prefix,
                                  │   allocate KV pages, build prefill batch)
                                  ▼
                          Engine.forward(prefill) ─► sample ─► first token
                                  │
                                  ▼
                          PrefillManager.running  ──► DecodeManager.schedule_decode()
                                  │                 (one token per running request)
                                  ▼
                     step():  Engine.forward(decode) ─► sample ─► append_token
                                  │
              finished (EOS / max_tokens / max_seq_len) ──► remove_finished_batch()
                                  │
                                  ▼
                       OutputToken(finished=True, finish_reason) returned to caller

    Two-phase scheduling:
    1. Prefill: Process new requests' full prompts.
    2. Decode: Generate one token per running request.

    Cache strategy: radix (prefix-aware, default) or naive (simple LRU).

    Thread model (teaching implementation — deliberately not maximally
    concurrent):
      * ``step()`` runs on a single scheduler thread, and that thread is the
        *only* forwarder / sampler and the only one to mutate the decode
        lifecycle (running requests mid-generation).
      * ``add_request`` and ``abort_request`` are cross-thread entry points
        (HTTP / SSE worker threads call them). Their shared-state touchpoints
        (UID counter, the ``_aborted`` drain queue, and the ``running``
        snapshot in ``step``) are serialized behind ``PrefillManager._lock`` /
        ``Scheduler._lock`` — a small critical section around list/deque
        access, never around a model forward. ``abort_request`` is a
        best-effort cancel: it takes effect cleanly for waiting/prefill
        requests, and for a request already inside a decode forward it can
        only remove the request's bookkeeping once that step finishes (the
        static decode buffers make the page table read race-free; the KV
        content itself may briefly be re-read, a cost accepted by the
        single-scheduler-thread design).
      * Results flow out one-way through ``FrontendManager`` result queues
        (``queue.Queue``), which are internally locked.
      * There is no CUDA multi-stream or multi-step parallelism, so the
        forward itself needs no lock.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        engine: Engine,
        cache_strategy: str = "radix",
    ) -> None:
        self.args = server_args
        self.engine = engine
        self.device = engine.device

        # Share the engine's KV pool: the scheduler never allocates GPU/CPU
        # memory itself, it only decides which pages requests get (via the
        # cache managers below). Single-process teaching design — in SGLang
        # the scheduler owns the pool outright.
        self.pool = engine.kv_cache_pool
        if cache_strategy == "naive":
            self.cache_manager = NaiveCacheManager(self.pool)
        else:
            self.cache_manager = RadixCacheManager(self.pool, server_args.page_size)
        self.prefill_manager = PrefillManager(
            server_args, self.pool, self.cache_manager
        )
        self.decode_manager = DecodeManager(
            server_args, self.pool, device=engine.device
        )

        self._uid_counter = 0
        self.eos_token_id = self._load_eos_token()
        # Results for requests rejected at add time (e.g., prompt too long).
        self._aborted: deque[OutputToken] = deque()
        # Guards the cross-thread touchpoints only (UID counter, _aborted
        # drain, running snapshot) — never a model forward. See the class
        # docstring's thread-model note.
        self._lock = Lock()

    def _placeholder_token(self) -> int:
        """Meaningless token_id for terminal outputs (abort / error).

        The value is never decoded or sampled; it only fills the field so
        OutputToken stays a uniform dataclass. An EOS id is a safe choice.
        """
        return next(iter(self.eos_token_id))

    def _load_eos_token(self) -> set[int]:
        """Load EOS token ID(s) from model config, with tokenizer fallback.

        Returns a set of EOS token IDs to handle models with multiple EOS tokens
        (e.g., Qwen3 uses [151645, 151643]).
        """
        model_path = Path(self.args.model_path)

        raw_eos = None
        # Priority order: generation config > tokenizer config > model config.
        for fname in (
            "generation_config.json",
            "tokenizer_config.json",
            "config.json",
        ):
            if raw_eos is not None:
                break
            cfg_file = model_path / fname
            if not cfg_file.exists():
                continue
            with cfg_file.open() as f:
                cfg = json.load(f)
            # config.json is the last resort: a missing key means 0 there,
            # not "could not determine".
            raw_eos = cfg.get("eos_token_id", 0 if fname == "config.json" else None)

        eos = self._normalize_eos(raw_eos)
        if not eos:
            # raw_eos was unparseable or empty (e.g. a config dict without a
            # usable token_id) — fall back instead of running with no EOS.
            logger.warning("Could not determine EOS token ID, using %s", 0)
            return {0}
        return eos

    @staticmethod
    def _normalize_eos(raw_eos: int | list[int] | dict | None) -> set[int]:
        """Convert various EOS formats to a set of int IDs.

        Returns an empty set when the value cannot be parsed so the caller can
        fall back to a default; never fabricates a token.
        """
        if isinstance(raw_eos, int):
            return {raw_eos}
        if isinstance(raw_eos, list):
            return {int(x) for x in raw_eos if x is not None}
        if isinstance(raw_eos, dict):
            token_id = raw_eos.get("token_id", raw_eos.get("id"))
            if isinstance(token_id, int):
                return {token_id}
            if isinstance(token_id, list):
                return {int(x) for x in token_id if x is not None}
        return set()

    def add_request(self, input_ids: list[int], sampling_params: SamplingParams) -> int:
        """Add a new request and return its UID."""
        with self._lock:
            uid = self._uid_counter
            self._uid_counter += 1

            if len(input_ids) > self.args.max_seq_len:
                logger.warning(
                    "Rejecting request %s: prompt length %s exceeds max_seq_len %s",
                    uid,
                    len(input_ids),
                    self.args.max_seq_len,
                )
                self._aborted.append(self._abort_result(uid))
                return uid

            req = Req(
                # Copy: the scheduler appends generated tokens in-place, so the
                # caller's list must not be aliased.
                input_ids=list(input_ids),
                uid=uid,
                sampling_params=sampling_params,
                cached_len=0,
            )
        # enqueue outside the lock: prefill_manager has its own lock.
        self.prefill_manager.add_request(req)
        return uid

    def _abort_result(self, uid: int) -> OutputToken:
        """Result for a request aborted before it ever ran.

        The token_id is meaningless here; see ``_placeholder_token``.
        """
        return OutputToken(
            uid=uid,
            token_id=self._placeholder_token(),
            finished=True,
            finish_reason="abort",
        )

    def _error_result(self, uid: int) -> OutputToken:
        """Terminal result for a request that died mid-run (forward/sample).

        token_id is meaningless; see ``_placeholder_token``. The scheduler
        records the failure via logger.exception at the raise site, so the
        token carries no message payload (keeps OutputToken a plain value).
        """
        return OutputToken(
            uid=uid,
            token_id=self._placeholder_token(),
            finished=True,
            finish_reason="error",
        )

    def _run_phase(
        self,
        batch: Batch | None,
        results: list[OutputToken],
        finished_reqs: list[Req],
    ) -> None:
        """Forward + sample one batch, isolating per-batch failures.

        A forward/sampling exception kills only the requests in ``batch``:
        they are marked FINISHED and surfaced with an ``"error"`` terminal
        token, while the rest of the step (and future steps) proceed. Without
        this, one bad request would raise through ``step`` and the caller's
        event loop would either die or swallow the error, leaving every
        waiting request hung until timeout.
        """
        if batch is None:
            return
        try:
            logits = self.engine.forward(batch)
            next_tokens = self.engine.sample(logits, batch)
        except STEP_ERRORS:
            logger.exception(
                "Forward/sample failed for a %s batch of %d request(s); "
                "marking them finished with an error result",
                batch.phase,
                len(batch.reqs),
            )
            for req in batch.reqs:
                req.status = SequenceStatus.FINISHED
                finished_reqs.append(req)
                results.append(self._error_result(req.uid))
            return
        self._collect_results(batch, next_tokens, results, finished_reqs)

    def _finish_reason(self, req: Req, token_id: int) -> str | None:
        """Return why the request should stop after token_id, or None to continue."""
        if token_id in self.eos_token_id and not req.sampling_params.ignore_eos:
            return "stop"
        if (
            req.output_len >= req.sampling_params.max_tokens
            or len(req.input_ids) >= self.args.max_seq_len
        ):
            return "length"
        return None

    def abort_request(self, uid: int) -> bool:
        """Abort a pending or running request by UID, releasing its resources.

        Returns True if the request was found and aborted, False otherwise.
        """
        return self.prefill_manager.abort(uid)

    def _collect_results(
        self,
        batch: Batch,
        next_tokens: list[int],
        results: list[OutputToken],
        finished_reqs: list[Req],
    ) -> None:
        """Append sampled tokens to their requests and record OutputTokens."""
        for req, token_id in zip(batch.reqs, next_tokens, strict=True):
            req.append_token(token_id)
            reason = self._finish_reason(req, token_id)
            if reason is not None:
                req.status = SequenceStatus.FINISHED
                finished_reqs.append(req)
            results.append(
                OutputToken(
                    uid=req.uid,
                    token_id=token_id,
                    finished=reason is not None,
                    finish_reason=reason,
                )
            )

    def step(self) -> list[OutputToken]:
        """Run one scheduler iteration: prefill new requests, then decode running ones.

        Returns a list of OutputToken, one per produced token. finish_reason is
        None for in-progress tokens and one of "stop" / "length" / "abort" /
        "error" for the final token ("abort"/"error" mean token_id is
        meaningless).

        A forward/sampling failure on one phase does not propagate: the
        affected requests get an "error" terminal token and are removed, and
        the other phase (and later steps) still run.
        """
        results: list[OutputToken] = []
        with self._lock:
            while self._aborted:
                results.append(self._aborted.popleft())

        finished_reqs: list[Req] = []

        prefill_batch = self.prefill_manager.schedule_prefill()
        if prefill_batch is not None:
            self._run_phase(prefill_batch, results, finished_reqs)

        # Requests aborted during scheduling never ran; report them as finished
        # so callers waiting on them stop waiting. `aborted` is mutated under
        # the prefill manager's lock (schedule_prefill / abort), so drain it
        # under that same lock.
        with self.prefill_manager._lock:
            for req in self.prefill_manager.aborted:
                results.append(self._abort_result(req.uid))
            self.prefill_manager.aborted.clear()

        # Skip requests already finished by this step's prefill. Snapshot under
        # the prefill lock: abort_request (HTTP thread) removes from `running`
        # under that same lock, so this guard prevents iterating a list that a
        # concurrent abort is mutating.
        with self.prefill_manager._lock:
            running = [req for req in self.prefill_manager.running if not req.is_finished]
        if running:
            decode_batch = self.decode_manager.schedule_decode(running)
            if decode_batch is not None:
                self._run_phase(decode_batch, results, finished_reqs)

        if finished_reqs:
            self.prefill_manager.remove_finished_batch(finished_reqs)

        return results

    def is_idle(self) -> bool:
        # A request rejected at add time (prompt too long, etc.) lands in
        # self._aborted and is only surfaced by a step() call. The event loop
        # sleeps while is_idle() is true, so count pending abort results as
        # non-idle: otherwise a rejected request would never be reported and
        # the caller would block until timeout.
        return (
            not self._aborted
            and len(self.prefill_manager.pending) == 0
            and len(self.prefill_manager.running) == 0
        )

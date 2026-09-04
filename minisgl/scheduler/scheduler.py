"""Main scheduler: coordinates prefill/decode cycles and tokenizer communication."""

__all__ = ["Scheduler"]
import json
from collections import deque
from pathlib import Path

from minisgl.config import SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.engine.kvcache.naive import NaiveCacheManager
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Batch, OutputToken, Req, SequenceStatus
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.utils.logger import logger


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
        self.prefill_manager.add_request(req)
        return uid

    def _abort_result(self, uid: int) -> OutputToken:
        """Result for a request aborted before it ever ran.

        The token_id is meaningless here; an EOS id is used as a placeholder.
        """
        return OutputToken(
            uid=uid,
            token_id=next(iter(self.eos_token_id)),
            finished=True,
            finish_reason="abort",
        )

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
        None for in-progress tokens and one of "stop" / "length" / "abort" for
        the final token ("abort" means token_id is meaningless).
        """
        results: list[OutputToken] = []
        while self._aborted:
            results.append(self._aborted.popleft())

        finished_reqs: list[Req] = []

        prefill_batch = self.prefill_manager.schedule_prefill()
        if prefill_batch is not None:
            logits = self.engine.forward(prefill_batch)
            next_tokens = self.engine.sample(logits, prefill_batch)
            self._collect_results(prefill_batch, next_tokens, results, finished_reqs)

        # Requests aborted during scheduling never ran; report them as finished
        # so callers waiting on them stop waiting.
        if self.prefill_manager.aborted:
            for req in self.prefill_manager.aborted:
                results.append(self._abort_result(req.uid))
            self.prefill_manager.aborted.clear()

        # Skip requests already finished by this step's prefill.
        running = [req for req in self.prefill_manager.running if not req.is_finished]
        if running:
            decode_batch = self.decode_manager.schedule_decode(running)
            if decode_batch is not None:
                logits = self.engine.forward(decode_batch)
                next_tokens = self.engine.sample(logits, decode_batch)
                self._collect_results(decode_batch, next_tokens, results, finished_reqs)

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

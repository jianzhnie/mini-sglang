"""Main scheduler: coordinates prefill/decode cycles and tokenizer communication."""

__all__ = ["Scheduler"]
import json
from pathlib import Path

from minisgl.config import SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.engine.kvcache.naive import NaiveCacheManager
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Req, SequenceStatus
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.utils.logger import logger


class Scheduler:
    """Coordinates the prefill/decode lifecycle for inference requests.

    Two-phase scheduling:
    1. Prefill: Process new requests' full prompts
    2. Decode: Generate one token per running request

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

    def _load_eos_token(self) -> set[int]:
        """Load EOS token ID(s) from model config, with tokenizer fallback.

        Returns a set of EOS token IDs to handle models with multiple EOS tokens
        (e.g., Qwen3 uses [151645, 151643]).
        """
        model_path = Path(self.args.model_path)

        raw_eos = None
        gen_config = model_path / "generation_config.json"
        if gen_config.exists():
            with gen_config.open() as f:
                cfg = json.load(f)
            if "eos_token_id" in cfg:
                raw_eos = cfg["eos_token_id"]

        if raw_eos is None:
            tok_config = model_path / "tokenizer_config.json"
            if tok_config.exists():
                with tok_config.open() as f:
                    cfg = json.load(f)
                if "eos_token_id" in cfg:
                    raw_eos = cfg["eos_token_id"]

        if raw_eos is None:
            cfg_file = model_path / "config.json"
            if cfg_file.exists():
                with cfg_file.open() as f:
                    cfg = json.load(f)
                raw_eos = cfg.get("eos_token_id", 0)

        if raw_eos is None:
            logger.warning("Could not determine EOS token ID, using {0}")
            return {0}

        return self._normalize_eos(raw_eos)

    @staticmethod
    def _normalize_eos(raw_eos) -> set[int]:
        """Convert various EOS formats to a set of int IDs."""
        if isinstance(raw_eos, int):
            return {raw_eos}
        if isinstance(raw_eos, list):
            return {int(x) for x in raw_eos}
        if isinstance(raw_eos, dict):
            return {raw_eos.get("token_id", raw_eos.get("id", 0))}
        return {0}

    def add_request(self, input_ids: list[int], sampling_params: SamplingParams) -> int:
        """Add a new request and return its UID."""
        uid = self._uid_counter
        self._uid_counter += 1

        req = Req(
            input_ids=input_ids,
            uid=uid,
            sampling_params=sampling_params,
            cached_len=0,
        )
        self.prefill_manager.add_request(req)
        return uid

    def step(self) -> list[tuple]:
        """Run one scheduler iteration: prefill new requests, then decode running ones."""
        results: list[tuple] = []

        prefill_batch = self.prefill_manager.schedule_prefill()
        if prefill_batch is not None:
            logits = self.engine.forward(prefill_batch)
            next_tokens = self.engine.sample(logits, prefill_batch)
            for req, token_id in zip(prefill_batch.reqs, next_tokens, strict=False):
                req.append_token(token_id)
                results.append((req.uid, token_id, False))

        running = self.prefill_manager.running
        if running:
            decode_batch = self.decode_manager.schedule_decode(running)
            if decode_batch is not None:
                logits = self.engine.forward(decode_batch)
                next_tokens = self.engine.sample(logits, decode_batch)

                finished_reqs: list[Req] = []
                for req, token_id in zip(decode_batch.reqs, next_tokens, strict=False):
                    req.append_token(token_id)
                    finished = False
                    if (
                        token_id in self.eos_token_id
                        and not req.sampling_params.ignore_eos
                    ) or req.output_len >= req.sampling_params.max_tokens:
                        finished = True
                        req.status = SequenceStatus.FINISHED
                        finished_reqs.append(req)
                    results.append((req.uid, token_id, finished))

                if finished_reqs:
                    self.prefill_manager.remove_finished_batch(finished_reqs)

        return results

    def is_idle(self) -> bool:
        return (
            len(self.prefill_manager.pending) == 0
            and len(self.prefill_manager.running) == 0
        )

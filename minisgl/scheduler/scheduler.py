"""Main scheduler: coordinates prefill/decode cycles and tokenizer communication."""

__all__ = ["Scheduler"]
from minisgl.config import SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Req, SequenceStatus
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager


class Scheduler:
    """Coordinates the prefill/decode lifecycle for inference requests.


    Two-phase scheduling:
    1. Prefill: Process new requests' full prompts
    2. Decode: Generate one token per running request
    """

    def __init__(self, server_args: ServerArgs, engine: Engine):
        self.args = server_args
        self.engine = engine
        self.device = engine.device

        self.pool = engine.kv_cache_pool
        self.radix_cache = RadixCacheManager(self.pool, server_args.page_size)
        self.prefill_manager = PrefillManager(server_args, self.pool, self.radix_cache)
        self.decode_manager = DecodeManager(server_args, self.pool, self.radix_cache)

        self._uid_counter = 0
        self.eos_token_id = self._load_eos_token()

    def _load_eos_token(self) -> int:
        """Load EOS token ID from model's tokenizer config."""
        import json
        import os

        path = os.path.join(self.args.model_path, "generation_config.json")
        try:
            with open(path) as f:
                cfg = json.load(f)
            return cfg.get("eos_token_id", 151643)
        except FileNotFoundError:
            return 151643

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

        # Phase 1: Prefill — process new requests if any
        prefill_batch = self.prefill_manager.schedule_prefill()
        if prefill_batch is not None:
            logits = self.engine.forward(prefill_batch)
            next_tokens = self.engine.sample(logits, prefill_batch)
            for req, token_id in zip(prefill_batch.reqs, next_tokens, strict=False):
                req.append_token(token_id)
                results.append((req.uid, token_id, False))

        # Phase 2: Decode — generate one token per running request
        running = self.prefill_manager.running
        if running:
            decode_batch = self.decode_manager.schedule_decode(running)
            if decode_batch is not None:
                logits = self.engine.forward(decode_batch)
                next_tokens = self.engine.sample(logits, decode_batch)

                for req, token_id in zip(decode_batch.reqs, next_tokens, strict=False):
                    req.append_token(token_id)
                    finished = False
                    if (
                        token_id == self.eos_token_id
                        and not req.sampling_params.ignore_eos
                        or req.output_len >= req.sampling_params.max_tokens
                    ):
                        finished = True
                        req.status = SequenceStatus.FINISHED
                        self.prefill_manager.remove_finished(req)
                    results.append((req.uid, token_id, finished))

        return results

    def is_idle(self) -> bool:
        return (
            len(self.prefill_manager.pending) == 0
            and len(self.prefill_manager.running) == 0
        )

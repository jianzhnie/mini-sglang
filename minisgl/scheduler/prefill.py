"""Prefill manager: schedules new requests for initial prompt processing."""

__all__ = ["PrefillManager"]
import contextlib
from collections import deque
from threading import Lock

from minisgl.config import ServerArgs
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Batch, Req, SequenceStatus


class PrefillManager:
    """Manages the prefill queue and token budget."""

    def __init__(
        self,
        args: ServerArgs,
        pool: KVCachePool,
        radix_cache: RadixCacheManager,
    ) -> None:
        self.max_running_req = args.max_running_req
        self.max_seq_len = args.max_seq_len
        self.page_size = args.page_size
        self.token_budget = args.max_seq_len  # max tokens per prefill batch

        self.pool = pool
        self.radix_cache = radix_cache
        self.pending: deque[Req] = deque()
        self.running: list[Req] = []
        self._lock = Lock()

    def add_request(self, req: Req) -> None:
        with self._lock:
            self.pending.append(req)

    def schedule_prefill(self) -> Batch | None:
        """Select requests from pending queue, allocate KV cache, build prefill batch.

        Returns None if no requests can be scheduled.
        """
        with self._lock:
            if not self.pending:
                return None

            scheduled: list[Req] = []
            total_tokens = 0

            while self.pending and len(scheduled) < self.max_running_req:
                req = self.pending[0]

                # Check if total tokens would exceed budget
                uncached = req.uncached_len
                if total_tokens + uncached > self.token_budget and scheduled:
                    break

                # Try prefix matching (may reduce pages needed via cache sharing)
                matched_len = self.radix_cache.match_prefix(req.input_ids)
                req.cached_len = max(req.cached_len, matched_len)

                new_pages = self._pages_needed(req)

                if self.pool.free_count() < new_pages:
                    # Try eviction
                    self.radix_cache.evict(new_pages - self.pool.free_count())
                    if self.pool.free_count() < new_pages:
                        break

                # Allocate KV cache pages
                handle = self.pool.alloc(new_pages)
                req.cache_handle = handle
                req.table_idx = len(scheduled)

                # Insert into radix cache
                self.radix_cache.insert(req.input_ids, handle)

                self.pending.popleft()
                req.status = SequenceStatus.RUNNING
                scheduled.append(req)
                total_tokens += uncached

            if not scheduled:
                return None

            self.running.extend(scheduled)
            return Batch(reqs=scheduled, phase="prefill")

    def _pages_needed(self, req: Req) -> int:
        total = min(
            len(req.input_ids) + req.sampling_params.max_tokens,
            self.max_seq_len,
        )
        return (total + self.page_size - 1) // self.page_size

    def remove_finished(self, req: Req) -> None:
        with self._lock:
            self._remove_finished_nolock(req)

    def _remove_finished_nolock(self, req: Req) -> None:
        with contextlib.suppress(ValueError):
            self.running.remove(req)
        if req.cache_handle:
            self.radix_cache.remove(req.input_ids)
            self.pool.free(req.cache_handle)
            req.cache_handle = None

    def remove_finished_batch(self, reqs: list[Req]) -> None:
        with self._lock:
            for req in reqs:
                self._remove_finished_nolock(req)

"""Prefill manager: schedules new requests for initial prompt processing."""

__all__ = ["PrefillManager"]
import contextlib
from collections import deque
from threading import Lock

from minisgl.config import ServerArgs
from minisgl.engine.kvcache.pool import CacheManager, KVCachePool
from minisgl.scheduler.batch import Batch, Req, SequenceStatus
from minisgl.utils.logger import logger


class PrefillManager:
    """Manages the prefill queue and token budget."""

    def __init__(
        self,
        args: ServerArgs,
        pool: KVCachePool,
        radix_cache: CacheManager,
    ) -> None:
        self.max_running_req = args.max_running_req
        self.max_seq_len = args.max_seq_len
        self.page_size = args.page_size
        self.token_budget = args.max_seq_len  # max tokens per prefill batch

        self.pool = pool
        self.radix_cache = radix_cache
        self.pending: deque[Req] = deque()
        self.running: list[Req] = []
        # Requests that can never be satisfied (e.g., need more pages than the
        # whole pool); the Scheduler reports them as finished.
        self.aborted: list[Req] = []
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

            while (
                self.pending
                and len(self.running) + len(scheduled) < self.max_running_req
            ):
                req = self.pending[0]

                # Check if total tokens would exceed budget
                uncached = req.uncached_len
                if total_tokens + uncached > self.token_budget and scheduled:
                    break

                # Try prefix matching (may reduce pages needed via cache sharing)
                matched_len, shared_pages = self.radix_cache.match_prefix(
                    req.input_ids
                )
                req.cached_len = matched_len

                new_pages = self._pages_needed(req, matched_len)

                if new_pages > self.pool.num_pages:
                    # A single request larger than the whole pool can never be
                    # satisfied by eviction or waiting; abort it explicitly.
                    logger.warning(
                        "Aborting request %s: needs %s pages, pool has only %s",
                        req.uid,
                        new_pages,
                        self.pool.num_pages,
                    )
                    self.pending.popleft()
                    req.status = SequenceStatus.FINISHED
                    self.aborted.append(req)
                    continue

                if self.pool.free_count() < new_pages:
                    # Try eviction
                    self.radix_cache.evict(new_pages - self.pool.free_count())
                    if self.pool.free_count() < new_pages:
                        break

                # Allocate pages for the uncached suffix only; the full page
                # table is shared prefix pages followed by the new pages.
                handle = self.pool.alloc(new_pages)
                handle.page_ids = list(shared_pages) + handle.page_ids
                handle.num_shared = len(shared_pages)
                handle.cached_len = matched_len
                req.cache_handle = handle
                req.table_idx = len(scheduled)

                # Insert into radix cache (teaching simplification: the KV is
                # attached to the tree before it is actually computed).
                self.radix_cache.insert(req.input_ids, handle)

                self.pending.popleft()
                req.status = SequenceStatus.RUNNING
                scheduled.append(req)
                total_tokens += uncached

            if not scheduled:
                return None

            self.running.extend(scheduled)
            return Batch(reqs=scheduled, phase="prefill")

    def _pages_needed(self, req: Req, matched_len: int) -> int:
        """Pages to allocate for the uncached suffix [matched_len, upper)."""
        upper = min(
            len(req.input_ids) + req.sampling_params.max_tokens,
            self.max_seq_len,
        )
        return (upper - matched_len + self.page_size - 1) // self.page_size

    def remove_finished(self, req: Req) -> None:
        with self._lock:
            self._remove_finished_nolock(req)

    def _remove_finished_nolock(self, req: Req) -> None:
        with contextlib.suppress(ValueError):
            self.running.remove(req)
        if req.cache_handle:
            # Pages are owned by the cache manager (the radix tree); they are
            # reclaimed by evict() under memory pressure, not here.
            self.radix_cache.remove(req.input_ids, req.cache_handle)
            req.cache_handle = None

    def remove_finished_batch(self, reqs: list[Req]) -> None:
        with self._lock:
            for req in reqs:
                self._remove_finished_nolock(req)

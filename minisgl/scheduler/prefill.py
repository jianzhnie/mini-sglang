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

                # Try prefix matching (may reduce pages needed via cache sharing).
                matched_len, shared_pages = self.radix_cache.match_prefix(req.input_ids)
                req.cached_len = matched_len
                # Tokens this request will actually forward this prefill step:
                # its uncached suffix (prefix KV is read from the shared pages).
                uncached = req.uncached_len

                if total_tokens + uncached > self.token_budget and scheduled:
                    # Advancing this request would blow the per-step budget;
                    # stop scheduling and leave it pending for a later step.
                    break

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

                # NOTE: the shared prefix pages this request matched are owned
                # by the radix tree, but the request has not claimed a
                # reference on them yet (insert() below does that). If the pool
                # is short and eviction runs now, those just-matched pages can
                # be detached (they are ref_count == 0), returned to the free
                # list, and then handed right back to this same request as
                # "fresh" pages by alloc() — producing a duplicate page in the
                # request's page table (its own prefix KV collides with its new
                # writes). To close that race, evict first and RE-MATCH against
                # the surviving tree: re-match never reports a page that is no
                # longer cached, so the final shared pages are guaranteed to
                # still be held by the tree (and thus never in the free list).
                if self.pool.free_count() < new_pages:
                    shortfall = new_pages - self.pool.free_count()
                    self.radix_cache.evict(shortfall)
                    # Re-match after eviction so shared_pages reflect the tree
                    # that is actually left.
                    matched_len, shared_pages = self.radix_cache.match_prefix(
                        req.input_ids
                    )
                    req.cached_len = matched_len
                    new_pages = self._pages_needed(req, matched_len)

                if self.pool.free_count() < new_pages:
                    # Even after eviction + re-match we cannot satisfy the
                    # request this step (e.g. the cache is pinned by running
                    # requests). Keep it pending; retry once requests finish.
                    break

                # Allocate pages for the uncached suffix only; the full page
                # table is shared prefix pages followed by the new pages.
                handle = self.pool.alloc(new_pages)
                handle.page_ids = list(shared_pages) + handle.page_ids
                handle.num_shared = len(shared_pages)
                handle.cached_len = matched_len
                req.cache_handle = handle

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

    def abort(self, uid: int) -> bool:
        """Abort a pending or running request by UID, releasing its resources.

        Returns True if the request was found and aborted, False otherwise.
        """
        with self._lock:
            for queue_ in (self.pending, self.running):
                for req in queue_:
                    if req.uid == uid:
                        req.status = SequenceStatus.FINISHED
                        with contextlib.suppress(ValueError):
                            queue_.remove(req)
                        self._remove_finished_nolock(req)
                        logger.info("Aborted request %s", uid)
                        return True
        return False

    def _pages_needed(self, req: Req, matched_len: int) -> int:
        """Pages to allocate for the uncached suffix [matched_len, upper)."""
        upper = min(
            len(req.input_ids) + req.sampling_params.max_tokens,
            self.max_seq_len,
        )
        # match_prefix guarantees matched_len <= len(input_ids) - 1 and
        # max_tokens is clamped >= 1, so at least one token is always
        # forwarded; max(1, ...) guards any residual underflow (e.g. a prompt
        # already at max_seq_len) from collapsing into a 0-page allocation.
        return max(1, (upper - matched_len + self.page_size - 1) // self.page_size)

    def _remove_finished_nolock(self, req: Req) -> None:
        with contextlib.suppress(ValueError):
            self.running.remove(req)
        if req.cache_handle:
            # Pages are owned by the cache manager (the radix tree); they are
            # reclaimed by evict() under memory pressure, not here.
            self.radix_cache.remove(req.input_ids, req.cache_handle)
            req.cache_handle = None

    def _remove_failed_nolock(self, req: Req) -> None:
        """Clean up a request whose PREFILL forward failed.

        Distinct from ``_remove_finished_nolock``: a prefill that died mid-
        forward had its prompt inserted into the radix tree but its KV was never
        written, so ``radix.remove`` (which assumes written KV and extends the
        tree with the sequence) is wrong. We roll the insert back instead —
        detaching the never-written nodes and freeing their pages — so no later
        request can match a garbage prefix.
        """
        with contextlib.suppress(ValueError):
            self.running.remove(req)
        if req.cache_handle:
            self.radix_cache.rollback_insert(req.input_ids, req.cache_handle)
            req.cache_handle = None

    def remove_finished_batch(self, reqs: list[Req]) -> None:
        with self._lock:
            for req in reqs:
                self._remove_finished_nolock(req)

    def remove_failed_prefill_batch(self, reqs: list[Req]) -> None:
        """Remove requests whose prefill forward raised (rolls back the tree)."""
        with self._lock:
            for req in reqs:
                self._remove_failed_nolock(req)

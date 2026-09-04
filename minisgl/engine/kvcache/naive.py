"""Naive KV cache manager (no prefix sharing).

Provides the same interface as RadixCacheManager for drop-in compatibility,
but shares nothing: every request allocates its full page table from the pool
and frees it on finish. There are no cached pages, so eviction is a no-op.
"""

__all__ = ["NaiveCacheManager"]
from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class NaiveCacheManager:
    """No-op cache manager without prefix sharing.

    Implements the same interface as RadixCacheManager so it can be used
    as a drop-in replacement when prefix-aware caching is not needed.
    """

    def __init__(self, pool: KVCachePool, page_size: int = 16) -> None:
        self.pool = pool
        self.page_size = page_size

    def match_prefix(self, input_ids: list[int]) -> tuple[int, list[int]]:
        """No prefix sharing: never matches anything."""
        return 0, []

    def insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """No-op: without sharing there is nothing to track."""

    def evict(self, num_pages: int) -> list[int]:
        """No-op: naive keeps no cached pages, so there is nothing to evict.

        Pages are freed synchronously by ``remove`` when a request finishes,
        so eviction can never reclaim memory here.
        """
        return []

    def remove(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """Free the request's pages immediately (no sharing, no caching)."""
        self.pool.free(handle)

    def rollback_insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """No tree to roll back; free the never-written pages like remove()."""
        self.pool.free(handle)

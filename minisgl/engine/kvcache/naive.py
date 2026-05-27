"""Naive LRU-based KV cache manager (no prefix sharing).

Provides the same interface as RadixCacheManager for drop-in compatibility.
"""

__all__ = ["NaiveCacheManager"]
from collections import OrderedDict

from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class NaiveCacheManager:
    """Simple LRU cache manager without prefix sharing.

    Implements the same interface as RadixCacheManager so it can be used
    as a drop-in replacement when prefix-aware caching is not needed.
    """

    def __init__(self, pool: KVCachePool, page_size: int = 16) -> None:
        self.pool = pool
        self.page_size = page_size
        self.lru: OrderedDict = OrderedDict()

    def match_prefix(self, input_ids: list[int]) -> int:
        """No prefix sharing: always returns 0."""
        return 0

    def insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """Track a request's cache handle for LRU eviction."""
        uid = id(handle)
        self.lru[uid] = handle

    def evict(self, num_pages: int) -> None:
        """Evict the least recently used entries to free pages."""
        freed = 0
        while freed < num_pages and self.lru:
            _, handle = self.lru.popitem(last=False)
            n = handle.num_pages()
            self.pool.free(handle)
            freed += n

    def remove(self, input_ids: list[int]) -> None:
        """No-op: naive cache doesn't track by input_ids."""
        pass

    def allocate(self, uid: int, num_pages: int) -> BaseCacheHandle:
        """Allocate pages for a request, evicting LRU if needed."""
        while self.pool.free_count() < num_pages:
            if not self.lru:
                msg = f"Cannot allocate {num_pages} pages: all pages in use"
                raise RuntimeError(msg)
            _, old_handle = self.lru.popitem(last=False)
            self.pool.free(old_handle)

        handle = self.pool.alloc(num_pages)
        self.lru[uid] = handle
        return handle

    def touch(self, uid: int) -> None:
        """Mark a request as recently used."""
        if uid in self.lru:
            self.lru.move_to_end(uid)

    def free(self, uid: int) -> None:
        """Free all pages for a request."""
        if uid in self.lru:
            handle = self.lru.pop(uid)
            self.pool.free(handle)

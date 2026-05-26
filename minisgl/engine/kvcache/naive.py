"""Naive LRU-based KV cache manager (no prefix sharing)."""

from collections import OrderedDict
from typing import List

from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class NaiveCacheManager:
    """Simple LRU cache manager without prefix sharing."""

    def __init__(self, pool: KVCachePool):
        self.pool = pool
        self.lru: OrderedDict = OrderedDict()

    def allocate(self, uid: int, num_pages: int) -> BaseCacheHandle:
        """Allocate pages for a request, evicting LRU if needed."""
        while self.pool.free_count() < num_pages:
            if not self.lru:
                raise RuntimeError(f"Cannot allocate {num_pages} pages: all pages in use")
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

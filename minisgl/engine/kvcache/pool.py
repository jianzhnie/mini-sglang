"""KV Cache pool and cache handle base classes."""

__all__ = ["BaseCacheHandle", "CacheManager", "KVCachePool"]
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


@dataclass
class BaseCacheHandle:
    """Tracks allocated KV cache pages for a request."""

    page_ids: list[int] = field(default_factory=list)
    cached_len: int = 0

    def num_pages(self) -> int:
        return len(self.page_ids)


@runtime_checkable
class CacheManager(Protocol):
    """Interface for KV cache managers (RadixCacheManager, NaiveCacheManager)."""

    def match_prefix(self, input_ids: list[int]) -> int: ...
    def insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None: ...
    def evict(self, num_pages: int) -> object: ...
    def remove(self, input_ids: list[int]) -> None: ...


class KVCachePool:
    """Manages GPU memory for KV cache as pages.

    Memory layout: (2, num_layers, num_pages, page_size, num_kv_heads, head_dim)
    where dim 0 stores [k_cache, v_cache].
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # (2, num_layers, num_pages, page_size, num_heads, head_dim)
        self.buffer = torch.empty(
            2,
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.free_pages: list[int] = list(range(num_pages))

    def alloc(self, num_pages: int) -> BaseCacheHandle:
        """Allocate num_pages from the free pool."""
        if len(self.free_pages) < num_pages:
            msg = f"KV cache out of memory: requested {num_pages} pages, only {len(self.free_pages)} free"
            raise RuntimeError(msg)

        handle = BaseCacheHandle()
        for _ in range(num_pages):
            page_id = self.free_pages.pop()
            handle.page_ids.append(page_id)
        return handle

    def free(self, handle: BaseCacheHandle) -> None:
        """Return pages to the free pool."""
        for page_id in handle.page_ids:
            self.free_pages.append(page_id)
        handle.page_ids.clear()

    def free_count(self) -> int:
        return len(self.free_pages)

    def get_kv_cache(self, layer_idx: int) -> tuple:
        """Get (k_cache, v_cache) tensors for a specific layer."""
        return self.buffer[0, layer_idx], self.buffer[1, layer_idx]

    def get_all_kv_cache(self) -> tuple:
        """Get full (k_cache, v_cache) tensors (all layers)."""
        return self.buffer[0], self.buffer[1]

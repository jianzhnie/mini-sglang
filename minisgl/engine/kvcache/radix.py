"""Radix tree-based KV cache for prefix sharing.

Maintains a radix tree (trie) of token sequences. When a new request
arrives, the tree is traversed to find the longest common prefix,
avoiding redundant KV cache computation.
"""

__all__ = ["RadixNode", "RadixCacheManager"]
from typing import Optional

from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class RadixNode:
    """A node in the radix tree."""

    __slots__ = ("token", "children", "ref_count", "cache_handle", "parent")

    def __init__(self, token: int = -1):
        self.token = token
        self.children: dict[int, "RadixNode"] = {}
        self.ref_count: int = 0
        self.cache_handle: BaseCacheHandle | None = None
        self.parent: Optional["RadixNode"] = None


class RadixCacheManager:
    """Radix tree cache that matches common prefixes and shares KV cache pages."""

    def __init__(self, pool: KVCachePool, page_size: int):
        self.pool = pool
        self.page_size = page_size
        self.root = RadixNode()
        self.root.ref_count = 1  # Root always referenced

    def match_prefix(self, input_ids: list[int]) -> int:
        """Return the number of tokens that match a cached prefix."""
        node = self.root
        matched = 0
        for token_id in input_ids:
            if token_id in node.children:
                node = node.children[token_id]
                matched += 1
            else:
                break
        return matched - (matched % self.page_size)

    def insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """Insert a token sequence into the radix tree."""
        node = self.root
        for token_id in input_ids:
            if token_id not in node.children:
                child = RadixNode(token_id)
                child.parent = node
                node.children[token_id] = child
            node = node.children[token_id]
            node.ref_count += 1
        node.cache_handle = handle

    def evict(self, num_pages: int) -> list[BaseCacheHandle]:
        """Evict least recently used nodes to free pages."""
        evicted: list[BaseCacheHandle] = []
        pages_freed = 0

        def _collect_evictable(node: RadixNode) -> list[RadixNode]:
            candidates = []
            if node.ref_count == 0 and node.cache_handle is not None:
                candidates.append(node)
            for child in node.children.values():
                candidates.extend(_collect_evictable(child))
            return candidates

        candidates = _collect_evictable(self.root)
        for node in candidates:
            if pages_freed >= num_pages:
                break
            if node.cache_handle is not None:
                self.pool.free(node.cache_handle)
                pages_freed += node.cache_handle.num_pages()
                evicted.append(node.cache_handle)
                node.cache_handle = None
                # Remove leaf node
                if node.parent and not node.children:
                    del node.parent.children[node.token]

        return evicted

    def remove(self, input_ids: list[int]) -> None:
        """Decrement reference counts when a request finishes."""
        node = self.root
        for token_id in input_ids:
            if token_id not in node.children:
                return
            node = node.children[token_id]
            node.ref_count -= 1

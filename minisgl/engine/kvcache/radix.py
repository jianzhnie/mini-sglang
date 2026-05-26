"""Radix tree-based KV cache for prefix sharing.

Maintains a radix tree (trie) of token sequences. When a new request
arrives, the tree is traversed to find the longest common prefix,
avoiding redundant KV cache computation.
"""

__all__ = ["RadixCacheManager", "RadixNode"]
from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class RadixNode:
    """A node in the radix tree."""

    __slots__ = ("cache_handle", "children", "parent", "ref_count", "token")

    def __init__(self, token: int = -1) -> None:
        self.token = token
        self.children: dict[int, RadixNode] = {}
        self.ref_count: int = 0
        self.cache_handle: BaseCacheHandle | None = None
        self.parent: RadixNode | None = None


class RadixCacheManager:
    """Radix tree cache that matches common prefixes and shares KV cache pages."""

    def __init__(self, pool: KVCachePool, page_size: int) -> None:
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
        """Evict least recently used nodes to free pages.

        Traverses tree depth-first, stopping early once enough pages are freed.
        """
        evicted: list[BaseCacheHandle] = []
        remaining = num_pages

        def _evict_from(node: RadixNode) -> int:
            """Recursively evict from subtree. Returns pages freed from this subtree."""
            nonlocal remaining
            if remaining <= 0:
                return 0

            freed = 0
            # Evict children first (leaf-first eviction for better prefix sharing)
            for child in list(node.children.values()):
                freed += _evict_from(child)

            # Evict this node if it has a cache handle and is unreferenced
            if remaining > 0 and node.ref_count == 0 and node.cache_handle is not None:
                self.pool.free(node.cache_handle)
                pages = node.cache_handle.num_pages()
                remaining -= pages
                freed += pages
                evicted.append(node.cache_handle)
                node.cache_handle = None
                # Prune leaf node
                if node.parent and not node.children:
                    del node.parent.children[node.token]

            return freed

        _evict_from(self.root)
        return evicted

    def remove(self, input_ids: list[int]) -> None:
        """Decrement reference counts when a request finishes."""
        node = self.root
        for token_id in input_ids:
            if token_id not in node.children:
                return
            node = node.children[token_id]
            node.ref_count -= 1

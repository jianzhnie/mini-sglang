"""Radix tree-based KV cache for prefix sharing.

Maintains a radix tree (trie) of token sequences. When a new request
arrives, the tree is traversed to find the longest common prefix,
avoiding redundant KV cache computation.

Key invariants (teaching-simplified SGLang semantics):

- Page ownership belongs to the tree, not to requests. ``insert`` records on
  every node the page holding its token's KV (``page_id``); a request finishing
  only drops its reference counts (``remove``), it never frees pages.
- Eviction is page-granular: a page is owned by its *page-boundary node*
  (depth ``d`` with ``(d+1) % page_size == 0``), or by the chain's leaf for a
  partial tail page. A page is freed only when the whole page's nodes are
  detached from the tree, so no remaining node can reference a freed page
  (no stale hits, no double-free).
- ``match_prefix`` never reports the last token as cached
  (``matched_len <= len(input_ids) - 1``): the final token must be forwarded
  again so sampling has logits.
"""

__all__ = ["RadixCacheManager", "RadixNode"]
from minisgl.engine.kvcache.pool import BaseCacheHandle, KVCachePool


class RadixNode:
    """A node in the radix tree.

    ``page_id`` is the pool page holding this node's token KV. For the node
    reached after consuming token index ``d`` (0-indexed depth), that is
    ``handle.page_ids[d // page_size]`` of the inserting request.
    """

    __slots__ = ("children", "depth", "page_id", "parent", "ref_count", "token")

    def __init__(self, token: int = -1) -> None:
        self.token = token
        self.children: dict[int, RadixNode] = {}
        self.ref_count: int = 0
        self.page_id: int = -1
        self.parent: RadixNode | None = None
        # 0-indexed token depth (root is -1); maintained by insert/remove
        # when the node is linked under its parent.
        self.depth: int = -1


class RadixCacheManager:
    """Radix tree cache that matches common prefixes and shares KV cache pages."""

    def __init__(self, pool: KVCachePool, page_size: int) -> None:
        self.pool = pool
        self.page_size = page_size
        self.root = RadixNode()
        self.root.ref_count = 1  # Root always referenced

    def _depth(self, node: RadixNode) -> int:
        """0-indexed token depth of a node (root is -1), maintained on insert."""
        return node.depth

    def match_prefix(self, input_ids: list[int]) -> tuple[int, list[int]]:
        """Match a cached prefix; return (matched_len, shared_page_ids).

        ``matched_len`` is page-aligned and at most ``len(input_ids) - 1``
        (rounded down to a page boundary), so the last token is always
        re-computed. ``shared_page_ids[j]`` is the pool page holding the KV of
        tokens ``[j*page_size, (j+1)*page_size)``, read from the page-boundary
        node at depth ``(j+1)*page_size - 1`` along the matched path.
        """
        if not input_ids:
            return 0, []
        node = self.root
        path: list[RadixNode] = []  # path[d] = node at token depth d
        for token_id in input_ids:
            if token_id in node.children:
                node = node.children[token_id]
                path.append(node)
            else:
                break
        matched = len(path)
        matched_len = matched - (matched % self.page_size)
        # Keep at least the last token uncached so sampling produces logits.
        matched_len = min(
            matched_len, ((len(input_ids) - 1) // self.page_size) * self.page_size
        )
        shared_pages = [
            path[(j + 1) * self.page_size - 1].page_id
            for j in range(matched_len // self.page_size)
        ]
        return matched_len, shared_pages

    def insert(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """Insert a token sequence, claiming one reference per node on the path.

        ``handle.page_ids`` must be the full page table for ``input_ids``
        (shared prefix pages first, then freshly allocated pages).
        """
        node = self.root
        for d, token_id in enumerate(input_ids):
            if token_id not in node.children:
                child = RadixNode(token_id)
                child.parent = node
                child.depth = node.depth + 1
                node.children[token_id] = child
            node = node.children[token_id]
            node.ref_count += 1
            node.page_id = handle.page_ids[d // self.page_size]

    def remove(self, input_ids: list[int], handle: BaseCacheHandle) -> None:
        """Drop this request's references when it finishes. Never frees pages
        that hold token KV; those are owned by the tree and freed by ``evict``.

        Also extends the tree with the full final sequence (prompt + generated
        tokens, like SGLang) so generated tokens stay cacheable, and returns
        over-allocated tail pages (allocated for ``max_tokens`` headroom but
        never filled with tokens) directly to the pool.
        """
        node = self.root
        for d, token_id in enumerate(input_ids):
            child = node.children.get(token_id)
            if child is None:
                # Generated suffix: new node, owned by the tree (ref_count 0).
                child = RadixNode(token_id)
                child.parent = node
                child.depth = node.depth + 1
                child.page_id = handle.page_ids[d // self.page_size]
                node.children[token_id] = child
            else:
                # This request's insert() added one reference to prompt nodes.
                # The floor guards generated-suffix nodes created by another
                # request's remove(), which this request never referenced.
                child.ref_count = max(0, child.ref_count - 1)
            node = child

        # Pages beyond the last token were never written; the tree only owns
        # pages that hold token KV, so return the unused tail pages here.
        used_pages = (len(input_ids) + self.page_size - 1) // self.page_size
        extra = handle.page_ids[used_pages:]
        if extra:
            self.pool.free_pages_by_id(extra)

    def evict(self, num_pages: int) -> list[int]:
        """Evict unreferenced (ref_count == 0) leaf chains, freeing whole pages.

        Leaf-first, page-granular: for each candidate leaf, walk up while nodes
        are unreferenced and have no sibling branches, then detach whole pages
        only (a page once started is always fully detached). Returns the list
        of freed page IDs.
        """
        freed: list[int] = []
        remaining = num_pages
        while remaining > 0:
            leaves = [n for n in self._iter_leaves() if n.ref_count == 0]
            if not leaves:
                break
            # Deepest leaves free the most pages per detachment.
            leaves.sort(key=self._depth, reverse=True)
            progress = False
            for leaf in leaves:
                if remaining <= 0:
                    break
                n_freed = self._evict_leaf_chain(leaf, remaining, freed)
                if n_freed > 0:
                    remaining -= n_freed
                    progress = True
                    break
            if not progress:
                break
        return freed

    def _iter_leaves(self) -> list[RadixNode]:
        """Collect all leaf nodes (iterative to avoid stack overflow)."""
        leaves: list[RadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and not node.children:
                leaves.append(node)
            stack.extend(node.children.values())
        return leaves

    def _evict_leaf_chain(
        self, leaf: RadixNode, max_pages: int, freed: list[int]
    ) -> int:
        """Detach whole pages from an unreferenced leaf chain; return count."""
        # Collect the removable chain (leaf first): unreferenced nodes with no
        # sibling branches below the chain.
        chain: list[RadixNode] = []
        node = leaf
        while node is not self.root and node.ref_count == 0 and len(node.children) <= 1:
            chain.append(node)
            node = node.parent

        ps = self.page_size
        depth = self._depth(leaf)
        pos = 0
        pages_freed = 0
        while pos < len(chain) and pages_freed < max_pages:
            # First group is the partial tail page (tokens depth..j*ps); later
            # groups are full pages of exactly `ps` nodes ending at a
            # page-boundary node. Never start a page we cannot finish.
            group = (depth % ps) + 1 if pos == 0 else ps
            if pos + group > len(chain):
                break
            # The group's deepest node owns the page: the leaf for a partial
            # tail page, else the page-boundary node ((d+1) % ps == 0).
            owner = chain[pos]
            for victim in chain[pos : pos + group]:
                del victim.parent.children[victim.token]
            self.pool.free_pages_by_id([owner.page_id])
            freed.append(owner.page_id)
            pages_freed += 1
            pos += group
        return pages_freed

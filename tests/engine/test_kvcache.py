"""KV cache: paged pool, radix tree, naive manager unit tests.

Run: python3 tests/engine/test_kvcache.py   (or: python -m pytest tests/engine/test_kvcache.py)
"""

import sys
import unittest
from pathlib import Path

import torch

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



# ── TestKVCachePool ──
class TestKVCachePool(unittest.TestCase):
    def test_pool_alloc_free(self):
        from minisgl.engine.kvcache.pool import KVCachePool

        pool = KVCachePool(
            num_layers=2,
            num_pages=100,
            page_size=16,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(pool.free_count(), 100)
        h1 = pool.alloc(10)
        self.assertEqual(h1.num_pages(), 10)
        self.assertEqual(pool.free_count(), 90)
        h2 = pool.alloc(5)
        self.assertEqual(h2.num_pages(), 5)
        self.assertEqual(pool.free_count(), 85)
        pool.free(h1)
        self.assertEqual(pool.free_count(), 95)
        pool.free(h2)
        self.assertEqual(pool.free_count(), 100)

    def test_free_idempotent_and_free_pages_by_id(self):
        from minisgl.engine.kvcache.pool import KVCachePool

        pool = KVCachePool(
            num_layers=2,
            num_pages=10,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        h = pool.alloc(3)
        self.assertEqual(pool.free_count(), 7)
        pool.free(h)
        self.assertEqual(pool.free_count(), 10)
        pool.free(h)  # second free must be a no-op
        self.assertEqual(pool.free_count(), 10)

        h2 = pool.alloc(2)
        pool.free_pages_by_id(list(h2.page_ids))
        self.assertEqual(pool.free_count(), 10)

    def test_get_all_kv_cache(self):
        from minisgl.engine.kvcache.pool import KVCachePool

        pool = KVCachePool(
            num_layers=4,
            num_pages=20,
            page_size=16,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        k_all, v_all = pool.get_all_kv_cache()
        self.assertEqual(k_all.shape, (4, 20, 16, 4, 32))
        self.assertEqual(v_all.shape, (4, 20, 16, 4, 32))


# ── Test Radix Cache ──

# ── TestRadixCache ──
class TestRadixCache(unittest.TestCase):
    def test_match_prefix(self):
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.engine.kvcache.radix import RadixCacheManager

        pool = KVCachePool(
            num_layers=2,
            num_pages=100,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        radix = RadixCacheManager(pool, page_size=4)

        # Insert a sequence
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        handle = pool.alloc(2)
        radix.insert(tokens, handle)

        # Match full prefix
        matched, shared = radix.match_prefix([1, 2, 3, 4, 9, 0])
        self.assertEqual(matched, 4)  # Pages are size 4
        self.assertEqual(shared, [handle.page_ids[0]])

        # Match partial
        matched, shared = radix.match_prefix([1, 2, 5, 6])
        self.assertEqual(matched, 0)  # Diverges at token 3
        self.assertEqual(shared, [])

        # No match
        matched, shared = radix.match_prefix([99, 99])
        self.assertEqual(matched, 0)
        self.assertEqual(shared, [])

    def test_match_prefix_keeps_last_token_uncached(self):
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.engine.kvcache.radix import RadixCacheManager

        pool = KVCachePool(
            num_layers=2,
            num_pages=100,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        radix = RadixCacheManager(pool, page_size=4)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        handle = pool.alloc(2)
        radix.insert(tokens, handle)

        # Exact same sequence: the last token must be re-forwarded so that
        # sampling has logits, so matched_len is clamped to len - 1 (aligned).
        matched, shared = radix.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(matched, 4)
        self.assertEqual(shared, [handle.page_ids[0]])


# ── Test Scheduler Batch Logic ──

# ── TestNaiveCacheManager ──
class TestNaiveCacheManager(unittest.TestCase):
    def _make_pool(self, num_pages=50):
        from minisgl.engine.kvcache.pool import KVCachePool

        return KVCachePool(
            num_layers=2,
            num_pages=num_pages,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_remove_frees_pages(self):
        from minisgl.engine.kvcache.naive import NaiveCacheManager

        pool = self._make_pool(50)
        mgr = NaiveCacheManager(pool, page_size=4)
        handle = pool.alloc(5)
        mgr.insert([1, 2, 3], handle)  # no-op: nothing tracked
        self.assertEqual(pool.free_count(), 45)
        mgr.remove([1, 2, 3], handle)
        self.assertEqual(pool.free_count(), 50)

    def test_evict_is_noop(self):
        from minisgl.engine.kvcache.naive import NaiveCacheManager

        pool = self._make_pool(10)
        mgr = NaiveCacheManager(pool, page_size=4)
        handle = pool.alloc(6)
        mgr.insert([1, 2, 3], handle)
        # Naive keeps no cached pages, so eviction reclaims nothing.
        self.assertEqual(mgr.evict(4), [])
        self.assertEqual(pool.free_count(), 4)

    def test_match_prefix_always_zero(self):
        from minisgl.engine.kvcache.naive import NaiveCacheManager

        pool = self._make_pool(10)
        mgr = NaiveCacheManager(pool, page_size=4)
        self.assertEqual(mgr.match_prefix([1, 2, 3]), (0, []))


# ── Test Radix Cache Evict/Remove ──

# ── TestRadixCacheEvictRemove ──
class TestRadixCacheEvictRemove(unittest.TestCase):
    def _make_pool_and_radix(self, num_pages=50):
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.engine.kvcache.radix import RadixCacheManager

        pool = KVCachePool(
            num_layers=2,
            num_pages=num_pages,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        radix = RadixCacheManager(pool, page_size=4)
        return pool, radix

    def test_remove_decrements_refcount(self):
        pool, radix = self._make_pool_and_radix()
        tokens = [1, 2, 3, 4]
        handle = pool.alloc(1)
        radix.insert(tokens, handle)
        node = radix.root.children[1]
        self.assertEqual(node.ref_count, 1)
        radix.remove(tokens, handle)
        self.assertEqual(node.ref_count, 0)
        # remove() must not free pages; they are owned by the tree.
        self.assertEqual(pool.free_count(), 49)

    def test_evict_after_remove(self):
        pool, radix = self._make_pool_and_radix(20)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        handle = pool.alloc(2)
        radix.insert(tokens, handle)
        radix.remove(tokens, handle)
        evicted = radix.evict(2)
        self.assertEqual(sorted(evicted), sorted(handle.page_ids))
        self.assertEqual(pool.free_count(), 20)
        # The sequence is fully detached: no stale prefix match afterwards.
        self.assertEqual(radix.match_prefix(tokens), (0, []))

    def test_evict_partial_tail_page(self):
        # 6 tokens with page_size 4: one full page + one partial tail page.
        pool, radix = self._make_pool_and_radix(20)
        tokens = [1, 2, 3, 4, 5, 6]
        handle = pool.alloc(2)
        radix.insert(tokens, handle)
        radix.remove(tokens, handle)
        evicted = radix.evict(2)
        self.assertEqual(sorted(evicted), sorted(handle.page_ids))
        self.assertEqual(pool.free_count(), 20)

    def test_evict_shared_prefix_only_frees_once(self):
        # Two sequences sharing page 0; evicting everything frees each page once.
        pool, radix = self._make_pool_and_radix(20)
        h1 = pool.alloc(2)
        radix.insert([1, 2, 3, 4, 5, 6], h1)
        matched, shared = radix.match_prefix([1, 2, 3, 4, 7, 8])
        self.assertEqual(matched, 4)
        h2 = pool.alloc(1)
        h2.page_ids = shared + h2.page_ids
        h2.num_shared = len(shared)
        radix.insert([1, 2, 3, 4, 7, 8], h2)
        radix.remove([1, 2, 3, 4, 5, 6], h1)
        radix.remove([1, 2, 3, 4, 7, 8], h2)
        evicted = radix.evict(10)
        # Pages: h1[0] (shared page 0), h1[1], h2[1] — each exactly once.
        self.assertEqual(
            sorted(evicted), sorted({h1.page_ids[0], h1.page_ids[1], h2.page_ids[1]})
        )
        self.assertEqual(pool.free_count(), 20)

    def test_evict_respects_refcount(self):
        pool, radix = self._make_pool_and_radix(20)
        tokens = [1, 2, 3, 4]
        handle = pool.alloc(1)
        radix.insert(tokens, handle)
        evicted = radix.evict(1)
        self.assertEqual(len(evicted), 0)

    def test_rollback_insert_removes_never_written_chain(self):
        """rollback_insert undoes an insert whose KV was never written."""
        pool, radix = self._make_pool_and_radix(20)
        tokens = list(range(1, 9))  # 8 tokens = 2 pages @ page_size 4
        handle = pool.alloc(2)
        radix.insert(tokens, handle)
        self.assertEqual(pool.free_count(), 18)

        radix.rollback_insert(tokens, handle)
        # The whole chain is gone — no stale prefix to match.
        self.assertEqual(radix.match_prefix(tokens), (0, []))
        # And the request's exclusively-owned pages are returned.
        self.assertEqual(pool.free_count(), 20)

    def test_rollback_insert_preserves_shared_prefix(self):
        """Rolling back one request must not disturb another's shared prefix."""
        pool, radix = self._make_pool_and_radix(30)
        # A caches [1..8] across pages p0,p1 (2 pages @ page_size 4).
        h1 = pool.alloc(2)
        radix.insert(list(range(1, 9)), h1)
        self.assertEqual(pool.free_count(), 28)

        # B shares A's prefix pages and extends by [9..12] on one own page.
        matched, shared = radix.match_prefix(list(range(1, 13)))
        self.assertEqual(matched, 8)
        self.assertEqual(len(shared), 2)
        h2 = pool.alloc(1)
        h2.page_ids = shared + h2.page_ids
        h2.num_shared = len(shared)
        radix.insert(list(range(1, 13)), h2)
        self.assertEqual(pool.free_count(), 27)

        # A "fails its forward" -> roll back ONLY A's insert.
        radix.rollback_insert(list(range(1, 9)), h1)

        # B's shared prefix is untouched: still fully matched.
        matched2, shared2 = radix.match_prefix(list(range(1, 13)))
        self.assertEqual(matched2, 8)
        self.assertEqual(shared2, shared)
        # A's pages are shared with B, so they must NOT be freed (B still
        # references them). Only B's own single page is held by the tree.
        self.assertEqual(pool.free_count(), 27)
        # A alone no longer matches its prompt as a *complete cached prefix*
        # for a longer request... but B keeps it alive, so [1..8] as a prefix
        # of a NEW request would still match (it is B's live shared prefix).
        # The invariant that matters: B's data was not corrupted.
        self.assertEqual(len(shared2), 2)

    def test_rollback_insert_frees_own_pages_when_unshared(self):
        """Rolling back an unshared insert fully releases its pages."""
        pool, radix = self._make_pool_and_radix(30)
        h = pool.alloc(2)
        radix.insert(list(range(1, 9)), h)  # 8 tokens, 2 pages, no sharing
        self.assertEqual(pool.free_count(), 28)

        radix.rollback_insert(list(range(1, 9)), h)
        self.assertEqual(pool.free_count(), 30)
        self.assertEqual(radix.match_prefix(list(range(1, 9))), (0, []))
        """remove() over a handle too small for the sequence must not crash.

        Guard for the concurrent/abnormal case where a request's generated
        length exceeds the pages it was allocated (normally unreachable — the
        scheduler allocates `upper` covering the full length). The out-of-range
        suffix nodes get page_id -1, and eviction detaches them without freeing
        any bogus page.
        """
        pool, radix = self._make_pool_and_radix(20)
        tokens = list(range(1, 17))  # 16 tokens
        handle = pool.alloc(2)  # only 2 pages x 4 tokens = 8 slots
        radix.insert(tokens[:8], handle)  # only 8 tokens were cached/written

        # remove() with the full 16-token sequence extends the tree past the
        # allocated pages without raising (out-of-range nodes keep page_id -1).
        radix.remove(tokens, handle)

        # Eviction detaches the whole chain and frees only the two real pages
        # — never a (-1) placeholder — and never crashes.
        evicted = radix.evict(10)
        self.assertEqual(sorted(evicted), sorted(handle.page_ids))
        self.assertNotIn(-1, evicted)
        self.assertEqual(pool.free_count(), 20)

        # Fully detached: no stale prefix match afterwards.
        self.assertEqual(radix.match_prefix(tokens), (0, []))


# ── Test DecodeManager ──



if __name__ == '__main__':
    unittest.main(verbosity=2)

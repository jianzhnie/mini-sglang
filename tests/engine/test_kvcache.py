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


# ── Test DecodeManager ──



if __name__ == '__main__':
    unittest.main(verbosity=2)

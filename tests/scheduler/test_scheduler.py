"""Scheduler: batching, prefill/decode managers, end-to-end lifecycle.

Run: python3 tests/scheduler/test_scheduler.py   (or: python -m pytest tests/scheduler/test_scheduler.py)
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



# ── TestSchedulerBatch ──
class TestSchedulerBatch(unittest.TestCase):
    def test_batch_creation(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import Batch, Req

        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            sampling_params=SamplingParams(max_tokens=100),
        )
        batch = Batch(reqs=[req], phase="prefill")
        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(batch.phase, "prefill")
        self.assertEqual(len(req.input_ids), 5)
        self.assertEqual(req.uncached_len, 5)
        self.assertFalse(req.is_finished)

    def test_req_lifecycle(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import Req, SequenceStatus

        req = Req(
            input_ids=[1, 2, 3],
            uid=0,
            sampling_params=SamplingParams(max_tokens=10),
        )
        self.assertEqual(req.status, SequenceStatus.WAITING)
        req.status = SequenceStatus.RUNNING
        req.append_token(42)
        self.assertEqual(len(req.input_ids), 4)
        self.assertEqual(req.output_len, 1)

    def test_context_prepare(self):
        from minisgl.config import SamplingParams
        from minisgl.engine.batch_context import BatchContext
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.batch import Batch, Req

        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        handle = pool.alloc(2)

        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            cache_handle=handle,
            sampling_params=SamplingParams(),
        )
        batch = Batch(reqs=[req], phase="prefill")

        ctx = BatchContext(
            max_running_req=4,
            max_seq_len=16,
            page_size=4,
            device=torch.device("cpu"),
        )
        ctx.prepare(batch)

        self.assertEqual(batch.input_ids.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(batch.positions.tolist(), [0, 1, 2, 3, 4])
        self.assertIsNotNone(batch.attn_meta)
        self.assertIsNotNone(batch.attn_meta.write_loc)


# ── Test Distributed ──

# ── TestDecodeManager ──
class TestDecodeManager(unittest.TestCase):
    def test_schedule_decode_basic(self):
        from minisgl.config import SamplingParams, ServerArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.batch import Req, SequenceStatus
        from minisgl.scheduler.decode import DecodeManager

        args = ServerArgs(
            model_path="/tmp/test",
            max_running_req=8,
            max_seq_len=64,
            page_size=4,
        )
        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        handle = pool.alloc(4)
        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            cache_handle=handle,
            sampling_params=SamplingParams(max_tokens=10),
        )
        req.status = SequenceStatus.RUNNING

        dm = DecodeManager(args, pool, device=torch.device("cpu"))
        batch = dm.schedule_decode([req])
        self.assertIsNotNone(batch)
        self.assertEqual(batch.phase, "decode")
        self.assertEqual(batch.input_ids.tolist(), [[5]])
        self.assertEqual(batch.positions.tolist(), [[4]])
        meta = batch.attn_meta
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forward_mode, "decode")
        self.assertIsNotNone(meta.write_loc)
        self.assertIsNotNone(meta.block_table)
        self.assertIsNotNone(meta.cache_seqlens)
        # cache_seqlens semantics: total length including the current token.
        self.assertEqual(meta.cache_seqlens.tolist(), [5])
        self.assertEqual(meta.max_seqlen, 5)

    def test_schedule_decode_empty(self):
        from minisgl.config import ServerArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.decode import DecodeManager

        args = ServerArgs(model_path="/tmp/test", max_seq_len=64, page_size=4)
        pool = KVCachePool(
            num_layers=2,
            num_pages=10,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        dm = DecodeManager(args, pool, device=torch.device("cpu"))
        self.assertIsNone(dm.schedule_decode([]))


# ── Test PrefillManager ──

# ── TestPrefillManager ──
class TestPrefillManager(unittest.TestCase):
    def test_schedule_prefill_basic(self):
        from minisgl.config import SamplingParams, ServerArgs
        from minisgl.engine.kvcache.naive import NaiveCacheManager
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.batch import Req, SequenceStatus
        from minisgl.scheduler.prefill import PrefillManager

        args = ServerArgs(
            model_path="/tmp/test",
            max_running_req=8,
            max_seq_len=64,
            page_size=4,
        )
        pool = KVCachePool(
            num_layers=2,
            num_pages=50,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache = NaiveCacheManager(pool, page_size=4)
        pm = PrefillManager(args, pool, cache)

        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            sampling_params=SamplingParams(max_tokens=10),
        )
        pm.add_request(req)
        batch = pm.schedule_prefill()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.phase, "prefill")
        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(req.status, SequenceStatus.RUNNING)
        self.assertIsNotNone(req.cache_handle)

    def test_schedule_prefill_empty(self):
        from minisgl.config import ServerArgs
        from minisgl.engine.kvcache.naive import NaiveCacheManager
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.prefill import PrefillManager

        args = ServerArgs(model_path="/tmp/test", max_seq_len=64, page_size=4)
        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache = NaiveCacheManager(pool, page_size=4)
        pm = PrefillManager(args, pool, cache)
        self.assertIsNone(pm.schedule_prefill())

    def test_prefix_match_then_eviction_no_duplicate_pages(self):
        """Regression: sharing a cached prefix must not yield a duplicate page.

        Prefill matches a shared prefix, then — under memory pressure — evicts
        cache pages that turn out to be the very prefix pages just matched.
        The evicted pages return to the free list and are handed right back to
        the same request by alloc(), so its page table ends up with the same
        page twice. Its own prefix KV then collides with its fresh writes
        (silent corruption). schedule_prefill() re-matches after eviction so a
        request only reuses pages that are provably still held by the tree.
        """
        from minisgl.config import SamplingParams, ServerArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.engine.kvcache.radix import RadixCacheManager
        from minisgl.scheduler.batch import Req
        from minisgl.scheduler.prefill import PrefillManager

        args = ServerArgs(
            model_path="/tmp/test",
            max_running_req=8,
            max_seq_len=64,
            page_size=4,
        )
        pool = KVCachePool(
            num_layers=1,
            num_pages=8,
            page_size=4,
            num_kv_heads=1,
            head_dim=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        radix = RadixCacheManager(pool, page_size=4)
        pm = PrefillManager(args, pool, radix)

        def _finish(prompt: list[int], max_tokens: int) -> None:
            req = Req(
                input_ids=prompt,
                uid=0,
                sampling_params=SamplingParams(max_tokens=max_tokens),
            )
            pm.add_request(req)
            pm.schedule_prefill()
            pm.running = [req]
            pm.remove_finished_batch([req])

        # Request A caches a 20-token prefix in the tree (5 pages: 8 - 3 free).
        _finish(list(range(100, 120)), max_tokens=1)
        self.assertEqual(pool.free_count(), 3)

        # Tighten the pool so scheduling B requires eviction.
        occupied = pool.alloc(2)
        self.assertEqual(pool.free_count(), 1)

        # B shares A's full prefix and wants to extend far enough that its
        # fresh-page need exceeds the pool's free count -> eviction runs.
        B = Req(
            input_ids=list(range(100, 120)),
            uid=1,
            sampling_params=SamplingParams(max_tokens=8),
        )
        pm.add_request(B)
        pm.pending.clear()
        pm.pending.append(B)
        batch = pm.schedule_prefill()

        # The evictor may have detached B's just-matched prefix (its nodes are
        # ref_count == 0 until insert()). That is fine and expected; what must
        # NEVER happen is B ending up with the same page both as a "shared"
        # prefix page and as a freshly allocated page.
        if batch is not None:
            handle = B.cache_handle
            self.assertEqual(len(handle.page_ids), len(set(handle.page_ids)),
                             "page table contains a duplicate page")
            # Shared pages must still be owned by the tree (not in free list).
            for shared in handle.page_ids[: handle.num_shared]:
                self.assertNotIn(shared, pool.free_pages)
            self.assertEqual(B.cached_len, handle.cached_len)
        else:
            # B stayed pending: under pressure its prefix was evicted and not
            # enough other pages are free this step. It must schedule cleanly
            # once the pages held by the "other running request" are released.
            self.assertIn(B, pm.pending)
            pool.free_pages_by_id(occupied.page_ids)  # other request finishes
            pm.pending.clear()
            pm.pending.append(B)
            batch2 = pm.schedule_prefill()
            self.assertIsNotNone(batch2)
            handle = B.cache_handle
            self.assertEqual(len(handle.page_ids), len(set(handle.page_ids)),
                             "page table contains a duplicate page")
            for shared in handle.page_ids[: handle.num_shared]:
                self.assertNotIn(shared, pool.free_pages)


# ── Test Shared Decoder Base Classes ──

# ── TestEndToEndScheduler ──
class TestEndToEndScheduler(unittest.TestCase):
    """Integration tests for the full Engine+Scheduler pipeline on CPU."""

    @staticmethod
    def _make_engine_scheduler(
        cache_strategy: str = "radix", seed: int | None = None, **server_overrides
    ):
        import json
        import tempfile

        from minisgl.config import ModelArgs, ServerArgs
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        if seed is not None:
            # Same seed -> identical random weights across instances, so two
            # schedulers can be compared token-for-token under greedy decoding.
            torch.manual_seed(seed)

        tmpdir = tempfile.mkdtemp()
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 512,
            "vocab_size": 256,
            "max_position_embeddings": 64,
            "ffn_dim": 512,
            "eos_token_id": 2,
        }
        import os

        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        model_args = ModelArgs.from_pretrained(tmpdir)
        server_kwargs = {
            "model_path": tmpdir,
            "tp_size": 1,
            "attention_backend": "pt",
            "max_running_req": 4,
            "max_seq_len": 64,
            "page_size": 8,
            "memory_ratio": 0.5,
            "cuda_graph_bs": 0,
        }
        server_kwargs.update(server_overrides)
        server_args = ServerArgs(**server_kwargs)
        engine = Engine(server_args, model_args, tp_rank=0)
        for param in engine.model.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, std=0.02)
            elif param.dim() == 1:
                nn.init.ones_(param)
        scheduler = Scheduler(server_args, engine, cache_strategy=cache_strategy)
        return scheduler

    @staticmethod
    def _run_to_completion(scheduler, max_steps: int = 200) -> dict:
        """Step until idle; return {uid: [(token_id, finished, reason), ...]}."""
        results: dict = {}
        steps = 0
        while not scheduler.is_idle() and steps < max_steps:
            for out in scheduler.step():
                results.setdefault(out.uid, []).append(
                    (out.token_id, out.finished, out.finish_reason)
                )
            steps += 1
        return results

    def test_single_request_generation(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        scheduler.add_request([1, 5, 10], SamplingParams(temperature=0.0, max_tokens=5))
        generated = []
        steps = 0
        while not scheduler.is_idle() and steps < 100:
            for out in scheduler.step():
                generated.append(out.token_id)
            steps += 1
        self.assertGreater(len(generated), 0)
        self.assertLessEqual(len(generated), 5)

    def test_multi_request_concurrent(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        uid1 = scheduler.add_request(
            [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3)
        )
        uid2 = scheduler.add_request(
            [4, 5, 6], SamplingParams(temperature=0.0, max_tokens=3)
        )
        results = {uid1: [], uid2: []}
        steps = 0
        while not scheduler.is_idle() and steps < 100:
            for out in scheduler.step():
                results[out.uid].append(out.token_id)
            steps += 1
        self.assertGreater(len(results[uid1]), 0)
        self.assertGreater(len(results[uid2]), 0)

    def test_eos_terminates_early(self):
        scheduler = self._make_engine_scheduler()
        self.assertIsInstance(scheduler.eos_token_id, set)
        self.assertIn(2, scheduler.eos_token_id)

    def test_decode_forward_error_isolates_batch(self):
        """A forward/sample failure must kill only the offending requests.

        Regression: without per-phase error isolation, one bad decode forward
        would raise out of step() and (in the server) take down the whole
        event loop, leaving every other request hung until timeout.
        """
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        uid1 = scheduler.add_request(
            [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3)
        )
        uid2 = scheduler.add_request(
            [4, 5, 6], SamplingParams(temperature=0.0, max_tokens=3)
        )

        real_forward = scheduler.engine.forward
        decode_calls = {"n": 0}

        def flaky_forward(batch):
            if batch.phase == "decode":
                decode_calls["n"] += 1
                raise RuntimeError("simulated decode failure")
            return real_forward(batch)

        scheduler.engine.forward = flaky_forward
        try:
            results = scheduler.step()
        finally:
            scheduler.engine.forward = real_forward

        # The two requests shared one decode batch; both must be surfaced as
        # finished-with-error (not silently dropped, not re-queued forever).
        error_uids = {
            out.uid
            for out in results
            if out.finished and out.finish_reason == "error"
        }
        self.assertEqual(error_uids, {uid1, uid2})
        self.assertGreaterEqual(decode_calls["n"], 1)
        self.assertTrue(scheduler.is_idle())

        # The scheduler must still accept and complete later requests.
        uid3 = scheduler.add_request(
            [7, 8, 9], SamplingParams(temperature=0.0, max_tokens=3)
        )
        final_reasons = {}
        steps = 0
        while not scheduler.is_idle() and steps < 100:
            for out in scheduler.step():
                if out.finished:
                    final_reasons[out.uid] = out.finish_reason
            steps += 1
        self.assertIn(final_reasons.get(uid3), ("stop", "length"))
        self.assertNotIn(uid3, error_uids)

    def test_prefill_forward_error_isolates_batch(self):
        """A prefill failure surfaces an error result and never starts decode."""
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        uid = scheduler.add_request(
            [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3)
        )

        real_forward = scheduler.engine.forward

        def flaky_forward(batch):
            if batch.phase == "prefill":
                raise ValueError("simulated prefill failure")
            return real_forward(batch)

        scheduler.engine.forward = flaky_forward
        try:
            results = scheduler.step()
        finally:
            scheduler.engine.forward = real_forward

        self.assertTrue(
            any(
                out.uid == uid
                and out.finished
                and out.finish_reason == "error"
                for out in results
            )
        )
        # Nothing left running: the errored request never reached decode.
        self.assertTrue(scheduler.is_idle())

    def test_sampling_error_isolates_batch(self):
        """Failures in sample() are isolated the same way as forward()."""
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        uid = scheduler.add_request(
            [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3)
        )

        def flaky_sample(logits, batch):
            raise RuntimeError("simulated sampling failure")

        real_sample = scheduler.engine.sample
        scheduler.engine.sample = flaky_sample
        try:
            results = scheduler.step()
        finally:
            scheduler.engine.sample = real_sample

        self.assertTrue(
            any(
                out.uid == uid
                and out.finished
                and out.finish_reason == "error"
                for out in results
            )
        )
        self.assertTrue(scheduler.is_idle())


# ── Test PyTorch Attention Decode Path ──

# ── TestEOSNormalization ──
class TestEOSNormalization(unittest.TestCase):
    def test_int_eos(self):
        from minisgl.scheduler.scheduler import Scheduler

        result = Scheduler._normalize_eos(2)
        self.assertEqual(result, {2})

    def test_list_eos(self):
        from minisgl.scheduler.scheduler import Scheduler

        result = Scheduler._normalize_eos([151645, 151643])
        self.assertEqual(result, {151645, 151643})

    def test_dict_eos(self):
        from minisgl.scheduler.scheduler import Scheduler

        result = Scheduler._normalize_eos({"token_id": 2})
        self.assertEqual(result, {2})

    def test_dict_eos_via_id_key(self):
        from minisgl.scheduler.scheduler import Scheduler

        result = Scheduler._normalize_eos({"id": 7})
        self.assertEqual(result, {7})

    def test_unparseable_returns_empty(self):
        """None / empty dict / bad types yield an empty set, not a fake {0}."""
        from minisgl.scheduler.scheduler import Scheduler

        self.assertEqual(Scheduler._normalize_eos(None), set())
        self.assertEqual(Scheduler._normalize_eos({}), set())
        self.assertEqual(Scheduler._normalize_eos({"foo": 1}), set())
        self.assertEqual(Scheduler._normalize_eos("x"), set())

    def test_load_eos_token_falls_back_when_none(self):
        # A scheduler with an unparseable config must still end up with {0}.
        import json
        import tempfile
        from pathlib import Path

        from minisgl.config import ServerArgs
        from minisgl.scheduler.scheduler import Scheduler

        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(
                json.dumps({"architectures": ["Qwen3ForCausalLM"], "eos_token_id": None})
            )
            # Build a bare scheduler object to call the private loader.
            args = ServerArgs(model_path=d)
            sched = object.__new__(Scheduler)
            sched.args = args
            self.assertEqual(sched._load_eos_token(), {0})


# ── Test Sampling Edge Cases ──

# ── TestFinishReasonAndAbort ──
class TestFinishReasonAndAbort(unittest.TestCase):
    @staticmethod
    def _make_scheduler():
        return TestEndToEndScheduler._make_engine_scheduler()

    def test_finish_reason_stop_and_length(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import Req

        scheduler = self._make_scheduler()
        # EOS token (2 in this fixture) ends with "stop".
        req = Req(input_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))
        self.assertEqual(scheduler._finish_reason(req, 2), "stop")
        self.assertIsNone(scheduler._finish_reason(req, 5))
        # ignore_eos disables the "stop" reason.
        req_ignore = Req(
            input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10, ignore_eos=True),
        )
        self.assertIsNone(scheduler._finish_reason(req_ignore, 2))

        # Hitting max_tokens ends with "length".
        req_len = Req(input_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=4))
        req_len.output_len = 4
        self.assertEqual(scheduler._finish_reason(req_len, 5), "length")

        # Hitting max_seq_len (64 in this fixture) ends with "length".
        req_seq = Req(
            input_ids=[0] * 64, sampling_params=SamplingParams(max_tokens=100)
        )
        self.assertEqual(scheduler._finish_reason(req_seq, 5), "length")

    def test_long_prompt_aborted(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_scheduler()
        # Prompt longer than max_seq_len (64) is aborted at add time.
        uid = scheduler.add_request(list(range(100)), SamplingParams(max_tokens=4))
        results = scheduler.step()
        self.assertEqual(len(results), 1)
        out = results[0]
        self.assertEqual(
            (out.uid, out.finished, out.finish_reason), (uid, True, "abort")
        )
        self.assertTrue(scheduler.is_idle())

    def test_abort_pending_request(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_scheduler()
        uid = scheduler.add_request([1, 2, 3], SamplingParams(max_tokens=4))
        self.assertEqual(len(scheduler.prefill_manager.pending), 1)

        self.assertTrue(scheduler.abort_request(uid))
        self.assertEqual(len(scheduler.prefill_manager.pending), 0)
        self.assertTrue(scheduler.is_idle())
        # Aborting an unknown UID is a no-op.
        self.assertFalse(scheduler.abort_request(uid))

    def test_abort_running_request_releases_resources(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_scheduler()
        uid = scheduler.add_request(
            [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
        )
        scheduler.step()  # prefill moves the request to running
        self.assertEqual(len(scheduler.prefill_manager.running), 1)
        req = scheduler.prefill_manager.running[0]
        self.assertIsNotNone(req.cache_handle)

        self.assertTrue(scheduler.abort_request(uid))
        self.assertEqual(len(scheduler.prefill_manager.running), 0)
        self.assertIsNone(req.cache_handle)  # cache handle released
        self.assertEqual(req.status.name, "FINISHED")
        self.assertTrue(scheduler.is_idle())
        # No further results are produced for the aborted request.
        self.assertEqual(scheduler.step(), [])


# ── Test Radix Prefix Sharing End-to-End (Regression) ──

# ── TestRadixPrefixSharingE2E ──
class TestRadixPrefixSharingE2E(unittest.TestCase):
    """Regression tests for the radix-cache rewrite (tree owns pages)."""

    def test_shared_prefix_matches_naive_baseline(self):
        # Guards the radix rewrite: two concurrent requests sharing >= 1 page
        # of prefix must generate token-for-token the same output as with the
        # naive (no-sharing) cache. Same seed -> identical weights, greedy
        # decoding -> any divergence is a sharing bug.
        from minisgl.config import SamplingParams

        prefix = list(range(10, 18))  # exactly one page (page_size=8)
        prompts = [prefix + [30, 31], prefix + [40, 41]]

        def run(cache_strategy):
            scheduler = TestEndToEndScheduler._make_engine_scheduler(
                cache_strategy=cache_strategy, seed=0
            )
            uids = [
                scheduler.add_request(p, SamplingParams(temperature=0.0, max_tokens=4))
                for p in prompts
            ]
            # Keep a reference before stepping; finished requests leave the
            # running list, but their Req objects retain cached_len.
            reqs = list(scheduler.prefill_manager.pending)
            results = TestEndToEndScheduler._run_to_completion(scheduler)
            return reqs, uids, results

        radix_reqs, radix_uids, radix_results = run("radix")
        _naive_reqs, naive_uids, naive_results = run("naive")

        # The second request must actually have reused the shared prefix.
        self.assertGreaterEqual(radix_reqs[1].cached_len, 8)

        for r_uid, n_uid in zip(radix_uids, naive_uids, strict=True):
            radix_tokens = [t for t, _, _ in radix_results[r_uid]]
            naive_tokens = [t for t, _, _ in naive_results[n_uid]]
            self.assertEqual(len(radix_tokens), 4)
            self.assertEqual(radix_tokens, naive_tokens)

    def test_same_prompt_twice_consistent(self):
        # Guards the stale-prefix-hit bug (match_prefix used to report the
        # full sequence as cached, skipping the last token's forward): the
        # second identical request hits the cached prefix via the extend
        # path, and its output must equal the first cold run token-for-token.
        from minisgl.config import SamplingParams

        scheduler = TestEndToEndScheduler._make_engine_scheduler(seed=0)
        prompt = list(range(10, 26))
        params = SamplingParams(temperature=0.0, max_tokens=4)

        # NOTE: add_request stores the caller's list by reference and appends
        # generated tokens to it, so each run needs its own copy.
        uid1 = scheduler.add_request(list(prompt), params)
        first = TestEndToEndScheduler._run_to_completion(scheduler)[uid1]

        uid2 = scheduler.add_request(list(prompt), params)
        req2 = scheduler.prefill_manager.pending[0]
        second = TestEndToEndScheduler._run_to_completion(scheduler)[uid2]
        # The second run must have taken the extend path (cached prefix).
        self.assertGreaterEqual(req2.cached_len, 8)
        self.assertEqual(
            [t for t, _, _ in second],
            [t for t, _, _ in first],
        )


# ── Test Prefill Termination (Regression) ──

# ── TestPrefillTermination ──
class TestPrefillTermination(unittest.TestCase):
    """Regression: termination is checked immediately after prefill."""

    def test_eos_at_prefill_stops_immediately(self):
        # Guards the missing post-prefill termination check: when the very
        # first sampled token is EOS, the request must finish right after
        # prefill instead of producing extra decode tokens.
        from minisgl.config import SamplingParams

        scheduler = TestEndToEndScheduler._make_engine_scheduler(seed=0)
        # Probe: discover the greedy first token for this prompt.
        scheduler.add_request([5, 6, 7], SamplingParams(temperature=0.0, max_tokens=1))
        first_token = scheduler.step()[0].token_id
        self.assertTrue(scheduler.is_idle())

        # Make that token an EOS and re-run with a larger token budget.
        scheduler.eos_token_id.add(first_token)
        uid = scheduler.add_request(
            [5, 6, 7], SamplingParams(temperature=0.0, max_tokens=5)
        )
        results = TestEndToEndScheduler._run_to_completion(scheduler)[uid]
        self.assertEqual(len(results), 1)
        token_id, finished, reason = results[0]
        self.assertEqual(token_id, first_token)
        self.assertTrue(finished)
        self.assertEqual(reason, "stop")

    def test_max_tokens_one_generates_exactly_one(self):
        # Same bug class via the length path: max_tokens=1 must produce
        # exactly one token with finish_reason "length" at prefill time.
        from minisgl.config import SamplingParams

        scheduler = TestEndToEndScheduler._make_engine_scheduler(seed=0)
        uid = scheduler.add_request(
            [8, 9], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        )
        results = TestEndToEndScheduler._run_to_completion(scheduler)[uid]
        self.assertEqual(len(results), 1)
        _token_id, finished, reason = results[0]
        self.assertTrue(finished)
        self.assertEqual(reason, "length")


# ── Test write_loc Guard (Regression) ──

# ── TestWriteLocGuard ──
class TestWriteLocGuard(unittest.TestCase):
    def test_negative_write_loc_does_not_touch_cache_tail(self):
        # Guards the silent negative-index write bug: write_loc == -1 entries
        # must be skipped; otherwise they write into the LAST rows of the KV
        # cache and corrupt whatever sequence lives there.
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3Attention

        config = ModelArgs(
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=16,
        )
        attn = Qwen3Attention(config)

        num_kv_heads, head_dim = 2, 16
        k_cache = torch.zeros(2, 4, num_kv_heads, head_dim)  # 8 flat slots
        v_cache = torch.zeros(2, 4, num_kv_heads, head_dim)
        k = torch.randn(1, num_kv_heads, 3, head_dim)
        v = torch.randn(1, num_kv_heads, 3, head_dim)
        write_loc = torch.tensor([0, -1, 5], dtype=torch.int32)

        attn.set_kv_cache(k_cache, v_cache)
        attn._write_kv_cache(k, v, write_loc)

        flat_k = k_cache.view(-1, num_kv_heads, head_dim)
        flat_v = v_cache.view(-1, num_kv_heads, head_dim)
        # Valid slots were written.
        self.assertTrue(torch.equal(flat_k[0], k[0, :, 0, :]))
        self.assertTrue(torch.equal(flat_v[5], v[0, :, 2, :]))
        # The -1 entry must not land in the last flat slot.
        self.assertTrue(torch.all(flat_k[-1] == 0))
        self.assertTrue(torch.all(flat_v[-1] == 0))


# ── Test Radix Eviction End-to-End (Regression) ──

# ── TestRadixEvictionE2E ──
class TestRadixEvictionE2E(unittest.TestCase):
    def test_eviction_frees_pages_for_new_request(self):
        # Guards two bugs at once: (1) finish() used to free pages directly
        # (double-free risk under prefix sharing) — now the tree owns them;
        # (2) evict() used to detach nodes without returning their pages to
        # the pool. Here a 3-page pool fills up, and the second request can
        # only be scheduled if evict() truly returns pages.
        from minisgl.config import SamplingParams

        scheduler = TestEndToEndScheduler._make_engine_scheduler(
            seed=0, max_running_req=1, max_seq_len=16
        )
        pool = scheduler.pool
        radix = scheduler.cache_manager
        self.assertEqual(pool.num_pages, 3)

        prompt_a = [10 + i for i in range(14)]
        params = SamplingParams(temperature=0.0, max_tokens=2)
        uid1 = scheduler.add_request(prompt_a, params)
        results1 = TestEndToEndScheduler._run_to_completion(scheduler)[uid1]
        self.assertEqual(len(results1), 2)
        # Finish does NOT free pages: the tree still owns them (1 page left).
        self.assertEqual(pool.free_count(), 1)
        matched, _shared = radix.match_prefix(prompt_a)
        self.assertEqual(matched, 8)

        # Second request (disjoint prefix) needs 2 pages; only 1 is free, so
        # scheduling it requires a real eviction.
        prompt_b = [100 + i for i in range(14)]
        uid2 = scheduler.add_request(prompt_b, params)
        results2 = TestEndToEndScheduler._run_to_completion(scheduler)[uid2]
        self.assertEqual(len(results2), 2)
        # Eviction detached the tail of prompt_a's chain (tokens 8..13).
        node = radix.root
        for token in prompt_a[:8]:
            node = node.children[token]
        self.assertNotIn(prompt_a[8], node.children)


# ── Test GraphRunner (CPU-safe paths) ──
class TestGraphRunner(unittest.TestCase):
    """Graph capture needs CUDA/NPU, but the eager-fallback + cleanup logic is CPU-testable."""

    def test_engine_creates_no_graph_runner_on_cpu(self):
        scheduler = TestEndToEndScheduler._make_engine_scheduler(cuda_graph_bs=8)
        # The engine gates graph capture to cuda/npu, so CPU stays eager.
        self.assertIsNone(scheduler.engine.graph_runner)

    def test_replay_empty_returns_none(self):
        """With no captured graphs, replay() falls back to eager (returns None)."""
        from minisgl.engine.graph import GraphRunner

        runner = object.__new__(GraphRunner)
        runner.graphs = {}
        runner.inputs = {}
        runner.outputs = {}
        # Batch with any request count: no graph large enough exists.
        from minisgl.scheduler.batch import Batch, Req

        batch = Batch(reqs=[Req()], phase="decode")
        self.assertIsNone(runner.replay(batch))

    def test_replay_no_fit_returns_none(self):
        """If every captured graph is smaller than the batch, fall back to eager."""
        from minisgl.engine.graph import GraphRunner

        runner = object.__new__(GraphRunner)
        runner.graphs = {1: object(), 2: object()}
        runner.inputs = {1: {}, 2: {}}
        runner.outputs = {1: None, 2: None}
        from minisgl.scheduler.batch import Batch, Req

        batch = Batch(reqs=[Req() for _ in range(4)], phase="decode")
        self.assertIsNone(runner.replay(batch))

    def test_clear_drops_state(self):
        from minisgl.engine.graph import GraphRunner

        runner = object.__new__(GraphRunner)
        runner.graphs = {1: object()}
        runner.inputs = {1: {"k": 1}}
        runner.outputs = {1: object()}
        runner.clear()
        self.assertEqual(runner.graphs, {})
        self.assertEqual(runner.inputs, {})
        self.assertEqual(runner.outputs, {})


# ── Test Device Utilities (NPU-aware) ──


if __name__ == '__main__':
    unittest.main(verbosity=2)

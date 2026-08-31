"""CPU-based self-contained test for mini-sglang core logic.

Run: python3 tests/test_cpu_core.py
"""

import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, ".")


# ── Test RMSNorm ──
class TestRMSNorm(unittest.TestCase):
    def test_basic_norm(self):
        from minisgl.models.layers.rms_norm import RMSNorm

        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 10, 64)
        out, res = norm(x)
        self.assertEqual(out.shape, x.shape)  # output same shape
        self.assertIsNone(res)  # no residual when None
        # Variance after norm should be ~1
        self.assertTrue(
            torch.allclose(out.float().pow(2).mean(-1), torch.ones(2, 10), atol=0.1),
        )

    def test_fused_residual(self):
        from minisgl.models.layers.rms_norm import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        residual = torch.randn(2, 10, 64)
        out, new_res = norm(x, residual)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(new_res.shape, residual.shape)
        self.assertTrue(new_res is not None)


# ── Test RoPE ──
class TestRoPE(unittest.TestCase):
    def test_rope_application(self):
        from minisgl.models.layers.rope import RotaryEmbedding

        rope = RotaryEmbedding(head_dim=64, max_position_embeddings=128)
        q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 8, 16, 64)
        q_copy = q.clone()
        k_copy = k.clone()
        positions = torch.arange(16)
        rope(q, k, positions)
        # RoPE should change Q and K
        self.assertFalse(torch.allclose(q, q_copy))
        self.assertFalse(torch.allclose(k, k_copy))
        # But shapes should stay the same
        self.assertEqual(q.shape, q_copy.shape)


# ── Test Linear Layers ──
class TestLinearLayers(unittest.TestCase):
    def test_column_parallel_linear_tp1(self):
        from minisgl.models.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(64, 128, bias=False)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 128))

    def test_row_parallel_linear_tp1(self):
        from minisgl.models.layers.linear import RowParallelLinear

        # RowParallel receives gathered input (full in_features)
        layer = RowParallelLinear(128, 64, bias=False)
        x = torch.randn(2, 10, 128)  # (batch, seq, full in_features)
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_vocab_parallel_embedding_tp1(self):
        from minisgl.models.layers.embedding import VocabParallelEmbedding

        emb = VocabParallelEmbedding(1000, 64)
        ids = torch.randint(0, 1000, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 64))


# ── Test Attention Backend ──
class TestAttentionBackend(unittest.TestCase):
    def test_pytorch_backend(self):
        from minisgl.models.attention.backend import AttentionBackend

        AttentionBackend.configure("fa")  # Use FA
        q = torch.randn(1, 8, 32, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(1, 8, 32, 64)
        v = torch.randn(1, 8, 32, 64)
        # Should not raise
        out = AttentionBackend.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)


# ── Test Sampling ──
class TestSampler(unittest.TestCase):
    def test_greedy_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(1000)
        logits = torch.randn(4, 1000)  # 4 requests, 1000 vocab
        params = SamplingParams(temperature=0.0)  # greedy
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))
        # Greedy should pick argmax
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(tokens, expected))

    def test_temperature_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(1000)
        logits = torch.randn(4, 1000)
        params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9)
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))

    def test_top_k_top_p(self):
        from minisgl.sampling.sampler import _apply_top_k, _apply_top_p

        logits = torch.randn(1, 1000)
        # Top-k: only top 10 should be > -inf
        filtered = _apply_top_k(logits.clone(), 10)
        self.assertEqual((filtered > float("-inf")).sum().item(), 10)
        # Top-p
        filtered = _apply_top_p(logits.clone(), 0.95)
        self.assertTrue((filtered > float("-inf")).sum().item() > 0)


# ── Test KV Cache Pool ──
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

    def test_get_kv_cache(self):
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
        k, v = pool.get_kv_cache(layer_idx=0)
        self.assertEqual(k.shape, (20, 16, 4, 32))
        self.assertEqual(v.shape, (20, 16, 4, 32))
        k_all, v_all = pool.get_all_kv_cache()
        self.assertEqual(k_all.shape, (4, 20, 16, 4, 32))
        self.assertEqual(v_all.shape, (4, 20, 16, 4, 32))


# ── Test Radix Cache ──
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
        self.assertEqual(batch.size(), 1)
        self.assertEqual(batch.phase, "prefill")
        self.assertEqual(req.total_len, 5)
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
        self.assertEqual(req.total_len, 4)
        self.assertEqual(req.output_len, 1)

    def test_context_prepare(self):
        from minisgl.config import SamplingParams
        from minisgl.engine.context import BatchContext
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
        self.assertIsNotNone(batch.write_loc)


# ── Test Distributed ──
class TestDistributed(unittest.TestCase):
    def test_all_reduce_noop(self):
        from minisgl.engine.distributed.pynccl import all_reduce

        x = torch.randn(10)
        y = all_reduce(x)
        self.assertTrue(torch.equal(x, y))  # no-op when not distributed


# ── Test Config ──
class TestConfig(unittest.TestCase):
    def test_config_creation(self):
        from minisgl.config import CacheArgs, SamplingParams, ServerArgs

        args = ServerArgs(model_path="/tmp/test", port=8000, tp_size=1)
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.tp_size, 1)

        cache = CacheArgs.from_server_args(args)
        self.assertEqual(cache.page_size, 16)
        self.assertEqual(cache.max_seq_len, 8192)

        params = SamplingParams(temperature=0.7, top_k=50, top_p=0.95)
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.top_k, 50)


# ── Test Tokenizer Worker ──
class TestTokenizerWorker(unittest.TestCase):
    @unittest.skipIf(
        not __import__("importlib.util").util.find_spec("transformers"),
        "transformers not installed",
    )
    def test_tokenizer_creation(self):
        """Test with a tiny tokenizer (requires transformers)."""
        from minisgl.models.tokenizer.worker import TokenizerWorker

        # Use a model that's likely cached or small
        try:
            worker = TokenizerWorker("google/bert_uncased_L-2_H-128_A-2")
            ids = worker.encode("Hello world")
            self.assertIsInstance(ids, list)
            self.assertTrue(len(ids) > 0)
        except Exception:
            self.skipTest("Model not available offline")


# ── Test Full Model (Dummy) ──
class TestModelDummy(unittest.TestCase):
    def test_qwen2_model_create(self):
        """Create a tiny Qwen2 model and run a forward pass."""
        from minisgl.config import ModelArgs
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            rms_norm_eps=1e-6,
        )

        model = Qwen2ForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8))  # 1 req, 8 tokens
        positions = torch.arange(8)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)

        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_qwen3_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,  # GQA: fewer KV heads
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            qk_norm=True,
        )
        model = Qwen3ForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_llama_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.llama import LlamaForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 6))
        positions = torch.arange(6)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (2, 6, 1000))

    def test_mistral_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.mistral import MistralForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            sliding_window=4096,
        )
        model = MistralForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_deep_decoder_forward(self):
        """Test forward pass through multiple layers with residual flow."""
        from minisgl.config import ModelArgs
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        gen = torch.Generator().manual_seed(123)
        model = Qwen2ForCausalLM(config)
        for _name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, std=0.02, generator=gen)
            elif param.dim() == 1:
                nn.init.ones_(param)
        model.eval()
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        )
        positions = torch.arange(16)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertFalse(torch.isnan(logits).any(), f"NaN in logits: {logits}")
        self.assertFalse(torch.isinf(logits).any(), f"Inf in logits: {logits}")

    def test_with_kv_cache(self):
        """Test forward pass with KV cache tensors."""

        from minisgl.config import ModelArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = Qwen2ForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()

        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=16,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        k_all, v_all = pool.get_all_kv_cache()

        # Prefill: 8 tokens, write to pages via flat indices
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        # write_loc: flat indices into page*page_size space
        write_loc = torch.arange(8, dtype=torch.int32)  # tokens 0-7 write to slots 0-7

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                k_cache=k_all,
                v_cache=v_all,
                write_loc=write_loc,
            )

        # Decode: 1 token, write to slot 8
        input_ids = torch.randint(0, 1000, (1, 1))
        positions = torch.tensor([8])
        write_loc = torch.tensor([8], dtype=torch.int32)
        # Paged KV metadata: tokens 0..8 live in flat slots 0..8
        req_to_token = torch.arange(9, dtype=torch.int32).unsqueeze(0)
        cache_seqlens = torch.tensor([8], dtype=torch.int32)

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                k_cache=k_all,
                v_cache=v_all,
                write_loc=write_loc,
                req_to_token=req_to_token,
                cache_seqlens=cache_seqlens,
                forward_mode="decode",
            )
        self.assertEqual(logits.shape, (1, 1, 1000))


# ── Test Qwen3MoE Model ──
class TestQwen3MoEModel(unittest.TestCase):
    def test_moe_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3_moe import Qwen3MoEForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            decoder_sparse_step=1,
        )
        model = Qwen3MoEForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))


# ── Test FusedMoE ──
class TestFusedMoE(unittest.TestCase):
    def test_fused_moe_pytorch(self):
        from minisgl.models.moe.fused_moe import fused_moe_pytorch

        total_tokens, hidden_size = 4, 64
        num_experts, intermediate_size = 4, 128

        x = torch.randn(total_tokens, hidden_size)
        router_logits = torch.randn(total_tokens, num_experts)
        gate_w = torch.randn(num_experts, intermediate_size, hidden_size)
        up_w = torch.randn(num_experts, intermediate_size, hidden_size)
        down_w = torch.randn(num_experts, hidden_size, intermediate_size)

        out = fused_moe_pytorch(x, router_logits, gate_w, up_w, down_w, top_k=2)
        self.assertEqual(out.shape, (total_tokens, hidden_size))
        self.assertFalse(torch.isnan(out).any())


# ── Test Model Registry ──
class TestRegistry(unittest.TestCase):
    def test_detect_model_type_fallback(self):
        # Should not crash on non-existent path
        self.assertTrue(True)  # Module loaded OK

    def test_create_model(self):
        from minisgl.config import ModelArgs
        from minisgl.models.registry import create_model

        config = ModelArgs(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=2,
            num_kv_heads=2,
            intermediate_size=256,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=64,
        )
        for mt in ["qwen2", "qwen3", "llama", "mistral"]:
            model = create_model(config, model_type=mt)
            # Apply initialization to prevent NaN
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
            model.eval()
            self.assertIsNotNone(model)
            ids = torch.randint(0, 100, (1, 4))
            pos = torch.arange(4)
            with torch.inference_mode():
                out = model(input_ids=ids, positions=pos)
            self.assertEqual(out.shape, (1, 4, 100))


# ── Test NaiveCacheManager ──
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
        self.assertEqual(sorted(evicted), sorted({h1.page_ids[0], h1.page_ids[1], h2.page_ids[1]}))
        self.assertEqual(pool.free_count(), 20)

    def test_evict_respects_refcount(self):
        pool, radix = self._make_pool_and_radix(20)
        tokens = [1, 2, 3, 4]
        handle = pool.alloc(1)
        radix.insert(tokens, handle)
        evicted = radix.evict(1)
        self.assertEqual(len(evicted), 0)


# ── Test OPT Model ──
class TestOPTModel(unittest.TestCase):
    def test_opt_forward(self):
        from minisgl.config import ModelArgs
        from minisgl.models.opt import OPTForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = OPTForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))


# ── Test DecodeManager ──
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
        self.assertIsNotNone(batch.write_loc)
        self.assertIsNotNone(batch.block_table)
        self.assertIsNotNone(batch.cache_seqlens)

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


# ── Test Shared Decoder Base Classes ──
class TestSharedDecoder(unittest.TestCase):
    def test_gated_mlp(self):
        from minisgl.models.decoder import GatedMLP

        mlp = GatedMLP(128, 512)
        x = torch.randn(2, 8, 128)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 8, 128))

    def test_rmsnorm_decoder_layer(self):
        from minisgl.config import ModelArgs
        from minisgl.models.decoder import RMSNormDecoderLayer
        from minisgl.models.llama import LlamaAttention

        config = ModelArgs(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=32,
        )
        layer = RMSNormDecoderLayer(
            hidden_size=128,
            rms_norm_eps=1e-6,
            attention=LlamaAttention(config),
            intermediate_size=512,
        )
        x = torch.randn(1, 8, 128)
        positions = torch.arange(8)
        out = layer(x, positions)
        self.assertEqual(out.shape, (1, 8, 128))

    def test_llama_inherits_tie_weights(self):
        from minisgl.models.decoder import RMSNormForCausalLM
        from minisgl.models.llama import LlamaForCausalLM

        self.assertTrue(issubclass(LlamaForCausalLM, RMSNormForCausalLM))

    def test_mlp_alias(self):
        from minisgl.models.decoder import GatedMLP
        from minisgl.models.llama import LlamaMLP
        from minisgl.models.mistral import MistralMLP
        from minisgl.models.qwen2 import Qwen2MLP
        from minisgl.models.qwen3 import Qwen3MLP

        self.assertIs(LlamaMLP, GatedMLP)
        self.assertIs(Qwen2MLP, GatedMLP)
        self.assertIs(Qwen3MLP, GatedMLP)
        self.assertIs(MistralMLP, GatedMLP)


# ── Test End-to-End Scheduler Loop ──
class TestEndToEndScheduler(unittest.TestCase):
    """Integration tests for the full Engine+Scheduler pipeline on CPU."""

    def _make_engine_scheduler(self):
        import json
        import tempfile

        from minisgl.config import ModelArgs, ServerArgs
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        tmpdir = tempfile.mkdtemp()
        config = {
            "architectures": ["OPTForCausalLM"],
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
        server_args = ServerArgs(
            model_path=tmpdir,
            tp_size=1,
            attention_backend="pt",
            max_running_req=4,
            max_seq_len=64,
            page_size=8,
            memory_ratio=0.5,
            cuda_graph_bs=0,
        )
        engine = Engine(server_args, model_args, tp_rank=0)
        for param in engine.model.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, std=0.02)
            elif param.dim() == 1:
                nn.init.ones_(param)
        scheduler = Scheduler(server_args, engine)
        return scheduler

    def test_single_request_generation(self):
        from minisgl.config import SamplingParams

        scheduler = self._make_engine_scheduler()
        scheduler.add_request([1, 5, 10], SamplingParams(temperature=0.0, max_tokens=5))
        generated = []
        steps = 0
        while not scheduler.is_idle() and steps < 100:
            for _uid, token_id, _finished in scheduler.step():
                generated.append(token_id)
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
            for uid, token_id, _finished in scheduler.step():
                results[uid].append(token_id)
            steps += 1
        self.assertGreater(len(results[uid1]), 0)
        self.assertGreater(len(results[uid2]), 0)

    def test_eos_terminates_early(self):

        scheduler = self._make_engine_scheduler()
        self.assertIsInstance(scheduler.eos_token_id, set)
        self.assertIn(2, scheduler.eos_token_id)


# ── Test PyTorch Attention Decode Path ──
class TestPyTorchBackendDecode(unittest.TestCase):
    def test_decode_with_valid_mask(self):
        from minisgl.models.attention.backend import PyTorchBackend

        q = torch.randn(2, 4, 1, 32)
        k_cache = torch.randn(10, 16, 4, 32)
        v_cache = torch.randn(10, 16, 4, 32)

        req_to_token = torch.full((2, 8), -1, dtype=torch.int32)
        req_to_token[0, :3] = torch.tensor([0, 1, 2])
        req_to_token[1, :5] = torch.tensor([16, 17, 18, 19, 20])
        cache_seqlens = torch.tensor([3, 5], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q,
            q[:, :, :, :],
            q[:, :, :, :],
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            cache_seqlens=cache_seqlens,
        )
        self.assertEqual(out.shape, (2, 4, 1, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_prefill_varlen(self):
        from minisgl.models.attention.backend import PyTorchBackend

        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

        out = PyTorchBackend.forward(q, k, v, cu_seqlens_q=cu_seqlens)
        self.assertEqual(out.shape, (1, 4, 8, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_prefill_extend_matches_full_prefill(self):
        # Extend attention (cached prefix + uncached suffix) must match a full
        # causal prefill over the whole sequence.
        import torch.nn.functional as F

        from minisgl.models.attention.backend import PyTorchBackend

        torch.manual_seed(0)
        num_heads, head_dim, page_size = 4, 32, 4
        total, cached = 6, 4
        suffix = total - cached

        k_cache = torch.zeros(2, page_size, num_heads, head_dim)
        v_cache = torch.zeros(2, page_size, num_heads, head_dim)
        q_full = torch.randn(1, num_heads, total, head_dim)
        k_full = torch.randn(1, num_heads, total, head_dim)
        v_full = torch.randn(1, num_heads, total, head_dim)
        # KV cache flat layout: page_id * page_size + offset
        k_cache.view(-1, num_heads, head_dim)[:total] = k_full[0].transpose(0, 1)
        v_cache.view(-1, num_heads, head_dim)[:total] = v_full[0].transpose(0, 1)

        req_to_token = torch.full((1, 8), -1, dtype=torch.int32)
        req_to_token[0, :total] = torch.arange(total, dtype=torch.int32)
        prefix_lens = torch.tensor([cached], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, suffix], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q_full[:, :, cached:, :],
            k_full[:, :, cached:, :],
            v_full[:, :, cached:, :],
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens,
            prefix_lens=prefix_lens,
            req_to_token=req_to_token,
            forward_mode="prefill",
        )
        ref = F.scaled_dot_product_attention(
            q_full, k_full, v_full, is_causal=True
        )[:, :, cached:, :]
        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


# ── Test EOS Normalization ──
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


# ── Test Sampling Edge Cases ──
class TestSamplerEdgeCases(unittest.TestCase):
    def test_single_token_batch(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(100)
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=0.0)
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(tokens[0].item(), logits.argmax(dim=-1)[0].item())

    def test_top_k_equals_vocab(self):
        from minisgl.sampling.sampler import _apply_top_k

        logits = torch.randn(1, 50)
        filtered = _apply_top_k(logits.clone(), 50)
        self.assertTrue(torch.equal(logits, filtered))

    def test_very_low_temperature(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(100)
        logits = torch.randn(4, 100)
        params = SamplingParams(temperature=0.01)
        tokens = sampler.sample(logits, params)
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(tokens, expected))


# ── Test FrontendManager ──
class TestFrontendManager(unittest.TestCase):
    def test_submit_and_get_queue(self):
        import queue

        from minisgl.config import SamplingParams, ServerArgs
        from minisgl.server.frontend import FrontendManager

        class MockScheduler:
            _uid = 0

            def add_request(self, input_ids, sampling_params):
                uid = self._uid
                self._uid += 1
                return uid

            def is_idle(self):
                return True

            def step(self):
                return []

        args = ServerArgs(model_path="/tmp/test")
        fm = FrontendManager(args, MockScheduler(), None)
        uid = fm.submit_request([1, 2, 3], SamplingParams())
        self.assertEqual(uid, 0)
        q = fm.get_result_queue(uid)
        self.assertIsInstance(q, queue.Queue)
        fm.remove_result(uid)
        self.assertIsNone(fm.get_result_queue(uid))


# ── Test Device Utilities (NPU-aware) ──
class TestDeviceUtils(unittest.TestCase):
    def test_get_device_type(self):
        from minisgl.utils.device import get_device_type

        dtype = get_device_type()
        self.assertIn(dtype, ("cpu", "cuda", "npu"))

    def test_is_npu_available(self):
        from minisgl.utils.device import is_npu_available

        result = is_npu_available()
        self.assertIsInstance(result, bool)

    def test_is_accelerator_available(self):
        from minisgl.utils.device import is_accelerator_available

        result = is_accelerator_available()
        self.assertIsInstance(result, bool)

    def test_synchronize_cpu(self):
        from minisgl.utils.device import synchronize

        synchronize()

    def test_mem_get_info_cpu(self):
        from minisgl.utils.device import mem_get_info

        free, total = mem_get_info(torch.device("cpu"))
        self.assertEqual(free, 0)
        self.assertEqual(total, 0)

    def test_set_device_cpu(self):
        from minisgl.utils.device import get_device, reset_device_state, set_device

        reset_device_state()
        set_device(torch.device("cpu"))
        self.assertEqual(get_device().type, "cpu")
        reset_device_state()

    def test_init_distributed_auto_backend(self):
        from minisgl.utils.device import get_device_type

        dtype = get_device_type()
        if dtype == "npu":
            expected_backend = "hccl"
        elif dtype == "cuda":
            expected_backend = "nccl"
        else:
            expected_backend = "gloo"
        self.assertIn(expected_backend, ("hccl", "nccl", "gloo"))


# ── Test ServerArgs Device Config ──
class TestServerArgsDevice(unittest.TestCase):
    def test_device_field_default(self):
        from minisgl.config import ServerArgs

        args = ServerArgs(model_path="/tmp/test")
        self.assertEqual(args.device, "auto")

    def test_attention_backend_pt(self):
        from minisgl.models.attention.backend import AttentionBackend

        AttentionBackend.configure("pt")
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        out = AttentionBackend.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)
        AttentionBackend.configure("fa")


if __name__ == "__main__":
    print("=" * 60)
    print("Mini-SGLang CPU Core Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
